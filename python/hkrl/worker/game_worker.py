"""GameWorker: the local sampling loop (PRD §8.1, invariant #1).

Runs entirely on the Game PC: local inference, env stepping, rollout buffering,
batch upload, checkpoint pulling, and crash/reconnect handling. The action loop
NEVER crosses the remote network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any

import numpy as np
import torch

from hkrl.models.base import ActorCritic
from hkrl.models.heads import ACTION_TENSOR_DIM_NO_MACRO
from hkrl.spaces import N_BUTTONS, action_mask_layout
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.worker.checkpoint_client import CheckpointClient


class GameWorker:
    """Owns one (or a few) HKRLEnv, a local policy, and a rollout buffer.

    Loop: ``act -> step -> buffer.add``; on full buffer upload a RolloutBatch; on a
    new checkpoint hot-swap weights (PRD Phase 6 milestone).
    """

    def __init__(
        self,
        env: Any,
        model: ActorCritic,
        config: TrainConfig,
        checkpoint_client: CheckpointClient | None = None,
        learner_endpoint: str | None = None,
        batch_uploader: Callable[[RolloutBatch], None] | None = None,
        heartbeat_sink: Callable[[dict[str, Any]], None] | None = None,
        task_provider: Callable[[], Any | None] | None = None,
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures < 0:
            raise ValueError("max_consecutive_failures must be non-negative")

        self.env = env
        self.model = model
        self.cfg = config
        self.checkpoint_client = checkpoint_client
        self.learner_endpoint = learner_endpoint
        self.batch_uploader = batch_uploader
        self.heartbeat_sink = heartbeat_sink
        self.task_provider = task_provider
        self.max_consecutive_failures = max_consecutive_failures
        self.device = _model_device(model)
        action_space: Any = env.action_space
        observation_space: Any = env.observation_space
        self.enable_macro = "macro" in action_space.spaces
        self.n_macros = int(action_space["macro"].n - 1) if self.enable_macro else 0
        self.action_dim = ACTION_TENSOR_DIM_NO_MACRO + (1 if self.enable_macro else 0)
        self.action_mask_dim = len(
            action_mask_layout(enable_macro=self.enable_macro, n_macros=self.n_macros)
        )
        obs_spec = {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
            "action": (self.action_dim,),
            "action_mask": (self.action_mask_dim,),
        }
        if config.algorithm == "recurrent_ppo":
            self.buffer: RolloutBuffer | RecurrentRolloutBuffer = RecurrentRolloutBuffer(
                capacity=config.rollout_steps,
                num_envs=1,
                sequence_length=config.sequence_length,
                burn_in=config.burn_in,
                obs_spec=obs_spec,
            )
        else:
            self.buffer = RolloutBuffer(
                capacity=config.rollout_steps,
                num_envs=1,
                obs_spec=obs_spec,
            )
        self.policy_version = 0
        self.checkpoint_version = -1
        self._obs: Any | None = None
        self._info: dict[str, Any] = {}
        self._rnn_state = model.initial_state(batch_size=1, device=self.device)
        self.last_batch: RolloutBatch | None = None
        self.worker_crash_count = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None

    def run(self, total_steps: int | None = None) -> None:
        """Sampling loop. Handles reset failures, reconnect, and weight reloads.

        A transient env/transport failure clears partial rollout state, emits a
        heartbeat with ``worker_crash_count``, reconnects when the env exposes a
        transport, and resumes sampling. Repeated failures eventually surface to
        avoid an infinite retry loop.
        """
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps must be positive")

        steps = 0
        while total_steps is None or steps < total_steps:
            try:
                batch = self.collect_rollout()
                self.last_batch = batch
                self._upload_batch(batch)
                self._emit_heartbeat(batch)
                self.consecutive_failures = 0
                self.last_error = None
                steps += int(batch.rewards.size)
            except Exception as exc:
                self._handle_runtime_failure(exc)

    def collect_rollout(self) -> RolloutBatch:
        """Fill one flat rollout and return a GAE-ready RolloutBatch."""
        self.buffer.clear()
        self._maybe_reload_checkpoint()
        self._maybe_switch_task()
        self.model.eval()
        self._ensure_reset()

        for _ in range(self.cfg.rollout_steps):
            assert self._obs is not None
            obs = self._obs
            info = self._info
            action_mask = self._action_mask(info)
            obs_tensor = _obs_to_tensor(obs, self.device)
            action_mask_tensor = torch.as_tensor(
                action_mask[None, :],
                dtype=torch.bool,
                device=self.device,
            )
            rnn_state = self._rnn_state
            action, log_prob, value, next_rnn_state = self.model.act(
                obs_tensor,
                rnn_state=rnn_state,
                action_mask=action_mask_tensor,
            )
            env_action = action_tensor_to_env_action(action[0], enable_macro=self.enable_macro)
            next_obs, reward, terminated, truncated, next_info = self.env.step(env_action)

            self.buffer.add(
                obs=obs,
                action=action.detach().cpu().numpy(),
                log_prob=log_prob.detach().cpu().numpy(),
                value=value.detach().cpu().numpy(),
                reward=np.array([reward], dtype=np.float32),
                done=np.array([terminated], dtype=bool),
                truncated=np.array([truncated], dtype=bool),
                action_mask=action_mask,
                rnn_state=rnn_state,
                episode_id=np.array([int(info.get("episode_id", 0))], dtype=np.uint64),
                task_id=np.array([int(info.get("task_id", 0))], dtype=np.int64),
            )

            self._obs = next_obs
            self._info = next_info
            self._rnn_state = next_rnn_state
            if terminated or truncated:
                self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
                if self.buffer.is_full():
                    self._obs = None
                    self._info = {}
                else:
                    self._reset()

        last_value = self._bootstrap_value()
        self.buffer.compute_returns(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        return self.buffer.to_batch(policy_version=self.policy_version)

    def _maybe_reload_checkpoint(self) -> bool:
        if self.checkpoint_client is None:
            return False

        latest_version = self.checkpoint_client.latest_version()
        if latest_version < 0 or latest_version <= self.checkpoint_version:
            return False

        state = self.checkpoint_client.pull(latest_version)
        model_state = state.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint missing model_state_dict")
        self.model.load_state_dict(model_state)
        self.checkpoint_version = latest_version
        policy_version = state.get("policy_version", latest_version)
        if not isinstance(policy_version, Integral):
            raise ValueError("checkpoint policy_version must be an integer")
        self.policy_version = int(policy_version)
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
        return True

    def _maybe_switch_task(self) -> bool:
        if self.task_provider is None:
            return False

        task = self.task_provider()
        if task is None:
            return False

        current_task = _find_attr(self.env, "task")
        current_wire_id = getattr(current_task, "wire_id", None)
        next_wire_id = getattr(task, "wire_id", None)
        if next_wire_id is not None and current_wire_id == next_wire_id:
            return False

        set_task = _find_set_task(self.env)
        if set_task is None:
            raise RuntimeError("task_provider requires env.set_task(task)")

        self._obs, self._info = set_task(task)
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
        return True

    def _upload_batch(self, batch: RolloutBatch) -> None:
        if self.batch_uploader is not None:
            self.batch_uploader(batch)

    def _emit_heartbeat(self, batch: RolloutBatch) -> None:
        if self.heartbeat_sink is None:
            return
        self.heartbeat_sink(
            {
                "checkpoint_version": self.checkpoint_version,
                "learner_endpoint": self.learner_endpoint,
                "policy_version": self.policy_version,
                "rollout_steps": int(batch.rewards.size),
                "status": "running",
                "worker_crash_count": self.worker_crash_count,
            }
        )

    def _emit_crash_heartbeat(self, exc: Exception) -> None:
        if self.heartbeat_sink is None:
            return
        self.heartbeat_sink(
            {
                "checkpoint_version": self.checkpoint_version,
                "error": f"{type(exc).__name__}: {exc}",
                "learner_endpoint": self.learner_endpoint,
                "policy_version": self.policy_version,
                "rollout_steps": 0,
                "status": "recovering",
                "worker_crash_count": self.worker_crash_count,
            }
        )

    def _handle_runtime_failure(self, exc: Exception) -> None:
        self.worker_crash_count += 1
        self.consecutive_failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        self._emit_crash_heartbeat(exc)
        if self.consecutive_failures > self.max_consecutive_failures:
            raise RuntimeError(
                f"game worker exceeded max_consecutive_failures={self.max_consecutive_failures}"
            ) from exc
        self._recover_env()

    def _recover_env(self) -> None:
        self.buffer.clear()
        self._obs = None
        self._info = {}
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)

        reconnect = _find_reconnect(self.env)
        if reconnect is not None:
            reconnect()

    def _ensure_reset(self) -> None:
        if self._obs is None:
            self._reset()

    def _reset(self) -> None:
        self._obs, self._info = self.env.reset()
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)

    def _action_mask(self, info: dict[str, Any]) -> np.ndarray:
        action_mask = info.get("action_mask")
        if action_mask is None:
            return np.ones((self.action_mask_dim,), dtype=bool)

        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape != (self.action_mask_dim,):
            raise ValueError(
                f"action_mask shape must be ({self.action_mask_dim},), got {mask.shape}"
            )
        return mask

    def _bootstrap_value(self) -> np.ndarray:
        if self._obs is None:
            return np.zeros((1,), dtype=np.float32)

        with torch.no_grad():
            _, value, _ = self.model.forward(
                _obs_to_tensor(self._obs, self.device),
                rnn_state=self._rnn_state,
                action_mask=torch.as_tensor(
                    self._action_mask(self._info)[None, :],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        return value.detach().cpu().numpy().reshape(1).astype(np.float32)


def action_tensor_to_env_action(
    action: torch.Tensor | np.ndarray, *, enable_macro: bool
) -> dict[str, Any]:
    values = np.asarray(action.detach().cpu() if isinstance(action, torch.Tensor) else action)
    values = values.reshape(-1).astype(np.int64, copy=False)
    expected_dim = ACTION_TENSOR_DIM_NO_MACRO + (1 if enable_macro else 0)
    if values.shape != (expected_dim,):
        raise ValueError(f"action tensor shape must be ({expected_dim},), got {values.shape}")

    offset = 0
    movement_x = int(values[offset])
    offset += 1
    aim_y = int(values[offset])
    offset += 1
    buttons = values[offset : offset + N_BUTTONS].astype(np.int8, copy=True)
    offset += N_BUTTONS
    duration = int(values[offset])
    offset += 1

    env_action: dict[str, Any] = {
        "movement_x": movement_x,
        "aim_y": aim_y,
        "buttons": buttons,
        "duration": duration,
    }
    if enable_macro:
        env_action["macro"] = int(values[offset])
    return env_action


def _obs_to_tensor(obs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "global": torch.as_tensor(obs["global"][None, :], dtype=torch.float32, device=device),
        "player": torch.as_tensor(obs["player"][None, :], dtype=torch.float32, device=device),
        "entities": torch.as_tensor(
            obs["entities"][None, :, :], dtype=torch.float32, device=device
        ),
        "entity_mask": torch.as_tensor(
            obs["entity_mask"][None, :], dtype=torch.bool, device=device
        ),
    }


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _find_reconnect(env: Any) -> Callable[[], None] | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        transport = getattr(current, "transport", None)
        transport_reconnect = getattr(transport, "reconnect", None)
        if callable(transport_reconnect):
            return lambda: transport_reconnect(timeout_s=10.0)

        env_reconnect = getattr(current, "reconnect", None)
        if callable(env_reconnect):
            return env_reconnect

        current = getattr(current, "env", None)
    return None


def _find_set_task(env: Any) -> Callable[[Any], tuple[Any, dict[str, Any]]] | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        set_task = getattr(current, "set_task", None)
        if callable(set_task):
            return set_task
        current = getattr(current, "env", None)
    return None


def _find_attr(env: Any, name: str) -> Any | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "env", None)
    return None
