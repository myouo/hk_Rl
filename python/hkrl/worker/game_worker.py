"""GameWorker: the local sampling loop (PRD §8.1, invariant #1).

Runs entirely on the Game PC: local inference, env stepping, rollout buffering,
batch upload, checkpoint pulling, and crash/reconnect handling. The action loop
NEVER crosses the remote network.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any

import numpy as np
import torch

from hkrl.models.base import ActorCritic
from hkrl.models.heads import ACTION_TENSOR_DIM_NO_MACRO
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X, action_mask_layout
from hkrl.training.numerics import require_finite_tensor
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
        batch_uploader: Callable[[RolloutBatch], bool | None] | None = None,
        heartbeat_sink: Callable[[dict[str, Any]], None] | None = None,
        task_provider: Callable[[], Any | None] | None = None,
        max_consecutive_failures: int = 3,
        clock: Callable[[], float] | None = None,
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
        self._clock = clock or time.monotonic
        self.device = _model_device(model)
        initial_rnn_state = model.initial_state(batch_size=1, device=self.device)
        uses_recurrent_state = initial_rnn_state is not None
        if config.algorithm == "appo" and isinstance(initial_rnn_state, tuple):
            raise ValueError("APPO worker rollout upload supports tensor/GRU rnn_state, not LSTM")
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
        if config.algorithm == "recurrent_ppo" or (
            config.algorithm == "appo" and uses_recurrent_state
        ):
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
        self._rnn_state = initial_rnn_state
        self._prev_action: np.ndarray = np.zeros((1, self.action_dim), dtype=np.int64)
        self._prev_reward: np.ndarray = np.zeros((1,), dtype=np.float32)
        self.last_batch: RolloutBatch | None = None
        self.worker_crash_count = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.last_rollout_duration_s = 0.0
        self.last_sps = 0.0
        self.learner_upload_submitted_batches = 0
        self.learner_upload_accepted_batches = 0
        self.learner_upload_rejected_batches = 0
        self.learner_upload_failed_batches = 0

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
        started_at = self._clock()
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
            obs_tensor["prev_action"] = torch.as_tensor(
                self._prev_action,
                dtype=torch.float32,
                device=self.device,
            )
            obs_tensor["prev_reward"] = torch.as_tensor(
                self._prev_reward,
                dtype=torch.float32,
                device=self.device,
            )
            rnn_state = self._rnn_state
            with torch.no_grad():
                action, log_prob, value, next_rnn_state = self.model.act(
                    obs_tensor,
                    rnn_state=rnn_state,
                    action_mask=action_mask_tensor,
                )
            require_finite_tensor("worker log_prob", log_prob)
            require_finite_tensor("worker value", value)
            action_array = action.detach().cpu().numpy()
            env_action = action_tensor_to_env_action(
                action[0],
                enable_macro=self.enable_macro,
                n_macros=self.n_macros,
                action_mask=action_mask,
            )
            next_obs, reward, terminated, truncated, next_info = self.env.step(env_action)

            self.buffer.add(
                obs=obs,
                action=action_array,
                log_prob=log_prob.detach().cpu().numpy(),
                value=value.detach().cpu().numpy(),
                reward=np.array([reward], dtype=np.float32),
                done=np.array([terminated], dtype=bool),
                truncated=np.array([truncated], dtype=bool),
                action_mask=action_mask,
                prev_action=self._prev_action,
                prev_reward=self._prev_reward,
                rnn_state=rnn_state,
                episode_id=np.array([int(info.get("episode_id", 0))], dtype=np.uint64),
                task_id=np.array([int(info.get("task_id", 0))], dtype=np.int64),
            )

            self._obs = next_obs
            self._info = next_info
            self._rnn_state = next_rnn_state
            self._prev_action = action_array.astype(np.int64, copy=True)
            self._prev_reward = np.array([reward], dtype=np.float32)
            if terminated or truncated:
                self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
                self._clear_memory_context()
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
        batch = self.buffer.to_batch(policy_version=self.policy_version)
        self._record_rollout_timing(batch, started_at)
        return batch

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
        if _task_identity(current_task) == _task_identity(task):
            return False

        set_task = _find_set_task(self.env)
        if set_task is None:
            raise RuntimeError("task_provider requires env.set_task(task)")

        self._obs, self._info = set_task(task)
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
        self._clear_memory_context()
        return True

    def _upload_batch(self, batch: RolloutBatch) -> None:
        if self.batch_uploader is None:
            return

        track_learner_upload = self.learner_endpoint is not None
        if track_learner_upload:
            self.learner_upload_submitted_batches += 1
        try:
            accepted = self.batch_uploader(batch)
        except Exception:
            if track_learner_upload:
                self.learner_upload_failed_batches += 1
            raise
        if not track_learner_upload:
            return
        if not isinstance(accepted, bool):
            self.learner_upload_failed_batches += 1
            raise ValueError("batch_uploader must return a bool when learner_endpoint is set")
        if accepted:
            self.learner_upload_accepted_batches += 1
        else:
            self.learner_upload_rejected_batches += 1

    def _emit_heartbeat(self, batch: RolloutBatch) -> None:
        if self.heartbeat_sink is None:
            return
        self.heartbeat_sink(
            {
                "checkpoint_version": self.checkpoint_version,
                "learner_endpoint": self.learner_endpoint,
                **self._learner_upload_metrics(),
                "policy_version": self.policy_version,
                "rollout_duration_s": self.last_rollout_duration_s,
                "rollout_steps": int(batch.rewards.size),
                "sps": self.last_sps,
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
                **self._learner_upload_metrics(),
                "policy_version": self.policy_version,
                "rollout_duration_s": 0.0,
                "rollout_steps": 0,
                "sps": 0.0,
                "status": "recovering",
                "worker_crash_count": self.worker_crash_count,
            }
        )

    def _learner_upload_metrics(self) -> dict[str, int]:
        return {
            "learner_upload_accepted_batches": self.learner_upload_accepted_batches,
            "learner_upload_failed_batches": self.learner_upload_failed_batches,
            "learner_upload_rejected_batches": self.learner_upload_rejected_batches,
            "learner_upload_submitted_batches": self.learner_upload_submitted_batches,
        }

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
        self._clear_memory_context()

        reconnect = _find_reconnect(self.env)
        if reconnect is not None:
            reconnect()

    def _ensure_reset(self) -> None:
        if self._obs is None:
            self._reset()

    def _reset(self) -> None:
        self._obs, self._info = self.env.reset()
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
        self._clear_memory_context()

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
                _obs_to_tensor(
                    self._obs,
                    self.device,
                    prev_action=self._prev_action,
                    prev_reward=self._prev_reward,
                ),
                rnn_state=self._rnn_state,
                action_mask=torch.as_tensor(
                    self._action_mask(self._info)[None, :],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        require_finite_tensor("bootstrap value", value)
        return value.detach().cpu().numpy().reshape(1).astype(np.float32)

    def _clear_memory_context(self) -> None:
        self._prev_action = np.zeros((1, self.action_dim), dtype=np.int64)
        self._prev_reward = np.zeros((1,), dtype=np.float32)

    def _record_rollout_timing(self, batch: RolloutBatch, started_at: float) -> None:
        duration = max(0.0, self._clock() - started_at)
        self.last_rollout_duration_s = duration
        self.last_sps = float(batch.rewards.size) / duration if duration > 0.0 else 0.0


def action_tensor_to_env_action(
    action: torch.Tensor | np.ndarray,
    *,
    enable_macro: bool,
    n_macros: int = 0,
    action_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if n_macros < 0:
        raise ValueError("n_macros must be non-negative")

    raw_values = np.asarray(action.detach().cpu() if isinstance(action, torch.Tensor) else action)
    flat_values = raw_values.reshape(-1)
    expected_dim = ACTION_TENSOR_DIM_NO_MACRO + (1 if enable_macro else 0)
    if flat_values.shape != (expected_dim,):
        raise ValueError(f"action tensor shape must be ({expected_dim},), got {flat_values.shape}")
    try:
        finite = np.isfinite(flat_values)
    except TypeError as exc:
        raise ValueError("action tensor must contain numeric values") from exc
    if not finite.all():
        raise ValueError("action tensor contains non-finite values")
    if not np.equal(flat_values, np.trunc(flat_values)).all():
        raise ValueError("action tensor values must be integer-coded")

    values = flat_values.astype(np.int64, copy=False)

    offset = 0
    movement_x = int(values[offset])
    _require_discrete_range("movement_x", movement_x, N_MOVEMENT_X)
    offset += 1
    aim_y = int(values[offset])
    _require_discrete_range("aim_y", aim_y, N_AIM_Y)
    offset += 1
    buttons = values[offset : offset + N_BUTTONS].astype(np.int8, copy=True)
    if not np.logical_or(buttons == 0, buttons == 1).all():
        raise ValueError("button action values must be binary")
    offset += N_BUTTONS
    duration = int(values[offset])
    _require_discrete_range("duration", duration, N_DURATION)
    offset += 1

    env_action: dict[str, Any] = {
        "movement_x": movement_x,
        "aim_y": aim_y,
        "buttons": buttons,
        "duration": duration,
    }
    if enable_macro:
        macro = int(values[offset])
        _require_discrete_range("macro", macro, n_macros + 1)
        env_action["macro"] = macro
    if action_mask is not None:
        _require_action_mask_allows(values, action_mask, enable_macro, n_macros)
    return env_action


def _require_discrete_range(name: str, value: int, size: int) -> None:
    if value < 0 or value >= size:
        raise ValueError(f"{name} must be in [0, {size}), got {value}")


def _require_action_mask_allows(
    values: np.ndarray,
    action_mask: np.ndarray,
    enable_macro: bool,
    n_macros: int,
) -> None:
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    expected_dim = N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION
    if enable_macro:
        expected_dim += n_macros + 1
    if mask.shape != (expected_dim,):
        raise ValueError(f"action_mask shape must be ({expected_dim},), got {mask.shape}")

    offset = 0
    movement_x = int(values[offset])
    if not mask[offset + movement_x]:
        raise ValueError(f"action_mask disallows movement_x={movement_x}")
    offset += N_MOVEMENT_X

    aim_y = int(values[1])
    if not mask[offset + aim_y]:
        raise ValueError(f"action_mask disallows aim_y={aim_y}")
    offset += N_AIM_Y

    button_values = values[2 : 2 + N_BUTTONS]
    button_mask = mask[offset : offset + N_BUTTONS]
    blocked_buttons = np.nonzero((button_values == 1) & ~button_mask)[0]
    if blocked_buttons.size:
        raise ValueError(f"action_mask disallows button index {int(blocked_buttons[0])}")
    offset += N_BUTTONS

    duration = int(values[2 + N_BUTTONS])
    if not mask[offset + duration]:
        raise ValueError(f"action_mask disallows duration={duration}")
    offset += N_DURATION

    if enable_macro:
        macro = int(values[3 + N_BUTTONS])
        if not mask[offset + macro]:
            raise ValueError(f"action_mask disallows macro={macro}")


def _obs_to_tensor(
    obs: Any,
    device: torch.device,
    *,
    prev_action: np.ndarray | None = None,
    prev_reward: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    tensors = {
        "global": torch.as_tensor(obs["global"][None, :], dtype=torch.float32, device=device),
        "player": torch.as_tensor(obs["player"][None, :], dtype=torch.float32, device=device),
        "entities": torch.as_tensor(
            obs["entities"][None, :, :], dtype=torch.float32, device=device
        ),
        "entity_mask": torch.as_tensor(
            obs["entity_mask"][None, :], dtype=torch.bool, device=device
        ),
    }
    if prev_action is not None:
        tensors["prev_action"] = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
    if prev_reward is not None:
        tensors["prev_reward"] = torch.as_tensor(prev_reward, dtype=torch.float32, device=device)
    return tensors


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


def _task_identity(task: Any) -> tuple[Any, ...] | None:
    if task is None:
        return None
    return (
        getattr(task, "task_id", None),
        getattr(task, "wire_id", None),
        getattr(task, "scene", None),
        getattr(task, "difficulty", None),
    )
