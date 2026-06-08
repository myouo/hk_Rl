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
    ) -> None:
        self.env = env
        self.model = model
        self.cfg = config
        self.checkpoint_client = checkpoint_client
        self.learner_endpoint = learner_endpoint
        self.batch_uploader = batch_uploader
        self.heartbeat_sink = heartbeat_sink
        self.device = _model_device(model)
        action_space: Any = env.action_space
        observation_space: Any = env.observation_space
        self.enable_macro = "macro" in action_space.spaces
        self.n_macros = int(action_space["macro"].n - 1) if self.enable_macro else 0
        self.action_dim = ACTION_TENSOR_DIM_NO_MACRO + (1 if self.enable_macro else 0)
        self.action_mask_dim = len(
            action_mask_layout(enable_macro=self.enable_macro, n_macros=self.n_macros)
        )
        self.buffer = RolloutBuffer(
            capacity=config.rollout_steps,
            num_envs=1,
            obs_spec={
                "global": observation_space["global"].shape,
                "player": observation_space["player"].shape,
                "entities": observation_space["entities"].shape,
                "entity_mask": observation_space["entity_mask"].shape,
                "action": (self.action_dim,),
                "action_mask": (self.action_mask_dim,),
            },
        )
        self.policy_version = 0
        self.checkpoint_version = -1
        self._obs: Any | None = None
        self._info: dict[str, Any] = {}
        self._rnn_state = model.initial_state(batch_size=1, device=self.device)
        self.last_batch: RolloutBatch | None = None

    def run(self, total_steps: int | None = None) -> None:
        """Sampling loop. Handles reset failures, reconnect, and weight reloads.

        Phase 3 keeps this local-only and stores the most recent batch on
        ``last_batch``. Phase 6 wires upload and checkpoint reload.
        """
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps must be positive")

        steps = 0
        while total_steps is None or steps < total_steps:
            batch = self.collect_rollout()
            self.last_batch = batch
            self._upload_batch(batch)
            self._emit_heartbeat(batch)
            steps += int(batch.rewards.size)

    def collect_rollout(self) -> RolloutBatch:
        """Fill one flat rollout and return a GAE-ready RolloutBatch."""
        self.buffer.clear()
        self._maybe_reload_checkpoint()
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
            action, log_prob, value, self._rnn_state = self.model.act(
                obs_tensor,
                rnn_state=self._rnn_state,
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
                episode_id=np.array([int(info.get("episode_id", 0))], dtype=np.uint64),
                task_id=np.array([int(info.get("task_id", 0))], dtype=np.int64),
            )

            self._obs = next_obs
            self._info = next_info
            if terminated or truncated:
                self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
                if not self.buffer.is_full():
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
            }
        )

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
