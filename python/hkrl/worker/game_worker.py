"""GameWorker: the local sampling loop (PRD §8.1, invariant #1).

Runs entirely on the Game PC: local inference, env stepping, rollout buffering,
batch upload, checkpoint pulling, and crash/reconnect handling. The action loop
NEVER crosses the remote network.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

from hkrl.models.base import ActorCritic
from hkrl.models.heads import ACTION_TENSOR_DIM_NO_MACRO
from hkrl.spaces import (
    N_AIM_Y,
    N_BUTTONS,
    N_DURATION,
    N_MOVEMENT_X,
    action_mask_layout,
    canonical_noop_action_values,
)
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.utils.live_tuning import LiveTuning
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
        checkpoint_poll_interval_s: float = 2.0,
        time_scale: float | None = None,
    ) -> None:
        if max_consecutive_failures < 0:
            raise ValueError("max_consecutive_failures must be non-negative")
        if time_scale is not None and (
            isinstance(time_scale, bool)
            or not isinstance(time_scale, Real)
            or not math.isfinite(float(time_scale))
            or float(time_scale) <= 0.0
        ):
            raise ValueError("time_scale must be a positive finite number")
        if (
            isinstance(checkpoint_poll_interval_s, bool)
            or not math.isfinite(float(checkpoint_poll_interval_s))
            or float(checkpoint_poll_interval_s) <= 0.0
        ):
            raise ValueError("checkpoint_poll_interval_s must be positive and finite")

        self.env = env
        self.model = model
        self.cfg = config
        self.checkpoint_client = checkpoint_client
        self.learner_endpoint = learner_endpoint
        self.batch_uploader = batch_uploader
        self.heartbeat_sink = heartbeat_sink
        self.task_provider = task_provider
        self.max_consecutive_failures = max_consecutive_failures
        self.checkpoint_poll_interval_s = float(checkpoint_poll_interval_s)
        self._base_time_scale: float | None = None if time_scale is None else float(time_scale)
        self.time_scale: float | None = None if time_scale is None else float(time_scale)
        self._time_scale_needs_apply = self.time_scale is not None
        self.live_tuning: LiveTuning | None = None
        self.tuning_version = 0
        self._checkpoint_lock = threading.Lock()
        self._pending_checkpoint: tuple[int, dict[str, object]] | None = None
        self._checkpoint_monitor_stop = threading.Event()
        self._checkpoint_monitor_thread: threading.Thread | None = None
        self._checkpoint_monitor_seen_version = -1
        self.last_checkpoint_poll_error: str | None = None
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
        self._prev_action = np.asarray(
            [canonical_noop_action_values(enable_macro=self.enable_macro)],
            dtype=np.int64,
        )
        self._prev_reward: np.ndarray = np.zeros((1,), dtype=np.float32)
        self.last_batch: RolloutBatch | None = None
        self.worker_crash_count = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.last_rollout_duration_s = 0.0
        self.last_sps = 0.0
        self.arena_auto_reset_count = 0
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
        self._start_checkpoint_monitor()
        try:
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
        finally:
            self._stop_checkpoint_monitor()

    def collect_rollout(self) -> RolloutBatch:
        """Fill one flat rollout and return a GAE-ready RolloutBatch."""
        started_at = self._clock()
        self.buffer.clear()
        self._maybe_reload_checkpoint()
        self._maybe_switch_task()
        self.model.eval()
        self._ensure_reset()

        collected = 0
        while collected < self.cfg.rollout_steps:
            if collected > 0 and self._maybe_apply_pending_tuning():
                # The prefix was collected under a different objective/version.
                # Discard it instead of mixing definitions in one learner batch.
                self.buffer.clear()
                collected = 0
                self._obs = None
                self._info = {}
                self._rnn_state = self.model.initial_state(
                    batch_size=1,
                    device=self.device,
                )
                self._clear_memory_context()
                self._ensure_reset()
                continue

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
            with torch.inference_mode():
                action, log_prob, value, next_rnn_state = self.model.act(
                    obs_tensor,
                    rnn_state=rnn_state,
                    action_mask=action_mask_tensor,
                )
            action_array, log_prob_array, value_array = _policy_outputs_to_numpy(
                action,
                log_prob,
                value,
                action_dim=self.action_dim,
            )
            env_action = action_tensor_to_env_action(
                action_array[0],
                enable_macro=self.enable_macro,
                n_macros=self.n_macros,
                action_mask=action_mask,
            )
            next_obs, reward, terminated, truncated, next_info = self.env.step(env_action)
            discount_exponent = _discount_exponent(self.env, next_info)
            truncation_value = (
                self._observation_value(
                    next_obs,
                    next_info,
                    next_rnn_state,
                    action_array,
                    np.array([reward], dtype=np.float32),
                    name="truncation value",
                )
                if truncated
                else np.zeros((1,), dtype=np.float32)
            )

            self.buffer.add(
                obs=obs,
                action=action_array,
                log_prob=log_prob_array,
                value=value_array,
                reward=np.array([reward], dtype=np.float32),
                done=np.array([terminated], dtype=bool),
                truncated=np.array([truncated], dtype=bool),
                action_mask=action_mask,
                prev_action=self._prev_action,
                prev_reward=self._prev_reward,
                rnn_state=rnn_state,
                episode_id=np.array([int(info.get("episode_id", 0))], dtype=np.uint64),
                task_id=np.array([int(info.get("task_id", 0))], dtype=np.int64),
                discount_exponent=np.array([discount_exponent], dtype=np.float32),
                truncation_value=truncation_value,
            )

            self._obs = next_obs
            self._info = next_info
            self._rnn_state = next_rnn_state
            self._prev_action = action_array.astype(np.int64, copy=True)
            self._prev_reward = np.array([reward], dtype=np.float32)
            if terminated or truncated:
                self.arena_auto_reset_count += 1
                self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
                self._clear_memory_context()
                if self.buffer.is_full():
                    self._obs = None
                    self._info = {}
                else:
                    self._reset()
            collected += 1

        last_value = self._bootstrap_value()
        self.buffer.compute_returns(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        batch = self.buffer.to_batch(
            policy_version=self.policy_version,
            tuning_version=self.tuning_version,
        )
        self._record_rollout_timing(batch, started_at)
        return batch

    def _maybe_reload_checkpoint(self) -> bool:
        if self.checkpoint_client is None:
            return False

        pending = self._peek_pending_checkpoint()
        latest_version = self.checkpoint_client.latest_version()
        pending_version = -1 if pending is None else pending[0]
        selected_version = max(latest_version, pending_version)
        if selected_version < 0 or selected_version <= self.checkpoint_version:
            return False

        state = (
            pending[1]
            if pending is not None and pending[0] == selected_version
            else self.checkpoint_client.pull(selected_version)
        )
        self._discard_pending_checkpoint(selected_version)
        return self._load_checkpoint_state(selected_version, state)

    def _load_checkpoint_state(
        self,
        checkpoint_version: int,
        state: Mapping[str, object],
    ) -> bool:
        model_state = state.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint missing model_state_dict")
        self.model.load_state_dict(model_state)
        _, tuning = _checkpoint_tuning(state)
        if tuning is not None:
            self._apply_live_tuning(tuning)
        self.checkpoint_version = checkpoint_version
        policy_version = state.get("policy_version", checkpoint_version)
        if not isinstance(policy_version, Integral):
            raise ValueError("checkpoint policy_version must be an integer")
        self.policy_version = int(policy_version)
        self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
        return True

    def _maybe_apply_pending_tuning(self) -> bool:
        pending = self._peek_pending_checkpoint()
        if pending is None:
            return False
        checkpoint_version, state = pending
        tuning_version, _ = _checkpoint_tuning(state)
        if tuning_version <= self.tuning_version:
            return False
        self._discard_pending_checkpoint(checkpoint_version)
        return self._load_checkpoint_state(checkpoint_version, state)

    def _peek_pending_checkpoint(self) -> tuple[int, dict[str, object]] | None:
        with self._checkpoint_lock:
            return self._pending_checkpoint

    def _discard_pending_checkpoint(self, through_version: int) -> None:
        with self._checkpoint_lock:
            pending = self._pending_checkpoint
            if pending is not None and pending[0] <= through_version:
                self._pending_checkpoint = None

    def _queue_checkpoint(
        self,
        checkpoint_version: int,
        state: dict[str, object],
    ) -> None:
        with self._checkpoint_lock:
            pending = self._pending_checkpoint
            if pending is None or checkpoint_version > pending[0]:
                self._pending_checkpoint = (checkpoint_version, state)

    def _start_checkpoint_monitor(self) -> None:
        if self.checkpoint_client is None:
            return
        thread = self._checkpoint_monitor_thread
        if thread is not None and thread.is_alive():
            return
        self._checkpoint_monitor_stop.clear()
        self._checkpoint_monitor_seen_version = self.checkpoint_version
        thread = threading.Thread(
            target=self._checkpoint_monitor_loop,
            name="hkrl-checkpoint-monitor",
            daemon=True,
        )
        self._checkpoint_monitor_thread = thread
        thread.start()

    def _stop_checkpoint_monitor(self) -> None:
        self._checkpoint_monitor_stop.set()
        thread = self._checkpoint_monitor_thread
        self._checkpoint_monitor_thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def _checkpoint_monitor_loop(self) -> None:
        assert self.checkpoint_client is not None
        while not self._checkpoint_monitor_stop.wait(self.checkpoint_poll_interval_s):
            try:
                version = self.checkpoint_client.latest_version()
                if version <= max(
                    self.checkpoint_version,
                    self._checkpoint_monitor_seen_version,
                ):
                    self.last_checkpoint_poll_error = None
                    continue
                state = self.checkpoint_client.pull(version)
                self._queue_checkpoint(version, state)
                self._checkpoint_monitor_seen_version = version
                self.last_checkpoint_poll_error = None
            except Exception as exc:
                self.last_checkpoint_poll_error = f"{type(exc).__name__}: {exc}"

    def _apply_live_tuning(self, tuning: LiveTuning) -> bool:
        if tuning.version < self.tuning_version:
            raise ValueError(
                f"checkpoint tuning version regressed from {self.tuning_version} "
                f"to {tuning.version}"
            )
        if tuning.version == self.tuning_version:
            if self.live_tuning is not None and tuning.digest != self.live_tuning.digest:
                raise ValueError(f"tuning version {tuning.version} changed content")
            return False

        previous_reward = (
            {}
            if self.live_tuning is None
            else self.live_tuning.reward.model_dump(exclude_none=True)
        )
        reward_overrides = tuning.reward.model_dump(exclude_none=True)
        if reward_overrides != previous_reward:
            set_reward_overrides = _find_attr(self.env, "set_reward_overrides")
            if not callable(set_reward_overrides):
                raise RuntimeError(
                    "live reward tuning requires env.set_reward_overrides(overrides)"
                )
            set_reward_overrides(reward_overrides)
            # Do not let one episode straddle two reward definitions.
            self._obs = None
            self._info = {}
            self._rnn_state = self.model.initial_state(batch_size=1, device=self.device)
            self._clear_memory_context()

        previous_scale = self.time_scale
        configured_scale = tuning.worker.time_scale
        self.time_scale = (
            self._base_time_scale if configured_scale is None else float(configured_scale)
        )
        if self.time_scale is None and previous_scale is not None:
            self.time_scale = 1.0
        if self.time_scale is not None and self.time_scale != previous_scale:
            set_timescale = _find_attr(self.env, "set_timescale")
            if not callable(set_timescale):
                raise RuntimeError("live worker tuning requires env.set_timescale(scale)")
            set_timescale(self.time_scale)
            self._time_scale_needs_apply = False

        self.live_tuning = tuning
        self.tuning_version = tuning.version
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
                "arena_auto_reset_count": self.arena_auto_reset_count,
                "learner_endpoint": self.learner_endpoint,
                **self._learner_upload_metrics(),
                "policy_version": self.policy_version,
                "tuning_version": self.tuning_version,
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
                "arena_auto_reset_count": self.arena_auto_reset_count,
                "error": f"{type(exc).__name__}: {exc}",
                "learner_endpoint": self.learner_endpoint,
                **self._learner_upload_metrics(),
                "policy_version": self.policy_version,
                "tuning_version": self.tuning_version,
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
        self._time_scale_needs_apply = self.time_scale is not None

    def _ensure_reset(self) -> None:
        if self._obs is None:
            self._reset()

    def _reset(self) -> None:
        if self._time_scale_needs_apply:
            set_timescale = _find_attr(self.env, "set_timescale")
            if not callable(set_timescale):
                raise RuntimeError("configured time_scale requires env.set_timescale(scale)")
            set_timescale(self.time_scale)
            self._time_scale_needs_apply = False
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
        _validate_categorical_action_mask(mask, enable_macro=self.enable_macro)
        return mask

    def _bootstrap_value(self) -> np.ndarray:
        if self._obs is None:
            return np.zeros((1,), dtype=np.float32)

        return self._observation_value(
            self._obs,
            self._info,
            self._rnn_state,
            self._prev_action,
            self._prev_reward,
            name="bootstrap value",
        )

    def _observation_value(
        self,
        obs: Any,
        info: dict[str, Any],
        rnn_state: Any,
        prev_action: np.ndarray,
        prev_reward: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        with torch.inference_mode():
            _, value, _ = self.model.forward(
                _obs_to_tensor(
                    obs,
                    self.device,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                ),
                rnn_state=rnn_state,
                action_mask=torch.as_tensor(
                    self._action_mask(info)[None, :],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        value_array = value.detach().float().reshape(-1).cpu().numpy()
        if not np.isfinite(value_array).all():
            raise ValueError(f"{name} contains non-finite values")
        if value_array.shape != (1,):
            raise ValueError(f"{name} must have one element, got {value_array.shape}")
        return value_array.astype(np.float32, copy=False)

    def _clear_memory_context(self) -> None:
        self._prev_action = np.asarray(
            [canonical_noop_action_values(enable_macro=self.enable_macro)],
            dtype=np.int64,
        )
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
        if macro > 0 and (movement_x != 1 or aim_y != 1 or bool(buttons.any()) or duration != 0):
            raise ValueError(
                "macro actions must use canonical primitive fields "
                "(neutral movement/aim, no buttons, duration=0)"
            )
        env_action["macro"] = macro
    if action_mask is not None:
        _require_action_mask_allows(values, action_mask, enable_macro, n_macros)
    return env_action


def _policy_outputs_to_numpy(
    action: torch.Tensor,
    log_prob: torch.Tensor,
    value: torch.Tensor,
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy one policy decision to CPU with a single device synchronization."""
    if action.numel() != action_dim:
        raise ValueError(f"worker action must have {action_dim} elements")
    if log_prob.numel() != 1:
        raise ValueError("worker log_prob must have one element")
    if value.numel() != 1:
        raise ValueError("worker value must have one element")
    packed = (
        torch.cat(
            (
                action.detach().reshape(-1).to(dtype=torch.float32),
                log_prob.detach().reshape(-1).to(dtype=torch.float32),
                value.detach().reshape(-1).to(dtype=torch.float32),
            )
        )
        .cpu()
        .numpy()
    )
    action_values = packed[:action_dim]
    log_prob_values = packed[action_dim : action_dim + 1]
    value_values = packed[action_dim + 1 :]
    if not np.isfinite(action_values).all():
        raise ValueError("worker action contains non-finite values")
    if not np.equal(action_values, np.trunc(action_values)).all():
        raise ValueError("worker action values must be integer-coded")
    if not np.isfinite(log_prob_values).all():
        raise ValueError("worker log_prob contains non-finite values")
    if not np.isfinite(value_values).all():
        raise ValueError("worker value contains non-finite values")
    return (
        action_values.reshape(1, action_dim).astype(np.int64, copy=False),
        log_prob_values.astype(np.float32, copy=False),
        value_values.astype(np.float32, copy=False),
    )


def _validate_categorical_action_mask(mask: np.ndarray, *, enable_macro: bool) -> None:
    groups = (
        ("movement_x", mask[:N_MOVEMENT_X]),
        ("aim_y", mask[N_MOVEMENT_X : N_MOVEMENT_X + N_AIM_Y]),
    )
    for name, group in groups:
        if not group.any():
            raise ValueError(f"action_mask {name} group has no valid action")
    duration_start = N_MOVEMENT_X + N_AIM_Y + N_BUTTONS
    if not mask[duration_start : duration_start + N_DURATION].any():
        raise ValueError("action_mask duration group has no valid action")
    if enable_macro and not mask[duration_start + N_DURATION :].any():
        raise ValueError("action_mask macro group has no valid action")


def _discount_exponent(env: Any, next_info: Mapping[str, Any]) -> float:
    """Convert variable option duration to the task's base decision-time unit."""
    task = _find_attr(env, "task")
    action_config = None if task is None else getattr(task, "action", None)
    base_repeat = 1 if action_config is None else getattr(action_config, "action_repeat", 1)
    if isinstance(base_repeat, bool) or not isinstance(base_repeat, Integral):
        raise ValueError("task action_repeat must be an integer")
    base_repeat = int(base_repeat)
    if base_repeat <= 0:
        raise ValueError("task action_repeat must be positive")

    elapsed = next_info.get(
        "elapsed_ticks",
        next_info.get("action_repeat", base_repeat),
    )
    if isinstance(elapsed, bool) or not isinstance(elapsed, Integral):
        raise ValueError("elapsed_ticks must be an integer")
    elapsed_ticks = int(elapsed)
    if elapsed_ticks <= 0:
        raise ValueError("elapsed_ticks must be positive")
    return float(elapsed_ticks) / float(base_repeat)


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

    duration_offset = N_MOVEMENT_X + N_AIM_Y + N_BUTTONS
    if enable_macro:
        macro = int(values[3 + N_BUTTONS])
        macro_offset = duration_offset + N_DURATION
        if not mask[macro_offset + macro]:
            raise ValueError(f"action_mask disallows macro={macro}")
        # macro>0 is the option branch: every primitive field is ignored by the
        # Mod and canonicalized only as recurrent context.
        if macro > 0:
            return

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


def _checkpoint_tuning(
    state: Mapping[str, object],
) -> tuple[int, LiveTuning | None]:
    raw_version = state.get("tuning_version", 0)
    if isinstance(raw_version, bool) or not isinstance(raw_version, Integral):
        raise ValueError("checkpoint tuning_version must be an integer")
    version = int(raw_version)
    if version < 0:
        raise ValueError("checkpoint tuning_version must be non-negative")

    payload = state.get("live_tuning")
    if payload is None:
        if version != 0:
            raise ValueError("checkpoint tuning_version requires live_tuning")
        return version, None
    tuning = LiveTuning.model_validate(payload)
    if tuning.version != version:
        raise ValueError("checkpoint live_tuning version does not match tuning_version")
    return version, tuning


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
