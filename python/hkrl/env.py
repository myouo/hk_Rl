"""Gymnasium environment wrapping one HKRLEnvMod instance.

Implements PRD §2.1 / Phase 2 and docs/architecture.md. Composes a Transport, the
reward function, and the spaces. ``reset()`` drives the clean-lifecycle handshake
(docs/episode_lifecycle.md); ``step()`` sends an action and returns the decoded
observation, reward, termination flags, and info (including the action mask).
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

from hkrl import protocol
from hkrl.reward import DefaultReward
from hkrl.spaces import (
    ENTITY_FEATURE_DIMS,
    GLOBAL_FEATURE_DIM,
    PLAYER_FEATURE_DIMS,
    action_mask_layout,
    make_action_space,
    make_observation_space,
)
from hkrl.transport.base import Transport
from hkrl.utils.config import TaskConfig


class EnvProtocolError(RuntimeError):
    """Raised when the mod returns an invalid lifecycle/protocol response."""


class HKRLEnv(gym.Env):
    """Single-env Gymnasium interface to the game.

    The real-time action loop is local (invariant #1): this env talks to the mod
    over a Transport, never to a remote learner. ``info["action_mask"]`` carries
    the per-step mask for masked policies. ``info["lifecycle_state"]`` and
    ``info["error_code"]`` surface reset health.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transport: Transport,
        task: TaskConfig,
        reward_fn: DefaultReward | None = None,
    ) -> None:
        super().__init__()
        self.transport = transport
        self._custom_reward_fn = reward_fn is not None
        self.reward_fn = reward_fn or DefaultReward(task.reward)
        self._configure_task(task)
        self._tick_id = 0
        self._episode_id = 0
        self._running = False
        self._env_id = 0
        self._step_timeout_s = 5.0
        self._last_server_tick: int | None = None

    # -- Gym API --------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Issue RESET and poll until LifecycleState.RUNNING (or an error code).

        Returns ``(obs, info)``. On a reset failure (non-OK StatusCode) raises /
        surfaces the code rather than returning a contaminated observation
        (docs/episode_lifecycle.md §4). Increments reset metrics.
        """
        super().reset(seed=seed)
        options = options or {}
        timeout_s = float(options.get("reset_timeout_s", options.get("timeout_s", 30.0)))
        recv_timeout_s = float(options.get("recv_timeout_s", self._step_timeout_s))

        if not self.transport.is_connected():
            self.transport.connect(timeout_s=timeout_s)

        self._running = False
        response = self._exchange(
            protocol.Command.RESET,
            action=None,
            action_repeat=1,
            timeout_s=recv_timeout_s,
        )
        self._raise_for_error(response, context="reset")

        if response.lifecycle_state != protocol.LifecycleState.RUNNING:
            response = self._await_running_response(
                timeout_s=timeout_s,
                recv_timeout_s=recv_timeout_s,
            )
            self._raise_for_error(response, context="reset")

        if response.lifecycle_state != protocol.LifecycleState.RUNNING:
            raise EnvProtocolError(f"reset did not reach RUNNING: {response.lifecycle_state.name}")
        if response.observation is None:
            raise EnvProtocolError("reset reached RUNNING without an observation")

        self._validate_action_mask(response.action_mask)
        self._running = True
        self._validate_observation(response.observation)
        obs = self._to_gym_observation(response.observation)
        self._episode_id = self._episode_id_from(obs)
        self._last_server_tick = response.server_tick
        return obs, self._info_from_response(response)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Send a STEP (with action_repeat), decode the response, compute reward.

        Returns ``(obs, reward, terminated, truncated, info)`` per Gymnasium.
        ``info`` includes ``action_mask``, ``reward_events``, ``lifecycle_state``,
        ``episode_id``, ``sps`` hints.
        """
        if not self._running:
            raise EnvProtocolError("reset must complete before step()")

        response = self._exchange(
            protocol.Command.STEP,
            action=action,
            action_repeat=self.task.action.action_repeat,
            timeout_s=self._step_timeout_s,
        )
        self._raise_for_error(response, context="step")

        if response.observation is None:
            raise EnvProtocolError("step response did not include an observation")

        self._validate_action_mask(response.action_mask)
        self._validate_observation(response.observation)
        obs = self._to_gym_observation(response.observation)
        self._episode_id = self._episode_id_from(obs)
        elapsed_ticks = self._server_tick_delta(response)
        reward = self.reward_fn(response.reward_events, dt=float(elapsed_ticks))
        self._last_server_tick = response.server_tick
        terminated = bool(response.terminated or self._is_terminal(response.lifecycle_state))
        truncated = bool(response.truncated)
        self._running = not (terminated or truncated)
        return obs, reward, terminated, truncated, self._info_from_response(response)

    def close(self) -> None:
        """Close the transport. Idempotent."""
        self.transport.close()

    def set_task(
        self,
        task: TaskConfig,
        *,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Switch to a new task, trigger mod reset, and wait until RUNNING."""
        self._configure_task(task)
        options = options or {}
        timeout_s = float(options.get("reset_timeout_s", options.get("timeout_s", 30.0)))
        recv_timeout_s = float(options.get("recv_timeout_s", self._step_timeout_s))

        if not self.transport.is_connected():
            self.transport.connect(timeout_s=timeout_s)

        self._running = False
        response = self._exchange(
            protocol.Command.SET_TASK,
            action=None,
            action_repeat=1,
            timeout_s=recv_timeout_s,
        )
        self._raise_for_error(response, context="set_task")

        if response.lifecycle_state != protocol.LifecycleState.RUNNING:
            response = self._await_running_response(
                timeout_s=timeout_s,
                recv_timeout_s=recv_timeout_s,
            )
            self._raise_for_error(response, context="set_task")

        if response.lifecycle_state != protocol.LifecycleState.RUNNING:
            raise EnvProtocolError(
                f"set_task did not reach RUNNING: {response.lifecycle_state.name}"
            )
        if response.observation is None:
            raise EnvProtocolError("set_task reached RUNNING without an observation")

        self._validate_action_mask(response.action_mask)
        self._running = True
        self._validate_observation(response.observation)
        obs = self._to_gym_observation(response.observation)
        self._episode_id = self._episode_id_from(obs)
        self._last_server_tick = response.server_tick
        return obs, self._info_from_response(response)

    def pause(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Pause the game simulation via the mod's SimControl."""
        return self._control(protocol.Command.PAUSE, timeout_s=timeout_s)

    def resume(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Resume the game simulation via the mod's SimControl."""
        return self._control(protocol.Command.RESUME, timeout_s=timeout_s)

    def ping(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Send a liveness probe and return the decoded response info."""
        return self._control(protocol.Command.PING, timeout_s=timeout_s)

    def set_timescale(self, scale: float, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Set Unity Time.timeScale / fixedDeltaTime through the mod."""
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        return self._control(
            protocol.Command.SET_TIMESCALE,
            timeout_s=timeout_s,
            time_scale=float(scale),
        )

    # -- helpers --------------------------------------------------------------
    def _configure_task(self, task: TaskConfig) -> None:
        self.task = task
        if not self._custom_reward_fn:
            self.reward_fn = DefaultReward(task.reward)
        self.observation_space = make_observation_space(
            max_entities=task.observation.max_entities,
            tier=task.observation.tier,
        )
        self.action_space = make_action_space(
            enable_macro=task.action.enable_macro_actions,
            n_macros=task.action.n_macro_actions,
        )

    def _await_running(self, timeout_s: float) -> protocol.StatusCode:
        """Poll the mod with no-op STEPs until RUNNING or error/timeout."""
        response = self._await_running_response(
            timeout_s=timeout_s,
            recv_timeout_s=self._step_timeout_s,
        )
        return response.error_code

    def _await_running_response(
        self,
        *,
        timeout_s: float,
        recv_timeout_s: float,
    ) -> protocol.DecodedStepResponse:
        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"reset did not reach RUNNING within {timeout_s:.1f}s")

            response = self._exchange(
                protocol.Command.STEP,
                action=None,
                action_repeat=1,
                timeout_s=min(recv_timeout_s, remaining),
            )
            if response.error_code != protocol.StatusCode.OK:
                return response
            if response.lifecycle_state == protocol.LifecycleState.RUNNING:
                return response
            if self._is_terminal(response.lifecycle_state):
                return response

    @staticmethod
    def _is_terminal(lifecycle: protocol.LifecycleState) -> bool:
        return lifecycle in (
            protocol.LifecycleState.TERMINATING,
            protocol.LifecycleState.REPORT_DONE,
        )

    def _exchange(
        self,
        command: protocol.Command,
        *,
        action: Any,
        action_repeat: int,
        timeout_s: float,
        time_scale: float = 0.0,
    ) -> protocol.DecodedStepResponse:
        tick_id = self._next_tick_id()
        request = protocol.encode_step_request(
            command=command,
            action=action,
            env_id=self._env_id,
            tick_id=tick_id,
            action_repeat=action_repeat,
            task_id=self.task.wire_id,
            time_scale=time_scale,
        )
        self.transport.send(request)
        response = protocol.decode_step_response(self.transport.recv(timeout_s=timeout_s))
        if response.env_id != self._env_id:
            raise EnvProtocolError(
                f"env_id mismatch: sent={self._env_id}, received={response.env_id}"
            )
        if response.tick_id != tick_id:
            raise EnvProtocolError(f"tick_id mismatch: sent={tick_id}, received={response.tick_id}")
        return response

    def _control(
        self,
        command: protocol.Command,
        *,
        timeout_s: float | None,
        time_scale: float = 0.0,
    ) -> dict[str, Any]:
        timeout = self._step_timeout_s if timeout_s is None else float(timeout_s)
        if not self.transport.is_connected():
            self.transport.connect(timeout_s=timeout)
        response = self._exchange(
            command,
            action=None,
            action_repeat=1,
            timeout_s=timeout,
            time_scale=time_scale,
        )
        self._raise_for_error(response, context=command.name.lower())
        self._last_server_tick = response.server_tick
        return self._info_from_response(response)

    def _validate_observation(self, observation: protocol.DecodedObservation) -> None:
        arrays = (
            np.asarray(observation.global_state, dtype=np.float32),
            np.asarray(observation.player_state, dtype=np.float32),
            np.asarray(observation.entities, dtype=np.float32),
        )
        for name, array in zip(("global", "player", "entities"), arrays, strict=True):
            if not np.isfinite(array).all():
                raise EnvProtocolError(f"observation {name} contains non-finite values")

        entities = arrays[2]
        if entities.size == 0:
            entities = entities.reshape((0, 0))
        elif entities.ndim != 2:
            raise EnvProtocolError(f"entities must be rank-2, got shape {entities.shape}")

        mask = np.asarray(observation.entity_mask, dtype=bool).reshape(-1)
        if len(mask) != entities.shape[0]:
            raise EnvProtocolError(
                f"entity_mask length {len(mask)} != entities length {entities.shape[0]}"
            )
        if entities.shape[1] > 13:
            hp = entities[:, 12]
            max_hp = entities[:, 13]
            invalid_hp = (max_hp > 0.0) & (hp > max_hp)
            if bool(invalid_hp.any()):
                raise EnvProtocolError("entity hp exceeds max_hp")

        if self._task_requires_boss():
            if entities.shape[1] <= 1:
                raise EnvProtocolError("boss task observation contains no boss entity")
            valid_entities = entities[mask]
            has_boss = bool((valid_entities[:, 1] == float(protocol.EntityType.BOSS)).any())
            if not has_boss:
                raise EnvProtocolError("boss task observation contains no boss entity")

    def _validate_action_mask(self, action_mask: np.ndarray) -> None:
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        expected = len(
            action_mask_layout(
                enable_macro=self.task.action.enable_macro_actions,
                n_macros=self.task.action.n_macro_actions,
            )
        )
        if mask.shape != (expected,):
            raise EnvProtocolError(f"action_mask length {mask.size} != expected {expected}")

    def _next_tick_id(self) -> int:
        tick_id = self._tick_id
        self._tick_id += 1
        return tick_id

    def _server_tick_delta(self, response: protocol.DecodedStepResponse) -> int:
        if self._last_server_tick is None:
            return int(self.task.action.action_repeat)
        delta = int(response.server_tick) - int(self._last_server_tick)
        if delta <= 0:
            return int(self.task.action.action_repeat)
        return delta

    @staticmethod
    def _raise_for_error(response: protocol.DecodedStepResponse, *, context: str) -> None:
        if response.error_code != protocol.StatusCode.OK:
            raise EnvProtocolError(f"{context} failed with {response.error_code.name}")

    def _to_gym_observation(
        self, observation: protocol.DecodedObservation
    ) -> dict[str, np.ndarray]:
        tier = self.task.observation.tier
        max_entities = self.task.observation.max_entities
        player_dim = PLAYER_FEATURE_DIMS[tier]
        entity_dim = ENTITY_FEATURE_DIMS[tier]

        global_state = np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32)
        global_count = min(GLOBAL_FEATURE_DIM, len(observation.global_state))
        global_state[:global_count] = observation.global_state[:global_count]

        player_state = np.zeros((player_dim,), dtype=np.float32)
        player_count = min(player_dim, len(observation.player_state))
        player_state[:player_count] = observation.player_state[:player_count]

        raw_entities = np.asarray(observation.entities, dtype=np.float32)
        if raw_entities.size == 0:
            raw_entities = raw_entities.reshape((0, 0))
        elif raw_entities.ndim != 2:
            raise EnvProtocolError(f"entities must be rank-2, got shape {raw_entities.shape}")

        raw_mask = np.asarray(observation.entity_mask, dtype=bool).reshape(-1)
        if len(raw_mask) != raw_entities.shape[0]:
            raise EnvProtocolError(
                f"entity_mask length {len(raw_mask)} != entities length {raw_entities.shape[0]}"
            )

        entities = np.zeros((max_entities, entity_dim), dtype=np.float32)
        entity_count = min(max_entities, raw_entities.shape[0])
        feature_count = min(entity_dim, raw_entities.shape[1]) if entity_count > 0 else 0
        if entity_count > 0 and feature_count > 0:
            entities[:entity_count, :feature_count] = raw_entities[:entity_count, :feature_count]

        entity_mask = np.zeros((max_entities,), dtype=np.int8)
        entity_mask[:entity_count] = raw_mask[:entity_count].astype(np.int8, copy=False)

        return {
            "global": global_state,
            "player": player_state,
            "entities": entities,
            "entity_mask": entity_mask,
        }

    def _info_from_response(self, response: protocol.DecodedStepResponse) -> dict[str, Any]:
        info: dict[str, Any] = {
            "schema_version": response.schema_version,
            "env_id": response.env_id,
            "tick_id": response.tick_id,
            "server_tick": response.server_tick,
            "action_mask": response.action_mask,
            "reward_events": response.reward_events,
            "lifecycle_state": response.lifecycle_state,
            "error_code": response.error_code,
            "episode_id": self._episode_id,
            "task_id": self.task.wire_id,
            "task_name": self.task.task_id,
        }
        if response.info is not None:
            info["raw_info"] = response.info
        return info

    @staticmethod
    def _episode_id_from(obs: dict[str, np.ndarray]) -> int:
        global_state = obs["global"]
        if len(global_state) <= 8:
            return 0
        return int(global_state[8])

    def _task_requires_boss(self) -> bool:
        return bool(self.task.task_id and self.task.task_id != "smoke")
