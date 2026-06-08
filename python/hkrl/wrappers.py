"""Gymnasium wrappers: observation tiers, normalization, frame ops.

Composable wrappers around HKRLEnv. The observation-tier wrappers implement the
privileged/reduced/human-visible ablations (docs/observation_schema.md §7) so the
same env can be evaluated at different information levels (PRD §9.8).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from hkrl import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Apply the player-centric normalization from hkrl.spaces (ARENA/VEL/T_MAX).

    Uses the constants in ``hkrl.spaces`` so ablations stay consistent.
    """

    def set_task(
        self, task: Any, *, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = _call_set_task(self.env, task, options=options)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        return self.observation(obs), info

    def observation(self, observation: Any) -> Any:
        if not isinstance(observation, dict):
            return observation

        normalized = dict(observation)
        for key in ("global", "player", "entities"):
            if key in observation:
                normalized[key] = np.asarray(observation[key], dtype=np.float32).copy()
        if "entity_mask" in observation:
            normalized["entity_mask"] = np.asarray(observation["entity_mask"]).copy()

        if "player" in normalized:
            _normalize_player(normalized["player"])
        if "entities" in normalized:
            mask = normalized.get("entity_mask")
            _normalize_entities(normalized["entities"], mask)
        return normalized


class ObservationTier(gym.ObservationWrapper):
    """Mask observation fields down to a tier: privileged | reduced | human_visible."""

    def __init__(self, env: gym.Env, tier: str = "privileged") -> None:
        super().__init__(env)
        if tier not in spaces.PLAYER_FEATURE_DIMS:
            known = ", ".join(sorted(spaces.PLAYER_FEATURE_DIMS))
            raise ValueError(f"unknown observation tier {tier!r}; expected one of: {known}")

        self.tier = tier
        self.observation_space = _tier_observation_space(env.observation_space, tier)

    def set_task(
        self, task: Any, *, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = _call_set_task(self.env, task, options=options)
        self.observation_space = _tier_observation_space(self.env.observation_space, self.tier)
        self.action_space = self.env.action_space
        return self.observation(obs), info

    def observation(self, observation: Any) -> Any:
        if not isinstance(observation, dict):
            return observation

        tiered = dict(observation)
        player_dim = spaces.PLAYER_FEATURE_DIMS[self.tier]
        entity_dim = spaces.ENTITY_FEATURE_DIMS[self.tier]
        if "player" in observation:
            player = np.asarray(observation["player"], dtype=np.float32)
            if player.shape[-1] < player_dim:
                raise ValueError(f"player feature dim {player.shape[-1]} < target {player_dim}")
            tiered["player"] = player[..., :player_dim].copy()
        if "entities" in observation:
            entities = np.asarray(observation["entities"], dtype=np.float32)
            if entities.shape[-1] < entity_dim:
                raise ValueError(f"entity feature dim {entities.shape[-1]} < target {entity_dim}")
            tiered["entities"] = entities[..., :entity_dim].copy()
        if "entity_mask" in observation:
            tiered["entity_mask"] = np.asarray(observation["entity_mask"]).copy()
        return tiered


class FrameStack(gym.Wrapper):
    """Optional short-history stacking. Note: NOT a substitute for the recurrent
    memory (docs/troubleshooting.md §9.1) — use sparingly for the MLP baseline.
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        super().__init__(env)
        self.k = k
        self._frames: deque[Any] = deque(maxlen=k)
        self.observation_space = _stack_observation_space(env.observation_space, k)

    def set_task(
        self, task: Any, *, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = _call_set_task(self.env, task, options=options)
        self.observation_space = _stack_observation_space(self.env.observation_space, self.k)
        self.action_space = self.env.action_space
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(_copy_observation(obs))
        return self._stack_frames(), info

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(_copy_observation(obs))
        return self._stack_frames(), info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(_copy_observation(obs))
        return self._stack_frames(), float(reward), terminated, truncated, info

    def _stack_frames(self) -> Any:
        return _stack_observations(list(self._frames))


def _call_set_task(
    env: gym.Env, task: Any, *, options: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    set_task = getattr(env, "set_task", None)
    if not callable(set_task):
        raise AttributeError(f"{type(env).__name__} does not expose set_task(task)")
    if options is None:
        return set_task(task)
    return set_task(task, options=options)


def _normalize_player(player: np.ndarray) -> None:
    if player.shape[0] < spaces.PLAYER_FEATURE_DIMS["human_visible"]:
        return

    player[0:2] = player[0:2] / spaces.ARENA_SCALE
    player[2:4] = player[2:4] / spaces.VEL_SCALE
    _normalize_current_and_max(player, current_idx=4, max_idx=5)
    _normalize_current_and_max(player, current_idx=6, max_idx=7)

    for idx in (16, 17, 18, 20):
        if idx < player.shape[0]:
            player[idx] = _clamped_timer(float(player[idx]))


def _normalize_entities(entities: np.ndarray, entity_mask: Any | None) -> None:
    if entities.ndim != 2 or entities.shape[1] < spaces.ENTITY_FEATURE_DIMS["human_visible"]:
        return

    if entity_mask is None:
        valid = np.ones((entities.shape[0],), dtype=bool)
    else:
        valid = np.asarray(entity_mask, dtype=bool).reshape(-1)
        if len(valid) != entities.shape[0]:
            raise ValueError(
                f"entity_mask length {len(valid)} != entities length {entities.shape[0]}"
            )

    for idx in np.flatnonzero(valid):
        row = entities[idx]
        row[6:10] = row[6:10] / spaces.ARENA_SCALE
        row[10:12] = row[10:12] / spaces.VEL_SCALE
        _normalize_current_and_max(row, current_idx=12, max_idx=13)

        if row.shape[0] >= spaces.ENTITY_FEATURE_DIMS["reduced"]:
            row[14:18] = row[14:18] / spaces.ARENA_SCALE
        if row.shape[0] > 20:
            row[20] = _clamped_timer(float(row[20]))


def _normalize_current_and_max(values: np.ndarray, *, current_idx: int, max_idx: int) -> None:
    max_value = float(values[max_idx])
    if max_value <= 0.0:
        return
    values[current_idx] = values[current_idx] / max_value
    values[max_idx] = 1.0


def _clamped_timer(value: float) -> float:
    return float(np.clip(value / spaces.T_MAX, 0.0, 1.0))


def _stack_observation_space(observation_space: gym.Space, k: int) -> gym.Space:
    from gymnasium import spaces as gym_spaces

    if not isinstance(observation_space, gym_spaces.Dict):
        return observation_space

    stacked_spaces = dict(observation_space.spaces)
    for key in ("global", "player", "entities"):
        space = stacked_spaces.get(key)
        if isinstance(space, gym_spaces.Box):
            stacked_spaces[key] = _stack_box_space(space, k)
    return gym_spaces.Dict(stacked_spaces)


def _tier_observation_space(observation_space: gym.Space, tier: str) -> gym.Space:
    from gymnasium import spaces as gym_spaces

    if not isinstance(observation_space, gym_spaces.Dict):
        return observation_space

    tiered_spaces = dict(observation_space.spaces)
    player_dim = spaces.PLAYER_FEATURE_DIMS[tier]
    entity_dim = spaces.ENTITY_FEATURE_DIMS[tier]
    if "player" in tiered_spaces:
        tiered_spaces["player"] = gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(player_dim,),
            dtype=np.float32,
        )
    if "entities" in tiered_spaces:
        entity_space = tiered_spaces["entities"]
        entity_shape = getattr(entity_space, "shape", None)
        max_entities = int(entity_shape[0]) if entity_shape else 0
        tiered_spaces["entities"] = gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_entities, entity_dim),
            dtype=np.float32,
        )
    return gym_spaces.Dict(tiered_spaces)


def _stack_box_space(space: gym.spaces.Box, k: int) -> gym.spaces.Box:
    from gymnasium import spaces as gym_spaces

    low = np.concatenate([np.asarray(space.low)] * k, axis=-1)
    high = np.concatenate([np.asarray(space.high)] * k, axis=-1)
    return gym_spaces.Box(low=low, high=high, dtype=np.dtype(space.dtype).type)  # type: ignore[arg-type]


def _copy_observation(observation: Any) -> Any:
    if not isinstance(observation, dict):
        return np.asarray(observation).copy()
    return {
        key: np.asarray(value).copy() if isinstance(value, (np.ndarray, list, tuple)) else value
        for key, value in observation.items()
    }


def _stack_observations(frames: list[Any]) -> Any:
    if not frames:
        raise RuntimeError("FrameStack has no frames; call reset() first")

    if not isinstance(frames[-1], dict):
        return np.concatenate([np.asarray(frame) for frame in frames], axis=-1)

    stacked: dict[str, Any] = {}
    keys = frames[-1].keys()
    for key in keys:
        if key == "entity_mask":
            stacked[key] = np.asarray(frames[-1][key]).copy()
        elif key in {"global", "player", "entities"}:
            stacked[key] = np.concatenate([np.asarray(frame[key]) for frame in frames], axis=-1)
        else:
            value = frames[-1][key]
            stacked[key] = np.asarray(value).copy() if isinstance(value, np.ndarray) else value
    return stacked
