"""Gymnasium wrappers: observation tiers, normalization, frame ops.

Composable wrappers around HKRLEnv. The observation-tier wrappers implement the
privileged/reduced/human-visible ablations (docs/observation_schema.md §7) so the
same env can be evaluated at different information levels (PRD §9.8).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from hkrl import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Apply the player-centric normalization from hkrl.spaces (ARENA/VEL/T_MAX).

    Uses the constants in ``hkrl.spaces`` so ablations stay consistent.
    """

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
        self.tier = tier
        # TODO(phase-5): adjust observation_space + drop fields per tier.

    def observation(self, observation: Any) -> Any:
        raise NotImplementedError  # TODO(phase-5)


class FrameStack(gym.Wrapper):
    """Optional short-history stacking. Note: NOT a substitute for the recurrent
    memory (docs/troubleshooting.md §9.1) — use sparingly for the MLP baseline.

    TODO(phase-3): implement deque-based stacking of the flat feature vectors.
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = k

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError  # TODO(phase-3)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError  # TODO(phase-3)


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
