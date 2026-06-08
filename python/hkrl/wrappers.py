"""Gymnasium wrappers: observation tiers, normalization, frame ops.

Composable wrappers around HKRLEnv. The observation-tier wrappers implement the
privileged/reduced/human-visible ablations (docs/observation_schema.md §7) so the
same env can be evaluated at different information levels (PRD §9.8).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class NormalizeObservation(gym.ObservationWrapper):
    """Apply the player-centric normalization from hkrl.spaces (ARENA/VEL/T_MAX).

    TODO(phase-2): implement using hkrl.spaces constants; keep it the single place
    normalization happens so ablations stay consistent.
    """

    def observation(self, observation: Any) -> Any:
        raise NotImplementedError  # TODO(phase-2)


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
