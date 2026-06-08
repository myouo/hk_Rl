"""Reward function: RewardEvent[] -> scalar (docs/reward_design.md).

The mod reports typed events; this module composes the config-weighted scalar.
Keeping this Python-side and decoupled is the core anti-reward-hacking measure
(PRD §9.4): terminal rewards dominate shaping, and the evaluator can always
recompute shaping-free metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

from hkrl.protocol import RewardEvent, RewardEventKind
from hkrl.utils.config import RewardWeights
from hkrl.utils.registry import register_reward


@register_reward("default")
class DefaultReward:
    """Linear combination of event amounts by configured weights.

    reward = +w.boss_damage*dealt - |w.player_damage|*taken + w.soul*soul
             + w.heal*heal + w.boss_kill*kill - |w.player_death|*death
             + w.time_penalty*1 + w.invalid_action*invalid  (+ optional shaping)
    """

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.w = weights or RewardWeights()

    def __call__(self, events: Sequence[RewardEvent], *, dt: float = 1.0) -> float:
        """Compute the per-step scalar reward from this step's events.

        ``dt`` scales the time penalty. Returns a float; terminal events
        (boss_kill / player_death) dominate by weight magnitude.

        TODO(phase-2): implement the sum over RewardEventKind using self.w.
        """
        raise NotImplementedError

    def shaping_free(self, events: Sequence[RewardEvent]) -> dict[str, float]:
        """Return shaping-free outcome stats (damage ratio, kill, death) for the
        evaluator. Never includes distance/positioning shaping. docs/metrics.md §2.

        TODO(phase-3): implement.
        """
        raise NotImplementedError


# Re-export for convenience / discoverability.
__all__ = ["DefaultReward", "RewardEvent", "RewardEventKind"]
