"""Reward function: RewardEvent[] -> scalar (docs/reward_design.md).

The mod reports typed events; this module composes the config-weighted scalar.
Keeping this Python-side and decoupled is the core anti-reward-hacking measure
(PRD §9.4): terminal rewards dominate shaping, and the evaluator can always
recompute shaping-free metrics.
"""

from __future__ import annotations

import math
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
        """
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("reward dt must be finite and non-negative")
        reward = self.w.time_penalty * dt

        for event in events:
            amount = _event_amount(event)

            if event.kind == RewardEventKind.DAMAGE_DEALT:
                reward += self.w.boss_damage * amount
            elif event.kind == RewardEventKind.DAMAGE_TAKEN:
                reward += self.w.player_damage * amount
            elif event.kind == RewardEventKind.SOUL_GAINED:
                reward += self.w.soul_gained * amount
            elif event.kind == RewardEventKind.HEAL:
                reward += self.w.heal_amount * amount
            elif event.kind == RewardEventKind.BOSS_KILLED:
                reward += self.w.boss_kill
            elif event.kind == RewardEventKind.PLAYER_DEATH:
                reward += self.w.player_death
            elif event.kind == RewardEventKind.INVALID_ACTION:
                reward += self.w.invalid_action

        return float(reward)

    def shaping_free(self, events: Sequence[RewardEvent]) -> dict[str, float]:
        """Return shaping-free outcome stats (damage ratio, kill, death) for the
        evaluator. Never includes distance/positioning shaping. docs/metrics.md §2.
        """
        stats = {
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "heal_amount": 0.0,
            "soul_gained": 0.0,
            "boss_killed": 0.0,
            "player_death": 0.0,
            "invalid_actions": 0.0,
        }

        for event in events:
            amount = _event_amount(event)

            if event.kind == RewardEventKind.DAMAGE_DEALT:
                stats["damage_dealt"] += amount
            elif event.kind == RewardEventKind.DAMAGE_TAKEN:
                stats["damage_taken"] += amount
            elif event.kind == RewardEventKind.HEAL:
                stats["heal_amount"] += amount
            elif event.kind == RewardEventKind.SOUL_GAINED:
                stats["soul_gained"] += amount
            elif event.kind == RewardEventKind.BOSS_KILLED:
                stats["boss_killed"] += 1.0
            elif event.kind == RewardEventKind.PLAYER_DEATH:
                stats["player_death"] += 1.0
            elif event.kind == RewardEventKind.INVALID_ACTION:
                stats["invalid_actions"] += 1.0

        return stats


def _event_amount(event: RewardEvent) -> float:
    amount = float(event.amount)
    if not math.isfinite(amount):
        raise ValueError("reward event amount must be finite")
    if event.kind in _NON_NEGATIVE_AMOUNT_KINDS and amount < 0.0:
        raise ValueError(f"{event.kind.name} reward event amount must be non-negative")
    return amount


_NON_NEGATIVE_AMOUNT_KINDS: frozenset[RewardEventKind] = frozenset(
    {
        RewardEventKind.DAMAGE_DEALT,
        RewardEventKind.DAMAGE_TAKEN,
        RewardEventKind.HEAL,
        RewardEventKind.SOUL_GAINED,
    }
)


# Re-export for convenience / discoverability.
__all__ = ["DefaultReward", "RewardEvent", "RewardEventKind"]
