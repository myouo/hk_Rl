"""Reward composition tests."""

from __future__ import annotations

from hkrl.protocol import RewardEvent, RewardEventKind
from hkrl.reward import DefaultReward
from hkrl.utils.config import RewardWeights


def test_default_reward_composes_weighted_events() -> None:
    reward = DefaultReward(
        RewardWeights(
            boss_damage=1.0,
            player_damage=-8.0,
            soul_gained=0.5,
            heal_amount=2.0,
            boss_kill=100.0,
            player_death=-100.0,
            time_penalty=-0.001,
            invalid_action=-0.01,
        )
    )

    events = [
        RewardEvent(RewardEventKind.DAMAGE_DEALT, amount=10.0),
        RewardEvent(RewardEventKind.DAMAGE_TAKEN, amount=2.0),
        RewardEvent(RewardEventKind.SOUL_GAINED, amount=4.0),
        RewardEvent(RewardEventKind.HEAL, amount=1.0),
        RewardEvent(RewardEventKind.INVALID_ACTION),
        RewardEvent(RewardEventKind.BOSS_KILLED),
    ]

    assert reward(events, dt=2.0) == 10.0 - 16.0 + 2.0 + 2.0 - 0.01 + 100.0 - 0.002


def test_shaping_free_stats_ignore_scalar_weights() -> None:
    reward = DefaultReward(RewardWeights(boss_damage=99.0, player_damage=-99.0))
    events = [
        RewardEvent(RewardEventKind.DAMAGE_DEALT, amount=3.0),
        RewardEvent(RewardEventKind.DAMAGE_TAKEN, amount=2.0),
        RewardEvent(RewardEventKind.PLAYER_DEATH),
        RewardEvent(RewardEventKind.INVALID_ACTION),
    ]

    assert reward.shaping_free(events) == {
        "damage_dealt": 3.0,
        "damage_taken": 2.0,
        "heal_amount": 0.0,
        "soul_gained": 0.0,
        "boss_killed": 0.0,
        "player_death": 1.0,
        "invalid_actions": 1.0,
    }
