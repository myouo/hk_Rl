"""Boss arena terminal preservation and clean auto-reset tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from hkrl import protocol, spaces
from hkrl.arena import BossArenaSupervisor


def test_arena_preserves_death_then_auto_resets_to_new_episode() -> None:
    env = FakeArenaEnv(outcome="death")
    supervisor = BossArenaSupervisor(
        env,
        policy=lambda _obs, _info: _noop(),
        max_steps_per_episode=4,
    )

    results = supervisor.run(episodes=2)

    assert len(results) == 2
    assert env.reset_calls == 3
    assert all(result.died for result in results)
    assert all(not result.won for result in results)
    assert all(result.reset_succeeded for result in results)
    assert [(result.episode_id, result.next_episode_id) for result in results] == [
        (1, 2),
        (2, 3),
    ]


def test_hitless_speed_target_requires_win_no_damage_and_deadline() -> None:
    hitless = FakeArenaEnv(outcome="win", damage_taken=0.0, elapsed_seconds=2.0)
    hit = FakeArenaEnv(outcome="win", damage_taken=1.0, elapsed_seconds=2.0)

    hitless_result = BossArenaSupervisor(
        hitless,
        policy=lambda _obs, _info: _noop(),
    ).run(1)[0]
    hit_result = BossArenaSupervisor(
        hit,
        policy=lambda _obs, _info: _noop(),
    ).run(1)[0]

    assert hitless_result.hitless is True
    assert hitless_result.target_met is True
    assert hit_result.hitless is False
    assert hit_result.target_met is False


class FakeArenaEnv:
    def __init__(
        self,
        *,
        outcome: str,
        damage_taken: float = 1.0,
        elapsed_seconds: float = 2.0,
    ) -> None:
        self.outcome = outcome
        self.damage_taken = damage_taken
        self.elapsed_seconds = elapsed_seconds
        self.reset_calls = 0
        self.steps = 0
        self.task = SimpleNamespace(
            action=SimpleNamespace(action_repeat=2),
            arena=SimpleNamespace(
                objective="hitless_speedrun",
                target_kill_time_seconds=3.0,
            ),
            time_limit_seconds=5,
        )

    def reset(
        self,
        *,
        options: dict[str, float],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        assert options["reset_timeout_s"] > 0.0
        self.reset_calls += 1
        self.steps = 0
        return _obs(0.0), {"episode_id": self.reset_calls, "reward_events": []}

    def step(
        self,
        _action: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        if self.steps == 1:
            return _obs(1.0), 0.0, False, False, {"reward_events": []}

        events = []
        if self.damage_taken:
            events.append(
                protocol.RewardEvent(
                    protocol.RewardEventKind.DAMAGE_TAKEN,
                    amount=self.damage_taken,
                )
            )
        if self.outcome == "death":
            events.append(protocol.RewardEvent(protocol.RewardEventKind.PLAYER_DEATH))
        else:
            events.append(protocol.RewardEvent(protocol.RewardEventKind.BOSS_KILLED))
        return (
            _obs(self.elapsed_seconds),
            1.0,
            True,
            False,
            {"reward_events": events},
        )


def _obs(elapsed_seconds: float) -> dict[str, np.ndarray]:
    global_state = np.zeros((spaces.GLOBAL_FEATURE_DIM,), dtype=np.float32)
    global_state[4] = elapsed_seconds
    global_state[6] = 0.02
    return {
        "global": global_state,
        "player": np.zeros(
            (spaces.PLAYER_FEATURE_DIMS["privileged"],),
            dtype=np.float32,
        ),
        "entities": np.zeros((1, spaces.ENTITY_FEATURE_DIMS["privileged"])),
        "entity_mask": np.ones((1,), dtype=np.int8),
    }


def _noop() -> dict[str, Any]:
    return {
        "movement_x": 1,
        "aim_y": 1,
        "buttons": {},
        "duration": 0,
    }
