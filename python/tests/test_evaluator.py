"""Evaluator tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from hkrl import protocol, spaces
from hkrl.eval.evaluator import Evaluator
from hkrl.eval.scripted_policies import ScriptedAggroPolicy
from hkrl.utils.config import TaskConfig


def test_evaluator_runs_fixed_seed_episodes_and_aggregates_metrics() -> None:
    task = TaskConfig(task_id="fake_boss", scene="FakeScene")
    action_space = spaces.make_action_space(enable_macro=False)
    policy = ScriptedAggroPolicy(action_space)
    made_envs: list[FakeEvalEnv] = []

    def make_env(received_task: TaskConfig) -> FakeEvalEnv:
        assert received_task is task
        env = FakeEvalEnv()
        made_envs.append(env)
        return env

    evaluator = Evaluator(
        policy,
        tasks=[task],
        seeds=[0, 1],
        env_factory=make_env,
        max_steps_per_episode=4,
    )

    results = evaluator.evaluate(episodes_per_task=2)

    metrics = results["fake_boss"]
    assert metrics["win_rate"] == 0.5
    assert metrics["episode_reward"] == 2.0
    assert metrics["episode_length"] == 2.0
    assert metrics["damage_dealt"] == 3.0
    assert metrics["damage_taken"] == 1.0
    assert metrics["heal_count"] == 1.0
    assert metrics["heal_amount"] == 1.0
    assert metrics["invalid_action_ratio"] == 0.25
    assert metrics["death_rate"] == 0.5
    assert metrics["death_reason"] == 1.0
    assert metrics["time_to_kill"] == 2.0
    assert made_envs[0].closed


def test_evaluator_regression_report_returns_win_rate_delta() -> None:
    evaluator = Evaluator(
        model=object(),
        tasks=[TaskConfig(task_id="a", scene="A")],
        seeds=[0],
        env_factory=lambda task: None,
    )

    report = evaluator.regression_report(
        baseline={"a": {"win_rate": 0.75}, "b": {"win_rate": 0.25}},
        current={"a": {"win_rate": 0.50}, "c": {"win_rate": 1.0}},
    )

    assert report == {"a": -0.25, "b": -0.25, "c": 1.0}


class FakeEvalEnv:
    def __init__(self) -> None:
        self.observation_space = spaces.make_observation_space(max_entities=2, tier="privileged")
        self.action_space = spaces.make_action_space(enable_macro=False)
        self.seed = 0
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.seed = 0 if seed is None else seed
        self.step_count = 0
        return _obs(), {"action_mask": np.ones((19,), dtype=bool)}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        self.step_count += 1
        events = []
        if self.step_count == 1:
            events.append(protocol.RewardEvent(protocol.RewardEventKind.DAMAGE_DEALT, amount=1.0))
            events.append(protocol.RewardEvent(protocol.RewardEventKind.HEAL, amount=1.0))
        if self.step_count >= 2:
            if self.seed % 2 == 0:
                events.extend(
                    [
                        protocol.RewardEvent(protocol.RewardEventKind.DAMAGE_DEALT, amount=4.0),
                        protocol.RewardEvent(protocol.RewardEventKind.BOSS_KILLED),
                    ]
                )
            else:
                events.extend(
                    [
                        protocol.RewardEvent(protocol.RewardEventKind.DAMAGE_TAKEN, amount=2.0),
                        protocol.RewardEvent(protocol.RewardEventKind.INVALID_ACTION),
                        protocol.RewardEvent(protocol.RewardEventKind.PLAYER_DEATH),
                    ]
                )
        return (
            _obs(),
            1.0,
            self.step_count >= 2,
            False,
            {
                "action_mask": np.ones((19,), dtype=bool),
                "reward_events": events,
            },
        )

    def close(self) -> None:
        self.closed = True


def _obs() -> dict[str, np.ndarray]:
    entities = np.zeros((2, spaces.ENTITY_FEATURE_DIMS["privileged"]), dtype=np.float32)
    entities[0, 1] = 1.0
    entities[0, 8] = 0.05
    return {
        "global": np.zeros((spaces.GLOBAL_FEATURE_DIM,), dtype=np.float32),
        "player": np.zeros((spaces.PLAYER_FEATURE_DIMS["privileged"],), dtype=np.float32),
        "entities": entities,
        "entity_mask": np.array([1, 0], dtype=np.int8),
    }
