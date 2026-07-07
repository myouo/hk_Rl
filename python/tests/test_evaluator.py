"""Evaluator tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from hkrl import protocol, spaces
from hkrl.eval.evaluator import Evaluator, _metrics_from_info
from hkrl.eval.scripted_policies import ScriptedAggroPolicy
from hkrl.models.base import ActorCritic, RnnState
from hkrl.utils.config import TaskConfig
from torch import Tensor


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
    assert metrics["per_boss_win_rate"] == 0.5
    assert metrics["per_boss_damage_ratio"] == 1.0 / 3.0
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


def test_evaluator_worker_pool_runs_tasks_and_closes_envs() -> None:
    tasks = [
        TaskConfig(task_id="fake_a", scene="FakeScene"),
        TaskConfig(task_id="fake_b", scene="FakeScene"),
    ]
    action_space = spaces.make_action_space(enable_macro=False)
    made_envs: list[tuple[str, FakeEvalEnv]] = []

    def make_env(task: TaskConfig) -> FakeEvalEnv:
        env = FakeEvalEnv()
        made_envs.append((task.task_id, env))
        return env

    evaluator = Evaluator(
        ScriptedAggroPolicy(action_space),
        tasks=tasks,
        seeds=[0, 1],
        env_factory=make_env,
        max_steps_per_episode=4,
        num_workers=2,
    )

    results = evaluator.evaluate(episodes_per_task=2)

    assert list(results) == ["fake_a", "fake_b"]
    assert results["fake_a"]["win_rate"] == 0.5
    assert results["fake_b"]["win_rate"] == 0.5
    assert sorted(task_id for task_id, _ in made_envs) == ["fake_a", "fake_b"]
    assert all(env.closed for _, env in made_envs)


def test_evaluator_rejects_non_positive_worker_count() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        Evaluator(
            model=object(),
            tasks=[TaskConfig(task_id="a", scene="A")],
            seeds=[0],
            env_factory=lambda task: None,
            num_workers=0,
        )


def test_evaluator_regression_report_accepts_per_boss_win_rate() -> None:
    evaluator = Evaluator(
        model=object(),
        tasks=[TaskConfig(task_id="a", scene="A")],
        seeds=[0],
        env_factory=lambda task: None,
    )

    report = evaluator.regression_report(
        baseline={"a": {"per_boss_win_rate": 0.75}},
        current={"a": {"per_boss_win_rate": 0.50}},
    )

    assert report == {"a": -0.25}


@pytest.mark.parametrize(
    "baseline_metrics, match",
    [
        ({}, "must include win_rate"),
        ({"win_rate": 1.2}, r"must be in \[0, 1\]"),
        ({"win_rate": float("nan")}, "must be finite"),
        ({"win_rate": "0.5"}, "must be numeric"),
        ({"per_boss_win_rate": True}, "must be numeric"),
        ("not an object", "must be an object"),
        (None, "must be an object"),
    ],
)
def test_evaluator_regression_report_rejects_invalid_baseline_win_rate(
    baseline_metrics: object,
    match: str,
) -> None:
    evaluator = Evaluator(
        model=object(),
        tasks=[TaskConfig(task_id="a", scene="A")],
        seeds=[0],
        env_factory=lambda task: None,
    )

    with pytest.raises(ValueError, match=match):
        evaluator.regression_report(
            baseline={"a": baseline_metrics},
            current={"a": {"win_rate": 0.5}},
        )


def test_evaluator_regression_report_rejects_invalid_current_win_rate() -> None:
    evaluator = Evaluator(
        model=object(),
        tasks=[TaskConfig(task_id="a", scene="A")],
        seeds=[0],
        env_factory=lambda task: None,
    )

    with pytest.raises(ValueError, match="current win_rate"):
        evaluator.regression_report(
            baseline={"a": {"win_rate": 0.5}},
            current={"a": {"win_rate": float("inf")}},
        )


def test_evaluator_emits_step_replay_records() -> None:
    task = TaskConfig(task_id="fake_boss", wire_id=9, scene="FakeScene")
    action_space = spaces.make_action_space(enable_macro=False)
    records: list[dict[str, Any]] = []
    evaluator = Evaluator(
        ScriptedAggroPolicy(action_space),
        tasks=[task],
        seeds=[0],
        env_factory=lambda _: FakeEvalEnv(),
        max_steps_per_episode=3,
        replay_sink=records.append,
    )

    evaluator.evaluate(episodes_per_task=1)

    assert len(records) == 2
    assert records[0]["type"] == "eval_step"
    assert records[0]["task_id"] == "fake_boss"
    assert records[0]["wire_id"] == 9
    assert records[0]["seed"] == 0
    assert records[0]["episode"] == 0
    assert records[0]["step"] == 1
    assert isinstance(records[0]["action"], dict)
    assert records[0]["damage_dealt"] == 1.0
    assert records[1]["won"] is True
    assert records[1]["terminated"] is True


@pytest.mark.parametrize(
    ("info", "match"),
    [
        (
            {
                "reward_events": [
                    protocol.RewardEvent(
                        protocol.RewardEventKind.DAMAGE_DEALT,
                        amount=float("nan"),
                    )
                ]
            },
            "reward event amount must be finite",
        ),
        (
            {
                "reward_events": [
                    protocol.RewardEvent(
                        protocol.RewardEventKind.DAMAGE_TAKEN,
                        amount=-1.0,
                    )
                ]
            },
            "DAMAGE_TAKEN reward event amount must be non-negative",
        ),
        (
            {
                "reward_events": [
                    object(),
                ]
            },
            "reward event kind is invalid",
        ),
        ({"damage_taken": float("inf")}, "damage_taken must be finite"),
        ({"heal_count": 1.5}, "heal_count must be an integer"),
        ({"invalid_actions": -1}, "invalid_actions must be non-negative"),
    ],
)
def test_evaluator_rejects_malformed_metric_inputs(
    info: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _metrics_from_info(info)


def test_evaluator_player_death_event_defaults_to_positive_reason() -> None:
    metrics = _metrics_from_info(
        {
            "reward_events": [
                protocol.RewardEvent(
                    protocol.RewardEventKind.PLAYER_DEATH,
                    aux_int=-7,
                )
            ]
        }
    )

    assert metrics.death_reason == 1


def test_evaluator_preserves_actor_critic_rnn_state_across_steps() -> None:
    task = TaskConfig(task_id="fake_boss", scene="FakeScene")
    model = StatefulActorCritic()
    evaluator = Evaluator(
        model,
        tasks=[task],
        seeds=[0],
        env_factory=lambda _: FakeEvalEnv(),
        max_steps_per_episode=2,
    )

    evaluator.evaluate(episodes_per_task=1)

    assert model.seen_states == [0.0, 1.0]
    np.testing.assert_array_equal(model.seen_prev_actions[0], np.zeros((1, 12), dtype=np.float32))
    np.testing.assert_array_equal(
        model.seen_prev_actions[1],
        np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
    )
    assert model.seen_prev_rewards == [0.0, 1.0]


def test_evaluator_sets_actor_critic_eval_mode_and_restores_training() -> None:
    task = TaskConfig(task_id="fake_boss", scene="FakeScene")
    model = StatefulActorCritic()
    model.train()
    evaluator = Evaluator(
        model,
        tasks=[task],
        seeds=[0],
        env_factory=lambda _: FakeEvalEnv(),
        max_steps_per_episode=1,
    )

    evaluator.evaluate(episodes_per_task=1)

    assert model.seen_training_modes == [False]
    assert model.training is True


def test_evaluator_model_policy_preserves_macro_action_ids() -> None:
    task = TaskConfig(task_id="fake_boss", scene="FakeScene")
    env = MacroEvalEnv(n_macros=3)
    evaluator = Evaluator(
        MacroActorCritic(macro=2),
        tasks=[task],
        seeds=[0],
        env_factory=lambda _: env,
        max_steps_per_episode=1,
    )

    results = evaluator.evaluate(episodes_per_task=1)

    assert results["fake_boss"]["win_rate"] == 1.0
    assert env.actions[0]["macro"] == 2
    assert env.closed


def test_evaluator_model_policy_rejects_masked_macro_before_env_step() -> None:
    task = TaskConfig(task_id="fake_boss", scene="FakeScene")
    env = MacroEvalEnv(n_macros=3, blocked_macro=2)
    evaluator = Evaluator(
        MacroActorCritic(macro=2),
        tasks=[task],
        seeds=[0],
        env_factory=lambda _: env,
        max_steps_per_episode=1,
    )

    with pytest.raises(ValueError, match="macro=2"):
        evaluator.evaluate(episodes_per_task=1)

    assert env.actions == []
    assert env.closed


class StatefulActorCritic(ActorCritic):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.seen_states: list[float] = []
        self.seen_prev_actions: list[np.ndarray] = []
        self.seen_prev_rewards: list[float] = []
        self.seen_training_modes: list[bool] = []

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        return torch.zeros((1, batch_size, 1), device=device)

    def forward(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[object, Tensor, RnnState]:
        del obs, action_mask
        return object(), torch.zeros((1,)), rnn_state

    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        del action_mask, deterministic
        assert rnn_state is not None
        self.seen_training_modes.append(self.training)
        self.seen_states.append(float(rnn_state.reshape(-1)[0].detach().cpu()))
        self.seen_prev_actions.append(obs["prev_action"].detach().cpu().numpy().copy())
        self.seen_prev_rewards.append(float(obs["prev_reward"].reshape(-1)[0].detach().cpu()))
        action = torch.zeros((1, 12), dtype=torch.long)
        action[:, 0] = 1
        action[:, 1] = 1
        return action, torch.zeros((1,)), torch.zeros((1,)), rnn_state + 1

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del obs, actions, rnn_state, action_mask
        return torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))


class MacroActorCritic(ActorCritic):
    def __init__(self, *, macro: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.macro = macro

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        del batch_size, device
        return None

    def forward(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[object, Tensor, RnnState]:
        del obs, action_mask
        return object(), torch.zeros((1,)), rnn_state

    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        del obs, action_mask, deterministic
        action = torch.zeros((1, 13), dtype=torch.long)
        action[:, 0] = 1
        action[:, 1] = 1
        action[:, 12] = self.macro
        return action, torch.zeros((1,)), torch.zeros((1,)), rnn_state

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del obs, actions, rnn_state, action_mask
        return torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))


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


class MacroEvalEnv:
    def __init__(self, *, n_macros: int, blocked_macro: int | None = None) -> None:
        self.observation_space = spaces.make_observation_space(max_entities=2, tier="privileged")
        self.action_space = spaces.make_action_space(enable_macro=True, n_macros=n_macros)
        self.n_macros = n_macros
        self.blocked_macro = blocked_macro
        self.actions: list[dict[str, Any]] = []
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del seed
        return _obs(), {"action_mask": self._action_mask()}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return (
            _obs(),
            1.0,
            True,
            False,
            {
                "action_mask": self._action_mask(),
                "reward_events": [protocol.RewardEvent(protocol.RewardEventKind.BOSS_KILLED)],
            },
        )

    def close(self) -> None:
        self.closed = True

    def _action_mask(self) -> np.ndarray:
        layout = spaces.action_mask_layout(enable_macro=True, n_macros=self.n_macros)
        mask = np.ones((len(layout),), dtype=bool)
        if self.blocked_macro is not None:
            mask[layout.index(f"macro:{self.blocked_macro}")] = False
        return mask


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
