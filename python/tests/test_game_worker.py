"""GameWorker rollout tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import make_action_space, make_observation_space
from hkrl.utils.config import TaskConfig, TrainConfig
from hkrl.worker.game_worker import GameWorker, action_tensor_to_env_action


def test_action_tensor_to_env_action_without_macro() -> None:
    action = torch.tensor([2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 3])

    env_action = action_tensor_to_env_action(action, enable_macro=False)

    assert env_action["movement_x"] == 2
    assert env_action["aim_y"] == 0
    assert env_action["duration"] == 3
    np.testing.assert_array_equal(
        env_action["buttons"],
        np.array([1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    )
    assert "macro" not in env_action


def test_game_worker_collect_rollout_returns_batch() -> None:
    env = FakeEnv()
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=4, gamma=0.99, gae_lambda=0.95),
    )

    batch = worker.collect_rollout()

    assert batch.obs_global.shape == (4, 1, 9)
    assert batch.actions.shape == (4, 1, 12)
    assert batch.action_masks.shape == (4, 1, 19)
    assert batch.returns.shape == (4, 1)
    assert batch.policy_version == 0
    assert env.reset_count == 2
    assert len(env.actions) == 4
    assert set(env.actions[0]) == {"movement_x", "aim_y", "buttons", "duration"}


def test_game_worker_hot_swaps_new_checkpoint_before_rollout() -> None:
    env = FakeEnv()
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    checkpoint_client = FakeCheckpointClient(
        {
            "model_state_dict": model.state_dict(),
            "policy_version": 7,
        }
    )
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=2),
        checkpoint_client=checkpoint_client,  # type: ignore[arg-type]
    )

    batch = worker.collect_rollout()

    assert checkpoint_client.latest_calls == 1
    assert checkpoint_client.pulled_versions == [1]
    assert worker.checkpoint_version == 1
    assert worker.policy_version == 7
    assert batch.policy_version == 7


def test_game_worker_run_uploads_batches_and_heartbeats() -> None:
    env = FakeEnv()
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    uploaded: list[Any] = []
    heartbeats: list[dict[str, Any]] = []
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=2),
        learner_endpoint="learner:5600",
        batch_uploader=uploaded.append,
        heartbeat_sink=heartbeats.append,
    )

    worker.run(total_steps=4)

    assert len(uploaded) == 2
    assert [int(batch.rewards.size) for batch in uploaded] == [2, 2]
    assert heartbeats == [
        {
            "checkpoint_version": -1,
            "learner_endpoint": "learner:5600",
            "policy_version": 0,
            "rollout_steps": 2,
            "status": "running",
            "worker_crash_count": 0,
        },
        {
            "checkpoint_version": -1,
            "learner_endpoint": "learner:5600",
            "policy_version": 0,
            "rollout_steps": 2,
            "status": "running",
            "worker_crash_count": 0,
        },
    ]


def test_game_worker_recovers_after_runtime_failure() -> None:
    env = FailOnceEnv()
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    uploaded: list[Any] = []
    heartbeats: list[dict[str, Any]] = []
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=2),
        batch_uploader=uploaded.append,
        heartbeat_sink=heartbeats.append,
        max_consecutive_failures=2,
    )

    worker.run(total_steps=2)

    assert env.transport.reconnect_calls == 1
    assert env.reset_count == 2
    assert len(uploaded) == 1
    assert worker.worker_crash_count == 1
    assert worker.consecutive_failures == 0
    assert worker.last_error is None
    assert heartbeats[0] == {
        "checkpoint_version": -1,
        "error": "TimeoutError: simulated transport timeout",
        "learner_endpoint": None,
        "policy_version": 0,
        "rollout_steps": 0,
        "status": "recovering",
        "worker_crash_count": 1,
    }
    assert heartbeats[-1]["status"] == "running"
    assert heartbeats[-1]["worker_crash_count"] == 1


def test_game_worker_recovery_reconnects_wrapped_env_transport() -> None:
    inner = FailOnceEnv()
    env = EnvWrapper(inner)
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=2),
        max_consecutive_failures=2,
    )

    worker.run(total_steps=2)

    assert inner.transport.reconnect_calls == 1
    assert worker.worker_crash_count == 1


def test_game_worker_switches_task_from_provider_through_wrappers() -> None:
    inner = SwitchableEnv(TaskConfig(task_id="a", wire_id=1, scene="A"))
    env = EnvWrapper(inner)
    target_task = TaskConfig(task_id="b", wire_id=2, scene="B")
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=1),
        task_provider=lambda: target_task,
    )

    batch = worker.collect_rollout()

    assert inner.set_task_wire_ids == [2]
    assert inner.reset_count == 1
    assert int(batch.task_ids[0, 0]) == 2


def test_game_worker_limits_repeated_runtime_failures() -> None:
    env = AlwaysFailEnv()
    model = MlpActorCritic(
        {
            "global": env.observation_space["global"].shape,
            "player": env.observation_space["player"].shape,
            "entities": env.observation_space["entities"].shape,
            "entity_mask": env.observation_space["entity_mask"].shape,
        },
        hidden=16,
        enable_macro=False,
    )
    worker = GameWorker(
        env=env,  # type: ignore[arg-type]
        model=model,
        config=TrainConfig(algorithm="ppo", rollout_steps=2),
        max_consecutive_failures=1,
    )

    with pytest.raises(RuntimeError, match="exceeded max_consecutive_failures=1"):
        worker.run(total_steps=2)

    assert env.transport.reconnect_calls == 1
    assert worker.worker_crash_count == 2
    assert worker.consecutive_failures == 2
    assert worker.last_error == "TimeoutError: persistent transport timeout"


class FakeEnv:
    def __init__(self) -> None:
        self.observation_space = make_observation_space(max_entities=4, tier="privileged")
        self.action_space = make_action_space(enable_macro=False)
        self.transport = FakeTransport()
        self.reset_count = 0
        self.step_count = 0
        self.actions: list[dict[str, Any]] = []

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.reset_count += 1
        return self._obs(), self._info(episode_id=self.reset_count)

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self.step_count += 1
        terminated = self.step_count == 3
        return self._obs(), 1.0, terminated, False, self._info(episode_id=self.reset_count)

    def _obs(self) -> dict[str, np.ndarray]:
        obs = self.observation_space.sample()
        return {
            "global": np.asarray(obs["global"], dtype=np.float32),
            "player": np.asarray(obs["player"], dtype=np.float32),
            "entities": np.asarray(obs["entities"], dtype=np.float32),
            "entity_mask": np.ones((4,), dtype=np.int8),
        }

    @staticmethod
    def _info(episode_id: int) -> dict[str, Any]:
        return {
            "episode_id": episode_id,
            "action_mask": np.ones((19,), dtype=bool),
        }


class FailOnceEnv(FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.failed:
            self.failed = True
            raise TimeoutError("simulated transport timeout")
        return super().step(action)


class AlwaysFailEnv(FakeEnv):
    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        raise TimeoutError("persistent transport timeout")


class SwitchableEnv(FakeEnv):
    def __init__(self, task: TaskConfig) -> None:
        super().__init__()
        self.task = task
        self.set_task_wire_ids: list[int] = []

    def set_task(self, task: TaskConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.task = task
        self.set_task_wire_ids.append(task.wire_id)
        self.reset_count += 1
        return self._obs(), {
            "episode_id": self.reset_count,
            "task_id": task.wire_id,
            "action_mask": np.ones((19,), dtype=bool),
        }


class FakeTransport:
    def __init__(self) -> None:
        self.reconnect_calls = 0

    def reconnect(self, timeout_s: float = 10.0) -> None:
        del timeout_s
        self.reconnect_calls += 1


class EnvWrapper:
    def __init__(self, env: FakeEnv) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self.env.reset()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        return self.env.step(action)


class FakeCheckpointClient:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.latest_calls = 0
        self.pulled_versions: list[int] = []

    def latest_version(self) -> int:
        self.latest_calls += 1
        return 1

    def pull(self, version: int) -> dict[str, Any]:
        self.pulled_versions.append(version)
        return self.state
