"""GameWorker rollout tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import make_action_space, make_observation_space
from hkrl.utils.config import TrainConfig
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


class FakeEnv:
    def __init__(self) -> None:
        self.observation_space = make_observation_space(max_entities=4, tier="privileged")
        self.action_space = make_action_space(enable_macro=False)
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
