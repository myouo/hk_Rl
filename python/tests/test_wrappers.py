"""Observation wrapper tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from hkrl import spaces
from hkrl.wrappers import FrameStack, NormalizeObservation, ObservationTier


class DummyEnv(gym.Env):
    observation_space = spaces.make_observation_space(max_entities=2, tier="privileged")
    action_space = spaces.make_action_space()


def test_normalize_observation_scales_player_and_entities_without_mutating_input() -> None:
    wrapper = NormalizeObservation(DummyEnv())
    observation = {
        "global": np.zeros((spaces.GLOBAL_FEATURE_DIM,), dtype=np.float32),
        "player": np.zeros((spaces.PLAYER_FEATURE_DIMS["privileged"],), dtype=np.float32),
        "entities": np.zeros((2, spaces.ENTITY_FEATURE_DIMS["privileged"]), dtype=np.float32),
        "entity_mask": np.array([1, 0], dtype=np.int8),
    }
    observation["player"][[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 20]] = [
        30.0,
        -60.0,
        20.0,
        -40.0,
        5.0,
        10.0,
        33.0,
        99.0,
        4.0,
        -1.0,
        1.0,
        0.5,
    ]
    observation["entities"][0, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]] = [
        30.0,
        -30.0,
        15.0,
        -15.0,
        20.0,
        -20.0,
        10.0,
        20.0,
        3.0,
        6.0,
        9.0,
        12.0,
        4.0,
    ]
    original_player = observation["player"].copy()
    original_entities = observation["entities"].copy()

    normalized = wrapper.observation(observation)

    np.testing.assert_allclose(normalized["player"][0:4], [1.0, -2.0, 1.0, -2.0])
    np.testing.assert_allclose(normalized["player"][[4, 5, 6, 7]], [0.5, 1.0, 1 / 3, 1.0])
    np.testing.assert_allclose(normalized["player"][[16, 17, 18, 20]], [1.0, 0.0, 0.5, 0.25])

    np.testing.assert_allclose(
        normalized["entities"][0, 6:14],
        [1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(normalized["entities"][0, 14:18], [0.1, 0.2, 0.3, 0.4])
    assert normalized["entities"][0, 20] == 1.0
    np.testing.assert_allclose(normalized["entities"][1], 0.0)
    np.testing.assert_array_equal(normalized["entity_mask"], observation["entity_mask"])

    np.testing.assert_array_equal(observation["player"], original_player)
    np.testing.assert_array_equal(observation["entities"], original_entities)


def test_normalize_observation_rejects_mismatched_entity_mask() -> None:
    wrapper = NormalizeObservation(DummyEnv())
    observation = {
        "entities": np.zeros((2, spaces.ENTITY_FEATURE_DIMS["privileged"]), dtype=np.float32),
        "entity_mask": np.array([1], dtype=np.int8),
    }

    with pytest.raises(ValueError, match="entity_mask length"):
        wrapper.observation(observation)


def test_observation_tier_slices_player_and_entity_features() -> None:
    wrapper = ObservationTier(DummyEnv(), tier="human_visible")
    observation = {
        "global": np.zeros((spaces.GLOBAL_FEATURE_DIM,), dtype=np.float32),
        "player": np.arange(spaces.PLAYER_FEATURE_DIMS["privileged"], dtype=np.float32),
        "entities": np.ones((2, spaces.ENTITY_FEATURE_DIMS["privileged"]), dtype=np.float32),
        "entity_mask": np.array([1, 0], dtype=np.int8),
    }

    tiered = wrapper.observation(observation)

    assert wrapper.observation_space["player"].shape == (
        spaces.PLAYER_FEATURE_DIMS["human_visible"],
    )
    assert wrapper.observation_space["entities"].shape == (
        2,
        spaces.ENTITY_FEATURE_DIMS["human_visible"],
    )
    assert tiered["player"].shape == (spaces.PLAYER_FEATURE_DIMS["human_visible"],)
    assert tiered["entities"].shape == (2, spaces.ENTITY_FEATURE_DIMS["human_visible"])
    np.testing.assert_array_equal(tiered["entity_mask"], observation["entity_mask"])
    assert observation["player"].shape == (spaces.PLAYER_FEATURE_DIMS["privileged"],)


def test_observation_tier_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError, match="unknown observation tier"):
        ObservationTier(DummyEnv(), tier="debug")


def test_frame_stack_stacks_feature_axes_and_updates_space() -> None:
    env = CountingEnv()
    wrapper = FrameStack(env, k=3)

    assert wrapper.observation_space["global"].shape == (6,)
    assert wrapper.observation_space["player"].shape == (9,)
    assert wrapper.observation_space["entities"].shape == (2, 3)
    assert wrapper.observation_space["entity_mask"].n == 2

    obs, _ = wrapper.reset()
    np.testing.assert_array_equal(obs["global"], np.array([0, 1, 0, 1, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(
        obs["entities"], np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(obs["entity_mask"], np.array([1, 0], dtype=np.int8))

    obs, reward, terminated, truncated, _ = wrapper.step(0)

    assert reward == 1.0
    assert not terminated
    assert not truncated
    np.testing.assert_array_equal(obs["global"], np.array([0, 1, 0, 1, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(
        obs["player"], np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    )


def test_frame_stack_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        FrameStack(CountingEnv(), k=0)


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "global": gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            "player": gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            "entities": gym.spaces.Box(low=-10, high=10, shape=(2, 1), dtype=np.float32),
            "entity_mask": gym.spaces.MultiBinary(2),
        }
    )
    action_space = gym.spaces.Discrete(1)

    def __init__(self) -> None:
        super().__init__()
        self.t = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del options
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(
        self, action: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        del action
        self.t += 1
        return self._obs(), 1.0, False, False, {}

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "global": np.array([self.t, self.t + 1], dtype=np.float32),
            "player": np.full((3,), self.t, dtype=np.float32),
            "entities": np.array([[self.t], [self.t + 1]], dtype=np.float32),
            "entity_mask": np.array([1, 0], dtype=np.int8),
        }
