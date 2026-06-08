"""Observation wrapper tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from hkrl import spaces
from hkrl.wrappers import NormalizeObservation


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
