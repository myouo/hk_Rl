"""Scripted/random policy tests."""

from __future__ import annotations

import numpy as np
import pytest
from hkrl import spaces
from hkrl.eval.scripted_policies import RandomPolicy, ScriptedAggroPolicy


def test_random_policy_samples_from_action_space_without_mask() -> None:
    action_space = spaces.make_action_space(enable_macro=True, n_macros=2)
    policy = RandomPolicy(action_space, seed=123)

    action = policy.act(obs=None)

    assert action_space.contains(action)


def test_random_policy_respects_flat_action_mask() -> None:
    action_space = spaces.make_action_space(enable_macro=True, n_macros=2)
    policy = RandomPolicy(action_space, seed=123)
    layout = spaces.action_mask_layout(enable_macro=True, n_macros=2)
    mask = np.zeros(len(layout), dtype=bool)

    for name in ("movement_x:2", "aim_y:1", "duration:4", "macro:0"):
        mask[layout.index(name)] = True

    action = policy.act(obs=None, action_mask=mask)

    assert action["movement_x"] == 2
    assert action["aim_y"] == 1
    assert action["duration"] == 2  # index for duration tick 4
    assert action["macro"] == 0
    assert np.array_equal(action["buttons"], np.zeros(spaces.N_BUTTONS, dtype=np.int8))


def test_random_policy_uses_current_mask_layout_across_task_changes() -> None:
    action_space = spaces.make_action_space(enable_macro=True, n_macros=2)
    policy = RandomPolicy(action_space, seed=123)
    layout = spaces.action_mask_layout(enable_macro=False)
    mask = np.zeros(len(layout), dtype=bool)

    for name in ("movement_x:1", "aim_y:1", "duration:1"):
        mask[layout.index(name)] = True

    action = policy.act(obs=None, action_mask=mask)

    assert "macro" not in action
    assert action["movement_x"] == 1
    assert action["aim_y"] == 1
    assert action["duration"] == 0


def test_random_policy_rejects_mask_with_no_valid_discrete_choice() -> None:
    action_space = spaces.make_action_space(enable_macro=False)
    policy = RandomPolicy(action_space, seed=123)
    mask = np.zeros(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)

    with pytest.raises(ValueError, match="movement_x"):
        policy.act(obs=None, action_mask=mask)


def test_scripted_aggro_policy_attacks_near_boss() -> None:
    action_space = spaces.make_action_space(enable_macro=True, n_macros=2)
    policy = ScriptedAggroPolicy(action_space)
    obs = _obs_with_entity(rel_x=0.05, rel_y=0.3)

    action = policy.act(
        obs, action_mask=np.ones(len(spaces.action_mask_layout(True, 2)), dtype=bool)
    )

    assert action["movement_x"] == 1
    assert action["aim_y"] == 2
    assert action["buttons"][spaces.BUTTON_BITS["attack"]] == 1
    assert action["macro"] == 0


def test_scripted_aggro_policy_uses_current_mask_layout_across_task_changes() -> None:
    action_space = spaces.make_action_space(enable_macro=False)
    policy = ScriptedAggroPolicy(action_space)
    layout = spaces.action_mask_layout(enable_macro=True, n_macros=3)
    mask = np.zeros(len(layout), dtype=bool)

    for name in ("movement_x:1", "aim_y:1", "duration:1", "macro:0"):
        mask[layout.index(name)] = True

    action = policy.act(_obs_with_entity(rel_x=0.0, rel_y=0.0), action_mask=mask)

    assert action["movement_x"] == 1
    assert action["aim_y"] == 1
    assert action["duration"] == 0
    assert action["macro"] == 0


def test_scripted_aggro_policy_respects_action_mask_fallbacks() -> None:
    action_space = spaces.make_action_space(enable_macro=False)
    policy = ScriptedAggroPolicy(action_space)
    layout = spaces.action_mask_layout(enable_macro=False)
    mask = np.zeros(len(layout), dtype=bool)
    for name in ("movement_x:1", "aim_y:1", "duration:1"):
        mask[layout.index(name)] = True

    action = policy.act(_obs_with_entity(rel_x=2.0, rel_y=-2.0), action_mask=mask)

    assert action["movement_x"] == 1
    assert action["aim_y"] == 1
    assert action["duration"] == 0
    np.testing.assert_array_equal(action["buttons"], np.zeros(spaces.N_BUTTONS, dtype=np.int8))


def test_scripted_aggro_policy_uses_neutral_without_entities() -> None:
    action_space = spaces.make_action_space(enable_macro=False)
    policy = ScriptedAggroPolicy(action_space)
    action = policy.act({"entities": np.zeros((0, 24), dtype=np.float32)})

    assert action["movement_x"] == 1
    assert action["aim_y"] == 1


def _obs_with_entity(rel_x: float, rel_y: float) -> dict[str, np.ndarray]:
    entities = np.zeros((2, spaces.ENTITY_FEATURE_DIMS["privileged"]), dtype=np.float32)
    entities[0, 1] = 1.0  # EntityType.Boss
    entities[0, 8] = rel_x
    entities[0, 9] = rel_y
    entities[1, 1] = 2.0
    entities[1, 8] = -5.0
    entities[1, 9] = 0.0
    return {
        "entities": entities,
        "entity_mask": np.array([1, 1], dtype=np.int8),
    }
