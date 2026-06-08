"""Scripted/random policy tests."""

from __future__ import annotations

import numpy as np
import pytest
from hkrl import spaces
from hkrl.eval.scripted_policies import RandomPolicy


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


def test_random_policy_rejects_mask_with_no_valid_discrete_choice() -> None:
    action_space = spaces.make_action_space(enable_macro=False)
    policy = RandomPolicy(action_space, seed=123)
    mask = np.zeros(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)

    with pytest.raises(ValueError, match="movement_x"):
        policy.act(obs=None, action_mask=mask)
