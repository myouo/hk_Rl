"""Action-space / mask-layout consistency tests.

The action_mask layout MUST stay self-consistent and match the documented order
(docs/action_space.md §3). Drift here is the #1 cause of invalid_action_ratio.
"""

from __future__ import annotations

import gymnasium as gym
import pytest
from hkrl import spaces


def test_button_bits_are_unique_and_contiguous() -> None:
    bits = list(spaces.BUTTON_BITS.values())
    assert sorted(bits) == list(range(spaces.N_BUTTONS))


def test_action_mask_layout_length() -> None:
    layout = spaces.action_mask_layout(enable_macro=False)
    expected = spaces.N_MOVEMENT_X + spaces.N_AIM_Y + spaces.N_BUTTONS + spaces.N_DURATION
    assert len(layout) == expected


def test_action_mask_layout_with_macros_grows() -> None:
    base = len(spaces.action_mask_layout(enable_macro=False))
    with_macro = len(spaces.action_mask_layout(enable_macro=True, n_macros=spaces.DEFAULT_N_MACROS))
    assert with_macro == base + spaces.DEFAULT_N_MACROS + 1


def test_duration_ticks_match_count() -> None:
    assert len(spaces.DURATION_TICKS) == spaces.N_DURATION


def test_make_action_space() -> None:
    action_space = spaces.make_action_space(
        enable_macro=True,
        n_macros=spaces.DEFAULT_N_MACROS,
    )

    assert isinstance(action_space, gym.spaces.Dict)
    assert action_space["movement_x"].n == spaces.N_MOVEMENT_X
    assert action_space["aim_y"].n == spaces.N_AIM_Y
    assert action_space["buttons"].n == spaces.N_BUTTONS
    assert action_space["duration"].n == spaces.N_DURATION
    assert action_space["macro"].n == spaces.DEFAULT_N_MACROS + 1


def test_make_action_space_without_macro() -> None:
    action_space = spaces.make_action_space(enable_macro=False)

    assert "macro" not in action_space


def test_make_observation_space() -> None:
    observation_space = spaces.make_observation_space(max_entities=32, tier="privileged")

    assert observation_space["global"].shape == (spaces.GLOBAL_FEATURE_DIM,)
    assert observation_space["player"].shape == (spaces.PLAYER_FEATURE_DIMS["privileged"],)
    assert observation_space["entities"].shape == (32, spaces.ENTITY_FEATURE_DIMS["privileged"])
    assert observation_space["entity_mask"].n == 32


def test_make_observation_space_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError, match="unknown observation tier"):
        spaces.make_observation_space(tier="debug")
