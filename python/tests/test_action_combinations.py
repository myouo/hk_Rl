"""Semantic action-combination catalog and dynamic bitset tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
from hkrl.action_combinations import (
    ACTION_COMBINATIONS,
    CombinationCoverageBandit,
    decode_combination_bits,
    get_action_combination,
    startable_combination_bits,
)
from hkrl.spaces import (
    BUTTON_BITS,
    PLAYER_FEATURE_DIMS,
    PLAYER_FEATURE_INDEX,
    action_mask_layout,
)


def test_catalog_is_stable_complete_and_uses_only_primitive_controls() -> None:
    assert [combo.combo_id for combo in ACTION_COMBINATIONS] == list(
        range(len(ACTION_COMBINATIONS))
    )
    assert len({combo.name for combo in ACTION_COMBINATIONS}) == len(ACTION_COMBINATIONS)
    assert {
        "jump_up_slash",
        "jump_down_slash",
        "jump_cyclone_up",
        "jump_cyclone_down",
        "jump_quake_down",
        "dash_slash_right",
    } <= {combo.name for combo in ACTION_COMBINATIONS}

    for combo in ACTION_COMBINATIONS:
        assert combo.phases
        for phase in combo.phases:
            assert phase.movement_x in {0, 1, 2}
            assert phase.aim_y in {0, 1, 2}
            assert set(phase.buttons) <= set(BUTTON_BITS)
            assert 0 <= phase.duration <= 3
            assert phase.ticks > 0
            action = phase.to_action(enable_macro=False)
            assert "macro" not in action


def test_dynamic_bitset_filters_ground_state_soul_and_live_action_mask() -> None:
    layout = action_mask_layout(enable_macro=False, n_macros=0)
    mask = np.ones((len(layout),), dtype=bool)
    player = np.zeros(
        (PLAYER_FEATURE_DIMS["privileged"],),
        dtype=np.float32,
    )
    player[PLAYER_FEATURE_INDEX["on_ground"]] = 1.0
    player[PLAYER_FEATURE_INDEX["soul"]] = 0.0

    bits = startable_combination_bits(
        mask,
        enable_macro=False,
        n_macros=0,
        player_state=player,
    )
    names = {combo.name for combo in decode_combination_bits(bits)}
    assert "jump_up_slash" in names
    assert "jump_cyclone_up" in names
    assert "jump_scream_up" not in names

    player[PLAYER_FEATURE_INDEX["soul"]] = 33.0
    bits_with_soul = startable_combination_bits(
        mask,
        enable_macro=False,
        n_macros=0,
        player_state=player,
    )
    assert "jump_scream_up" in {combo.name for combo in decode_combination_bits(bits_with_soul)}

    mask[layout.index("button:nail_art_hold")] = False
    bits_without_nail_art = startable_combination_bits(
        mask,
        enable_macro=False,
        n_macros=0,
        player_state=player,
    )
    assert "jump_cyclone_up" not in {
        combo.name for combo in decode_combination_bits(bits_without_nail_art)
    }


def test_combination_catalog_rejects_invalid_inputs() -> None:
    with pytest.raises(KeyError, match="teleport_boss"):
        get_action_combination("teleport_boss")
    with pytest.raises(ValueError, match="unknown catalog ids"):
        decode_combination_bits(1 << len(ACTION_COMBINATIONS))
    with pytest.raises(ValueError, match="action_mask length"):
        startable_combination_bits(
            [True],
            enable_macro=False,
            n_macros=0,
        )


def test_coverage_bandit_covers_unseen_startable_combinations_first() -> None:
    bandit = CombinationCoverageBandit()
    candidates = (1 << 3) - 1
    rng = random.Random(7)

    selected = []
    for _ in range(3):
        combo = bandit.select(candidates, rng=rng)
        selected.append(combo.combo_id)
        bandit.observe(combo.combo_id, score=1.0)

    assert set(selected) == {0, 1, 2}
    assert bandit.counts[:3] == (1, 1, 1)
