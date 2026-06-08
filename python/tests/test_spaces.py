"""Action-space / mask-layout consistency tests.

The action_mask layout MUST stay self-consistent and match the documented order
(docs/action_space.md §3). Drift here is the #1 cause of invalid_action_ratio.
"""

from __future__ import annotations

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
    with_macro = len(spaces.action_mask_layout(enable_macro=True, n_macros=11))
    assert with_macro == base + 12  # n_macros + 1 (none)


def test_duration_ticks_match_count() -> None:
    assert len(spaces.DURATION_TICKS) == spaces.N_DURATION


@pytest.mark.xfail(reason="gym space construction lands in phase 2", strict=True)
def test_make_action_space() -> None:
    spaces.make_action_space()
