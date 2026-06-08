"""Action-space / mask-layout consistency tests.

The action_mask layout MUST stay self-consistent and match the documented order
(docs/action_space.md §3). Drift here is the #1 cause of invalid_action_ratio.
"""

from __future__ import annotations

import re
from pathlib import Path

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


def test_action_mask_layout_rejects_negative_macro_count() -> None:
    with pytest.raises(ValueError, match="n_macros"):
        spaces.action_mask_layout(enable_macro=True, n_macros=-1)


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


def test_csharp_action_masker_constants_match_python_layout() -> None:
    constants = _csharp_int_constants(
        Path(__file__).parents[2] / "mod/HKRLEnvMod/Action/ActionMasker.cs"
    )

    assert constants["MovementCount"] == spaces.N_MOVEMENT_X
    assert constants["AimCount"] == spaces.N_AIM_Y
    assert constants["ButtonCount"] == spaces.N_BUTTONS
    assert constants["DurationCount"] == spaces.N_DURATION
    assert constants["DefaultMacroCount"] == spaces.DEFAULT_N_MACROS

    assert constants["MovementOffset"] == 0
    assert constants["AimOffset"] == spaces.N_MOVEMENT_X
    assert constants["ButtonOffset"] == spaces.N_MOVEMENT_X + spaces.N_AIM_Y
    assert constants["DurationOffset"] == (spaces.N_MOVEMENT_X + spaces.N_AIM_Y + spaces.N_BUTTONS)
    assert constants["MacroOffset"] == len(spaces.action_mask_layout(enable_macro=False))

    assert constants["ButtonJumpTap"] == spaces.BUTTON_BITS["jump_tap"]
    assert constants["ButtonJumpHold"] == spaces.BUTTON_BITS["jump_hold"]
    assert constants["ButtonDash"] == spaces.BUTTON_BITS["dash"]
    assert constants["ButtonAttack"] == spaces.BUTTON_BITS["attack"]
    assert constants["ButtonCast"] == spaces.BUTTON_BITS["cast"]
    assert constants["ButtonFocusHold"] == spaces.BUTTON_BITS["focus_hold"]
    assert constants["ButtonDreamNail"] == spaces.BUTTON_BITS["dream_nail"]
    assert constants["ButtonNailArtHold"] == spaces.BUTTON_BITS["nail_art_hold"]
    assert constants["ButtonNailArtRelease"] == spaces.BUTTON_BITS["nail_art_release"]


def test_csharp_macro_scheduler_cases_match_python_macro_count() -> None:
    text = (Path(__file__).parents[2] / "mod/HKRLEnvMod/Action/MacroActionScheduler.cs").read_text(
        encoding="utf-8"
    )
    cases = sorted({int(value) for value in re.findall(r"^\s*(\d+)\s*=>", text, re.MULTILINE)})

    assert cases == list(range(spaces.DEFAULT_N_MACROS))


def test_csharp_action_masker_masks_unavailable_macros_after_noop_slot() -> None:
    path = Path(__file__).parents[2] / "mod/HKRLEnvMod/Action/ActionMasker.cs"
    text = path.read_text(encoding="utf-8")
    constants = _csharp_int_constants(path)

    macro_names = [
        "MacroApproach",
        "MacroRetreat",
        "MacroJumpAttack",
        "MacroPogo",
        "MacroDashAway",
        "MacroDashThrough",
        "MacroCastForward",
        "MacroCastUp",
        "MacroFocusWhenSafe",
        "MacroShortHop",
        "MacroLongJump",
    ]
    assert [constants[name] for name in macro_names] == list(range(spaces.DEFAULT_N_MACROS))

    # action-space macro:0 is "no macro"; mod macro id 0 starts at flat mask macro:1.
    assert "MacroOffset + 1 + macroId" in text

    for name in macro_names[2:]:
        assert f"MaskMacro(mask, {name})" in text


def test_csharp_input_injector_button_mask_matches_python_layout() -> None:
    text = (Path(__file__).parents[2] / "mod/HKRLEnvMod/Action/InputInjector.cs").read_text(
        encoding="utf-8"
    )
    match = re.search(r"ButtonMask\s*=\s*\(1u\s*<<\s*(\d+)\)\s*-\s*1u", text)
    assert match is not None
    assert int(match.group(1)) == spaces.N_BUTTONS


def _csharp_int_constants(path: Path) -> dict[str, int]:
    constants: dict[str, int] = {}
    text = path.read_text(encoding="utf-8")
    for name, expression in re.findall(r"public const int (\w+) = ([^;]+);", text):
        parts = [part.strip() for part in expression.split("+")]
        value = 0
        for part in parts:
            value += int(part) if part.isdigit() else constants[part]
        constants[name] = value
    return constants
