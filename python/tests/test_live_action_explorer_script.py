"""Offline contracts for the input-only live combat-action explorer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from hkrl.spaces import BUTTON_BITS, DEFAULT_N_MACROS

ROOT = Path(__file__).parents[2]


def test_catalog_covers_every_requested_semantic_action_family() -> None:
    module = _load_script()
    names = {case.name for case in module.ACTION_CASES}
    families = {case.family for case in module.ACTION_CASES}

    assert {
        "movement",
        "jump",
        "dash",
        "ordinary_attack",
        "dream_nail",
        "spell",
        "focus",
        "nail_art",
        "combination",
        "duration",
        "macro",
    } <= families
    assert {
        "ground_up_slash",
        "aerial_up_slash",
        "aerial_down_slash",
        "fireball_left",
        "fireball_right",
        "scream_up",
        "quake_down",
        "focus_heal",
        "great_slash_left",
        "cyclone_slash_up",
        "dash_slash_right",
        "double_jump",
        "combo_jump_up_slash",
        "combo_jump_down_slash",
        "combo_jump_cyclone_up",
        "combo_jump_cyclone_down",
    } <= names

    macro_names = [name for name in names if name.startswith("macro_")]
    assert len(macro_names) == DEFAULT_N_MACROS
    assert {
        "duration_1_ticks",
        "duration_2_ticks",
        "duration_4_ticks",
        "duration_8_ticks",
    } <= names

    duration_cases = [case for case in module.ACTION_CASES if case.family == "duration"]
    assert [case.expected_hold_steps for case in duration_cases] == [1, 2, 4, 8]
    assert [case.phases[0].duration for case in duration_cases] == [0, 1, 2, 3]

    macros = {
        case.name.removeprefix("macro_"): case
        for case in module.ACTION_CASES
        if case.family == "macro"
    }
    assert macros["pogo"].phases[0].ticks == 5
    assert macros["focus_when_safe"].phases[0].ticks == 120
    assert macros["focus_when_safe"].phases[-1].label == "release"
    assert macros["short_hop"].phases[0].ticks == 4
    assert macros["long_jump"].phases[0].ticks == 4


def test_catalog_phases_use_only_policy_callable_controls() -> None:
    module = _load_script()

    for case in module.ACTION_CASES:
        assert case.phases
        for phase in case.phases:
            assert phase.movement in {0, 1, 2}
            assert phase.aim in {0, 1, 2}
            assert set(phase.buttons) <= set(BUTTON_BITS)
            assert 0 <= phase.duration <= 3
            assert 1 <= phase.ticks <= module.MAX_SAFE_TICKS
            assert 0 <= phase.macro <= DEFAULT_N_MACROS


def test_explorer_source_has_no_privileged_game_control_path() -> None:
    source = (ROOT / "scripts/live_action_explorer.py").read_text(encoding="utf-8")

    assert ".pause(" not in source
    assert ".resume(" not in source
    assert ".set_timescale(" not in source
    assert "BossSceneController" not in source
    assert "HealthManager" not in source
    assert "PlayMakerFSM" not in source
    assert "SetValue" not in source
    assert "boss_mutation_allowed" in source
    assert "not bool(player[P_DASHING])" in source


def test_case_selection_rejects_unknown_names_and_can_exclude_macros() -> None:
    module = _load_script()

    selected = module.select_cases(
        names=[],
        families=[],
        exclude_macros=True,
    )
    assert selected
    assert all(case.family != "macro" for case in selected)

    try:
        module.select_cases(
            names=["teleport_boss"],
            families=[],
            exclude_macros=False,
        )
    except ValueError as exc:
        assert "teleport_boss" in str(exc)
    else:
        raise AssertionError("unknown case was accepted")


def _load_script() -> ModuleType:
    path = ROOT / "scripts/live_action_explorer.py"
    spec = importlib.util.spec_from_file_location("live_action_explorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
