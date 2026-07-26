"""Offline contracts for the live Hero movement-continuity benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]


def test_walk_smoothness_argument_validation() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match=r"\[2, 200\]"):
        module.main(["--active-ticks", "1"])
    with pytest.raises(ValueError, match="divisible"):
        module.main(["--active-ticks", "5", "--decision-repeat", "2"])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        module.main(["--min-speed-retention", "0"])


def test_walk_comparison_detects_and_accepts_speed_retention() -> None:
    module = _load_script()
    continuous = _trial(module, label="continuous", speed=8.0)
    stuttering = _trial(module, label="stepped", speed=5.2)
    smooth = _trial(module, label="stepped", speed=7.6)

    failed = module.compare_walk_trials(
        continuous,
        stuttering,
        min_speed_retention=0.9,
    )
    passed = module.compare_walk_trials(
        continuous,
        smooth,
        min_speed_retention=0.9,
    )

    assert failed["speed_retention"] == 0.65
    assert not failed["smooth"]
    assert passed["speed_retention"] == 0.95
    assert passed["smooth"]
    assert not passed["boss_mutation_allowed"]
    assert not passed["simulation_control_allowed"]


def test_walk_source_uses_server_time_and_cannot_mutate_simulation() -> None:
    source = (ROOT / "scripts/live_walk_smoothness.py").read_text(encoding="utf-8")

    assert 'int(info["server_tick"]) - start_tick' in source
    assert "server_ticks * fixed_delta" in source
    assert '"boss_mutation_allowed": False' in source
    assert '"simulation_control_allowed": False' in source
    assert '"tested_mod"' in source
    assert "fingerprint_file(args.mod_dll)" in source
    assert ".pause(" not in source
    assert ".set_timescale(" not in source
    assert "HealthManager" not in source
    assert "BossSceneController" not in source


def _trial(module: ModuleType, *, label: str, speed: float) -> object:
    return module.WalkTrial(
        label=label,
        direction="right",
        action_repeat=2,
        decisions=12,
        commanded_ticks=24,
        server_ticks=36,
        fixed_delta_time_s=0.02,
        wall_time_s=0.72,
        start_x=10.0,
        end_x=10.0 + speed * 0.72,
        signed_displacement=speed * 0.72,
        distance=speed * 0.72,
        game_time_s=0.72,
        game_speed_units_s=speed,
        wall_speed_units_s=speed,
        response_velocity_x=(speed,),
        event_kinds=(),
        start_hp=9,
        end_hp=9,
    )


def _load_script() -> ModuleType:
    path = ROOT / "scripts/live_walk_smoothness.py"
    spec = importlib.util.spec_from_file_location("live_walk_smoothness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
