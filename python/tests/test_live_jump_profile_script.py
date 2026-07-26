"""Offline contracts for the live jump-amplitude profiler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]


def test_jump_relationship_requires_monotonic_distinct_amplitudes() -> None:
    module = _load_script()
    profiles = tuple(
        module.JumpProfile(
            requested_hold_ticks=ticks,
            status="verified",
            clean_trials=1,
            attempts=1,
            median_height=height,
            median_peak_vy=10.0,
            median_airtime_seconds=1.0,
            trials=(),
        )
        for ticks, height in ((1, 2.0), (2, 2.4), (4, 3.1), (8, 3.8))
    )

    relationship = module.summarize_relationship(profiles)

    assert relationship["verified_profiles"] == 4
    assert relationship["monotonic_non_decreasing_height"] is True
    assert relationship["short_long_height_distinct"] is True
    assert relationship["height_delta_long_minus_short"] == 1.8
    assert relationship["valid"] is True


def test_jump_profiler_source_has_no_privileged_control_path() -> None:
    source = (ROOT / "scripts/live_jump_profile.py").read_text(encoding="utf-8")

    assert ".pause(" not in source
    assert ".set_timescale(" not in source
    assert "BossSceneController" not in source
    assert "HealthManager" not in source
    assert "boss_mutation_allowed" in source
    assert "DAMAGE_TAKEN" in source


def _load_script() -> ModuleType:
    path = ROOT / "scripts/live_jump_profile.py"
    spec = importlib.util.spec_from_file_location("live_jump_profile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
