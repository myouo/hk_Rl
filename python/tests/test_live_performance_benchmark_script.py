"""Import-level checks for the real-game performance benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_live_performance_benchmark_argument_validation() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="steps must be positive"):
        module.main(["--steps", "0"])
    with pytest.raises(ValueError, match=r"\[1, 200\]"):
        module.main(["--action-repeat", "201"])
    with pytest.raises(ValueError, match="time-scale must be positive"):
        module.main(["--time-scale", "0"])


def test_benchmark_reports_slow_step_diagnostics() -> None:
    source = (ROOT / "scripts" / "live_performance_benchmark.py").read_text(encoding="utf-8")

    assert '"slow_step_count_over_100ms"' in source
    assert '"server_tick_delta": tick_deltas[index]' in source
    assert '"damage_taken_step_count"' in source
    assert '"non_damage_latency_ms"' in source
    assert '"event_kinds": event_kinds[index]' in source


def _load_script():
    path = ROOT / "scripts" / "live_performance_benchmark.py"
    spec = importlib.util.spec_from_file_location("live_performance_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
