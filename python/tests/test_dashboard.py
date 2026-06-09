"""Phase 8 dashboard tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.coordinator.dashboard import build_dashboard_model, render_dashboard_html


def test_dashboard_model_reports_degraded_worker_lag_and_tasks() -> None:
    model = build_dashboard_model(_phase8_summary())

    assert model["health"] == {
        "status": "degraded",
        "reasons": ["workers recovering", "worker crashes", "stale policy workers"],
    }
    assert model["metrics"]["sps"] == 12.5
    assert model["metrics"]["worker_policy_lag_max"] == 2.0
    assert model["metrics"]["worker_without_policy_version_count"] == 0.0
    assert model["metrics"]["worker_without_checkpoint_version_count"] == 0.0
    assert model["workers"] == [
        {
            "alive": True,
            "assigned_task": "gruz",
            "checkpoint_lag": 0.0,
            "checkpoint_version": 3.0,
            "policy_lag": 0.0,
            "policy_version": 7.0,
            "sps": 12.5,
            "status": "running",
            "worker_crash_count": 0.0,
            "worker_id": "worker-a",
        },
        {
            "alive": True,
            "assigned_task": "hornet",
            "checkpoint_lag": 1.0,
            "checkpoint_version": 2.0,
            "policy_lag": 2.0,
            "policy_version": 5.0,
            "sps": 0.0,
            "status": "recovering",
            "worker_crash_count": 1.0,
            "worker_id": "worker-b",
        },
    ]
    assert model["tasks"] == [
        {
            "mastered": True,
            "sampler_weight": 0.1,
            "task_id": "gruz",
            "win_rate": 0.9,
        },
        {
            "mastered": False,
            "sampler_weight": 0.8,
            "task_id": "hornet",
            "win_rate": 0.2,
        },
    ]


def test_dashboard_html_escapes_worker_and_task_values() -> None:
    summary = _phase8_summary()
    summary["coordinator"]["workers"]["<worker>"] = summary["coordinator"]["workers"].pop(
        "worker-a"
    )
    summary["coordinator"]["workers"]["<worker>"]["assigned_task"] = "<task>"

    html = render_dashboard_html(build_dashboard_model(summary))

    assert "&lt;worker&gt;" in html
    assert "&lt;task&gt;" in html
    assert "<worker>" not in html


def test_dashboard_rejects_summary_without_metrics() -> None:
    with pytest.raises(ValueError, match="metrics object"):
        build_dashboard_model({"coordinator": {}})


def test_dashboard_health_flags_worker_crashes_without_active_recovery() -> None:
    summary = _phase8_summary()
    metrics = summary["coordinator"]["metrics"]
    assert isinstance(metrics, dict)
    metrics["recovering_worker_count"] = 0.0
    metrics["stale_policy_worker_count"] = 0.0
    worker_b = summary["coordinator"]["workers"]["worker-b"]
    assert isinstance(worker_b, dict)
    worker_b["info"] = {"status": "running"}

    model = build_dashboard_model(summary)

    assert model["health"] == {
        "status": "degraded",
        "reasons": ["worker crashes"],
    }


def test_dashboard_health_flags_workers_missing_versions() -> None:
    summary = _phase8_summary()
    metrics = summary["coordinator"]["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(
        {
            "recovering_worker_count": 0.0,
            "stale_policy_worker_count": 0.0,
            "worker_crash_count": 0.0,
            "worker_without_checkpoint_version_count": 1.0,
            "worker_without_policy_version_count": 1.0,
        }
    )
    worker_b = summary["coordinator"]["workers"]["worker-b"]
    assert isinstance(worker_b, dict)
    worker_b["info"] = {"status": "running"}
    worker_metrics = worker_b["metrics"]
    assert isinstance(worker_metrics, dict)
    worker_metrics.pop("checkpoint_version")
    worker_metrics.pop("policy_version")
    worker_metrics["worker_crash_count"] = 0

    model = build_dashboard_model(summary)

    assert model["health"] == {
        "status": "degraded",
        "reasons": [
            "workers missing policy version",
            "workers missing checkpoint version",
        ],
    }
    assert model["workers"][1]["policy_version"] is None
    assert model["workers"][1]["checkpoint_version"] is None


def test_render_phase8_dashboard_script_writes_html_and_json(tmp_path: Path) -> None:
    module = _load_script("render_phase8_dashboard.py")
    summary_path = tmp_path / "summary.json"
    html_path = tmp_path / "dashboard.html"
    json_path = tmp_path / "dashboard.json"
    summary_path.write_text(json.dumps(_phase8_summary()), encoding="utf-8")
    args = argparse.Namespace(
        summary=str(summary_path),
        output_html=str(html_path),
        output_json=str(json_path),
        eval_metrics=None,
    )

    model = module.run_from_args(args)

    assert model["health"]["status"] == "degraded"
    assert "HKRL Phase 8 Dashboard" in html_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["sps"] == 12.5


def _phase8_summary() -> dict[str, object]:
    return {
        "coordinator": {
            "eval_winrates": {"gruz": 0.9, "hornet": 0.2},
            "metrics": {
                "active_worker_count": 2.0,
                "lost_worker_count": 0.0,
                "recovering_worker_count": 1.0,
                "sps": 12.5,
                "sps_mean": 6.25,
                "stale_checkpoint_worker_count": 0.0,
                "stale_policy_worker_count": 1.0,
                "worker_checkpoint_lag_max": 1.0,
                "worker_checkpoint_version_max": 3.0,
                "worker_count": 2.0,
                "worker_crash_count": 1.0,
                "worker_policy_lag_max": 2.0,
                "worker_policy_version_max": 7.0,
                "worker_without_checkpoint_version_count": 0.0,
                "worker_without_policy_version_count": 0.0,
            },
            "sampler_mastered_tasks": ["gruz"],
            "sampler_weights": {"gruz": 0.1, "hornet": 0.8},
            "task_ids": ["gruz", "hornet"],
            "workers": {
                "worker-a": {
                    "alive": True,
                    "assigned_task": "gruz",
                    "info": {"status": "running"},
                    "metrics": {
                        "checkpoint_version": 3,
                        "policy_version": 7,
                        "sps": 12.5,
                        "worker_crash_count": 0,
                    },
                },
                "worker-b": {
                    "alive": True,
                    "assigned_task": "hornet",
                    "info": {"status": "recovering"},
                    "metrics": {
                        "checkpoint_version": 2,
                        "policy_version": 5,
                        "sps": 0.0,
                        "worker_crash_count": 1,
                    },
                },
            },
        }
    }


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
