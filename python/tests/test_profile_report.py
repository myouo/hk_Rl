"""Phase 8 profiling report tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from hkrl.coordinator.profiling import build_profile_report, render_profile_markdown


def test_profile_report_summarizes_worker_timing_and_findings() -> None:
    report = build_profile_report(_summary(include_timing=True))

    assert report["source"] == "phase8_smoke"
    assert report["metrics"]["assigned_worker_count"] == 2.0
    assert report["metrics"]["learner_accepted_batches"] == 1.0
    assert report["metrics"]["learner_rejected_batches"] == 0.0
    assert report["metrics"]["sps"] == 32.0
    assert report["metrics"]["rollout_duration_s_mean"] == 2.0
    assert report["metrics"]["rollout_duration_s_max"] == 4.0
    assert report["metrics"]["rollout_steps_total"] == 128.0
    assert report["metrics"]["worker_learner_upload_accepted_batches"] == 1.0
    assert report["metrics"]["worker_learner_upload_failed_batches"] == 0.0
    assert report["metrics"]["worker_learner_upload_rejected_batches"] == 0.0
    assert report["metrics"]["worker_learner_upload_submitted_batches"] == 1.0
    assert report["metrics"]["worker_without_policy_version_count"] == 0.0
    assert report["metrics"]["worker_without_checkpoint_version_count"] == 0.0
    assert [finding["code"] for finding in report["findings"]] == [
        "recovering_workers",
        "worker_crashes",
        "stale_policy_workers",
        "stale_checkpoint_workers",
    ]
    assert report["workers"][0]["worker_id"] == "worker-0"
    assert report["workers"][0]["learner_upload_submitted_batches"] == 1.0
    assert report["workers"][0]["rollout_duration_s"] == 4.0


def test_profile_report_flags_unassigned_workers() -> None:
    summary = _summary(include_timing=True)
    metrics = summary["coordinator"]["metrics"]
    assert isinstance(metrics, dict)
    metrics["assigned_worker_count"] = 1.0

    report = build_profile_report(summary)

    assert report["metrics"]["unassigned_worker_count"] == 1.0
    assert "unassigned_workers" in {finding["code"] for finding in report["findings"]}


def test_profile_report_flags_learner_backpressure() -> None:
    summary = _summary(include_timing=True)
    learner = summary["learner"]
    assert isinstance(learner, dict)
    learner["queued_batches"] = 2
    learner["rejected_batches"] = 1

    report = build_profile_report(summary)

    assert report["metrics"]["learner_queued_batches"] == 2.0
    assert report["metrics"]["learner_rejected_batches"] == 1.0
    assert "learner_queued_batches" in {finding["code"] for finding in report["findings"]}
    assert "learner_rejected_batches" in {finding["code"] for finding in report["findings"]}


def test_profile_report_flags_worker_learner_upload_issues() -> None:
    summary = _summary(include_timing=True)
    metrics = summary["coordinator"]["metrics"]
    assert isinstance(metrics, dict)
    metrics["worker_learner_upload_failed_batches"] = 1.0
    metrics["worker_learner_upload_rejected_batches"] = 2.0

    report = build_profile_report(summary)

    assert report["metrics"]["worker_learner_upload_failed_batches"] == 1.0
    assert report["metrics"]["worker_learner_upload_rejected_batches"] == 2.0
    assert "worker_learner_upload_failures" in {finding["code"] for finding in report["findings"]}
    assert "worker_learner_upload_rejections" in {finding["code"] for finding in report["findings"]}


def test_profile_report_flags_missing_worker_timing() -> None:
    report = build_profile_report(_summary(include_timing=False))

    assert "missing_rollout_timing" in {finding["code"] for finding in report["findings"]}


def test_profile_report_flags_missing_worker_versions() -> None:
    summary = _summary(include_timing=True)
    metrics = summary["coordinator"]["metrics"]
    assert isinstance(metrics, dict)
    metrics["worker_without_checkpoint_version_count"] = 1.0
    metrics["worker_without_policy_version_count"] = 1.0

    report = build_profile_report(summary)

    assert report["metrics"]["worker_without_policy_version_count"] == 1.0
    assert report["metrics"]["worker_without_checkpoint_version_count"] == 1.0
    assert "missing_policy_versions" in {finding["code"] for finding in report["findings"]}
    assert "missing_checkpoint_versions" in {finding["code"] for finding in report["findings"]}


def test_profile_report_flags_version_lag_without_stale_counts() -> None:
    summary = _summary(include_timing=True)
    coordinator = summary["coordinator"]
    assert isinstance(coordinator, dict)
    metrics = coordinator["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(
        {
            "recovering_worker_count": 0.0,
            "stale_checkpoint_worker_count": 0.0,
            "stale_policy_worker_count": 0.0,
            "worker_checkpoint_lag_max": 2.0,
            "worker_crash_count": 0.0,
            "worker_policy_lag_max": 1.0,
        }
    )
    workers = coordinator["workers"]
    assert isinstance(workers, dict)
    worker_1 = workers["worker-1"]
    assert isinstance(worker_1, dict)
    worker_1["info"] = {"status": "running"}
    worker_metrics = worker_1["metrics"]
    assert isinstance(worker_metrics, dict)
    worker_metrics["worker_crash_count"] = 0

    report = build_profile_report(summary)

    assert [finding["code"] for finding in report["findings"]] == [
        "stale_policy_workers",
        "stale_checkpoint_workers",
    ]


def test_profile_report_markdown_contains_findings_and_worker_table() -> None:
    markdown = render_profile_markdown(build_profile_report(_summary(include_timing=True)))

    assert "# HKRL Phase 8 Profile" in markdown
    assert "`recovering_workers`" in markdown
    assert "| worker-0 | running | 32 | 4 | 128 | 0 | 1 | 1 | 0 | 0 |" in markdown


def test_render_profile_report_script_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_script("render_profile_report.py")
    summary_path = tmp_path / "summary.json"
    json_path = tmp_path / "profile.json"
    markdown_path = tmp_path / "profile.md"
    summary_path.write_text(json.dumps(_summary(include_timing=True)), encoding="utf-8")
    args = argparse.Namespace(
        summary=str(summary_path),
        output_json=str(json_path),
        output_md=str(markdown_path),
    )

    report = module.run_from_args(args)

    assert report["metrics"]["sps_per_active_worker"] == 16.0
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["sps"] == 32.0
    assert "HKRL Phase 8 Profile" in markdown_path.read_text(encoding="utf-8")


def _summary(*, include_timing: bool) -> dict[str, object]:
    workers = {
        "worker-0": _worker(
            status="running",
            checkpoint_version=2,
            policy_version=2,
            rollout_duration_s=4.0 if include_timing else None,
            rollout_steps=128,
            sps=32.0,
            crashes=0,
        ),
        "worker-1": _worker(
            status="recovering",
            checkpoint_version=1,
            policy_version=1,
            rollout_duration_s=0.0 if include_timing else None,
            rollout_steps=0,
            sps=0.0,
            crashes=1,
        ),
    }
    return {
        "coordinator": {
            "metrics": {
                "active_worker_count": 2.0,
                "assigned_worker_count": 2.0,
                "lost_worker_count": 0.0,
                "recovering_worker_count": 1.0,
                "sps": 32.0,
                "stale_checkpoint_worker_count": 1.0,
                "stale_policy_worker_count": 1.0,
                "worker_checkpoint_lag_max": 1.0,
                "worker_count": 2.0,
                "worker_crash_count": 1.0,
                "worker_learner_upload_accepted_batches": 1.0,
                "worker_learner_upload_failed_batches": 0.0,
                "worker_learner_upload_rejected_batches": 0.0,
                "worker_learner_upload_submitted_batches": 1.0,
                "worker_policy_lag_max": 1.0,
                "worker_without_checkpoint_version_count": 0.0,
                "worker_without_policy_version_count": 0.0,
            },
            "workers": workers,
        },
        "learner": {
            "accepted_batches": 1,
            "network_accepted_batches": 1,
            "network_submitted_batches": 1,
            "queued_batches": 0,
            "rejected_batches": 0,
            "submitted_batches": 1,
        },
    }


def _worker(
    *,
    status: str,
    checkpoint_version: int,
    policy_version: int,
    rollout_duration_s: float | None,
    rollout_steps: int,
    sps: float,
    crashes: int,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "checkpoint_version": checkpoint_version,
        "learner_upload_accepted_batches": 1 if status == "running" else 0,
        "learner_upload_failed_batches": 0,
        "learner_upload_rejected_batches": 0,
        "learner_upload_submitted_batches": 1 if status == "running" else 0,
        "policy_version": policy_version,
        "rollout_steps": rollout_steps,
        "sps": sps,
        "worker_crash_count": crashes,
    }
    if rollout_duration_s is not None:
        metrics["rollout_duration_s"] = rollout_duration_s
    return {
        "alive": True,
        "info": {"status": status},
        "metrics": metrics,
    }


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
