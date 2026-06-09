"""Release evidence manifest tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.coordinator.dashboard import render_dashboard_html
from hkrl.coordinator.profiling import render_profile_markdown
from hkrl.eval.report import render_eval_report_markdown
from hkrl.utils.release import (
    PHASE8_RELEASE_ARTIFACTS,
    build_release_checklist,
    build_release_evidence_manifest,
    release_evidence_to_json,
    render_release_evidence_markdown,
    render_release_markdown,
    verify_release_evidence_manifest,
)

FULL_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_FULL_GIT_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def test_release_evidence_manifest_hashes_artifacts(tmp_path: Path) -> None:
    summary = tmp_path / "runs" / "phase8-smoke" / "summary.json"
    checklist = tmp_path / "runs" / "release" / "checklist.md"
    _write(summary, '{"ok": true}\n')
    _write(checklist, "# Checklist\n")

    manifest = build_release_evidence_manifest(
        root=tmp_path,
        version="phase8",
        git_sha="abc123",
        artifacts=[
            "runs/phase8-smoke/summary.json",
            "runs/release/checklist.md",
        ],
    )

    assert manifest["version"] == "phase8"
    assert manifest["git_sha"] == "abc123"
    assert manifest["manifest_version"] == 1
    assert manifest["artifact_count"] == 2
    assert manifest["total_bytes"] == summary.stat().st_size + checklist.stat().st_size
    assert manifest["artifacts"][0] == {
        "bytes": summary.stat().st_size,
        "path": "runs/phase8-smoke/summary.json",
        "sha256": _sha256(summary),
    }


def test_release_evidence_manifest_includes_existing_eval_artifacts(tmp_path: Path) -> None:
    required = [
        "runs/phase8-smoke/summary.json",
        "runs/phase8-smoke/dashboard.html",
        "runs/phase8-smoke/dashboard.json",
        "runs/phase8-smoke/profile.md",
        "runs/phase8-smoke/profile.json",
        "runs/release/checklist.md",
        "runs/release/checklist.json",
    ]
    for path in required:
        _write(tmp_path / path, "{}\n")
    _write(tmp_path / "runs" / "eval.json", '{"metrics": {}}\n')
    _write(tmp_path / "runs" / "eval-report.md", "# Eval\n")
    _write(tmp_path / "runs" / "eval-report.json", '{"source": "run_eval"}\n')

    manifest = build_release_evidence_manifest(root=tmp_path)

    paths = [artifact["path"] for artifact in manifest["artifacts"]]
    assert paths[-3:] == [
        "runs/eval.json",
        "runs/eval-report.md",
        "runs/eval-report.json",
    ]


def test_release_evidence_manifest_skips_missing_eval_artifacts(tmp_path: Path) -> None:
    required = [
        "runs/phase8-smoke/summary.json",
        "runs/phase8-smoke/dashboard.html",
        "runs/phase8-smoke/dashboard.json",
        "runs/phase8-smoke/profile.md",
        "runs/phase8-smoke/profile.json",
        "runs/release/checklist.md",
        "runs/release/checklist.json",
    ]
    for path in required:
        _write(tmp_path / path, "{}\n")

    manifest = build_release_evidence_manifest(root=tmp_path)

    paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert "runs/eval.json" not in paths
    assert "runs/eval-report.md" not in paths
    assert "runs/eval-report.json" not in paths


def test_release_evidence_manifest_skips_partial_eval_artifacts(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "eval.json", '{"metrics": {}}\n')

    manifest = build_release_evidence_manifest(root=tmp_path)

    paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert "runs/eval.json" not in paths
    assert "runs/eval-report.md" not in paths
    assert "runs/eval-report.json" not in paths


def test_release_evidence_markdown_contains_hash_table(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "release" / "checklist.json"
    _write(artifact, "{}\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        artifacts=["runs/release/checklist.json"],
    )

    markdown = render_release_evidence_markdown(manifest)

    assert "# HKRL Release Evidence" in markdown
    assert "- Manifest version: `1`" in markdown
    assert "| Path | Bytes | SHA256 |" in markdown
    assert "runs/release/checklist.json" in markdown
    assert _sha256(artifact) in markdown


def test_release_evidence_markdown_tolerates_non_object_artifacts() -> None:
    manifest = {
        "artifact_count": 1,
        "artifacts": ["runs/phase8-smoke/summary.json"],
        "git_sha": FULL_GIT_SHA,
        "total_bytes": 0,
        "version": "phase8",
    }

    markdown = render_release_evidence_markdown(manifest)

    assert "HKRL Release Evidence" in markdown
    assert "&lt;invalid" not in markdown
    assert "<invalid artifact 0>" in markdown


def test_release_evidence_rejects_artifacts_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-release-artifact.txt"
    outside.write_text("bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes root"):
        build_release_evidence_manifest(root=tmp_path, artifacts=[outside])


def test_release_evidence_manifest_normalizes_absolute_artifact_inputs(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runs" / "phase8-smoke" / "summary.json"
    _write(artifact, '{"ok": true}\n')

    manifest = build_release_evidence_manifest(root=tmp_path, artifacts=[artifact])

    assert manifest["artifacts"][0]["path"] == "runs/phase8-smoke/summary.json"


def test_release_evidence_verifier_accepts_matching_manifest(tmp_path: Path) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is True
    assert result["artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == []


def test_release_evidence_verifier_accepts_matching_release_evidence_markdown(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(
        tmp_path / "runs" / "release" / "evidence.md",
        render_release_evidence_markdown(manifest),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is True
    assert result["failures"] == []


def test_release_evidence_verifier_rejects_invalid_release_evidence_markdown(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(tmp_path / "runs" / "release" / "evidence.md", "not evidence\n")

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "title",
            "ok": False,
            "path": "runs/release/evidence.md",
            "reason": "release_evidence_markdown_title_missing",
        }
    ]


def test_release_evidence_verifier_rejects_release_evidence_markdown_metadata_drift(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(
        tmp_path / "runs" / "release" / "evidence.md",
        render_release_evidence_markdown(manifest).replace(
            f"- Git SHA: `{FULL_GIT_SHA}`",
            f"- Git SHA: `{OTHER_FULL_GIT_SHA}`",
        ),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "actual_metadata": [
                "- Version: `phase8`",
                f"- Git SHA: `{OTHER_FULL_GIT_SHA}`",
                "- Manifest version: `1`",
                f"- Artifact count: `{manifest['artifact_count']}`",
                f"- Total bytes: `{manifest['total_bytes']}`",
            ],
            "expected_metadata": [
                "- Version: `phase8`",
                f"- Git SHA: `{FULL_GIT_SHA}`",
                "- Manifest version: `1`",
                f"- Artifact count: `{manifest['artifact_count']}`",
                f"- Total bytes: `{manifest['total_bytes']}`",
            ],
            "field": "metadata",
            "ok": False,
            "path": "runs/release/evidence.md",
            "reason": "release_evidence_markdown_metadata_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_release_evidence_markdown_missing_manifest_version(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(
        tmp_path / "runs" / "release" / "evidence.md",
        render_release_evidence_markdown(manifest).replace(
            "- Manifest version: `1`\n",
            "",
        ),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "actual_metadata": [
                "- Version: `phase8`",
                f"- Git SHA: `{FULL_GIT_SHA}`",
                f"- Artifact count: `{manifest['artifact_count']}`",
                f"- Total bytes: `{manifest['total_bytes']}`",
            ],
            "expected_metadata": [
                "- Version: `phase8`",
                f"- Git SHA: `{FULL_GIT_SHA}`",
                "- Manifest version: `1`",
                f"- Artifact count: `{manifest['artifact_count']}`",
                f"- Total bytes: `{manifest['total_bytes']}`",
            ],
            "field": "metadata",
            "ok": False,
            "path": "runs/release/evidence.md",
            "reason": "release_evidence_markdown_metadata_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_release_evidence_markdown_missing_artifact_row(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    artifact = manifest["artifacts"][0]
    row = f"| {artifact['path']} | {artifact['bytes']} | `{artifact['sha256']}` |"
    _write(
        tmp_path / "runs" / "release" / "evidence.md",
        render_release_evidence_markdown(manifest).replace(row, ""),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "artifacts",
            "missing_paths": ["runs/phase8-smoke/summary.json"],
            "ok": False,
            "path": "runs/release/evidence.md",
            "reason": "release_evidence_markdown_artifact_rows_missing",
        }
    ]


def test_release_evidence_verifier_rejects_release_evidence_markdown_extra_artifact_row(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    unexpected_row = (
        "| runs/old-artifact.json | 10 | "
        "`0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` |"
    )
    _write(
        tmp_path / "runs" / "release" / "evidence.md",
        render_release_evidence_markdown(manifest) + unexpected_row + "\n",
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "artifacts",
            "ok": False,
            "path": "runs/release/evidence.md",
            "reason": "release_evidence_markdown_artifact_rows_unexpected",
            "unexpected_rows": [unexpected_row],
        }
    ]


def test_release_evidence_verifier_rejects_release_evidence_markdown_reordered_rows(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    rows = [
        f"| {artifact['path']} | {artifact['bytes']} | `{artifact['sha256']}` |"
        for artifact in manifest["artifacts"]
    ]
    _write(
        tmp_path / "runs" / "release" / "evidence.md",
        render_release_evidence_markdown(manifest).replace(
            rows[0] + "\n" + rows[1],
            rows[1] + "\n" + rows[0],
        ),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "artifacts",
            "ok": False,
            "path": "runs/release/evidence.md",
            "reason": "release_evidence_markdown_artifact_rows_order_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_failed_phase8_smoke_summary(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", '{"ok": false}\n')
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "ok",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_not_ok",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_phase8_smoke_summary_json(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", "not json\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "ok",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_json_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_incomplete_phase8_smoke_summary(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", '{"ok": true}\n')
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "coordinator",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_section_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_smoke_without_metrics(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    summary["coordinator"] = {}
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "coordinator.metrics",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_metrics_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_smoke_empty_metrics(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    coordinator = summary["coordinator"]
    assert isinstance(coordinator, dict)
    coordinator["metrics"] = {}
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "coordinator.metrics",
            "malformed_metrics": ["active_worker_count", "sps", "worker_count"],
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_metrics_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_smoke_metrics(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    coordinator = summary["coordinator"]
    assert isinstance(coordinator, dict)
    metrics = coordinator["metrics"]
    assert isinstance(metrics, dict)
    metrics["active_worker_count"] = -1.0
    metrics["sps"] = "fast"
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "coordinator.metrics",
            "malformed_metrics": ["active_worker_count", "sps"],
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_metrics_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_smoke_without_worker_rows(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    coordinator = summary["coordinator"]
    assert isinstance(coordinator, dict)
    coordinator.pop("workers")
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "coordinator.workers",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_workers_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_smoke_missing_worker_rows(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    summary["worker_ids"] = ["worker-0", "worker-1", "worker-missing"]
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "worker_ids",
            "missing_worker_ids": ["worker-missing"],
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_worker_rows_missing",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_smoke_worker_rows(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    coordinator = summary["coordinator"]
    assert isinstance(coordinator, dict)
    workers = coordinator["workers"]
    assert isinstance(workers, dict)
    worker_0 = workers["worker-0"]
    assert isinstance(worker_0, dict)
    worker_0["alive"] = "yes"
    worker_1 = workers["worker-1"]
    assert isinstance(worker_1, dict)
    metrics = worker_1["metrics"]
    assert isinstance(metrics, dict)
    metrics["worker_crash_count"] = -1.0
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "coordinator.workers",
            "malformed_worker_ids": ["worker-0", "worker-1"],
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_worker_rows_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_smoke_checkpoint_versions(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    summary = _phase8_smoke_summary()
    summary["checkpoint_versions"] = [1, -1, "latest"]
    _write(tmp_path / "runs" / "phase8-smoke" / "summary.json", json.dumps(summary) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "checkpoint_versions",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_checkpoint_versions_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_phase8_dashboard_json(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "dashboard.json", "not json\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "health",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_json_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_incomplete_phase8_dashboard(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    dashboard = _phase8_dashboard_model()
    del dashboard["health"]
    _write(tmp_path / "runs" / "phase8-smoke" / "dashboard.json", json.dumps(dashboard) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "health",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_section_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_dashboard_tasks(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    dashboard = _phase8_dashboard_model()
    tasks = dashboard["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    del task["mastered"]
    _write(tmp_path / "runs" / "phase8-smoke" / "dashboard.json", json.dumps(dashboard) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "tasks",
            "indexes": [0],
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_tasks_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_dashboard_workers(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    dashboard = _phase8_dashboard_model()
    workers = dashboard["workers"]
    assert isinstance(workers, list)
    worker = workers[0]
    assert isinstance(worker, dict)
    worker["sps"] = -1.0
    _write(tmp_path / "runs" / "phase8-smoke" / "dashboard.json", json.dumps(dashboard) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "workers",
            "indexes": [0],
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_workers_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_phase8_dashboard_html(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "dashboard.html", "not a dashboard\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "title",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.html",
            "reason": "phase8_dashboard_html_title_missing",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_dashboard_html_without_sections(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(
        tmp_path / "runs" / "phase8-smoke" / "dashboard.html",
        "<title>HKRL Phase 8 Dashboard</title><h1>HKRL Phase 8 Dashboard</h1>\n",
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "sections",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.html",
            "reason": "phase8_dashboard_html_sections_missing",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_dashboard_html_missing_worker_row(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    dashboard = _phase8_dashboard_model()
    workers = dashboard["workers"]
    assert isinstance(workers, list)
    dashboard["workers"] = workers[:1]
    _write(
        tmp_path / "runs" / "phase8-smoke" / "dashboard.html",
        render_dashboard_html(dashboard),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "workers",
            "missing_worker_ids": ["worker-1"],
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.html",
            "reason": "phase8_dashboard_html_worker_rows_missing",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_dashboard_html_missing_task_row(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    dashboard = _phase8_dashboard_model()
    tasks = dashboard["tasks"]
    assert isinstance(tasks, list)
    dashboard["tasks"] = tasks[:1]
    _write(
        tmp_path / "runs" / "phase8-smoke" / "dashboard.html",
        render_dashboard_html(dashboard),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "tasks",
            "missing_task_ids": ["hornet_protector_attuned"],
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.html",
            "reason": "phase8_dashboard_html_task_rows_missing",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_phase8_profile_json(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "profile.json", "not json\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "metrics",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_json_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_incomplete_phase8_profile(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    profile = _phase8_profile_report()
    del profile["metrics"]
    _write(tmp_path / "runs" / "phase8-smoke" / "profile.json", json.dumps(profile) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "metrics",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_metrics_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_profile_findings(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    profile = _phase8_profile_report()
    profile["findings"] = [{"code": "stale_policy_workers", "severity": "warning"}]
    _write(tmp_path / "runs" / "phase8-smoke" / "profile.json", json.dumps(profile) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "findings",
            "indexes": [0],
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_findings_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_phase8_profile_workers(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    profile = _phase8_profile_report()
    workers = profile["workers"]
    assert isinstance(workers, list)
    worker = workers[0]
    assert isinstance(worker, dict)
    del worker["status"]
    _write(tmp_path / "runs" / "phase8-smoke" / "profile.json", json.dumps(profile) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "workers",
            "indexes": [0],
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_workers_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_phase8_profile_markdown(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "profile.md", "not a profile\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "title",
            "ok": False,
            "path": "runs/phase8-smoke/profile.md",
            "reason": "phase8_profile_markdown_title_missing",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_profile_markdown_without_workers(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "phase8-smoke" / "profile.md", "# HKRL Phase 8 Profile\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "workers",
            "ok": False,
            "path": "runs/phase8-smoke/profile.md",
            "reason": "phase8_profile_markdown_workers_missing",
        }
    ]


def test_release_evidence_verifier_rejects_phase8_profile_markdown_missing_worker_row(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    profile = _phase8_profile_report()
    workers = profile["workers"]
    assert isinstance(workers, list)
    profile["workers"] = workers[:1]
    _write(
        tmp_path / "runs" / "phase8-smoke" / "profile.md",
        render_profile_markdown(profile),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "workers",
            "missing_worker_ids": ["worker-1"],
            "ok": False,
            "path": "runs/phase8-smoke/profile.md",
            "reason": "phase8_profile_markdown_worker_rows_missing",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_release_checklist_json(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "release" / "checklist.json", "not json\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "checks",
            "ok": False,
            "path": "runs/release/checklist.json",
            "reason": "release_checklist_json_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_release_checklist_checks(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    checklist = _release_checklist()
    checks = checklist["checks"]
    assert isinstance(checks, list)
    check = checks[0]
    assert isinstance(check, dict)
    del check["command"]
    _write(tmp_path / "runs" / "release" / "checklist.json", json.dumps(checklist) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "checks",
            "indexes": [0],
            "ok": False,
            "path": "runs/release/checklist.json",
            "reason": "release_checklist_checks_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_release_checklist_missing_required_gate(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    checklist = _release_checklist()
    checks = checklist["checks"]
    assert isinstance(checks, list)
    checklist["checks"] = [
        check
        for check in checks
        if isinstance(check, dict) and check["id"] != "release_evidence_verification"
    ]
    checklist["required_count"] = len(checklist["checks"])
    _write(tmp_path / "runs" / "release" / "checklist.json", json.dumps(checklist) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "checks",
            "missing_check_ids": ["release_evidence_verification"],
            "ok": False,
            "path": "runs/release/checklist.json",
            "reason": "release_checklist_required_checks_missing",
        }
    ]


def test_release_evidence_verifier_rejects_release_checklist_git_sha_drift(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    checklist = _release_checklist(git_sha=OTHER_FULL_GIT_SHA)
    _write(tmp_path / "runs" / "release" / "checklist.json", json.dumps(checklist) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_git_sha": OTHER_FULL_GIT_SHA,
            "expected_git_sha": FULL_GIT_SHA,
            "field": "git_sha",
            "ok": False,
            "path": "runs/release/checklist.json",
            "reason": "release_checklist_git_sha_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_release_checklist_required_count_drift(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    checklist = _release_checklist()
    checklist["required_count"] = 1
    _write(tmp_path / "runs" / "release" / "checklist.json", json.dumps(checklist) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_required_count": 13,
            "expected_required_count": 1,
            "field": "required_count",
            "ok": False,
            "path": "runs/release/checklist.json",
            "reason": "release_checklist_required_count_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_invalid_release_checklist_markdown(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "release" / "checklist.md", "not a checklist\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "title",
            "ok": False,
            "path": "runs/release/checklist.md",
            "reason": "release_checklist_markdown_title_missing",
        }
    ]


def test_release_evidence_verifier_rejects_release_checklist_markdown_git_sha_drift(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    checklist = _release_checklist(git_sha=OTHER_FULL_GIT_SHA)
    _write(tmp_path / "runs" / "release" / "checklist.md", render_release_markdown(checklist))
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "git_sha",
            "ok": False,
            "path": "runs/release/checklist.md",
            "reason": "release_checklist_markdown_git_sha_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_release_checklist_markdown_missing_gate(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    checklist = _release_checklist()
    checks = checklist["checks"]
    assert isinstance(checks, list)
    checklist["checks"] = [
        check for check in checks if isinstance(check, dict) and check["id"] != "offline_profile"
    ]
    _write(tmp_path / "runs" / "release" / "checklist.md", render_release_markdown(checklist))
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "checks",
            "missing_check_ids": ["offline_profile"],
            "ok": False,
            "path": "runs/release/checklist.md",
            "reason": "release_checklist_markdown_required_checks_missing",
        }
    ]


def test_release_evidence_verifier_accepts_clean_eval_report(tmp_path: Path) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            findings=[
                {
                    "code": "low_win_rate",
                    "severity": "warning",
                }
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is True
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 3
    assert result["failures"] == []


def test_release_evidence_verifier_rejects_invalid_eval_report_markdown(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(tmp_path, _eval_report())
    _write(tmp_path / "runs" / "eval-report.md", "not an eval report\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 3
    assert result["failures"] == [
        {
            "field": "title",
            "ok": False,
            "path": "runs/eval-report.md",
            "reason": "eval_report_markdown_title_missing",
        }
    ]


def test_release_evidence_verifier_rejects_eval_report_markdown_without_sections(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(tmp_path, _eval_report())
    _write(tmp_path / "runs" / "eval-report.md", "# HKRL Eval Report\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 3
    assert result["failures"] == [
        {
            "field": "sections",
            "ok": False,
            "path": "runs/eval-report.md",
            "reason": "eval_report_markdown_sections_missing",
        }
    ]


def test_release_evidence_verifier_rejects_eval_report_markdown_missing_task_row(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    report = _eval_report(
        tasks=[
            {
                "metrics_valid": True,
                "task_id": "gruz_mother",
            },
            {
                "metrics_valid": True,
                "task_id": "hornet_protector_attuned",
            },
        ],
        summary={
            "malformed_task_count": 0.0,
            "task_count": 2.0,
            "valid_task_count": 2.0,
        },
    )
    _write_eval_artifacts(tmp_path, report)
    markdown_report = _eval_report(
        tasks=[
            {
                "metrics_valid": True,
                "task_id": "gruz_mother",
            }
        ]
    )
    _write(
        tmp_path / "runs" / "eval-report.md",
        render_eval_report_markdown(markdown_report),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 3
    assert result["failures"] == [
        {
            "field": "tasks",
            "missing_task_ids": ["hornet_protector_attuned"],
            "ok": False,
            "path": "runs/eval-report.md",
            "reason": "eval_report_markdown_task_rows_missing",
        }
    ]


def test_release_evidence_verifier_rejects_eval_report_without_findings(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    report = _eval_report()
    del report["findings"]
    _write_eval_artifacts(tmp_path, report)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_findings_missing",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_eval_report_findings(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            findings=[
                {"code": "low_win_rate", "severity": "warning"},
                {"code": "missing severity"},
                "not an object",
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "findings",
            "indexes": [1, 2],
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_findings_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_eval_report_without_valid_tasks(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "malformed_task_count": 1.0,
                "task_count": 1.0,
                "valid_task_count": 0.0,
            },
            tasks=[
                {
                    "metrics_valid": False,
                    "task_id": "gruz_mother",
                }
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_no_valid_tasks",
        }
    ]


def test_release_evidence_verifier_rejects_eval_report_task_count_mismatch(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "malformed_task_count": 0.0,
                "task_count": 2.0,
                "valid_task_count": 1.0,
            },
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "actual_task_count": 1,
            "expected_task_count": 2.0,
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_task_count_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_eval_report_tasks(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "malformed_task_count": 2.0,
                "task_count": 3.0,
                "valid_task_count": 1.0,
            },
            tasks=[
                {
                    "metrics_valid": True,
                    "task_id": "gruz_mother",
                },
                {
                    "task_id": "missing metrics_valid",
                },
                "not an object",
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "tasks",
            "indexes": [1, 2],
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_tasks_malformed",
        }
    ]


def test_release_evidence_verifier_rejects_valid_task_count_mismatch(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "malformed_task_count": 1.0,
                "task_count": 2.0,
                "valid_task_count": 2.0,
            },
            tasks=[
                {
                    "metrics_valid": True,
                    "task_id": "gruz_mother",
                },
                {
                    "metrics_valid": False,
                    "task_id": "hornet_protector_attuned",
                },
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "actual_valid_task_count": 1,
            "expected_valid_task_count": 2.0,
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_valid_task_count_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_missing_malformed_task_count(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "task_count": 1.0,
                "valid_task_count": 1.0,
            },
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_summary_counts_invalid",
        }
    ]


def test_release_evidence_verifier_rejects_malformed_task_count_mismatch(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "malformed_task_count": 0.0,
                "task_count": 2.0,
                "valid_task_count": 1.0,
            },
            tasks=[
                {
                    "metrics_valid": True,
                    "task_id": "gruz_mother",
                },
                {
                    "metrics_valid": False,
                    "task_id": "hornet_protector_attuned",
                },
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "actual_malformed_task_count": 1,
            "expected_malformed_task_count": 0.0,
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_malformed_task_count_mismatch",
        }
    ]


def test_release_evidence_verifier_rejects_duplicate_eval_report_task_ids(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            summary={
                "malformed_task_count": 0.0,
                "task_count": 2.0,
                "valid_task_count": 2.0,
            },
            tasks=[
                {
                    "metrics_valid": True,
                    "task_id": "gruz_mother",
                },
                {
                    "metrics_valid": True,
                    "task_id": "gruz_mother",
                },
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["failures"] == [
        {
            "duplicate_task_ids": ["gruz_mother"],
            "field": "tasks",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_task_ids_duplicate",
        }
    ]


def test_release_evidence_verifier_reports_non_object_artifact_entries(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    manifest["artifacts"].append("runs/phase8-smoke/summary.json")
    manifest["artifact_count"] = len(PHASE8_RELEASE_ARTIFACTS) + 1

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 1
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "index": len(PHASE8_RELEASE_ARTIFACTS),
            "ok": False,
            "path": "<missing>",
            "reason": "artifact_entry_invalid",
        }
    ]


def test_release_evidence_verifier_reports_absolute_manifest_path(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    artifact = tmp_path / PHASE8_RELEASE_ARTIFACTS[0]
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    manifest["artifacts"][0]["path"] = str(artifact)

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) - 1
    assert result["failures"][0] == {
        "ok": False,
        "path": str(artifact),
        "reason": "artifact_path_absolute",
    }


def test_release_evidence_verifier_reports_non_normalized_manifest_path(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    manifest["artifacts"][0]["path"] = "./runs/phase8-smoke/summary.json"

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) - 1
    assert result["failures"][0] == {
        "actual_path": "runs/phase8-smoke/summary.json",
        "expected_path": "./runs/phase8-smoke/summary.json",
        "ok": False,
        "path": "./runs/phase8-smoke/summary.json",
        "reason": "artifact_path_not_normalized",
    }


def test_release_evidence_verifier_reports_sha_mismatch(tmp_path: Path) -> None:
    _write_required_release_artifacts(tmp_path)
    artifact = tmp_path / PHASE8_RELEASE_ARTIFACTS[0]
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(artifact, "x" * artifact.stat().st_size)

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) - 1
    assert result["failures"][0]["path"] == "runs/phase8-smoke/summary.json"
    assert result["failures"][0]["reason"] == "artifact_sha256_mismatch"


def test_release_evidence_verifier_reports_artifact_count_mismatch(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    manifest["artifact_count"] = len(PHASE8_RELEASE_ARTIFACTS) + 1

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_artifact_count": len(PHASE8_RELEASE_ARTIFACTS),
            "expected_artifact_count": len(PHASE8_RELEASE_ARTIFACTS) + 1,
            "field": "artifact_count",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_artifact_count_mismatch",
        }
    ]


def test_release_evidence_verifier_reports_manifest_version_mismatch(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    manifest["manifest_version"] = 2

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_manifest_version": 2,
            "expected_manifest_version": 1,
            "field": "manifest_version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_version_mismatch",
        }
    ]


def test_release_evidence_verifier_reports_missing_release_version(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    del manifest["version"]

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_release_version_missing",
        }
    ]


def test_release_evidence_verifier_reports_unsupported_release_version(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        version="phase7",
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_version": "phase7",
            "expected_versions": ["phase8"],
            "field": "version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_release_version_unsupported",
        }
    ]


def test_release_evidence_verifier_reports_missing_git_sha(tmp_path: Path) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_missing",
        }
    ]


def test_release_evidence_verifier_reports_invalid_git_sha(tmp_path: Path) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha="deadbeef",
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_git_sha": "deadbeef",
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_invalid",
        }
    ]


def test_release_evidence_verifier_reports_expected_git_sha_mismatch(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(
        root=tmp_path,
        manifest=manifest,
        expected_git_sha=OTHER_FULL_GIT_SHA,
    )

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_git_sha": FULL_GIT_SHA,
            "expected_git_sha": OTHER_FULL_GIT_SHA,
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_mismatch",
        }
    ]


def test_release_evidence_verifier_reports_invalid_expected_git_sha(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(
        root=tmp_path,
        manifest=manifest,
        expected_git_sha="deadbeef",
    )

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_git_sha": "deadbeef",
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "expected_git_sha_invalid",
        }
    ]


def test_release_evidence_verifier_reports_missing_required_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runs" / "phase8-smoke" / "summary.json"
    _write(artifact, json.dumps(_phase8_smoke_summary()) + "\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
        artifacts=["runs/phase8-smoke/summary.json"],
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == 1
    assert result["failures"] == [
        {
            "field": "artifacts",
            "missing_paths": list(PHASE8_RELEASE_ARTIFACTS[1:]),
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_required_artifacts_missing",
        }
    ]


def test_release_evidence_verifier_reports_partial_eval_artifacts(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write(tmp_path / "runs" / "eval.json", '{"metrics": {}}\n')
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
        artifacts=(*PHASE8_RELEASE_ARTIFACTS, "runs/eval.json"),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 1
    assert result["failures"] == [
        {
            "field": "artifacts",
            "group": "phase8_eval",
            "missing_paths": ["runs/eval-report.md", "runs/eval-report.json"],
            "ok": False,
            "path": "<manifest>",
            "present_paths": ["runs/eval.json"],
            "reason": "manifest_optional_artifacts_partial",
        }
    ]


def test_release_evidence_verifier_rejects_critical_eval_report_findings(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    _write_eval_artifacts(
        tmp_path,
        _eval_report(
            findings=[
                {
                    "code": "no_valid_eval_tasks",
                    "severity": "critical",
                },
                {
                    "code": "low_win_rate",
                    "severity": "warning",
                },
            ],
        ),
    )
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 3
    assert result["failures"] == [
        {
            "critical_codes": ["no_valid_eval_tasks"],
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_critical_findings",
        }
    ]


def test_release_evidence_verifier_reports_duplicate_artifact_paths(
    tmp_path: Path,
) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
        artifacts=(*PHASE8_RELEASE_ARTIFACTS, PHASE8_RELEASE_ARTIFACTS[0]),
    )

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 1
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS) + 1
    assert result["failures"] == [
        {
            "duplicate_paths": ["runs/phase8-smoke/summary.json"],
            "field": "artifacts",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_artifact_paths_duplicate",
        }
    ]


def test_release_evidence_verifier_reports_total_bytes_mismatch(tmp_path: Path) -> None:
    _write_required_release_artifacts(tmp_path)
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    manifest["total_bytes"] = int(manifest["total_bytes"]) + 1

    result = verify_release_evidence_manifest(root=tmp_path, manifest=manifest)

    assert result["ok"] is False
    assert result["checked_artifact_count"] == len(PHASE8_RELEASE_ARTIFACTS)
    assert result["failures"] == [
        {
            "actual_total_bytes": int(manifest["total_bytes"]) - 1,
            "expected_total_bytes": int(manifest["total_bytes"]),
            "field": "total_bytes",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_total_bytes_mismatch",
        }
    ]


def test_render_release_evidence_script_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_script("render_release_evidence.py")
    artifact = tmp_path / "runs" / "phase8-smoke" / "summary.json"
    json_path = tmp_path / "runs" / "release" / "evidence.json"
    markdown_path = tmp_path / "runs" / "release" / "evidence.md"
    _write(artifact, '{"ok": true}\n')
    args = argparse.Namespace(
        version="phase8",
        git_sha="deadbeef",
        root=str(tmp_path),
        artifacts=["runs/phase8-smoke/summary.json"],
        output_json=str(json_path),
        output_md=str(markdown_path),
    )

    manifest = module.run_from_args(args)

    assert manifest["git_sha"] == "deadbeef"
    assert json.loads(json_path.read_text(encoding="utf-8"))["artifact_count"] == 1
    assert "HKRL Release Evidence" in markdown_path.read_text(encoding="utf-8")


def test_verify_release_evidence_script_writes_failure_report(tmp_path: Path) -> None:
    module = _load_script("verify_release_evidence.py")
    _write_required_release_artifacts(tmp_path)
    artifact = tmp_path / PHASE8_RELEASE_ARTIFACTS[0]
    manifest_path = tmp_path / "runs" / "release" / "evidence.json"
    report_path = tmp_path / "runs" / "release" / "verification.json"
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(manifest_path, release_evidence_to_json(manifest))
    _write(artifact, "x" * artifact.stat().st_size)

    exit_code = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--root",
            str(tmp_path),
            "--output-json",
            str(report_path),
        ]
    )

    assert exit_code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["failures"][0]["reason"] == "artifact_sha256_mismatch"


def test_verify_release_evidence_script_checks_expected_git_sha(tmp_path: Path) -> None:
    module = _load_script("verify_release_evidence.py")
    _write_required_release_artifacts(tmp_path)
    manifest_path = tmp_path / "runs" / "release" / "evidence.json"
    report_path = tmp_path / "runs" / "release" / "verification.json"
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        git_sha=FULL_GIT_SHA,
    )
    _write(manifest_path, release_evidence_to_json(manifest))

    exit_code = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--root",
            str(tmp_path),
            "--git-sha",
            OTHER_FULL_GIT_SHA,
            "--output-json",
            str(report_path),
        ]
    )

    assert exit_code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["failures"] == [
        {
            "actual_git_sha": FULL_GIT_SHA,
            "expected_git_sha": OTHER_FULL_GIT_SHA,
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_mismatch",
        }
    ]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_required_release_artifacts(root: Path) -> None:
    for artifact in PHASE8_RELEASE_ARTIFACTS:
        if artifact == "runs/phase8-smoke/summary.json":
            _write(root / artifact, json.dumps(_phase8_smoke_summary()) + "\n")
        elif artifact == "runs/phase8-smoke/dashboard.html":
            _write(root / artifact, render_dashboard_html(_phase8_dashboard_model()))
        elif artifact == "runs/phase8-smoke/dashboard.json":
            _write(root / artifact, json.dumps(_phase8_dashboard_model()) + "\n")
        elif artifact == "runs/phase8-smoke/profile.md":
            _write(root / artifact, render_profile_markdown(_phase8_profile_report()))
        elif artifact == "runs/phase8-smoke/profile.json":
            _write(root / artifact, json.dumps(_phase8_profile_report()) + "\n")
        elif artifact == "runs/release/checklist.md":
            _write(root / artifact, render_release_markdown(_release_checklist()))
        elif artifact == "runs/release/checklist.json":
            _write(root / artifact, json.dumps(_release_checklist()) + "\n")
        else:
            _write(root / artifact, f"{artifact}\n")


def _release_checklist(*, git_sha: str = FULL_GIT_SHA) -> dict[str, object]:
    return build_release_checklist(version="phase8", git_sha=git_sha)


def _phase8_smoke_summary() -> dict[str, object]:
    return {
        "checkpoint_versions": [1, 2],
        "coordinator": {
            "metrics": {
                "active_worker_count": 2.0,
                "sps": 32.0,
                "worker_count": 2.0,
            },
            "workers": {
                "worker-0": {
                    "alive": True,
                    "metrics": {"sps": 32.0, "worker_crash_count": 0.0},
                },
                "worker-1": {
                    "alive": True,
                    "metrics": {"sps": 0.0, "worker_crash_count": 1.0},
                },
            },
        },
        "learner": {
            "policy_version": 2.0,
        },
        "ok": True,
        "task_ids": ["gruz_mother", "hornet_protector_attuned"],
        "worker": {
            "dry_run": True,
        },
        "worker_ids": ["worker-0", "worker-1"],
    }


def _phase8_dashboard_model() -> dict[str, object]:
    return {
        "health": {
            "reasons": [],
            "status": "healthy",
        },
        "learner": {
            "policy_version": 2.0,
        },
        "metrics": {
            "sps": 32.0,
            "worker_count": 2.0,
        },
        "tasks": [
            {
                "mastered": True,
                "sampler_weight": 0.1,
                "task_id": "gruz_mother",
                "win_rate": 0.9,
            },
            {
                "mastered": False,
                "sampler_weight": 0.8,
                "task_id": "hornet_protector_attuned",
                "win_rate": 0.2,
            },
        ],
        "workers": [
            {
                "alive": True,
                "assigned_task": "hornet_protector_attuned",
                "checkpoint_lag": 0.0,
                "checkpoint_version": 2.0,
                "learner_upload_accepted_batches": 0.0,
                "learner_upload_failed_batches": 0.0,
                "learner_upload_rejected_batches": 0.0,
                "learner_upload_submitted_batches": 0.0,
                "policy_lag": 0.0,
                "policy_version": 2.0,
                "sps": 32.0,
                "status": "running",
                "worker_crash_count": 0.0,
                "worker_id": "worker-0",
            },
            {
                "alive": True,
                "assigned_task": "gruz_mother",
                "checkpoint_lag": 1.0,
                "checkpoint_version": 1.0,
                "learner_upload_accepted_batches": 0.0,
                "learner_upload_failed_batches": 0.0,
                "learner_upload_rejected_batches": 0.0,
                "learner_upload_submitted_batches": 0.0,
                "policy_lag": 1.0,
                "policy_version": 1.0,
                "sps": 0.0,
                "status": "recovering",
                "worker_crash_count": 1.0,
                "worker_id": "worker-1",
            },
        ],
    }


def _phase8_profile_report() -> dict[str, object]:
    return {
        "findings": [],
        "metrics": {
            "sps": 32.0,
            "worker_count": 2.0,
        },
        "source": "phase8_smoke",
        "workers": [
            {
                "alive": True,
                "learner_upload_accepted_batches": 1.0,
                "learner_upload_failed_batches": 0.0,
                "learner_upload_rejected_batches": 0.0,
                "learner_upload_submitted_batches": 1.0,
                "rollout_duration_s": 4.0,
                "rollout_steps": 128.0,
                "sps": 32.0,
                "status": "running",
                "worker_crash_count": 0.0,
                "worker_id": "worker-0",
            },
            {
                "alive": True,
                "learner_upload_accepted_batches": 0.0,
                "learner_upload_failed_batches": 0.0,
                "learner_upload_rejected_batches": 0.0,
                "learner_upload_submitted_batches": 0.0,
                "rollout_duration_s": 0.0,
                "rollout_steps": 0.0,
                "sps": 0.0,
                "status": "recovering",
                "worker_crash_count": 1.0,
                "worker_id": "worker-1",
            },
        ],
    }


def _write_eval_artifacts(root: Path, report: dict[str, object]) -> None:
    _write(root / "runs" / "eval.json", '{"metrics": {}}\n')
    _write(root / "runs" / "eval-report.md", render_eval_report_markdown(report))
    _write(root / "runs" / "eval-report.json", json.dumps(report) + "\n")


def _eval_report(
    *,
    findings: list[object] | None = None,
    summary: dict[str, object] | None = None,
    tasks: list[object] | None = None,
) -> dict[str, object]:
    return {
        "findings": [] if findings is None else findings,
        "metadata": {},
        "source": "run_eval",
        "summary": (
            {
                "malformed_task_count": 0.0,
                "task_count": 1.0,
                "valid_task_count": 1.0,
            }
            if summary is None
            else summary
        ),
        "tasks": (
            [
                {
                    "metrics_valid": True,
                    "task_id": "gruz_mother",
                }
            ]
            if tasks is None
            else tasks
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
