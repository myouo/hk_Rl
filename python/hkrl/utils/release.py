"""Release checklist rendering for Phase 8 engineering gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseCheck:
    id: str
    category: str
    title: str
    command: str
    evidence: str
    blocking: bool = True


PHASE8_RELEASE_CHECKS: tuple[ReleaseCheck, ...] = (
    ReleaseCheck(
        id="python_quality_gate",
        category="local",
        title="Run the full Python quality gate",
        command="make check",
        evidence="format, lint, mypy, FlatBuffers codegen, and pytest all pass",
    ),
    ReleaseCheck(
        id="offline_distributed_smoke",
        category="local",
        title="Run the offline distributed wiring smoke",
        command="make phase8-smoke",
        evidence="summary JSON reports ok=true for learner/worker/coordinator wiring",
    ),
    ReleaseCheck(
        id="offline_dashboard",
        category="local",
        title="Render the Phase 8 fleet dashboard",
        command="make phase8-dashboard",
        evidence="runs/phase8-smoke/dashboard.html and dashboard.json are generated",
    ),
    ReleaseCheck(
        id="offline_profile",
        category="local",
        title="Render the Phase 8 profile report",
        command="make phase8-profile",
        evidence="runs/phase8-smoke/profile.md and profile.json are generated",
    ),
    ReleaseCheck(
        id="release_evidence_manifest",
        category="local",
        title="Render the release evidence manifest",
        command="make phase8-release-evidence",
        evidence="runs/release/evidence.json records sha256 hashes for release artifacts",
    ),
    ReleaseCheck(
        id="release_evidence_verification",
        category="local",
        title="Verify the release evidence manifest",
        command="make phase8-verify-release-evidence",
        evidence="runs/release/evidence-verification.json reports ok=true",
    ),
    ReleaseCheck(
        id="github_ci",
        category="remote",
        title="Confirm the pushed commit is green in GitHub Actions",
        command="gh run list --branch main --limit 1",
        evidence="latest run conclusion is success and headSha matches the release commit",
    ),
    ReleaseCheck(
        id="mod_build",
        category="game_machine",
        title="Build HKRLEnvMod on a configured Hollow Knight machine",
        command="dotnet build mod/HKRLEnvMod/HKRLEnvMod.csproj",
        evidence="mod compiles against local Hollow Knight Managed assemblies and HK Modding API",
    ),
    ReleaseCheck(
        id="live_smoke",
        category="game_machine",
        title="Run a live env smoke against HKRLEnvMod",
        command=(
            "python scripts/train.py --config configs/train/ppo_mlp.yaml "
            "--task configs/tasks/gruz_mother.yaml --smoke"
        ),
        evidence="live smoke reaches RUNNING, steps the env, and exits without protocol errors",
    ),
    ReleaseCheck(
        id="fixed_seed_eval",
        category="game_machine",
        title="Run fixed-seed evaluator output for release evidence",
        command=(
            "python scripts/run_eval.py --policy scripted "
            "--tasks configs/tasks/gruz_mother.yaml --episodes 5 "
            "--output runs/eval.json"
        ),
        evidence="runs/eval.json contains shaping-free per-boss metrics",
    ),
    ReleaseCheck(
        id="fixed_seed_eval_report",
        category="game_machine",
        title="Render the fixed-seed eval regression report",
        command="make phase8-eval-report",
        evidence="runs/eval-report.json and eval-report.md summarize win rates/regressions",
    ),
    ReleaseCheck(
        id="security_scope",
        category="review",
        title="Review runtime network/security settings",
        command="review configs/train/remote_learner.yaml and deployment env",
        evidence="LAN/localhost scope, token auth, and checkpoint hash verification are enabled",
    ),
    ReleaseCheck(
        id="docs_changelog",
        category="review",
        title="Confirm docs and changelog describe the released behavior",
        command="review CHANGELOG.md README.md docs/",
        evidence=(
            "release notes mention new user-visible commands, metrics, and compatibility limits"
        ),
    ),
)


PHASE8_RELEASE_ARTIFACTS: tuple[str, ...] = (
    "runs/phase8-smoke/summary.json",
    "runs/phase8-smoke/dashboard.html",
    "runs/phase8-smoke/dashboard.json",
    "runs/phase8-smoke/profile.md",
    "runs/phase8-smoke/profile.json",
    "runs/release/checklist.md",
    "runs/release/checklist.json",
)

PHASE8_OPTIONAL_RELEASE_ARTIFACTS: tuple[str, ...] = (
    "runs/eval.json",
    "runs/eval-report.md",
    "runs/eval-report.json",
)

RELEASE_EVIDENCE_SUPPORTED_VERSIONS = frozenset({"phase8"})
RELEASE_EVIDENCE_MANIFEST_VERSION = 1


def build_release_checklist(
    *,
    version: str = "phase8",
    git_sha: str | None = None,
    checks: Sequence[ReleaseCheck] = PHASE8_RELEASE_CHECKS,
) -> dict[str, Any]:
    """Return a stable release checklist payload."""
    categories = sorted({check.category for check in checks})
    return {
        "categories": categories,
        "checks": [asdict(check) for check in checks],
        "git_sha": git_sha,
        "required_count": sum(1 for check in checks if check.blocking),
        "version": version,
    }


def render_release_markdown(payload: dict[str, Any]) -> str:
    """Render a release checklist payload as Markdown."""
    checks = [
        _release_check_markdown_item(check, index=index)
        for index, check in enumerate(list(payload.get("checks", [])))
    ]
    lines = [
        "# HKRL Release Checklist",
        "",
        f"- Version: `{payload.get('version', 'unknown')}`",
        f"- Git SHA: `{payload.get('git_sha') or 'unrecorded'}`",
        f"- Blocking checks: `{payload.get('required_count', 0)}`",
        "",
    ]
    categories = sorted({str(check.get("category", "uncategorized")) for check in checks})
    for category in categories:
        lines.extend([f"## {category.replace('_', ' ').title()}", ""])
        for check in checks:
            if check.get("category") != category:
                continue
            mark = "[ ]" if check.get("blocking", True) else "[-]"
            lines.extend(
                [
                    f"- {mark} **{check.get('title', '')}** (`{check.get('id', '')}`)",
                    f"  - Command: `{check.get('command', '')}`",
                    f"  - Evidence: {check.get('evidence', '')}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _release_check_markdown_item(check: Any, *, index: int) -> Mapping[str, Any]:
    if isinstance(check, Mapping):
        return check
    return {
        "blocking": True,
        "category": "uncategorized",
        "command": "",
        "evidence": "Checklist entry is not an object.",
        "id": f"invalid_check_{index}",
        "title": f"Invalid release check entry {index}",
    }


def release_checklist_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_release_evidence_manifest(
    *,
    root: str | Path = ".",
    version: str = "phase8",
    git_sha: str | None = None,
    artifacts: Sequence[str | Path] | None = None,
    optional_artifacts: Sequence[str | Path] = PHASE8_OPTIONAL_RELEASE_ARTIFACTS,
) -> dict[str, Any]:
    """Return a hash manifest for release evidence artifacts."""
    root_path = Path(root).expanduser().resolve()
    artifact_paths = (
        _default_release_artifacts(root_path, optional_artifacts=optional_artifacts)
        if artifacts is None
        else artifacts
    )
    artifact_rows = [_artifact_row(root_path, artifact) for artifact in artifact_paths]
    total_bytes = sum(int(row["bytes"]) for row in artifact_rows)
    return {
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "git_sha": git_sha,
        "manifest_version": RELEASE_EVIDENCE_MANIFEST_VERSION,
        "total_bytes": total_bytes,
        "version": version,
    }


def _default_release_artifacts(
    root: Path,
    *,
    optional_artifacts: Sequence[str | Path],
) -> tuple[str | Path, ...]:
    existing_optional = tuple(
        artifact for artifact in optional_artifacts if _artifact_exists(root, artifact)
    )
    if existing_optional and len(existing_optional) != len(optional_artifacts):
        return PHASE8_RELEASE_ARTIFACTS
    return (*PHASE8_RELEASE_ARTIFACTS, *existing_optional)


def render_release_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render a release evidence manifest as Markdown."""
    artifacts = list(payload.get("artifacts", []))
    lines = [
        "# HKRL Release Evidence",
        "",
        f"- Version: `{payload.get('version', 'unknown')}`",
        f"- Git SHA: `{payload.get('git_sha') or 'unrecorded'}`",
        f"- Artifact count: `{payload.get('artifact_count', 0)}`",
        f"- Total bytes: `{payload.get('total_bytes', 0)}`",
        "",
        "## Artifacts",
        "",
        "| Path | Bytes | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for index, artifact in enumerate(artifacts):
        item = _artifact_markdown_item(artifact, index=index)
        lines.append(
            "| "
            f"{_markdown_cell(str(item.get('path', '')))} | "
            f"{item.get('bytes', 0)} | "
            f"`{item.get('sha256', '')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def release_evidence_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _artifact_markdown_item(artifact: Any, *, index: int) -> Mapping[str, Any]:
    if isinstance(artifact, Mapping):
        return artifact
    return {
        "bytes": 0,
        "path": f"<invalid artifact {index}>",
        "sha256": "",
    }


def verify_release_evidence_manifest(
    *,
    root: str | Path = ".",
    manifest: Mapping[str, Any],
    expected_git_sha: str | None = None,
) -> dict[str, Any]:
    """Verify release evidence artifact hashes against a manifest."""
    root_path = Path(root).expanduser().resolve()
    artifacts = manifest.get("artifacts", [])
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        failures.append(
            {
                "path": "<manifest>",
                "reason": "artifacts_not_a_list",
            }
        )
        artifacts = []

    for index, artifact in enumerate(artifacts):
        result = _verify_artifact(root_path, artifact, index=index)
        results.append(result)
        if not result["ok"]:
            failures.append(result)

    version_failure = _verify_manifest_version(manifest)
    if version_failure is not None:
        failures.append(version_failure)

    release_version_failure = _verify_manifest_release_version(manifest)
    if release_version_failure is not None:
        failures.append(release_version_failure)

    git_sha_failure = _verify_manifest_git_sha(manifest)
    if git_sha_failure is not None:
        failures.append(git_sha_failure)

    expected_git_sha_failure = _verify_manifest_expected_git_sha(
        manifest,
        expected_git_sha=expected_git_sha,
    )
    if expected_git_sha_failure is not None:
        failures.append(expected_git_sha_failure)

    duplicate_paths_failure = _verify_manifest_unique_artifact_paths(results)
    if duplicate_paths_failure is not None:
        failures.append(duplicate_paths_failure)

    required_artifacts_failure = _verify_manifest_required_artifacts(results)
    if required_artifacts_failure is not None:
        failures.append(required_artifacts_failure)

    optional_artifacts_failure = _verify_manifest_optional_artifacts(results)
    if optional_artifacts_failure is not None:
        failures.append(optional_artifacts_failure)

    smoke_summary_failure = _verify_phase8_smoke_summary_artifact(root_path, results)
    if smoke_summary_failure is not None:
        failures.append(smoke_summary_failure)

    dashboard_failure = _verify_phase8_dashboard_artifact(root_path, results)
    if dashboard_failure is not None:
        failures.append(dashboard_failure)

    profile_failure = _verify_phase8_profile_artifact(root_path, results)
    if profile_failure is not None:
        failures.append(profile_failure)

    eval_report_failure = _verify_eval_report_artifact(root_path, results)
    if eval_report_failure is not None:
        failures.append(eval_report_failure)

    count_failure = _verify_manifest_artifact_count(manifest, actual_count=len(results))
    if count_failure is not None:
        failures.append(count_failure)

    total_bytes_failure = _verify_manifest_total_bytes(manifest, artifacts)
    if total_bytes_failure is not None:
        failures.append(total_bytes_failure)

    return {
        "artifact_count": len(results),
        "checked_artifact_count": sum(1 for result in results if result["ok"]),
        "failures": failures,
        "git_sha": manifest.get("git_sha"),
        "manifest_version": manifest.get("manifest_version"),
        "ok": not failures,
        "version": manifest.get("version"),
    }


def release_evidence_verification_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _artifact_row(root: Path, artifact: str | Path) -> dict[str, Any]:
    path, relative_path = _resolve_artifact_path(root, artifact)
    return {
        "bytes": path.stat().st_size,
        "path": relative_path,
        "sha256": _sha256_file(path),
    }


def _artifact_exists(root: Path, artifact: str | Path) -> bool:
    artifact_path = Path(artifact).expanduser()
    path = artifact_path if artifact_path.is_absolute() else root / artifact_path
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except ValueError:
        return False
    return resolved.is_file()


def _resolve_artifact_path(root: Path, artifact: str | Path) -> tuple[Path, str]:
    artifact_path = Path(artifact).expanduser()
    path = artifact_path if artifact_path.is_absolute() else root / artifact_path
    resolved = path.resolve()
    try:
        relative_path = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"release artifact path escapes root: {artifact}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"release artifact is missing: {relative_path.as_posix()}")
    return resolved, relative_path.as_posix()


def _verify_artifact(root: Path, artifact: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {
            "index": index,
            "ok": False,
            "path": "<missing>",
            "reason": "artifact_entry_invalid",
        }
    item = artifact
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return {
            "ok": False,
            "path": "<missing>",
            "reason": "artifact_path_missing",
        }
    if Path(raw_path).expanduser().is_absolute():
        return {
            "ok": False,
            "path": raw_path,
            "reason": "artifact_path_absolute",
        }

    try:
        path, relative_path = _resolve_artifact_path(root, raw_path)
    except FileNotFoundError:
        return {
            "ok": False,
            "path": raw_path,
            "reason": "artifact_missing",
        }
    except ValueError:
        return {
            "ok": False,
            "path": raw_path,
            "reason": "artifact_path_escapes_root",
        }
    if raw_path != relative_path:
        return {
            "actual_path": relative_path,
            "expected_path": raw_path,
            "ok": False,
            "path": raw_path,
            "reason": "artifact_path_not_normalized",
        }

    expected_bytes = item.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        return {
            "ok": False,
            "path": relative_path,
            "reason": "artifact_bytes_invalid",
        }

    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        return {
            "actual_bytes": actual_bytes,
            "expected_bytes": expected_bytes,
            "ok": False,
            "path": relative_path,
            "reason": "artifact_bytes_mismatch",
        }

    expected_sha256 = item.get("sha256")
    if not _is_sha256(expected_sha256):
        return {
            "ok": False,
            "path": relative_path,
            "reason": "artifact_sha256_invalid",
        }

    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        return {
            "actual_sha256": actual_sha256,
            "expected_sha256": expected_sha256,
            "ok": False,
            "path": relative_path,
            "reason": "artifact_sha256_mismatch",
        }

    return {
        "bytes": actual_bytes,
        "ok": True,
        "path": relative_path,
        "sha256": actual_sha256,
    }


def _verify_manifest_version(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    manifest_version = manifest.get("manifest_version")
    if _invalid_non_negative_int(manifest_version):
        return {
            "field": "manifest_version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_version_invalid",
        }
    if manifest_version != RELEASE_EVIDENCE_MANIFEST_VERSION:
        return {
            "actual_manifest_version": manifest_version,
            "expected_manifest_version": RELEASE_EVIDENCE_MANIFEST_VERSION,
            "field": "manifest_version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_version_mismatch",
        }
    return None


def _verify_manifest_release_version(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        return {
            "field": "version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_release_version_missing",
        }
    if version not in RELEASE_EVIDENCE_SUPPORTED_VERSIONS:
        return {
            "actual_version": version,
            "expected_versions": sorted(RELEASE_EVIDENCE_SUPPORTED_VERSIONS),
            "field": "version",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_release_version_unsupported",
        }
    return None


def _verify_manifest_git_sha(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    git_sha = manifest.get("git_sha")
    if not isinstance(git_sha, str) or not git_sha:
        return {
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_missing",
        }
    if not _is_git_sha(git_sha):
        return {
            "actual_git_sha": git_sha,
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_invalid",
        }
    return None


def _verify_manifest_expected_git_sha(
    manifest: Mapping[str, Any],
    *,
    expected_git_sha: str | None,
) -> dict[str, Any] | None:
    if expected_git_sha is None:
        return None
    if not _is_git_sha(expected_git_sha):
        return {
            "actual_git_sha": expected_git_sha,
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "expected_git_sha_invalid",
        }

    git_sha = manifest.get("git_sha")
    if not _is_git_sha(git_sha):
        return None
    if git_sha != expected_git_sha:
        return {
            "actual_git_sha": git_sha,
            "expected_git_sha": expected_git_sha,
            "field": "git_sha",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_git_sha_mismatch",
        }
    return None


def _verify_manifest_unique_artifact_paths(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for result in results:
        path = result.get("path")
        if not isinstance(path, str) or path == "<missing>":
            continue
        if path in seen:
            duplicates.add(path)
        seen.add(path)

    if duplicates:
        return {
            "duplicate_paths": sorted(duplicates),
            "field": "artifacts",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_artifact_paths_duplicate",
        }
    return None


def _verify_manifest_required_artifacts(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    paths = _manifest_artifact_paths(results)
    missing_paths = [path for path in PHASE8_RELEASE_ARTIFACTS if path not in paths]
    if missing_paths:
        return {
            "field": "artifacts",
            "missing_paths": missing_paths,
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_required_artifacts_missing",
        }
    return None


def _verify_manifest_optional_artifacts(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    paths = _manifest_artifact_paths(results)
    present_paths = [path for path in PHASE8_OPTIONAL_RELEASE_ARTIFACTS if path in paths]
    if not present_paths:
        return None
    missing_paths = [path for path in PHASE8_OPTIONAL_RELEASE_ARTIFACTS if path not in paths]
    if missing_paths:
        return {
            "field": "artifacts",
            "group": "phase8_eval",
            "missing_paths": missing_paths,
            "ok": False,
            "path": "<manifest>",
            "present_paths": present_paths,
            "reason": "manifest_optional_artifacts_partial",
        }
    return None


def _verify_eval_report_artifact(
    root: Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    eval_report = next(
        (result for result in results if result.get("path") == "runs/eval-report.json"),
        None,
    )
    if eval_report is None or eval_report.get("ok") is not True:
        return None

    path = root / "runs/eval-report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_json_invalid",
        }
    if not isinstance(payload, Mapping):
        return {
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_not_object",
        }

    structure_failure = _verify_eval_report_structure(payload)
    if structure_failure is not None:
        return structure_failure

    if "findings" not in payload:
        return {
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_findings_missing",
        }
    findings = payload.get("findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
        return {
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_findings_invalid",
        }

    malformed_indexes = [
        index for index, finding in enumerate(findings) if not _valid_eval_report_finding(finding)
    ]
    if malformed_indexes:
        return {
            "field": "findings",
            "indexes": malformed_indexes,
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_findings_malformed",
        }

    critical_codes = [
        str(finding.get("code", "unknown"))
        for finding in findings
        if isinstance(finding, Mapping) and finding.get("severity") == "critical"
    ]
    if critical_codes:
        return {
            "critical_codes": critical_codes,
            "field": "findings",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_critical_findings",
        }
    return None


def _verify_phase8_smoke_summary_artifact(
    root: Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    summary_result = next(
        (result for result in results if result.get("path") == "runs/phase8-smoke/summary.json"),
        None,
    )
    if summary_result is None or summary_result.get("ok") is not True:
        return None

    path = root / "runs/phase8-smoke/summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "field": "ok",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_json_invalid",
        }
    if not isinstance(payload, Mapping):
        return {
            "field": "ok",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_not_object",
        }
    if payload.get("ok") is not True:
        return {
            "field": "ok",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_not_ok",
        }
    return _verify_phase8_smoke_summary_structure(payload)


def _verify_phase8_smoke_summary_structure(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    for field in ("coordinator", "learner", "worker"):
        if not isinstance(payload.get(field), Mapping):
            return {
                "field": field,
                "ok": False,
                "path": "runs/phase8-smoke/summary.json",
                "reason": "phase8_smoke_summary_section_invalid",
            }

    coordinator = payload.get("coordinator")
    assert isinstance(coordinator, Mapping)
    if not isinstance(coordinator.get("metrics"), Mapping):
        return {
            "field": "coordinator.metrics",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_metrics_invalid",
        }
    workers = coordinator.get("workers")
    if not isinstance(workers, Mapping) or not workers:
        return {
            "field": "coordinator.workers",
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_workers_invalid",
        }

    for field in ("checkpoint_versions", "task_ids", "worker_ids"):
        value = payload.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            return {
                "field": field,
                "ok": False,
                "path": "runs/phase8-smoke/summary.json",
                "reason": "phase8_smoke_summary_list_invalid",
            }
        if field in {"task_ids", "worker_ids"} and not all(
            isinstance(item, str) and item for item in value
        ):
            return {
                "field": field,
                "ok": False,
                "path": "runs/phase8-smoke/summary.json",
                "reason": "phase8_smoke_summary_list_invalid",
            }
    worker_ids = payload.get("worker_ids")
    assert isinstance(worker_ids, Sequence)
    missing_worker_ids = sorted(
        str(worker_id) for worker_id in worker_ids if worker_id not in workers
    )
    if missing_worker_ids:
        return {
            "field": "worker_ids",
            "missing_worker_ids": missing_worker_ids,
            "ok": False,
            "path": "runs/phase8-smoke/summary.json",
            "reason": "phase8_smoke_summary_worker_rows_missing",
        }
    return None


def _verify_phase8_dashboard_artifact(
    root: Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    dashboard_result = next(
        (result for result in results if result.get("path") == "runs/phase8-smoke/dashboard.json"),
        None,
    )
    if dashboard_result is None or dashboard_result.get("ok") is not True:
        return None

    path = root / "runs/phase8-smoke/dashboard.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "field": "health",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_json_invalid",
        }
    if not isinstance(payload, Mapping):
        return {
            "field": "health",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_not_object",
        }

    for field in ("health", "learner", "metrics"):
        if not isinstance(payload.get(field), Mapping):
            return {
                "field": field,
                "ok": False,
                "path": "runs/phase8-smoke/dashboard.json",
                "reason": "phase8_dashboard_section_invalid",
            }
    health = payload.get("health")
    assert isinstance(health, Mapping)
    if not isinstance(health.get("status"), str) or not health.get("status"):
        return {
            "field": "health.status",
            "ok": False,
            "path": "runs/phase8-smoke/dashboard.json",
            "reason": "phase8_dashboard_health_invalid",
        }

    for field in ("tasks", "workers"):
        value = payload.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            return {
                "field": field,
                "ok": False,
                "path": "runs/phase8-smoke/dashboard.json",
                "reason": "phase8_dashboard_list_invalid",
            }
    return None


def _verify_phase8_profile_artifact(
    root: Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    profile_result = next(
        (result for result in results if result.get("path") == "runs/phase8-smoke/profile.json"),
        None,
    )
    if profile_result is None or profile_result.get("ok") is not True:
        return None

    path = root / "runs/phase8-smoke/profile.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "field": "metrics",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_json_invalid",
        }
    if not isinstance(payload, Mapping):
        return {
            "field": "metrics",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_not_object",
        }
    if payload.get("source") != "phase8_smoke":
        return {
            "field": "source",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_source_invalid",
        }
    if not isinstance(payload.get("metrics"), Mapping):
        return {
            "field": "metrics",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_metrics_invalid",
        }
    findings = payload.get("findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
        return {
            "field": "findings",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_findings_invalid",
        }
    workers = payload.get("workers")
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)) or not workers:
        return {
            "field": "workers",
            "ok": False,
            "path": "runs/phase8-smoke/profile.json",
            "reason": "phase8_profile_workers_invalid",
        }
    return None


def _verify_eval_report_structure(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if payload.get("source") != "run_eval":
        return {
            "field": "source",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_source_invalid",
        }

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return {
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_summary_invalid",
        }
    task_count = summary.get("task_count")
    valid_task_count = summary.get("valid_task_count")
    malformed_task_count = summary.get("malformed_task_count")
    if (
        not _is_non_negative_count(task_count)
        or not _is_non_negative_count(valid_task_count)
        or not _is_non_negative_count(malformed_task_count)
    ):
        return {
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_summary_counts_invalid",
        }
    assert isinstance(task_count, (int, float))
    assert isinstance(valid_task_count, (int, float))
    assert isinstance(malformed_task_count, (int, float))
    expected_task_count = int(task_count)
    valid_tasks = float(valid_task_count)
    malformed_tasks = float(malformed_task_count)

    tasks = payload.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        return {
            "field": "tasks",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_tasks_invalid",
        }
    malformed_task_indexes = [
        index for index, task in enumerate(tasks) if not _valid_eval_report_task(task)
    ]
    if malformed_task_indexes:
        return {
            "field": "tasks",
            "indexes": malformed_task_indexes,
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_tasks_malformed",
        }
    duplicate_task_ids = _duplicate_eval_report_task_ids(tasks)
    if duplicate_task_ids:
        return {
            "duplicate_task_ids": duplicate_task_ids,
            "field": "tasks",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_task_ids_duplicate",
        }
    if expected_task_count != len(tasks):
        return {
            "actual_task_count": len(tasks),
            "expected_task_count": task_count,
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_task_count_mismatch",
        }
    actual_valid_task_count = sum(
        1 for task in tasks if isinstance(task, Mapping) and task.get("metrics_valid") is not False
    )
    actual_malformed_task_count = len(tasks) - actual_valid_task_count
    if int(valid_tasks) != actual_valid_task_count:
        return {
            "actual_valid_task_count": actual_valid_task_count,
            "expected_valid_task_count": valid_task_count,
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_valid_task_count_mismatch",
        }
    if int(malformed_tasks) != actual_malformed_task_count:
        return {
            "actual_malformed_task_count": actual_malformed_task_count,
            "expected_malformed_task_count": malformed_task_count,
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_malformed_task_count_mismatch",
        }
    if valid_tasks <= 0.0:
        return {
            "field": "summary",
            "ok": False,
            "path": "runs/eval-report.json",
            "reason": "eval_report_no_valid_tasks",
        }
    return None


def _valid_eval_report_finding(finding: Any) -> bool:
    return (
        isinstance(finding, Mapping)
        and isinstance(finding.get("code"), str)
        and bool(finding.get("code"))
        and isinstance(finding.get("severity"), str)
        and bool(finding.get("severity"))
    )


def _valid_eval_report_task(task: Any) -> bool:
    return (
        isinstance(task, Mapping)
        and isinstance(task.get("task_id"), str)
        and bool(task.get("task_id"))
        and isinstance(task.get("metrics_valid"), bool)
    )


def _duplicate_eval_report_task_ids(tasks: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = task.get("task_id")
        if not isinstance(task_id, str):
            continue
        if task_id in seen:
            duplicates.add(task_id)
        seen.add(task_id)
    return sorted(duplicates)


def _manifest_artifact_paths(results: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        path
        for result in results
        if isinstance((path := result.get("path")), str) and path != "<missing>"
    }


def _verify_manifest_artifact_count(
    manifest: Mapping[str, Any],
    *,
    actual_count: int,
) -> dict[str, Any] | None:
    expected_count = manifest.get("artifact_count")
    if _invalid_non_negative_int(expected_count):
        return {
            "field": "artifact_count",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_artifact_count_invalid",
        }
    if expected_count != actual_count:
        return {
            "actual_artifact_count": actual_count,
            "expected_artifact_count": expected_count,
            "field": "artifact_count",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_artifact_count_mismatch",
        }
    return None


def _verify_manifest_total_bytes(
    manifest: Mapping[str, Any],
    artifacts: Sequence[Any],
) -> dict[str, Any] | None:
    expected_total = manifest.get("total_bytes")
    if _invalid_non_negative_int(expected_total):
        return {
            "field": "total_bytes",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_total_bytes_invalid",
        }

    declared_total = 0
    for artifact in artifacts:
        item = artifact if isinstance(artifact, Mapping) else {}
        declared_bytes = item.get("bytes")
        if _invalid_non_negative_int(declared_bytes):
            return None
        assert isinstance(declared_bytes, int)
        declared_total += declared_bytes

    if expected_total != declared_total:
        return {
            "actual_total_bytes": declared_total,
            "expected_total_bytes": expected_total,
            "field": "total_bytes",
            "ok": False,
            "path": "<manifest>",
            "reason": "manifest_total_bytes_mismatch",
        }
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _invalid_non_negative_int(value: Any) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 0


def _is_non_negative_count(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value).is_integer()
        and value >= 0.0
    )
