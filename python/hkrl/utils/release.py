"""Release checklist rendering for Phase 8 engineering gates."""

from __future__ import annotations

import hashlib
import json
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
    checks = list(payload.get("checks", []))
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
        "manifest_version": 1,
        "total_bytes": total_bytes,
        "version": version,
    }


def _default_release_artifacts(
    root: Path,
    *,
    optional_artifacts: Sequence[str | Path],
) -> tuple[str | Path, ...]:
    existing_optional = [
        artifact for artifact in optional_artifacts if _artifact_exists(root, artifact)
    ]
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
    for artifact in artifacts:
        item = dict(artifact)
        lines.append(
            "| "
            f"{_markdown_cell(str(item.get('path', '')))} | "
            f"{item.get('bytes', 0)} | "
            f"`{item.get('sha256', '')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def release_evidence_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def verify_release_evidence_manifest(
    *,
    root: str | Path = ".",
    manifest: Mapping[str, Any],
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

    for artifact in artifacts:
        result = _verify_artifact(root_path, artifact)
        results.append(result)
        if not result["ok"]:
            failures.append(result)

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


def _verify_artifact(root: Path, artifact: Any) -> dict[str, Any]:
    item = artifact if isinstance(artifact, Mapping) else {}
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return {
            "ok": False,
            "path": "<missing>",
            "reason": "artifact_path_missing",
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
