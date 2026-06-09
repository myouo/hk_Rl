"""Release checklist rendering for Phase 8 engineering gates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
