"""Release checklist tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from hkrl.utils.release import build_release_checklist, render_release_markdown


def test_release_checklist_contains_phase8_gates() -> None:
    checklist = build_release_checklist(version="phase8", git_sha="abc123")
    check_ids = {check["id"] for check in checklist["checks"]}
    commands = {check["command"] for check in checklist["checks"]}

    assert checklist["version"] == "phase8"
    assert checklist["git_sha"] == "abc123"
    assert checklist["required_count"] == len(checklist["checks"])
    assert {
        "python_quality_gate",
        "offline_distributed_smoke",
        "offline_dashboard",
        "offline_profile",
        "release_evidence_manifest",
        "release_evidence_verification",
        "github_ci",
        "mod_build",
        "live_smoke",
        "fixed_seed_eval",
        "fixed_seed_eval_report",
        "security_scope",
        "docs_changelog",
    } <= check_ids
    assert "make check" in commands
    assert "make phase8-smoke" in commands
    assert "make phase8-dashboard" in commands
    assert "make phase8-profile" in commands
    assert "make phase8-eval-report" in commands
    assert "make phase8-release-evidence" in commands
    assert "make phase8-verify-release-evidence" in commands


def test_release_checklist_markdown_groups_commands() -> None:
    markdown = render_release_markdown(build_release_checklist(version="phase8"))

    assert "# HKRL Release Checklist" in markdown
    assert "## Local" in markdown
    assert "## Game Machine" in markdown
    assert "`make check`" in markdown
    assert "`gh run list --branch main --limit 1`" in markdown


def test_render_release_checklist_script_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_script("render_release_checklist.py")
    json_path = tmp_path / "release.json"
    markdown_path = tmp_path / "release.md"
    args = argparse.Namespace(
        version="phase8",
        git_sha="deadbeef",
        output_json=str(json_path),
        output_md=str(markdown_path),
    )

    checklist = module.run_from_args(args)

    assert checklist["git_sha"] == "deadbeef"
    assert json.loads(json_path.read_text(encoding="utf-8"))["version"] == "phase8"
    assert "HKRL Release Checklist" in markdown_path.read_text(encoding="utf-8")


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
