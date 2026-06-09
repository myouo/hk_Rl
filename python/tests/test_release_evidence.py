"""Release evidence manifest tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.utils.release import (
    build_release_evidence_manifest,
    render_release_evidence_markdown,
)


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
    assert manifest["artifact_count"] == 2
    assert manifest["total_bytes"] == summary.stat().st_size + checklist.stat().st_size
    assert manifest["artifacts"][0] == {
        "bytes": summary.stat().st_size,
        "path": "runs/phase8-smoke/summary.json",
        "sha256": _sha256(summary),
    }


def test_release_evidence_markdown_contains_hash_table(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "release" / "checklist.json"
    _write(artifact, "{}\n")
    manifest = build_release_evidence_manifest(
        root=tmp_path,
        artifacts=["runs/release/checklist.json"],
    )

    markdown = render_release_evidence_markdown(manifest)

    assert "# HKRL Release Evidence" in markdown
    assert "| Path | Bytes | SHA256 |" in markdown
    assert "runs/release/checklist.json" in markdown
    assert _sha256(artifact) in markdown


def test_release_evidence_rejects_artifacts_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-release-artifact.txt"
    outside.write_text("bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes root"):
        build_release_evidence_manifest(root=tmp_path, artifacts=[outside])


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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
