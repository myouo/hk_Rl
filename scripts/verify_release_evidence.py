#!/usr/bin/env python3
"""Verify HKRL release evidence artifacts against a sha256 manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hkrl.utils.release import (
    release_evidence_verification_to_json,
    verify_release_evidence_manifest,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify HKRL release evidence manifest")
    p.add_argument("--manifest", required=True, help="release evidence JSON manifest")
    p.add_argument("--root", default=".")
    p.add_argument("--git-sha", help="expected full git SHA for this release")
    p.add_argument("--git-dirty", choices=("true", "false"), help="expected dirty-worktree flag")
    p.add_argument("--output-json", help="optional verification report path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    result = run_from_args(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _non_empty_path(getattr(args, "manifest", None), name="manifest")
    root = _non_empty_path(getattr(args, "root", "."), name="root")
    output_json = _optional_path(getattr(args, "output_json", None), name="output_json")
    manifest = _read_json(manifest_path)
    result = verify_release_evidence_manifest(
        root=root,
        manifest=manifest,
        expected_git_dirty=_optional_bool_arg(getattr(args, "git_dirty", None)),
        expected_git_sha=getattr(args, "git_sha", None),
    )
    if output_json is not None:
        _write_text(output_json, release_evidence_verification_to_json(result))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _optional_bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("git_dirty must be 'true' or 'false'")


def _non_empty_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str | os.PathLike) or not str(value).strip():
        raise ValueError(f"{name} must not be empty")
    return Path(value)


def _optional_path(value: Any, *, name: str) -> Path | None:
    if value is None:
        return None
    return _non_empty_path(value, name=name)


if __name__ == "__main__":
    raise SystemExit(main())
