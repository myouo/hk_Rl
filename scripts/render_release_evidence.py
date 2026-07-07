#!/usr/bin/env python3
"""Render a hash manifest for HKRL release evidence artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hkrl.utils.release import (
    build_release_evidence_manifest,
    release_evidence_to_json,
    render_release_evidence_markdown,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render HKRL release evidence manifest")
    p.add_argument("--version", default="phase8")
    p.add_argument("--git-sha", default=None)
    p.add_argument("--git-dirty", choices=("true", "false"), default="false")
    p.add_argument("--root", default=".")
    p.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="release artifact path relative to --root; may be repeated",
    )
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", help="optional Markdown manifest path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest = run_from_args(args)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    version = _non_empty_string(getattr(args, "version", None), name="version")
    root = _non_empty_path(getattr(args, "root", "."), name="root")
    output_json = _non_empty_path(getattr(args, "output_json", None), name="output_json")
    output_md = _optional_path(getattr(args, "output_md", None), name="output_md")
    artifacts = _optional_artifacts(getattr(args, "artifacts", None))
    manifest = build_release_evidence_manifest(
        root=root,
        version=version,
        git_dirty=_bool_arg(getattr(args, "git_dirty", "false")),
        git_sha=getattr(args, "git_sha", None),
        artifacts=artifacts,
    )
    _write_text(output_json, release_evidence_to_json(manifest))
    if output_md is not None:
        _write_text(output_md, render_release_evidence_markdown(manifest))
    return manifest


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bool_arg(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("git_dirty must be 'true' or 'false'")


def _optional_artifacts(value: Any) -> tuple[str | os.PathLike[str], ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, list | tuple):
        raise ValueError("artifacts must be a sequence of paths")
    artifacts: list[str | os.PathLike[str]] = []
    for index, artifact in enumerate(value):
        artifacts.append(_non_empty_path_like(artifact, name=f"artifacts[{index}]"))
    if not artifacts:
        return None
    return tuple(artifacts)


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _non_empty_path(value: Any, *, name: str) -> Path:
    return Path(_non_empty_path_like(value, name=name))


def _non_empty_path_like(value: Any, *, name: str) -> str | os.PathLike[str]:
    if not isinstance(value, str | os.PathLike) or not str(value).strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_path(value: Any, *, name: str) -> Path | None:
    if value is None:
        return None
    return _non_empty_path(value, name=name)


if __name__ == "__main__":
    raise SystemExit(main())
