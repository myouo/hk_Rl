#!/usr/bin/env python3
"""Render the HKRL release checklist as JSON/Markdown."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hkrl.utils.release import (
    build_release_checklist,
    release_checklist_to_json,
    render_release_markdown,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render HKRL release checklist")
    p.add_argument("--version", default="phase8")
    p.add_argument("--git-sha", default=None)
    p.add_argument("--git-dirty", choices=("true", "false"), default="false")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", help="optional Markdown checklist path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    checklist = run_from_args(args)
    print(json.dumps(checklist, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    version = _non_empty_string(getattr(args, "version", None), name="version")
    output_json = _non_empty_path(getattr(args, "output_json", None), name="output_json")
    output_md = _optional_path(getattr(args, "output_md", None), name="output_md")
    checklist = build_release_checklist(
        version=version,
        git_dirty=_bool_arg(getattr(args, "git_dirty", "false")),
        git_sha=getattr(args, "git_sha", None),
    )
    _write_text(output_json, release_checklist_to_json(checklist))
    if output_md is not None:
        _write_text(output_md, render_release_markdown(checklist))
    return checklist


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bool_arg(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("git_dirty must be 'true' or 'false'")


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


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
