#!/usr/bin/env python3
"""Render the HKRL release checklist as JSON/Markdown."""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", help="optional Markdown checklist path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    checklist = run_from_args(args)
    print(json.dumps(checklist, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    checklist = build_release_checklist(
        version=str(args.version),
        git_sha=getattr(args, "git_sha", None),
    )
    _write_text(Path(args.output_json), release_checklist_to_json(checklist))
    if getattr(args, "output_md", None):
        _write_text(Path(args.output_md), render_release_markdown(checklist))
    return checklist


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
