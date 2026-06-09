#!/usr/bin/env python3
"""Render a hash manifest for HKRL release evidence artifacts."""

from __future__ import annotations

import argparse
import json
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
    manifest = build_release_evidence_manifest(
        root=getattr(args, "root", "."),
        version=str(args.version),
        git_sha=getattr(args, "git_sha", None),
        artifacts=tuple(args.artifacts) if getattr(args, "artifacts", None) else None,
    )
    _write_text(Path(args.output_json), release_evidence_to_json(manifest))
    if getattr(args, "output_md", None):
        _write_text(Path(args.output_md), render_release_evidence_markdown(manifest))
    return manifest


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
