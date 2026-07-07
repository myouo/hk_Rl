#!/usr/bin/env python3
"""Render a Phase 8 profiling report from coordinator or smoke JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hkrl.coordinator.profiling import (
    build_profile_report,
    render_profile_markdown,
    report_to_json,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render HKRL Phase 8 profile report")
    p.add_argument("--summary", required=True, help="run_coordinator or phase8 smoke JSON")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", help="optional Markdown report path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report = run_from_args(args)
    print(json.dumps(report, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = _non_empty_path(getattr(args, "summary", None), name="summary")
    output_json = _non_empty_path(getattr(args, "output_json", None), name="output_json")
    output_md = _optional_path(getattr(args, "output_md", None), name="output_md")
    report = build_profile_report(_read_json(summary_path))
    _write_text(output_json, report_to_json(report))
    if output_md is not None:
        _write_text(output_md, render_profile_markdown(report))
    return report


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
