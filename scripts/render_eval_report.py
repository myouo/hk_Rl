#!/usr/bin/env python3
"""Render a fixed-seed evaluator report from run_eval JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hkrl.eval.report import (
    build_eval_report,
    eval_report_to_json,
    render_eval_report_markdown,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render HKRL fixed-seed eval report")
    p.add_argument("--eval-json", required=True, help="run_eval.py output JSON")
    p.add_argument("--output-json", help="optional normalized report JSON")
    p.add_argument("--output-md", help="optional Markdown report")
    p.add_argument("--min-win-rate", type=float, default=None)
    p.add_argument("--max-regression-drop", type=float, default=0.05)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report = run_from_args(args)
    print(json.dumps(report, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    report = build_eval_report(
        _read_json(Path(args.eval_json)),
        min_win_rate=args.min_win_rate,
        max_regression_drop=args.max_regression_drop,
    )
    if args.output_json:
        _write_text(Path(args.output_json), eval_report_to_json(report))
    if args.output_md:
        _write_text(Path(args.output_md), render_eval_report_markdown(report))
    return report


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("eval JSON must be an object")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
