#!/usr/bin/env python3
"""Render a fixed-seed evaluator report from run_eval JSON."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
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
    p.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="exit non-zero after writing artifacts if the report has critical findings",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report = run_from_args(args)
    print(json.dumps(report, sort_keys=True))
    if args.fail_on_critical and _has_critical_findings(report):
        return 1
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    eval_json = _non_empty_path(getattr(args, "eval_json", None), name="eval_json")
    output_json = _optional_path(getattr(args, "output_json", None), name="output_json")
    output_md = _optional_path(getattr(args, "output_md", None), name="output_md")
    report = build_eval_report(
        _read_json(eval_json),
        min_win_rate=args.min_win_rate,
        max_regression_drop=args.max_regression_drop,
    )
    if output_json is not None:
        _write_text(output_json, eval_report_to_json(report))
    if output_md is not None:
        _write_text(output_md, render_eval_report_markdown(report))
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


def _has_critical_findings(report: Mapping[str, Any]) -> bool:
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        return False
    return any(
        isinstance(finding, Mapping) and finding.get("severity") == "critical"
        for finding in findings
    )


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
