#!/usr/bin/env python3
"""Render a static Phase 8 dashboard from coordinator or smoke JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hkrl.coordinator.dashboard import build_dashboard_model, render_dashboard_html


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render HKRL Phase 8 dashboard")
    p.add_argument("--summary", required=True, help="run_coordinator or phase8 smoke JSON")
    p.add_argument("--output-html", required=True)
    p.add_argument("--output-json", help="optional normalized dashboard model JSON")
    p.add_argument("--eval-metrics", help="optional evaluator metrics JSON override")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    model = run_from_args(args)
    print(json.dumps(model, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = _non_empty_path(getattr(args, "summary", None), name="summary")
    output_html = _non_empty_path(getattr(args, "output_html", None), name="output_html")
    output_json = _optional_path(getattr(args, "output_json", None), name="output_json")
    eval_metrics_path = _optional_path(
        getattr(args, "eval_metrics", None),
        name="eval_metrics",
    )
    summary = _read_json(summary_path)
    eval_metrics = None if eval_metrics_path is None else _read_json(eval_metrics_path)
    model = build_dashboard_model(summary, eval_metrics=eval_metrics)
    html = render_dashboard_html(model)
    _write_text(output_html, html)
    if output_json is not None:
        _write_json(output_json, model)
    return model


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
