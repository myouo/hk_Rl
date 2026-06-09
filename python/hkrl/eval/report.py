"""Fixed-seed evaluator report rendering for Phase 8 regression evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

TASK_METRICS: tuple[str, ...] = (
    "win_rate",
    "per_boss_win_rate",
    "damage_taken",
    "damage_dealt",
    "per_boss_damage_ratio",
    "time_to_kill",
    "invalid_action_ratio",
    "death_rate",
)


def build_eval_report(
    payload: Mapping[str, Any],
    *,
    min_win_rate: float | None = None,
    max_regression_drop: float = 0.05,
) -> dict[str, Any]:
    """Build a stable report from ``scripts/run_eval.py`` output."""
    if max_regression_drop < 0.0:
        raise ValueError("max_regression_drop must be non-negative")
    if min_win_rate is not None and not 0.0 <= min_win_rate <= 1.0:
        raise ValueError("min_win_rate must be between 0 and 1")

    metadata = _mapping(payload.get("metadata", {}))
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("eval report input must contain a metrics object")

    regression = _mapping(payload.get("regression", {}))
    tasks = _task_rows(metrics, regression=regression)
    summary = _summary(tasks)
    findings = _findings(
        tasks,
        min_win_rate=min_win_rate,
        max_regression_drop=max_regression_drop,
    )
    return {
        "findings": findings,
        "metadata": _metadata_summary(metadata),
        "source": "run_eval",
        "summary": summary,
        "tasks": tasks,
    }


def render_eval_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for release artifacts."""
    summary = _mapping(report.get("summary", {}))
    metadata = _mapping(report.get("metadata", {}))
    findings = list(report.get("findings", []))
    tasks = list(report.get("tasks", []))
    lines = [
        "# HKRL Eval Report",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in sorted(metadata):
        lines.append(f"| `{key}` | {_markdown_cell(_format_value(metadata[key]))} |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key in sorted(summary):
        lines.append(f"| `{key}` | {_format_value(summary[key])} |")

    lines.extend(["", "## Findings", ""])
    if findings:
        for finding in findings:
            item = _mapping(finding)
            lines.append(
                "- "
                f"**{item.get('severity', 'info')}** `{item.get('code', 'unknown')}`: "
                f"{item.get('message', '')} "
                f"Recommendation: {item.get('recommendation', '')}"
            )
    else:
        lines.append("- No evaluator findings.")

    lines.extend(
        [
            "",
            "## Tasks",
            "",
            (
                "| Task | Metrics Valid | Regression Valid | Win Rate | Regression Delta | "
                "Damage Taken | TTK | Invalid Action Ratio | Death Rate |"
            ),
            "| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in (_mapping(item) for item in tasks):
        lines.append(
            "| "
            f"{_markdown_cell(str(task.get('task_id', '')))} | "
            f"{_format_value(task.get('metrics_valid'))} | "
            f"{_format_value(task.get('regression_valid'))} | "
            f"{_format_value(task.get('win_rate'))} | "
            f"{_format_value(task.get('regression_delta'))} | "
            f"{_format_value(task.get('damage_taken'))} | "
            f"{_format_value(task.get('time_to_kill'))} | "
            f"{_format_value(task.get('invalid_action_ratio'))} | "
            f"{_format_value(task.get('death_rate'))} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def eval_report_to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _task_rows(
    metrics: Mapping[str, Any],
    *,
    regression: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, raw_metrics in sorted(metrics.items()):
        metrics_valid = isinstance(raw_metrics, Mapping)
        task_metrics = raw_metrics if metrics_valid else {}
        row: dict[str, Any] = {key: _task_metric(task_metrics, key) for key in TASK_METRICS}
        row["metrics_valid"] = metrics_valid
        row["task_id"] = str(task_id)
        row["regression_delta"], row["regression_valid"] = _regression_delta(
            regression,
            str(task_id),
        )
        rows.append(row)
    return rows


def _task_metric(task_metrics: Mapping[str, Any], key: str) -> float:
    if key == "win_rate":
        win_rate = _finite_float_or_none(task_metrics.get("win_rate"))
        if win_rate is not None:
            return win_rate
        return _float(task_metrics.get("per_boss_win_rate", 0.0))
    return _float(task_metrics.get(key, 0.0))


def _regression_delta(
    regression: Mapping[str, Any],
    task_id: str,
) -> tuple[float | None, bool]:
    if task_id not in regression:
        return None, True
    raw_delta = regression.get(task_id)
    if raw_delta is None:
        return None, True
    delta = _finite_float_or_none(raw_delta)
    if delta is None:
        return None, False
    return delta, True


def _summary(tasks: list[dict[str, Any]]) -> dict[str, float]:
    valid_tasks = [task for task in tasks if task.get("metrics_valid") is not False]
    win_rates = [_float(task.get("win_rate", 0.0)) for task in valid_tasks]
    regressions = [
        float(task["regression_delta"])
        for task in valid_tasks
        if task.get("regression_valid") is not False and task.get("regression_delta") is not None
    ]
    return {
        "malformed_task_count": float(len(tasks) - len(valid_tasks)),
        "mean_win_rate": _mean(win_rates),
        "min_win_rate": min(win_rates, default=0.0),
        "task_count": float(len(tasks)),
        "valid_task_count": float(len(valid_tasks)),
        "worst_regression_delta": min(regressions, default=0.0),
    }


def _metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": _optional_str(metadata.get("checkpoint")),
        "checkpoint_dir": _optional_str(metadata.get("checkpoint_dir")),
        "episodes": _float(metadata.get("episodes", 0.0)),
        "eval_workers": _float(metadata.get("eval_workers", 0.0)),
        "model": _optional_str(metadata.get("model")),
        "policy": _optional_str(metadata.get("policy")),
        "seeds": list(metadata.get("seeds", [])) if isinstance(metadata.get("seeds"), list) else [],
        "task_ids": [str(task_id) for task_id in metadata.get("task_ids", [])]
        if isinstance(metadata.get("task_ids"), list)
        else [],
    }


def _findings(
    tasks: list[dict[str, Any]],
    *,
    min_win_rate: float | None,
    max_regression_drop: float,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not tasks:
        findings.append(
            _finding(
                "critical",
                "no_eval_tasks",
                "The evaluator report contains no task metrics.",
                "Run fixed-seed eval before producing release evidence.",
            )
        )
        return findings

    for task in tasks:
        if task.get("metrics_valid") is False:
            findings.append(
                _finding(
                    "critical",
                    "malformed_task_metrics",
                    f"{task['task_id']} has a non-object metric payload.",
                    "Re-run fixed-seed eval and check the evaluator JSON writer.",
                )
            )

    if not any(task.get("metrics_valid") is not False for task in tasks):
        findings.append(
            _finding(
                "critical",
                "no_valid_eval_tasks",
                "The evaluator report contains no valid task metric rows.",
                "Re-run fixed-seed eval and inspect malformed task metric payloads.",
            )
        )

    for task in tasks:
        if task.get("metrics_valid") is False or task.get("regression_valid") is not False:
            continue
        findings.append(
            _finding(
                "critical",
                "malformed_regression_delta",
                f"{task['task_id']} has a non-finite or non-numeric regression delta.",
                "Rebuild the regression baseline comparison before using this eval report.",
            )
        )

    if min_win_rate is not None:
        for task in tasks:
            if task.get("metrics_valid") is False:
                continue
            if _float(task.get("win_rate", 0.0)) < min_win_rate:
                findings.append(
                    _finding(
                        "warning",
                        "low_win_rate",
                        f"{task['task_id']} is below the configured win-rate floor.",
                        "Inspect policy capability before treating reward as progress.",
                    )
                )

    for task in tasks:
        if task.get("metrics_valid") is False:
            continue
        if task.get("regression_valid") is False:
            continue
        delta = task.get("regression_delta")
        if delta is not None and float(delta) < -max_regression_drop:
            findings.append(
                _finding(
                    "warning",
                    "win_rate_regression",
                    f"{task['task_id']} regressed versus the baseline.",
                    "Check anti-forgetting data and compare fixed-seed episodes.",
                )
            )
    return findings


def _finding(severity: str, code: str, message: str, recommendation: str) -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "recommendation": recommendation,
        "severity": severity,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
