"""Static monitoring dashboard model/rendering for Phase 8 worker fleets."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from typing import Any

KEY_METRICS: tuple[str, ...] = (
    "worker_count",
    "active_worker_count",
    "lost_worker_count",
    "assigned_worker_count",
    "unassigned_worker_count",
    "recovering_worker_count",
    "sps",
    "sps_mean",
    "worker_crash_count",
    "worker_learner_upload_submitted_batches",
    "worker_learner_upload_accepted_batches",
    "worker_learner_upload_rejected_batches",
    "worker_learner_upload_failed_batches",
    "worker_policy_lag_max",
    "worker_checkpoint_lag_max",
    "stale_policy_worker_count",
    "stale_checkpoint_worker_count",
    "worker_without_policy_version_count",
    "worker_without_checkpoint_version_count",
)

LEARNER_METRICS: tuple[str, ...] = (
    "accepted_batches",
    "network_accepted_batches",
    "network_submitted_batches",
    "queued_batches",
    "rejected_batches",
    "submitted_batches",
)


def build_dashboard_model(
    payload: Mapping[str, Any],
    *,
    eval_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact dashboard model from coordinator or Phase 8 smoke JSON."""
    coordinator = _coordinator_payload(payload)
    metrics = _metrics_payload(coordinator)
    workers = _worker_rows(coordinator.get("workers", {}), metrics)
    tasks = _task_rows(coordinator, eval_metrics=eval_metrics)
    dashboard_metrics = _dashboard_metrics(metrics)
    learner = _learner_summary(payload)
    health = _health(dashboard_metrics, learner, workers)

    return {
        "health": health,
        "learner": learner,
        "metrics": dashboard_metrics,
        "sampler_mastered_tasks": sorted(
            str(task_id) for task_id in coordinator.get("sampler_mastered_tasks", [])
        ),
        "tasks": tasks,
        "workers": workers,
    }


def render_dashboard_html(model: Mapping[str, Any]) -> str:
    """Render a standalone static HTML dashboard."""
    health = _mapping(model.get("health", {}))
    learner = _mapping(model.get("learner", {}))
    metrics = _mapping(model.get("metrics", {}))
    tasks = list(model.get("tasks", []))
    workers = list(model.get("workers", []))
    status = str(health.get("status", "unknown"))
    reasons = [str(reason) for reason in health.get("reasons", [])]

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>HKRL Phase 8 Dashboard</title>",
            "<style>",
            _STYLE,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            "<h1>HKRL Phase 8 Dashboard</h1>",
            f'<p class="status status-{_escape(status)}">{_escape(status.upper())}</p>',
            "</header>",
            _render_reasons(reasons),
            '<section class="metrics" aria-label="Fleet metrics">',
            *[_render_metric(key, metrics.get(key, 0.0)) for key in KEY_METRICS if key in metrics],
            "</section>",
            '<section aria-label="Learner">',
            "<h2>Learner</h2>",
            _render_learner_table(learner),
            "</section>",
            '<section aria-label="Workers">',
            "<h2>Workers</h2>",
            _render_worker_table(workers),
            "</section>",
            '<section aria-label="Tasks">',
            "<h2>Tasks</h2>",
            _render_task_table(tasks),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _coordinator_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("coordinator")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _metrics_payload(coordinator: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = coordinator.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("dashboard input must contain a coordinator metrics object")
    return metrics


def _dashboard_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = {key: _float(metrics.get(key, 0.0)) for key in KEY_METRICS}
    rows["unassigned_worker_count"] = max(
        0.0,
        rows["active_worker_count"] - rows["assigned_worker_count"],
    )
    return rows


def _learner_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    learner = _mapping(payload.get("learner"))
    summary: dict[str, Any] = {key: _float(learner.get(key, 0.0)) for key in LEARNER_METRICS}
    summary.update(
        {
            "algorithm": _optional_str(learner.get("algorithm")),
            "latest_checkpoint": _optional_float(learner.get("latest_checkpoint")),
            "model": _optional_str(learner.get("model")),
            "policy_version": _float(learner.get("policy_version", 0.0)),
        }
    )
    return summary


def _worker_rows(raw_workers: Any, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw_workers, Mapping):
        return []

    policy_max = _float(metrics.get("worker_policy_version_max", 0.0))
    checkpoint_max = _float(metrics.get("worker_checkpoint_version_max", 0.0))
    rows: list[dict[str, Any]] = []
    for worker_id, raw_record in sorted(raw_workers.items()):
        record = _mapping(raw_record)
        worker_metrics = _mapping(record.get("metrics", {}))
        info = _mapping(record.get("info", {}))
        policy_version = _optional_float(worker_metrics.get("policy_version"))
        checkpoint_version = _optional_float(worker_metrics.get("checkpoint_version"))
        rows.append(
            {
                "alive": _optional_bool(record.get("alive")),
                "assigned_task": _optional_str(record.get("assigned_task")),
                "checkpoint_lag": _lag(checkpoint_max, checkpoint_version),
                "checkpoint_version": checkpoint_version,
                "learner_upload_accepted_batches": _float(
                    worker_metrics.get("learner_upload_accepted_batches", 0.0)
                ),
                "learner_upload_failed_batches": _float(
                    worker_metrics.get("learner_upload_failed_batches", 0.0)
                ),
                "learner_upload_rejected_batches": _float(
                    worker_metrics.get("learner_upload_rejected_batches", 0.0)
                ),
                "learner_upload_submitted_batches": _float(
                    worker_metrics.get("learner_upload_submitted_batches", 0.0)
                ),
                "policy_lag": _lag(policy_max, policy_version),
                "policy_version": policy_version,
                "sps": _float(worker_metrics.get("sps", 0.0)),
                "status": str(info.get("status", "unknown")),
                "worker_crash_count": _float(worker_metrics.get("worker_crash_count", 0.0)),
                "worker_id": str(worker_id),
            }
        )
    return rows


def _task_rows(
    coordinator: Mapping[str, Any],
    *,
    eval_metrics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    task_ids = [str(task_id) for task_id in coordinator.get("task_ids", [])]
    weights = _mapping(coordinator.get("sampler_weights", {}))
    mastered = {str(task_id) for task_id in coordinator.get("sampler_mastered_tasks", [])}
    winrates = _eval_winrates(coordinator, eval_metrics=eval_metrics)
    return [
        {
            "mastered": task_id in mastered,
            "sampler_weight": _float(weights.get(task_id, 0.0)),
            "task_id": task_id,
            "win_rate": winrates.get(task_id),
        }
        for task_id in task_ids
    ]


def _eval_winrates(
    coordinator: Mapping[str, Any],
    *,
    eval_metrics: Mapping[str, Any] | None,
) -> dict[str, float]:
    raw = eval_metrics if eval_metrics is not None else coordinator.get("eval_winrates", {})
    if not isinstance(raw, Mapping):
        return {}

    metrics = raw.get("metrics", raw)
    if not isinstance(metrics, Mapping):
        return {}

    winrates: dict[str, float] = {}
    for task_id, value in metrics.items():
        if isinstance(value, Mapping):
            winrate = value.get("win_rate", value.get("per_boss_win_rate"))
        else:
            winrate = value
        if winrate is not None:
            winrates[str(task_id)] = _float(winrate)
    return winrates


def _health(
    metrics: Mapping[str, Any],
    learner: Mapping[str, Any],
    workers: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if _float(metrics.get("lost_worker_count", 0.0)) > 0.0 or _has_dead_worker(workers):
        reasons.append("lost workers")
    if _float(metrics.get("unassigned_worker_count", 0.0)) > 0.0:
        reasons.append("unassigned workers")
    if _float(metrics.get("recovering_worker_count", 0.0)) > 0.0:
        reasons.append("workers recovering")
    if _float(metrics.get("worker_crash_count", 0.0)) > 0.0:
        reasons.append("worker crashes")
    if _float(metrics.get("worker_learner_upload_failed_batches", 0.0)) > 0.0:
        reasons.append("worker learner upload failures")
    if _float(metrics.get("worker_learner_upload_rejected_batches", 0.0)) > 0.0:
        reasons.append("worker learner upload rejections")
    if (
        _float(metrics.get("stale_policy_worker_count", 0.0)) > 0.0
        or _float(metrics.get("worker_policy_lag_max", 0.0)) > 0.0
    ):
        reasons.append("stale policy workers")
    if (
        _float(metrics.get("stale_checkpoint_worker_count", 0.0)) > 0.0
        or _float(metrics.get("worker_checkpoint_lag_max", 0.0)) > 0.0
    ):
        reasons.append("stale checkpoint workers")
    if _float(metrics.get("worker_without_policy_version_count", 0.0)) > 0.0:
        reasons.append("workers missing policy version")
    if _float(metrics.get("worker_without_checkpoint_version_count", 0.0)) > 0.0:
        reasons.append("workers missing checkpoint version")
    if (
        _float(metrics.get("active_worker_count", 0.0)) > 0.0
        and _float(metrics.get("sps", 0.0)) <= 0.0
    ):
        reasons.append("zero fleet SPS")
    if _float(learner.get("rejected_batches", 0.0)) > 0.0:
        reasons.append("learner rejected batches")
    if _float(learner.get("queued_batches", 0.0)) > 0.0:
        reasons.append("learner queued batches")

    return {"reasons": reasons, "status": "degraded" if reasons else "healthy"}


def _has_dead_worker(workers: list[dict[str, Any]]) -> bool:
    return any(worker.get("alive") is False for worker in workers)


def _render_reasons(reasons: list[str]) -> str:
    if not reasons:
        return '<p class="summary">No fleet health issues reported.</p>'
    items = "".join(f"<li>{_escape(reason)}</li>" for reason in reasons)
    return f'<ul class="summary issues">{items}</ul>'


def _render_metric(key: str, value: Any) -> str:
    label = key.replace("_", " ")
    return (
        '<article class="metric">'
        f"<span>{_escape(label)}</span>"
        f"<strong>{_escape(_format_value(value))}</strong>"
        "</article>"
    )


def _render_worker_table(workers: list[Any]) -> str:
    headers = (
        "worker",
        "alive",
        "status",
        "task",
        "sps",
        "policy",
        "policy lag",
        "checkpoint",
        "checkpoint lag",
        "crashes",
        "upload submitted",
        "upload accepted",
        "upload rejected",
        "upload failed",
    )
    rows = [
        [
            row.get("worker_id"),
            row.get("alive"),
            row.get("status"),
            row.get("assigned_task"),
            row.get("sps"),
            row.get("policy_version"),
            row.get("policy_lag"),
            row.get("checkpoint_version"),
            row.get("checkpoint_lag"),
            row.get("worker_crash_count"),
            row.get("learner_upload_submitted_batches"),
            row.get("learner_upload_accepted_batches"),
            row.get("learner_upload_rejected_batches"),
            row.get("learner_upload_failed_batches"),
        ]
        for row in (_mapping(item) for item in workers)
    ]
    return _render_table(headers, rows)


def _render_learner_table(learner: Mapping[str, Any]) -> str:
    headers = (
        "algorithm",
        "model",
        "policy",
        "checkpoint",
        "accepted",
        "rejected",
        "queued",
        "network accepted",
        "network submitted",
    )
    rows = [
        [
            learner.get("algorithm"),
            learner.get("model"),
            learner.get("policy_version"),
            learner.get("latest_checkpoint"),
            learner.get("accepted_batches"),
            learner.get("rejected_batches"),
            learner.get("queued_batches"),
            learner.get("network_accepted_batches"),
            learner.get("network_submitted_batches"),
        ]
    ]
    return _render_table(headers, rows)


def _render_task_table(tasks: list[Any]) -> str:
    headers = ("task", "win rate", "sampler weight", "mastered")
    rows = [
        [
            row.get("task_id"),
            row.get("win_rate"),
            row.get("sampler_weight"),
            row.get("mastered"),
        ]
        for row in (_mapping(item) for item in tasks)
    ]
    return _render_table(headers, rows)


def _render_table(headers: tuple[str, ...], rows: list[list[Any]]) -> str:
    header_html = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    if not rows:
        body_html = f'<tr><td colspan="{len(headers)}">No records.</td></tr>'
    else:
        body_html = "".join(
            "<tr>" + "".join(f"<td>{_escape(_format_value(cell))}</td>" for cell in row) + "</tr>"
            for row in rows
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _lag(max_version: float, version: float | None) -> float | None:
    if version is None:
        return None
    return max(0.0, max_version - version)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


_STYLE = """
:root {
  color-scheme: light;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background: #f5f7fa;
  color: #17202a;
}
body {
  margin: 0;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid #d7dde5;
  padding-bottom: 16px;
}
h1, h2 {
  margin: 0;
  font-weight: 650;
}
h1 {
  font-size: 28px;
}
h2 {
  font-size: 18px;
  margin: 28px 0 12px;
}
.status {
  margin: 0;
  padding: 6px 10px;
  border-radius: 4px;
  font-weight: 700;
  background: #e7f4ee;
  color: #176c43;
}
.status-degraded {
  background: #fdebea;
  color: #9a3412;
}
.summary {
  margin: 18px 0;
}
.issues {
  color: #7c2d12;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-top: 18px;
}
.metric {
  background: #ffffff;
  border: 1px solid #dfe5ec;
  border-radius: 6px;
  padding: 12px;
}
.metric span {
  display: block;
  color: #5c6875;
  font-size: 12px;
  text-transform: uppercase;
}
.metric strong {
  display: block;
  margin-top: 8px;
  font-size: 24px;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border: 1px solid #dfe5ec;
}
th, td {
  padding: 9px 10px;
  border-bottom: 1px solid #e8edf2;
  text-align: left;
  font-size: 14px;
}
th {
  background: #eef2f6;
  color: #394553;
  font-size: 12px;
  text-transform: uppercase;
}
tr:last-child td {
  border-bottom: 0;
}
""".strip()
