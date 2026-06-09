"""Phase 8 fleet profiling report helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def build_profile_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a normalized profiling report from coordinator or smoke summary JSON."""
    coordinator = _coordinator_payload(payload)
    learner = _learner_payload(payload)
    metrics = _mapping(coordinator.get("metrics", {}))
    workers = _worker_profiles(coordinator.get("workers", {}))
    rollout_durations = [
        worker["rollout_duration_s"]
        for worker in workers
        if worker["rollout_duration_s"] is not None
    ]
    rollout_steps = [
        worker["rollout_steps"] for worker in workers if worker["rollout_steps"] is not None
    ]
    active_workers = _float(metrics.get("active_worker_count", 0.0))
    assigned_workers = _float(metrics.get("assigned_worker_count", 0.0))
    sps = _float(metrics.get("sps", 0.0))
    report_metrics = {
        "active_worker_count": active_workers,
        "assigned_worker_count": assigned_workers,
        "learner_accepted_batches": _float(learner.get("accepted_batches", 0.0)),
        "learner_network_accepted_batches": _float(learner.get("network_accepted_batches", 0.0)),
        "learner_network_submitted_batches": _float(learner.get("network_submitted_batches", 0.0)),
        "learner_queued_batches": _float(learner.get("queued_batches", 0.0)),
        "learner_rejected_batches": _float(learner.get("rejected_batches", 0.0)),
        "learner_submitted_batches": _float(learner.get("submitted_batches", 0.0)),
        "lost_worker_count": _float(metrics.get("lost_worker_count", 0.0)),
        "recovering_worker_count": _float(metrics.get("recovering_worker_count", 0.0)),
        "rollout_duration_s_max": max(rollout_durations, default=0.0),
        "rollout_duration_s_mean": _mean(rollout_durations),
        "rollout_steps_total": sum(rollout_steps),
        "sps": sps,
        "sps_per_active_worker": sps / active_workers if active_workers > 0.0 else 0.0,
        "stale_checkpoint_worker_count": _float(metrics.get("stale_checkpoint_worker_count", 0.0)),
        "stale_policy_worker_count": _float(metrics.get("stale_policy_worker_count", 0.0)),
        "unassigned_worker_count": max(0.0, active_workers - assigned_workers),
        "worker_checkpoint_lag_max": _float(metrics.get("worker_checkpoint_lag_max", 0.0)),
        "worker_count": _float(metrics.get("worker_count", 0.0)),
        "worker_crash_count": _float(metrics.get("worker_crash_count", 0.0)),
        "worker_learner_upload_accepted_batches": _float(
            metrics.get("worker_learner_upload_accepted_batches", 0.0)
        ),
        "worker_learner_upload_failed_batches": _float(
            metrics.get("worker_learner_upload_failed_batches", 0.0)
        ),
        "worker_learner_upload_rejected_batches": _float(
            metrics.get("worker_learner_upload_rejected_batches", 0.0)
        ),
        "worker_learner_upload_submitted_batches": _float(
            metrics.get("worker_learner_upload_submitted_batches", 0.0)
        ),
        "worker_without_checkpoint_version_count": _float(
            metrics.get("worker_without_checkpoint_version_count", 0.0)
        ),
        "worker_without_policy_version_count": _float(
            metrics.get("worker_without_policy_version_count", 0.0)
        ),
        "worker_policy_lag_max": _float(metrics.get("worker_policy_lag_max", 0.0)),
    }
    return {
        "findings": _findings(report_metrics, workers),
        "metrics": report_metrics,
        "source": _source(payload),
        "workers": workers,
    }


def render_profile_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown profile report."""
    metrics = _mapping(report.get("metrics", {}))
    findings = list(report.get("findings", []))
    workers = list(report.get("workers", []))
    lines = [
        "# HKRL Phase 8 Profile",
        "",
        "## Fleet Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in sorted(metrics):
        lines.append(f"| `{key}` | {_format_value(metrics[key])} |")

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
        lines.append("- No profiling findings.")

    lines.extend(
        [
            "",
            "## Workers",
            "",
            (
                "| Worker | Status | SPS | Rollout s | Steps | Crashes | "
                "Upload Submitted | Upload Accepted | Upload Rejected | Upload Failed |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for worker in (_mapping(item) for item in workers):
        lines.append(
            "| "
            f"{worker.get('worker_id', '')} | "
            f"{worker.get('status', '')} | "
            f"{_format_value(worker.get('sps'))} | "
            f"{_format_value(worker.get('rollout_duration_s'))} | "
            f"{_format_value(worker.get('rollout_steps'))} | "
            f"{_format_value(worker.get('worker_crash_count'))} | "
            f"{_format_value(worker.get('learner_upload_submitted_batches'))} | "
            f"{_format_value(worker.get('learner_upload_accepted_batches'))} | "
            f"{_format_value(worker.get('learner_upload_rejected_batches'))} | "
            f"{_format_value(worker.get('learner_upload_failed_batches'))} |"
        )

    return "\n".join(lines) + "\n"


def report_to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _coordinator_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    coordinator = payload.get("coordinator")
    if isinstance(coordinator, Mapping):
        return coordinator
    return payload


def _learner_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    learner = payload.get("learner")
    if isinstance(learner, Mapping):
        return learner
    return {}


def _source(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("coordinator"), Mapping):
        return "phase8_smoke"
    return "coordinator"


def _worker_profiles(raw_workers: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_workers, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for worker_id, raw_record in sorted(raw_workers.items()):
        record = _mapping(raw_record)
        info = _mapping(record.get("info", {}))
        metrics = _mapping(record.get("metrics", {}))
        rows.append(
            {
                "alive": bool(record.get("alive", False)),
                "rollout_duration_s": _optional_float(metrics.get("rollout_duration_s")),
                "rollout_steps": _optional_float(metrics.get("rollout_steps")),
                "sps": _float(metrics.get("sps", 0.0)),
                "status": str(info.get("status", "unknown")),
                "worker_crash_count": _float(metrics.get("worker_crash_count", 0.0)),
                "learner_upload_accepted_batches": _float(
                    metrics.get("learner_upload_accepted_batches", 0.0)
                ),
                "learner_upload_failed_batches": _float(
                    metrics.get("learner_upload_failed_batches", 0.0)
                ),
                "learner_upload_rejected_batches": _float(
                    metrics.get("learner_upload_rejected_batches", 0.0)
                ),
                "learner_upload_submitted_batches": _float(
                    metrics.get("learner_upload_submitted_batches", 0.0)
                ),
                "worker_id": str(worker_id),
            }
        )
    return rows


def _findings(metrics: Mapping[str, float], workers: list[dict[str, Any]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if metrics["active_worker_count"] <= 0.0:
        findings.append(
            _finding(
                "critical",
                "no_active_workers",
                "No active workers are reporting heartbeats.",
                "Start or reconnect GameWorker instances before profiling throughput.",
            )
        )
    elif metrics["sps"] <= 0.0:
        findings.append(
            _finding(
                "critical",
                "zero_sps",
                "Active workers report zero fleet SPS.",
                "Profile reset readiness, env.step timeouts, and transport reconnect loops first.",
            )
        )
    if metrics["unassigned_worker_count"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "unassigned_workers",
                "Some active workers do not have assigned tasks.",
                "Inspect coordinator registration, task sampler state, and worker assignment loop.",
            )
        )
    if metrics["learner_rejected_batches"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "learner_rejected_batches",
                "The learner rejected rollout batches.",
                "Inspect policy-version staleness, batch schema, and worker upload health.",
            )
        )
    if metrics["learner_queued_batches"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "learner_queued_batches",
                "The learner still has queued rollout batches after serving.",
                "Check learner update throughput and batch intake backpressure.",
            )
        )
    if metrics["recovering_worker_count"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "recovering_workers",
                "At least one active worker is in recovery.",
                "Inspect worker error heartbeats and transport timeout/reconnect behavior.",
            )
        )
    if metrics["worker_crash_count"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "worker_crashes",
                "Worker crash/recovery churn is visible in the fleet.",
                "Correlate crash count with learner intake, env transport, and reset failures.",
            )
        )
    if metrics["worker_learner_upload_failed_batches"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "worker_learner_upload_failures",
                "Workers failed to upload rollout batches to the learner.",
                "Inspect learner reachability, auth tokens, and worker upload retry behavior.",
            )
        )
    if metrics["worker_learner_upload_rejected_batches"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "worker_learner_upload_rejections",
                "The learner rejected batches submitted by workers.",
                "Inspect policy staleness, batch schema, and learner max-staleness settings.",
            )
        )
    if metrics["stale_policy_worker_count"] > 0.0 or metrics["worker_policy_lag_max"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "stale_policy_workers",
                "Some workers are behind the newest active policy version.",
                "Check checkpoint polling cadence and learner publication frequency.",
            )
        )
    if metrics["stale_checkpoint_worker_count"] > 0.0 or metrics["worker_checkpoint_lag_max"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "stale_checkpoint_workers",
                "Some workers are behind the newest active checkpoint version.",
                "Verify checkpoint registry reachability and hash verification failures.",
            )
        )
    if metrics["worker_without_policy_version_count"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "missing_policy_versions",
                "Some active workers have not reported a policy version.",
                "Check worker heartbeat payloads and policy hot-swap bookkeeping.",
            )
        )
    if metrics["worker_without_checkpoint_version_count"] > 0.0:
        findings.append(
            _finding(
                "warning",
                "missing_checkpoint_versions",
                "Some active workers have not reported a checkpoint version.",
                "Check checkpoint polling, registry probing, and heartbeat payloads.",
            )
        )
    if workers and all(worker["rollout_duration_s"] is None for worker in workers):
        findings.append(
            _finding(
                "info",
                "missing_rollout_timing",
                "Workers did not report rollout_duration_s.",
                "Enable worker heartbeat timing before comparing Python inference or env latency.",
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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value)


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)
