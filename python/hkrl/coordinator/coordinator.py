"""Coordinator: manage workers, assign tasks, aggregate metrics (PRD §4.2).

Owns the worker registry, distributes tasks (via the task sampler / curriculum),
keeps training and evaluation isolated, and recovers from worker crashes (a single
worker crash must not stall training — PRD Phase 8 milestone).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real

from hkrl.coordinator.task_sampler import TaskSampler


@dataclass
class WorkerRecord:
    worker_id: str
    info: dict[str, object]
    last_heartbeat: float
    alive: bool = True
    assigned_task: str | None = None
    lost_at: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class Coordinator:
    """Top-level orchestrator for distributed sampling/training."""

    def __init__(
        self,
        task_sampler: TaskSampler,
        bind: str = "127.0.0.1:5610",
        heartbeat_timeout_s: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if isinstance(heartbeat_timeout_s, bool) or not isinstance(heartbeat_timeout_s, Real):
            raise ValueError("heartbeat_timeout_s must be numeric")
        heartbeat_timeout = float(heartbeat_timeout_s)
        if not math.isfinite(heartbeat_timeout):
            raise ValueError("heartbeat_timeout_s must be finite")
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout_s must be positive")

        self.task_sampler: TaskSampler = task_sampler
        self.bind: str = bind
        self.heartbeat_timeout_s: float = heartbeat_timeout
        self._clock: Callable[[], float] = clock or time.monotonic
        self._workers: dict[str, WorkerRecord] = {}

    def register_worker(self, worker_id: str, info: dict[str, object]) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")

        now = self._clock()
        existing = self._workers.get(worker_id)
        if existing is None:
            self._workers[worker_id] = WorkerRecord(
                worker_id=worker_id,
                info=dict(info),
                last_heartbeat=now,
            )
            return

        existing.info = dict(info)
        existing.last_heartbeat = now
        existing.alive = True
        existing.lost_at = None

    def assign_task(self, worker_id: str) -> str:
        """Return a task_id for the worker (curriculum/balanced sampling)."""
        worker = self._require_worker(worker_id)
        if not worker.alive:
            raise RuntimeError(f"worker {worker_id!r} is not alive")
        task_id = self.task_sampler.sample()
        worker.assigned_task = task_id
        return task_id

    def on_worker_lost(self, worker_id: str) -> None:
        worker = self._require_worker(worker_id)
        if not worker.alive:
            return
        worker.alive = False
        worker.lost_at = self._clock()

    def heartbeat(
        self,
        worker_id: str,
        *,
        info: dict[str, object] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        worker = self._require_worker(worker_id)
        if info is not None:
            worker.info.update(info)
        if metrics is not None:
            _validate_metrics(metrics)
            worker.metrics.update(metrics)
        worker.last_heartbeat = self._clock()
        worker.alive = True
        worker.lost_at = None

    def ingest_heartbeat_payload(self, worker_id: str, payload: dict[str, object]) -> None:
        """Update worker state from a raw GameWorker heartbeat payload.

        GameWorker heartbeats carry both numeric metrics (SPS, policy/checkpoint
        versions, crash count) and non-numeric status fields. The coordinator
        stores numeric fields in ``WorkerRecord.metrics`` so monitoring can
        aggregate them, while preserving status/error fields in ``info``.
        """
        metrics: dict[str, float] = {}
        info: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, bool):
                info[key] = value
            elif isinstance(value, Real):
                if not math.isfinite(float(value)):
                    raise ValueError(f"heartbeat metric {key!r} must be finite")
                metrics[key] = float(value)
            else:
                info[key] = value
        self.heartbeat(worker_id, info=info, metrics=metrics)

    def expire_workers(self) -> list[str]:
        now = self._clock()
        expired: list[str] = []
        for worker_id, worker in self._workers.items():
            if worker.alive and now - worker.last_heartbeat > self.heartbeat_timeout_s:
                worker.alive = False
                worker.lost_at = now
                expired.append(worker_id)
        return expired

    def active_workers(self) -> list[str]:
        return sorted(worker_id for worker_id, worker in self._workers.items() if worker.alive)

    def lost_workers(self) -> list[str]:
        return sorted(worker_id for worker_id, worker in self._workers.items() if not worker.alive)

    def worker_record(self, worker_id: str) -> WorkerRecord:
        return self._require_worker(worker_id)

    def metrics_snapshot(self) -> dict[str, float]:
        """Return aggregate worker metrics for dashboards/logging.

        ``sps`` is summed across active workers because SPS is a fleet throughput
        metric; ``sps_mean`` is included for per-worker health inspection.
        Lost workers still contribute to ``worker_crash_count`` so restarts are
        not hidden once a worker is expired. Version lag metrics are calculated
        across active workers that reported a numeric version; missing-version
        counts stay explicit so dashboards can distinguish unknown from up to
        date workers.
        """
        self.expire_workers()
        records = list(self._workers.values())
        active = [worker for worker in records if worker.alive]
        lost = [worker for worker in records if not worker.alive]
        active_sps = [worker.metrics.get("sps", 0.0) for worker in active]
        sps_total = sum(active_sps)
        policy_versions = _metric_values(active, "policy_version")
        checkpoint_versions = _metric_values(active, "checkpoint_version")
        policy_version_max = max(policy_versions, default=0.0)
        policy_version_min = min(policy_versions, default=0.0)
        checkpoint_version_max = max(checkpoint_versions, default=0.0)
        checkpoint_version_min = min(checkpoint_versions, default=0.0)

        return {
            "worker_count": float(len(records)),
            "active_worker_count": float(len(active)),
            "lost_worker_count": float(len(lost)),
            "assigned_worker_count": float(
                sum(1 for worker in active if worker.assigned_task is not None)
            ),
            "sps": sps_total,
            "sps_mean": sps_total / len(active_sps) if active_sps else 0.0,
            "worker_crash_count": sum(
                worker.metrics.get("worker_crash_count", 0.0) for worker in records
            ),
            "recovering_worker_count": float(
                sum(1 for worker in active if worker.info.get("status") == "recovering")
            ),
            "worker_policy_version_min": policy_version_min,
            "worker_policy_version_max": policy_version_max,
            "worker_policy_lag_max": policy_version_max - policy_version_min,
            "stale_policy_worker_count": float(
                sum(1 for version in policy_versions if version < policy_version_max)
            ),
            "worker_without_policy_version_count": float(
                sum(1 for worker in active if "policy_version" not in worker.metrics)
            ),
            "worker_checkpoint_version_min": checkpoint_version_min,
            "worker_checkpoint_version_max": checkpoint_version_max,
            "worker_checkpoint_lag_max": checkpoint_version_max - checkpoint_version_min,
            "stale_checkpoint_worker_count": float(
                sum(1 for version in checkpoint_versions if version < checkpoint_version_max)
            ),
            "worker_without_checkpoint_version_count": float(
                sum(1 for worker in active if "checkpoint_version" not in worker.metrics)
            ),
            "worker_learner_upload_accepted_batches": _sum_metric(
                records, "learner_upload_accepted_batches"
            ),
            "worker_learner_upload_failed_batches": _sum_metric(
                records, "learner_upload_failed_batches"
            ),
            "worker_learner_upload_rejected_batches": _sum_metric(
                records, "learner_upload_rejected_batches"
            ),
            "worker_learner_upload_submitted_batches": _sum_metric(
                records, "learner_upload_submitted_batches"
            ),
        }

    def _require_worker(self, worker_id: str) -> WorkerRecord:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise KeyError(f"unknown worker {worker_id!r}") from exc


def _metric_values(workers: list[WorkerRecord], key: str) -> list[float]:
    return [float(worker.metrics[key]) for worker in workers if key in worker.metrics]


def _sum_metric(workers: list[WorkerRecord], key: str) -> float:
    return sum(float(worker.metrics.get(key, 0.0)) for worker in workers)


def _validate_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"heartbeat metric {key!r} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"heartbeat metric {key!r} must be finite")
