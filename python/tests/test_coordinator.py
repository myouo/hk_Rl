"""Coordinator tests."""

from __future__ import annotations

import pytest
from hkrl.coordinator.coordinator import Coordinator
from hkrl.coordinator.task_sampler import TaskSampler


def test_coordinator_registers_worker_and_assigns_task() -> None:
    coordinator = Coordinator(TaskSampler(["gruz"], seed=0), clock=FakeClock())

    coordinator.register_worker("worker-1", {"host": "game-pc"})
    task_id = coordinator.assign_task("worker-1")

    assert task_id == "gruz"
    assert coordinator.active_workers() == ["worker-1"]
    record = coordinator.worker_record("worker-1")
    assert record.info == {"host": "game-pc"}
    assert record.assigned_task == "gruz"


def test_coordinator_marks_lost_and_heartbeat_recovers_worker() -> None:
    clock = FakeClock()
    coordinator = Coordinator(TaskSampler(["gruz"], seed=0), clock=clock)
    coordinator.register_worker("worker-1", {})

    coordinator.on_worker_lost("worker-1")
    assert coordinator.active_workers() == []
    assert coordinator.lost_workers() == ["worker-1"]

    clock.advance(1.0)
    coordinator.heartbeat("worker-1", info={"host": "game-pc"}, metrics={"sps": 12.5})

    assert coordinator.active_workers() == ["worker-1"]
    record = coordinator.worker_record("worker-1")
    assert record.info == {"host": "game-pc"}
    assert record.metrics == {"sps": 12.5}
    assert record.lost_at is None


def test_coordinator_expires_workers_by_heartbeat_timeout() -> None:
    clock = FakeClock()
    coordinator = Coordinator(TaskSampler(["gruz"], seed=0), heartbeat_timeout_s=5.0, clock=clock)
    coordinator.register_worker("a", {})
    coordinator.register_worker("b", {})

    clock.advance(6.0)
    coordinator.heartbeat("b")
    expired = coordinator.expire_workers()

    assert expired == ["a"]
    assert coordinator.active_workers() == ["b"]
    assert coordinator.lost_workers() == ["a"]


def test_coordinator_ingests_heartbeat_payload_and_aggregates_metrics() -> None:
    clock = FakeClock()
    coordinator = Coordinator(TaskSampler(["gruz"], seed=0), heartbeat_timeout_s=5.0, clock=clock)
    coordinator.register_worker("a", {"host": "game-pc-a"})
    coordinator.register_worker("b", {"host": "game-pc-b"})
    coordinator.assign_task("a")

    coordinator.ingest_heartbeat_payload(
        "a",
        {
            "checkpoint_version": 2,
            "learner_upload_accepted_batches": 1,
            "learner_upload_failed_batches": 0,
            "learner_upload_rejected_batches": 1,
            "learner_upload_submitted_batches": 2,
            "learner_endpoint": "learner:5600",
            "policy_version": 7,
            "rollout_steps": 128,
            "sps": 12.5,
            "status": "running",
            "worker_crash_count": 1,
        },
    )
    coordinator.ingest_heartbeat_payload(
        "b",
        {
            "checkpoint_version": 2,
            "learner_upload_accepted_batches": 0,
            "learner_upload_failed_batches": 1,
            "learner_upload_rejected_batches": 0,
            "learner_upload_submitted_batches": 1,
            "policy_version": 7,
            "rollout_steps": 128,
            "sps": 7.5,
            "status": "running",
            "worker_crash_count": 0,
        },
    )

    record = coordinator.worker_record("a")
    assert record.info["status"] == "running"
    assert record.info["learner_endpoint"] == "learner:5600"
    assert record.metrics["policy_version"] == 7.0
    assert record.metrics["sps"] == 12.5
    assert record.metrics["worker_crash_count"] == 1.0
    assert record.metrics["learner_upload_submitted_batches"] == 2.0

    clock.advance(6.0)
    coordinator.ingest_heartbeat_payload(
        "a",
        {
            "policy_version": 7,
            "sps": 10.0,
            "status": "running",
            "worker_crash_count": 1,
        },
    )

    snapshot = coordinator.metrics_snapshot()

    assert snapshot == {
        "worker_count": 2.0,
        "active_worker_count": 1.0,
        "lost_worker_count": 1.0,
        "assigned_worker_count": 1.0,
        "sps": 10.0,
        "sps_mean": 10.0,
        "worker_crash_count": 1.0,
        "recovering_worker_count": 0.0,
        "worker_policy_version_min": 7.0,
        "worker_policy_version_max": 7.0,
        "worker_policy_lag_max": 0.0,
        "stale_policy_worker_count": 0.0,
        "worker_without_policy_version_count": 0.0,
        "worker_checkpoint_version_min": 2.0,
        "worker_checkpoint_version_max": 2.0,
        "worker_checkpoint_lag_max": 0.0,
        "stale_checkpoint_worker_count": 0.0,
        "worker_without_checkpoint_version_count": 0.0,
        "worker_learner_upload_accepted_batches": 1.0,
        "worker_learner_upload_failed_batches": 1.0,
        "worker_learner_upload_rejected_batches": 1.0,
        "worker_learner_upload_submitted_batches": 3.0,
    }
    assert coordinator.lost_workers() == ["b"]


def test_coordinator_metrics_snapshot_reports_version_lag_and_recovery() -> None:
    coordinator = Coordinator(TaskSampler(["gruz"], seed=0), clock=FakeClock())
    coordinator.register_worker("current", {})
    coordinator.register_worker("stale", {})
    coordinator.register_worker("unknown", {})

    coordinator.ingest_heartbeat_payload(
        "current",
        {
            "checkpoint_version": 5,
            "policy_version": 10,
            "sps": 12.0,
            "status": "running",
        },
    )
    coordinator.ingest_heartbeat_payload(
        "stale",
        {
            "checkpoint_version": 4,
            "policy_version": 8,
            "sps": 6.0,
            "status": "recovering",
        },
    )

    snapshot = coordinator.metrics_snapshot()

    assert snapshot["recovering_worker_count"] == 1.0
    assert snapshot["worker_policy_version_min"] == 8.0
    assert snapshot["worker_policy_version_max"] == 10.0
    assert snapshot["worker_policy_lag_max"] == 2.0
    assert snapshot["stale_policy_worker_count"] == 1.0
    assert snapshot["worker_without_policy_version_count"] == 1.0
    assert snapshot["worker_checkpoint_version_min"] == 4.0
    assert snapshot["worker_checkpoint_version_max"] == 5.0
    assert snapshot["worker_checkpoint_lag_max"] == 1.0
    assert snapshot["stale_checkpoint_worker_count"] == 1.0
    assert snapshot["worker_without_checkpoint_version_count"] == 1.0


def test_coordinator_rejects_unknown_or_dead_worker_assignment() -> None:
    coordinator = Coordinator(TaskSampler(["gruz"], seed=0), clock=FakeClock())

    with pytest.raises(KeyError, match="unknown worker"):
        coordinator.assign_task("missing")

    coordinator.register_worker("worker-1", {})
    coordinator.on_worker_lost("worker-1")
    with pytest.raises(RuntimeError, match="not alive"):
        coordinator.assign_task("worker-1")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
