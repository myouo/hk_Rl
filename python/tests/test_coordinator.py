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
