"""Coordinator: manage workers, assign tasks, aggregate metrics (PRD §4.2).

Owns the worker registry, distributes tasks (via the task sampler / curriculum),
keeps training and evaluation isolated, and recovers from worker crashes (a single
worker crash must not stall training — PRD Phase 8 milestone).
"""

from __future__ import annotations

from hkrl.coordinator.task_sampler import TaskSampler


class Coordinator:
    """Top-level orchestrator for distributed sampling/training."""

    def __init__(self, task_sampler: TaskSampler, bind: str = "0.0.0.0:5610") -> None:
        self.task_sampler = task_sampler
        self.bind = bind
        self._workers: dict[str, object] = {}
        # TODO(phase-8): registry, heartbeat tracking, crash recovery.

    def register_worker(self, worker_id: str, info: dict[str, object]) -> None:
        raise NotImplementedError  # TODO(phase-8)

    def assign_task(self, worker_id: str) -> str:
        """Return a task_id for the worker (curriculum/balanced sampling)."""
        raise NotImplementedError  # TODO(phase-8)

    def on_worker_lost(self, worker_id: str) -> None:
        raise NotImplementedError  # TODO(phase-8)
