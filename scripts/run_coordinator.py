#!/usr/bin/env python3
"""Run a Coordinator bootstrap/status snapshot (PRD Phase 8).

Usage:
    python scripts/run_coordinator.py --config configs/train/remote_learner.yaml \
        --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
        --dry-run

The current coordinator entry point is intentionally one-shot: it validates
config/task wiring, creates task assignments for the expected worker set, ingests
optional heartbeat JSONL, and prints a monitoring snapshot. A network service can
reuse the same Coordinator methods behind an authenticated endpoint later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hkrl.coordinator.coordinator import Coordinator, WorkerRecord
from hkrl.coordinator.task_sampler import TaskSampler
from hkrl.utils.config import TaskConfig, load_task_config, load_train_config


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Coordinator bootstrap/status")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", nargs="+", required=True, help="task YAML files available to sample")
    p.add_argument("--bind", default=None, help="override config.coordinator.bind")
    p.add_argument("--num-workers", type=int, default=None, help="override expected worker count")
    p.add_argument("--worker-id", action="append", dest="worker_ids", help="explicit worker id")
    p.add_argument("--heartbeat-timeout-s", type=float, default=None)
    p.add_argument("--heartbeat-jsonl", help="optional worker heartbeat JSONL to ingest")
    p.add_argument("--seed", type=int, default=None, help="override config seed for task sampling")
    p.add_argument("--dry-run", action="store_true", help="validate wiring and print summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    tasks = _load_tasks(args.tasks)
    task_ids = [task.task_id for task in tasks]
    coordinator = Coordinator(
        TaskSampler(task_ids, seed=cfg.seed if args.seed is None else args.seed),
        bind=args.bind or cfg.coordinator.bind,
        heartbeat_timeout_s=(
            cfg.coordinator.heartbeat_timeout_s
            if args.heartbeat_timeout_s is None
            else args.heartbeat_timeout_s
        ),
    )
    worker_ids = _worker_ids(args, default_count=cfg.coordinator.num_workers)
    assignments = _register_and_assign(coordinator, worker_ids)
    ingested_heartbeats = _ingest_heartbeat_jsonl(coordinator, args.heartbeat_jsonl)
    metrics = coordinator.metrics_snapshot()
    workers = _worker_records(coordinator)

    return {
        "assignments": assignments,
        "bind": coordinator.bind,
        "dry_run": bool(args.dry_run),
        "heartbeat_jsonl": args.heartbeat_jsonl,
        "heartbeat_timeout_s": coordinator.heartbeat_timeout_s,
        "ingested_heartbeats": ingested_heartbeats,
        "metrics": metrics,
        "num_workers": len(worker_ids),
        "task_ids": task_ids,
        "task_wire_ids": {task.task_id: task.wire_id for task in tasks},
        "workers": workers,
    }


def _load_tasks(paths: list[str]) -> list[TaskConfig]:
    tasks = [load_task_config(path) for path in paths]
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def _worker_ids(args: argparse.Namespace, *, default_count: int) -> list[str]:
    if args.worker_ids:
        worker_ids = [str(worker_id) for worker_id in args.worker_ids]
        if any(not worker_id for worker_id in worker_ids):
            raise ValueError("worker ids must not be empty")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker ids must be unique")
        return worker_ids

    count = default_count if args.num_workers is None else args.num_workers
    if count <= 0:
        raise ValueError("num_workers must be positive")
    return [f"worker-{idx}" for idx in range(count)]


def _register_and_assign(coordinator: Coordinator, worker_ids: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for worker_id in worker_ids:
        coordinator.register_worker(worker_id, {"source": "configured"})
        assignments[worker_id] = coordinator.assign_task(worker_id)
    return assignments


def _ingest_heartbeat_jsonl(coordinator: Coordinator, path: str | None) -> int:
    if path is None:
        return 0

    heartbeat_path = Path(path)
    if not heartbeat_path.exists():
        raise FileNotFoundError(f"heartbeat JSONL does not exist: {heartbeat_path}")

    count = 0
    with heartbeat_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"heartbeat JSONL line {line_no} must be an object")
            worker_id = str(record.get("worker_id", ""))
            if not worker_id:
                raise ValueError(f"heartbeat JSONL line {line_no} missing worker_id")
            payload = record.get("payload")
            if payload is None:
                payload = {key: value for key, value in record.items() if key != "worker_id"}
            if not isinstance(payload, dict):
                raise ValueError(f"heartbeat JSONL line {line_no} payload must be an object")
            _register_if_missing(coordinator, worker_id)
            coordinator.ingest_heartbeat_payload(worker_id, payload)
            count += 1
    return count


def _register_if_missing(coordinator: Coordinator, worker_id: str) -> None:
    try:
        coordinator.worker_record(worker_id)
    except KeyError:
        coordinator.register_worker(worker_id, {"source": "heartbeat"})


def _worker_records(coordinator: Coordinator) -> dict[str, dict[str, Any]]:
    worker_ids = sorted(set(coordinator.active_workers()) | set(coordinator.lost_workers()))
    return {
        worker_id: _serialize_worker_record(coordinator.worker_record(worker_id))
        for worker_id in worker_ids
    }


def _serialize_worker_record(record: WorkerRecord) -> dict[str, Any]:
    return {
        "alive": record.alive,
        "assigned_task": record.assigned_task,
        "info": record.info,
        "last_heartbeat": record.last_heartbeat,
        "lost_at": record.lost_at,
        "metrics": record.metrics,
    }


if __name__ == "__main__":
    raise SystemExit(main())
