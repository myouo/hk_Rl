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
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from hkrl.coordinator.coordinator import Coordinator, WorkerRecord
from hkrl.coordinator.task_sampler import TaskSampler
from hkrl.utils.config import (
    TaskConfig,
    load_task_config,
    load_train_config,
    validate_bind_address,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Coordinator bootstrap/status")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--tasks", nargs="+", required=True, help="task YAML files available to sample"
    )
    p.add_argument("--bind", default=None, help="override config.coordinator.bind")
    p.add_argument(
        "--num-workers", type=int, default=None, help="override expected worker count"
    )
    p.add_argument(
        "--worker-id", action="append", dest="worker_ids", help="explicit worker id"
    )
    p.add_argument("--heartbeat-timeout-s", type=float, default=None)
    p.add_argument(
        "--heartbeat-jsonl", help="optional worker heartbeat JSONL to ingest"
    )
    p.add_argument(
        "--eval-metrics", help="optional evaluator metrics JSON for sampler weights"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override config seed for task sampling"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="validate wiring and print summary"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    _validate_coordinator_args(args)
    cfg = load_train_config(args.config)
    tasks = _load_tasks(args.tasks)
    task_ids = [task.task_id for task in tasks]
    bind = validate_bind_address(
        args.bind or cfg.coordinator.bind, cfg.security.bind_scope
    )
    worker_ids = _worker_ids(args, default_count=cfg.coordinator.num_workers)
    heartbeat_timeout_s = _heartbeat_timeout_s(
        args,
        default_value=cfg.coordinator.heartbeat_timeout_s,
    )
    seed = cfg.seed if args.seed is None else _integer(args.seed, name="seed")
    sampler = TaskSampler(task_ids, seed=seed)
    eval_winrates = _load_eval_winrates(getattr(args, "eval_metrics", None))
    if eval_winrates:
        sampler.update_weights(eval_winrates)
    coordinator = Coordinator(
        sampler,
        bind=bind,
        heartbeat_timeout_s=heartbeat_timeout_s,
    )
    assignments = _register_and_assign(coordinator, worker_ids)
    ingested_heartbeats = _ingest_heartbeat_jsonl(coordinator, args.heartbeat_jsonl)
    metrics = coordinator.metrics_snapshot()
    workers = _worker_records(coordinator)

    return {
        "assignments": assignments,
        "bind": coordinator.bind,
        "dry_run": bool(args.dry_run),
        "eval_metrics": getattr(args, "eval_metrics", None),
        "eval_winrates": eval_winrates,
        "heartbeat_jsonl": args.heartbeat_jsonl,
        "heartbeat_timeout_s": coordinator.heartbeat_timeout_s,
        "ingested_heartbeats": ingested_heartbeats,
        "metrics": metrics,
        "num_workers": len(worker_ids),
        "sampler_mastered_tasks": sorted(sampler.mastered_tasks),
        "sampler_weights": {
            task_id: float(sampler.weights[task_id]) for task_id in task_ids
        },
        "task_ids": task_ids,
        "task_wire_ids": {task.task_id: task.wire_id for task in tasks},
        "workers": workers,
    }


def _load_tasks(paths: list[str]) -> list[TaskConfig]:
    tasks = [
        load_task_config(_non_empty_path_like(path, name=f"tasks[{index}]"))
        for index, path in enumerate(paths)
    ]
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def _load_eval_winrates(path: str | None) -> dict[str, float]:
    if path is None:
        return {}

    metrics_path = Path(_non_empty_path_like(path, name="eval_metrics"))
    if not metrics_path.exists():
        raise FileNotFoundError(f"eval metrics JSON does not exist: {metrics_path}")

    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("eval metrics JSON must be an object")

    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, Mapping):
        raise ValueError("eval metrics must be a task metrics object")

    winrates: dict[str, float] = {}
    for task_id, values in metrics.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"eval metrics for task {task_id!r} must be an object")
        winrate = values.get("win_rate")
        if winrate is None:
            winrate = values.get("per_boss_win_rate")
        if winrate is None:
            raise ValueError(
                f"eval metrics for task {task_id!r} must include win_rate or per_boss_win_rate"
            )
        winrates[str(task_id)] = _validate_winrate(str(task_id), winrate)
    return winrates


def _validate_winrate(task_id: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"eval win_rate for task {task_id!r} must be numeric")
    winrate = float(value)
    if not math.isfinite(winrate):
        raise ValueError(f"eval win_rate for task {task_id!r} must be finite")
    if not 0.0 <= winrate <= 1.0:
        raise ValueError(f"eval win_rate for task {task_id!r} must be in [0, 1]")
    return winrate


def _validate_coordinator_args(args: argparse.Namespace) -> None:
    _non_empty_path_like(getattr(args, "config", None), name="config")
    tasks = getattr(args, "tasks", None)
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        raise ValueError("at least one task is required")
    for index, task in enumerate(tasks):
        _non_empty_path_like(task, name=f"tasks[{index}]")

    _optional_non_empty_string(getattr(args, "bind", None), name="bind")
    _optional_non_empty_path_like(
        getattr(args, "heartbeat_jsonl", None), name="heartbeat_jsonl"
    )
    _optional_non_empty_path_like(
        getattr(args, "eval_metrics", None), name="eval_metrics"
    )

    if getattr(args, "num_workers", None) is not None:
        _positive_int(args.num_workers, name="num_workers")
    if getattr(args, "heartbeat_timeout_s", None) is not None:
        _positive_finite_float(args.heartbeat_timeout_s, name="heartbeat_timeout_s")
    if getattr(args, "seed", None) is not None:
        _integer(args.seed, name="seed")

    if (
        getattr(args, "worker_ids", None) is not None
        and getattr(args, "num_workers", None) is not None
    ):
        raise ValueError("num_workers cannot be combined with explicit worker ids")


def _worker_ids(args: argparse.Namespace, *, default_count: int) -> list[str]:
    if args.worker_ids is not None:
        if isinstance(args.worker_ids, (str, bytes)) or not isinstance(
            args.worker_ids, Sequence
        ):
            raise ValueError("worker_ids must be a sequence of worker ids")
        worker_ids: list[str] = []
        for worker_id in args.worker_ids:
            if not isinstance(worker_id, str) or not worker_id.strip():
                raise ValueError("worker ids must not be empty")
            if worker_id != worker_id.strip():
                raise ValueError("worker ids must not have surrounding whitespace")
            worker_ids.append(worker_id)
        if not worker_ids:
            raise ValueError("worker ids must not be empty")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker ids must be unique")
        return worker_ids

    count = default_count if args.num_workers is None else args.num_workers
    count = _positive_int(count, name="num_workers")
    return [f"worker-{idx}" for idx in range(count)]


def _heartbeat_timeout_s(args: argparse.Namespace, *, default_value: float) -> float:
    value = (
        default_value if args.heartbeat_timeout_s is None else args.heartbeat_timeout_s
    )
    return _positive_finite_float(value, name="heartbeat_timeout_s")


def _positive_int(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_empty_path_like(value: Any, *, name: str) -> str | os.PathLike[str]:
    if not isinstance(value, str | os.PathLike) or not str(value).strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_non_empty_path_like(
    value: Any,
    *,
    name: str,
) -> str | os.PathLike[str] | None:
    if value is None:
        return None
    return _non_empty_path_like(value, name=name)


def _optional_non_empty_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _register_and_assign(
    coordinator: Coordinator, worker_ids: list[str]
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for worker_id in worker_ids:
        coordinator.register_worker(worker_id, {"source": "configured"})
        assignments[worker_id] = coordinator.assign_task(worker_id)
    return assignments


def _ingest_heartbeat_jsonl(coordinator: Coordinator, path: str | None) -> int:
    if path is None:
        return 0

    heartbeat_path = Path(_non_empty_path_like(path, name="heartbeat_jsonl"))
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
                payload = {
                    key: value for key, value in record.items() if key != "worker_id"
                }
            if not isinstance(payload, dict):
                raise ValueError(
                    f"heartbeat JSONL line {line_no} payload must be an object"
                )
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
    worker_ids = sorted(
        set(coordinator.active_workers()) | set(coordinator.lost_workers())
    )
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
