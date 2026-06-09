#!/usr/bin/env python3
"""Run an offline Phase 8 distributed smoke check.

The smoke check does not connect to Hollow Knight or open learner/worker network
ports. It validates the config-driven distributed wiring by building the learner,
publishing a local checkpoint registry, probing that registry from a worker
dry-run, and ingesting synthetic worker heartbeats/evaluator metrics through the
coordinator monitoring path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.utils.config import load_task_config


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Phase 8 offline smoke")
    p.add_argument(
        "--config",
        default="configs/train/remote_learner.yaml",
        help="distributed train YAML",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "configs/tasks/gruz_mother.yaml",
            "configs/tasks/hornet_protector.yaml",
        ],
        help="task YAMLs used for learner/worker/coordinator wiring",
    )
    p.add_argument("--work-dir", help="directory for generated smoke artifacts")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="optional path to write the summary JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    if args.output:
        _write_json(Path(args.output), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not args.tasks:
        raise ValueError("at least one task is required")

    work_dir = _work_dir(getattr(args, "work_dir", None))
    _reset_generated_artifacts(work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    heartbeat_jsonl = work_dir / "worker-heartbeats.jsonl"
    eval_metrics_json = work_dir / "eval-metrics.json"
    task_paths = [str(path) for path in args.tasks]
    tasks = [load_task_config(path) for path in task_paths]
    worker_ids = [f"worker-{idx}" for idx in range(args.num_workers)]

    run_learner = _load_script_module("run_learner.py")
    run_worker = _load_script_module("run_worker.py")
    run_coordinator = _load_script_module("run_coordinator.py")

    learner_summary = run_learner.run_from_args(
        argparse.Namespace(
            config=args.config,
            bind="127.0.0.1:0",
            batch_dir=None,
            checkpoint_dir=str(checkpoint_dir),
            disable_macro_actions=False,
            intake_count=0,
            intake_timeout_s=1.0,
            max_entities=None,
            max_staleness=None,
            n_macro_actions=None,
            publish_every_updates=None,
            serve_forever=False,
            task=None,
            tasks=task_paths,
            tier=None,
        )
    )
    checkpoint_versions = _publish_smoke_checkpoints(checkpoint_dir)
    _write_heartbeat_jsonl(
        heartbeat_jsonl,
        worker_ids=worker_ids,
        latest_checkpoint=checkpoint_versions[-1],
    )
    _write_eval_metrics(eval_metrics_json, [task.task_id for task in tasks])

    worker_summary = run_worker.run_from_args(
        argparse.Namespace(
            config=args.config,
            task=task_paths[0],
            tasks=task_paths,
            learner=None,
            registry=str(checkpoint_dir),
            batch_dir=str(work_dir / "batches"),
            heartbeat_jsonl=str(heartbeat_jsonl),
            worker_id=worker_ids[0],
            steps=None,
            max_consecutive_failures=3,
            dry_run=True,
        )
    )
    coordinator_summary = run_coordinator.run_from_args(
        argparse.Namespace(
            config=args.config,
            tasks=task_paths,
            bind="127.0.0.1:0",
            num_workers=None,
            worker_ids=worker_ids,
            heartbeat_timeout_s=None,
            heartbeat_jsonl=str(heartbeat_jsonl),
            eval_metrics=str(eval_metrics_json),
            seed=args.seed,
            dry_run=True,
        )
    )

    return {
        "artifacts": {
            "checkpoint_dir": str(checkpoint_dir),
            "eval_metrics": str(eval_metrics_json),
            "heartbeat_jsonl": str(heartbeat_jsonl),
            "work_dir": str(work_dir),
        },
        "checkpoint_versions": checkpoint_versions,
        "config": args.config,
        "coordinator": coordinator_summary,
        "learner": learner_summary,
        "ok": True,
        "task_ids": [task.task_id for task in tasks],
        "worker": worker_summary,
        "worker_ids": worker_ids,
    }


def _publish_smoke_checkpoints(checkpoint_dir: Path) -> list[int]:
    registry = CheckpointRegistry(str(checkpoint_dir))
    versions: list[int] = []
    for policy_version in (1, 2):
        meta = registry.publish(
            {
                "model_state_dict": {
                    "smoke_weight": torch.tensor([float(policy_version)]),
                },
                "policy_version": policy_version,
            },
            policy_version=policy_version,
            step=policy_version,
        )
        versions.append(meta.version)
    return versions


def _write_heartbeat_jsonl(
    path: Path,
    *,
    worker_ids: list[str],
    latest_checkpoint: int,
) -> None:
    records: list[dict[str, Any]] = []
    for index, worker_id in enumerate(worker_ids):
        current = index == 0
        records.append(
            {
                "payload": {
                    "checkpoint_version": latest_checkpoint if current else latest_checkpoint - 1,
                    "policy_version": 2 if current else 1,
                    "rollout_duration_s": 4.0 if current else 0.0,
                    "rollout_steps": 128 if current else 0,
                    "sps": 32.0 if current else 0.0,
                    "status": "running" if current else "recovering",
                    "worker_crash_count": 0 if current else 1,
                },
                "worker_id": worker_id,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_eval_metrics(path: Path, task_ids: list[str]) -> None:
    metrics = {
        task_id: {"win_rate": 0.9 if index == 0 else 0.2}
        for index, task_id in enumerate(task_ids)
    }
    _write_json(path, {"metrics": metrics})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _work_dir(path: str | None) -> Path:
    if path is None:
        return Path(tempfile.mkdtemp(prefix="hkrl_phase8_smoke_")).resolve()
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _reset_generated_artifacts(work_dir: Path) -> None:
    for dirname in ("batches", "checkpoints"):
        path = work_dir / dirname
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for filename in ("eval-metrics.json", "worker-heartbeats.jsonl"):
        path = work_dir / filename
        if path.exists():
            path.unlink()


def _load_script_module(name: str) -> ModuleType:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load script module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
