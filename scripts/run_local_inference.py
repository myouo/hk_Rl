#!/usr/bin/env python3
"""Run the latency-sensitive GameWorker locally from one experiment manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from hkrl.utils.config import resolve_auth_token
from hkrl.utils.experiment import ResolvedExperiment, load_experiment


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKRL local inference worker")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/godhome_smart.yaml",
        help="aggregate experiment YAML",
    )
    parser.add_argument("--worker-id", default=None, help="override manifest worker id")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="positive finite transition count; omit to run continuously",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print a secret-free plan without contacting the game",
    )
    return parser


def build_plan(
    experiment: ResolvedExperiment,
    *,
    worker_id: str | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    local = experiment.config.local
    selected_worker_id = local.worker_id if worker_id is None else worker_id.strip()
    if not selected_worker_id:
        raise ValueError("worker_id must not be empty")
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive when provided")

    command = [
        sys.executable,
        str(experiment.project_root / "scripts/run_worker.py"),
        "--config",
        str(experiment.local_train_path),
        "--task",
        str(experiment.task_paths[0]),
        "--tasks",
        *(str(path) for path in experiment.task_paths),
        "--env-host",
        local.env_host,
        "--env-port",
        str(local.env_port),
        "--learner",
        local.learner_endpoint,
        "--registry",
        local.registry_endpoint,
        "--worker-id",
        selected_worker_id,
        "--batch-dir",
        str(experiment.batch_dir),
        "--heartbeat-jsonl",
        str(experiment.heartbeat_jsonl),
        "--inference-threads",
        str(local.inference_threads),
        "--checkpoint-poll-interval-s",
        str(local.checkpoint_poll_interval_s),
        "--time-scale",
        str(local.time_scale),
        "--max-consecutive-failures",
        str(local.max_consecutive_failures),
    ]
    if steps is not None:
        command.extend(("--steps", str(steps)))
    train = experiment.local_train
    return {
        "algorithm": train.algorithm,
        "auth_token_configured": bool(os.environ.get(train.security.auth_token_env)),
        "auth_token_env": train.security.auth_token_env,
        "auth_token_required": train.security.require_token,
        "batch_dir": str(experiment.batch_dir),
        "command": command,
        "env_endpoint": f"{local.env_host}:{local.env_port}",
        "experiment": experiment.config.name,
        "heartbeat_jsonl": str(experiment.heartbeat_jsonl),
        "inference_threads": local.inference_threads,
        "checkpoint_poll_interval_s": local.checkpoint_poll_interval_s,
        "learner_endpoint": local.learner_endpoint,
        "model": train.model.name,
        "registry_endpoint": local.registry_endpoint,
        "steps": steps,
        "task_ids": [task.task_id for task in experiment.tasks],
        "time_scale": local.time_scale,
        "worker_id": selected_worker_id,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    experiment = load_experiment(args.experiment)
    plan = build_plan(
        experiment,
        worker_id=args.worker_id,
        steps=args.steps,
    )
    if args.dry_run:
        print(json.dumps({**plan, "dry_run": True}, sort_keys=True))
        return 0

    resolve_auth_token(experiment.local_train)
    experiment.batch_dir.mkdir(parents=True, exist_ok=True)
    experiment.heartbeat_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({**plan, "dry_run": False}, sort_keys=True), flush=True)
    os.execvpe(
        sys.executable,
        plan["command"],
        {**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
