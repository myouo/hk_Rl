#!/usr/bin/env python3
"""Launch the remote GPU learner and checkpoint registry from one manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

from hkrl.utils.config import (
    resolve_auth_token,
    validate_bind_address,
    validate_service_auth,
)
from hkrl.utils.experiment import ResolvedExperiment, load_experiment


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKRL remote training stack")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/godhome_smart.yaml",
        help="aggregate experiment YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print a secret-free launch plan",
    )
    return parser


def build_plan(experiment: ResolvedExperiment) -> dict[str, Any]:
    root = experiment.project_root
    remote = experiment.config.remote
    train = experiment.remote_train
    learner_bind = validate_bind_address(
        remote.learner_bind,
        train.security.bind_scope,
    )
    registry_bind = validate_bind_address(
        remote.registry_bind,
        train.security.bind_scope,
    )
    validate_service_auth(learner_bind, train)
    validate_service_auth(registry_bind, train)
    checkpoint_command = [
        sys.executable,
        str(root / "scripts/run_checkpoint_server.py"),
        "--config",
        str(experiment.remote_train_path),
        "--bind",
        registry_bind,
    ]
    learner_command = [
        sys.executable,
        str(root / "scripts/run_learner.py"),
        "--config",
        str(experiment.remote_train_path),
        "--tasks",
        *(str(path) for path in experiment.task_paths),
        "--bind",
        learner_bind,
        "--serve-forever",
    ]
    return {
        "algorithm": train.algorithm,
        "amp_dtype": train.learner.amp_dtype,
        "auth_token_configured": bool(os.environ.get(train.security.auth_token_env)),
        "auth_token_env": train.security.auth_token_env,
        "auth_token_required": train.security.require_token,
        "batches_per_update": train.learner.batches_per_update,
        "checkpoint_command": checkpoint_command,
        "compile_mode": train.learner.compile_mode,
        "experiment": experiment.config.name,
        "learner_bind": learner_bind,
        "learner_command": learner_command,
        "model": train.model.name,
        "registry_bind": registry_bind,
        "sequence_length": train.sequence_length,
        "task_ids": [task.task_id for task in experiment.tasks],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    experiment = load_experiment(args.experiment)
    plan = build_plan(experiment)
    if args.dry_run:
        print(json.dumps({**plan, "dry_run": True}, sort_keys=True))
        return 0

    # Resolve before spawning so a missing token cannot leave a half-started stack.
    resolve_auth_token(experiment.remote_train)
    print(json.dumps({**plan, "dry_run": False}, sort_keys=True), flush=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    processes = [
        subprocess.Popen(
            plan["checkpoint_command"],
            cwd=experiment.project_root,
            env=environment,
        ),
        subprocess.Popen(
            plan["learner_command"],
            cwd=experiment.project_root,
            env=environment,
        ),
    ]
    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    return int(return_code)
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 130
    finally:
        _terminate(processes)


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
