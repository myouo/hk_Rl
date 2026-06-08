#!/usr/bin/env python3
"""Run a GameWorker (Game PC): local inference + rollout upload (PRD Phase 6).

Usage:
    python scripts/run_worker.py --config configs/train/remote_learner.yaml \
        --task configs/tasks/hornet_protector.yaml --learner host:5600

Use ``--dry-run`` to validate config/model/checkpoint wiring without connecting
to a live Hollow Knight mod instance.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Import model modules for registry side effects.
from hkrl.models import mlp as _mlp  # noqa: F401
from hkrl.models import recurrent_policy as _recurrent_policy  # noqa: F401
from hkrl.spaces import make_observation_space
from hkrl.training.batch_io import save_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch
from hkrl.utils.config import TaskConfig, load_task_config, load_train_config, resolve_auth_token
from hkrl.utils.registry import get
from hkrl.worker.checkpoint_client import CheckpointClient
from hkrl.worker.game_worker import GameWorker


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL GameWorker")
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--tasks", nargs="+", help="optional task list cycled per rollout")
    p.add_argument("--learner", help="learner endpoint host:port")
    p.add_argument("--registry", help="checkpoint registry endpoint")
    p.add_argument("--batch-dir", help="directory for NPZ rollout batch spooling")
    p.add_argument("--worker-id", default="worker-0", help="stable worker id for batch names")
    p.add_argument("--steps", type=int, default=None, help="optional finite rollout sample count")
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="validate wiring without env connection")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    tasks = _load_tasks(args)
    task = tasks[0]
    observation_space = make_observation_space(
        max_entities=task.observation.max_entities,
        tier=task.observation.tier,
    )
    obs_dims = {
        "global": observation_space["global"].shape,
        "player": observation_space["player"].shape,
        "entities": observation_space["entities"].shape,
        "entity_mask": observation_space["entity_mask"].shape,
    }
    model = _build_model(
        cfg,
        obs_dims,
        enable_macro=task.action.enable_macro_actions,
        max_entities=task.observation.max_entities,
    )
    checkpoint_client = CheckpointClient(args.registry) if args.registry else None

    if args.dry_run:
        auth_token_env = cfg.security.auth_token_env
        latest_checkpoint = (
            None if checkpoint_client is None else checkpoint_client.latest_version()
        )
        return {
            "algorithm": cfg.algorithm,
            "auth_token_configured": bool(os.environ.get(auth_token_env)),
            "auth_token_env": auth_token_env,
            "auth_token_required": cfg.security.require_token,
            "dry_run": True,
            "batch_dir": args.batch_dir,
            "learner": args.learner,
            "latest_checkpoint": latest_checkpoint,
            "max_consecutive_failures": args.max_consecutive_failures,
            "model": cfg.model.name,
            "registry": args.registry,
            "task_id": task.task_id,
            "task_ids": [item.task_id for item in tasks],
            "worker_id": args.worker_id,
        }

    if cfg.transport.name != "tcp":
        raise ValueError(f"worker currently supports tcp transport, got {cfg.transport.name!r}")

    from hkrl.env import HKRLEnv
    from hkrl.transport.tcp import TcpTransport
    from hkrl.wrappers import NormalizeObservation

    transport = TcpTransport(
        host=cfg.transport.host,
        port=cfg.transport.port,
        auth_token=resolve_auth_token(cfg),
    )
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    spooled_batches: list[str] = []
    worker = GameWorker(
        env=env,
        model=model,
        config=cfg,
        checkpoint_client=checkpoint_client,
        learner_endpoint=args.learner,
        batch_uploader=_make_batch_uploader(args.batch_dir, args.worker_id, spooled_batches),
        task_provider=_make_task_provider(tasks),
        max_consecutive_failures=args.max_consecutive_failures,
    )
    try:
        worker.run(total_steps=args.steps)
        last_batch = worker.last_batch
        return {
            "algorithm": cfg.algorithm,
            "batch_dir": args.batch_dir,
            "consecutive_failures": worker.consecutive_failures,
            "dry_run": False,
            "last_error": worker.last_error,
            "learner": args.learner,
            "model": cfg.model.name,
            "policy_version": worker.policy_version,
            "rollout_steps": 0 if last_batch is None else int(last_batch.rewards.size),
            "spooled_batches": spooled_batches,
            "task_id": task.task_id,
            "task_ids": [item.task_id for item in tasks],
            "worker_crash_count": worker.worker_crash_count,
            "worker_id": args.worker_id,
        }
    finally:
        env.close()


def _build_model(
    cfg: Any,
    obs_dims: dict[str, tuple[int, ...]],
    *,
    enable_macro: bool,
    max_entities: int,
) -> Any:
    model_cls = get("model", cfg.model.name)
    if cfg.model.name == "mlp":
        return model_cls(
            obs_dims,
            hidden=cfg.model.rnn_hidden,
            enable_macro=enable_macro,
        )
    return model_cls(
        obs_dims,
        entity_hidden=cfg.model.entity_hidden,
        attention_layers=cfg.model.attention_layers,
        attention_heads=cfg.model.attention_heads,
        rnn_type=cfg.model.rnn_type,
        rnn_hidden=cfg.model.rnn_hidden,
        enable_macro=enable_macro,
        max_entities=max_entities,
    )


def _load_tasks(args: argparse.Namespace) -> list[TaskConfig]:
    paths = args.tasks if args.tasks else [args.task]
    tasks = [load_task_config(path) for path in paths]
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def _make_task_provider(tasks: list[TaskConfig]) -> Callable[[], TaskConfig | None] | None:
    if len(tasks) <= 1:
        return None

    index = -1

    def provide() -> TaskConfig:
        nonlocal index
        index = (index + 1) % len(tasks)
        return tasks[index]

    return provide


def _make_batch_uploader(
    batch_dir: str | None,
    worker_id: str,
    written: list[str],
) -> Callable[[RolloutBatch], None] | None:
    if batch_dir is None:
        return None

    directory = Path(batch_dir)
    safe_worker = _safe_filename_component(worker_id)
    sequence = 0

    def upload(batch: RolloutBatch) -> None:
        nonlocal sequence
        sequence += 1
        path = directory / f"{safe_worker}_{sequence:08d}_v{batch.policy_version:06d}.npz"
        save_rollout_batch(path, batch)
        written.append(str(path))

    return upload


def _safe_filename_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    safe = safe.strip("._")
    return safe or "worker"


if __name__ == "__main__":
    raise SystemExit(main())
