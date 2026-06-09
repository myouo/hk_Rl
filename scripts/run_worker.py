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
from urllib.parse import urlparse

# Import model modules for registry side effects.
from hkrl.models import mlp as _mlp  # noqa: F401
from hkrl.models import recurrent_policy as _recurrent_policy  # noqa: F401
from hkrl.spaces import make_observation_space
from hkrl.learner.batch_intake import BatchIntakeClient
from hkrl.training.batch_io import save_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch
from hkrl.transport.factory import make_transport
from hkrl.utils.config import (
    TaskConfig,
    load_task_config,
    load_train_config,
    resolve_auth_token,
)
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
    p.add_argument("--heartbeat-jsonl", help="append worker heartbeats to JSONL")
    p.add_argument(
        "--worker-id", default="worker-0", help="stable worker id for batch names"
    )
    p.add_argument(
        "--steps", type=int, default=None, help="optional finite rollout sample count"
    )
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument(
        "--dry-run", action="store_true", help="validate wiring without env connection"
    )
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
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    checkpoint_client = (
        CheckpointClient(
            args.registry, auth_token=_checkpoint_auth_token(cfg, args.registry)
        )
        if args.registry
        else None
    )

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
            "enable_macro_actions": task.action.enable_macro_actions,
            "heartbeat_jsonl": args.heartbeat_jsonl,
            "learner": args.learner,
            "learner_upload_enabled": args.learner is not None,
            "latest_checkpoint": latest_checkpoint,
            "max_consecutive_failures": args.max_consecutive_failures,
            "model": cfg.model.name,
            "n_macro_actions": task.action.n_macro_actions,
            "registry": args.registry,
            "task_id": task.task_id,
            "task_ids": [item.task_id for item in tasks],
            "worker_id": args.worker_id,
        }

    from hkrl.env import HKRLEnv
    from hkrl.wrappers import NormalizeObservation

    transport = make_transport(cfg)
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    spooled_batches: list[str] = []
    heartbeats: list[None] = []
    uploaded_batches: list[bool] = []
    worker = GameWorker(
        env=env,
        model=model,
        config=cfg,
        checkpoint_client=checkpoint_client,
        learner_endpoint=args.learner,
        batch_uploader=_make_batch_uploader(
            args.batch_dir,
            args.worker_id,
            spooled_batches,
            learner_endpoint=args.learner,
            auth_token=resolve_auth_token(cfg) if args.learner is not None else None,
            uploaded=uploaded_batches,
        ),
        heartbeat_sink=_make_heartbeat_sink(
            args.heartbeat_jsonl, args.worker_id, heartbeats
        ),
        task_provider=_make_task_provider(tasks),
        max_consecutive_failures=args.max_consecutive_failures,
    )
    try:
        worker.run(total_steps=args.steps)
        last_batch = worker.last_batch
        upload_summary = _upload_summary(uploaded_batches)
        return {
            "algorithm": cfg.algorithm,
            "batch_dir": args.batch_dir,
            "consecutive_failures": worker.consecutive_failures,
            "dry_run": False,
            "enable_macro_actions": task.action.enable_macro_actions,
            "heartbeat_jsonl": args.heartbeat_jsonl,
            "heartbeats_written": len(heartbeats),
            "last_error": worker.last_error,
            **upload_summary,
            "learner": args.learner,
            "learner_upload_enabled": args.learner is not None,
            "model": cfg.model.name,
            "n_macro_actions": task.action.n_macro_actions,
            "policy_version": worker.policy_version,
            "rollout_duration_s": worker.last_rollout_duration_s,
            "rollout_steps": 0 if last_batch is None else int(last_batch.rewards.size),
            "sps": worker.last_sps,
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
    n_macros: int,
    max_entities: int,
) -> Any:
    model_cls = get("model", cfg.model.name)
    if cfg.model.name == "mlp":
        return model_cls(
            obs_dims,
            hidden=cfg.model.rnn_hidden or 256,
            enable_macro=enable_macro,
            n_macros=n_macros,
        )
    return model_cls(
        obs_dims,
        entity_hidden=cfg.model.entity_hidden,
        attention_layers=cfg.model.attention_layers,
        attention_heads=cfg.model.attention_heads,
        rnn_type=cfg.model.rnn_type,
        rnn_hidden=cfg.model.rnn_hidden,
        enable_macro=enable_macro,
        n_macros=n_macros,
        max_entities=max_entities,
    )


def _load_tasks(args: argparse.Namespace) -> list[TaskConfig]:
    paths = args.tasks if args.tasks else [args.task]
    tasks = [load_task_config(path) for path in paths]
    if not tasks:
        raise ValueError("at least one task is required")
    _validate_task_layouts(tasks)
    return tasks


def _validate_task_layouts(tasks: list[TaskConfig]) -> None:
    base = tasks[0]
    for task in tasks[1:]:
        if task.observation.max_entities != base.observation.max_entities:
            raise ValueError("all worker tasks must share observation.max_entities")
        if task.observation.tier != base.observation.tier:
            raise ValueError("all worker tasks must share observation.tier")
        if task.action.enable_macro_actions != base.action.enable_macro_actions:
            raise ValueError("all worker tasks must share action.enable_macro_actions")
        if task.action.n_macro_actions != base.action.n_macro_actions:
            raise ValueError("all worker tasks must share action.n_macro_actions")


def _make_task_provider(
    tasks: list[TaskConfig],
) -> Callable[[], TaskConfig | None] | None:
    if len(tasks) <= 1:
        return None

    index = -1

    def provide() -> TaskConfig:
        nonlocal index
        index = (index + 1) % len(tasks)
        return tasks[index]

    return provide


def _checkpoint_auth_token(cfg: Any, registry_endpoint: str | None) -> str | None:
    if registry_endpoint is None:
        return None
    if urlparse(registry_endpoint).scheme in ("http", "https"):
        return resolve_auth_token(cfg)
    return None


def _make_heartbeat_sink(
    path: str | None,
    worker_id: str,
    written: list[None],
) -> Callable[[dict[str, Any]], None] | None:
    if path is None:
        return None

    target = Path(path)
    safe_worker = _safe_filename_component(worker_id)

    def sink(payload: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "payload": payload,
                        "worker_id": safe_worker,
                    },
                    sort_keys=True,
                )
            )
            fh.write("\n")
        written.append(None)

    return sink


def _make_batch_uploader(
    batch_dir: str | None,
    worker_id: str,
    written: list[str],
    *,
    learner_endpoint: str | None = None,
    auth_token: str | None = None,
    uploaded: list[bool] | None = None,
) -> Callable[[RolloutBatch], bool | None] | None:
    if batch_dir is None and learner_endpoint is None:
        return None

    directory = None if batch_dir is None else Path(batch_dir)
    client = (
        None
        if learner_endpoint is None
        else BatchIntakeClient(learner_endpoint, auth_token=auth_token)
    )
    safe_worker = _safe_filename_component(worker_id)
    sequence = 0

    def upload(batch: RolloutBatch) -> bool | None:
        nonlocal sequence
        sequence += 1
        if directory is not None:
            path = (
                directory
                / f"{safe_worker}_{sequence:08d}_v{batch.policy_version:06d}.npz"
            )
            save_rollout_batch(path, batch)
            written.append(str(path))
        if client is not None:
            accepted = client.submit(batch)
            if uploaded is not None:
                uploaded.append(accepted)
            return accepted
        return None

    return upload


def _upload_summary(uploaded: list[bool]) -> dict[str, int]:
    accepted = sum(1 for value in uploaded if value)
    submitted = len(uploaded)
    return {
        "learner_accepted_batches": accepted,
        "learner_rejected_batches": submitted - accepted,
        "learner_submitted_batches": submitted,
        "uploaded_batches": submitted,
    }


def _safe_filename_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    safe = safe.strip("._")
    return safe or "worker"


if __name__ == "__main__":
    raise SystemExit(main())
