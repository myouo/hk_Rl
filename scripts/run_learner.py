#!/usr/bin/env python3
"""Run the remote Learner server (GPU): batch training + checkpoint publish.

Usage:
    python scripts/run_learner.py --config configs/train/remote_learner.yaml

Builds the learner core and checkpoint registry. For filesystem smoke tests,
``--batch-dir`` ingests NPZ RolloutBatch files through ``LearnerServer.submit``;
network intake can use the same server method behind an authenticated endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.learner.learner_server import LearnerServer

# Import model modules for registry side effects.
from hkrl.models import mlp as _mlp  # noqa: F401
from hkrl.models import recurrent_policy as _recurrent_policy  # noqa: F401
from hkrl.spaces import DEFAULT_N_MACROS, make_observation_space
from hkrl.training.batch_io import load_rollout_batch
from hkrl.utils.config import (
    TaskConfig,
    load_task_config,
    load_train_config,
    validate_bind_address,
)
from hkrl.utils.registry import get


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Learner server")
    p.add_argument("--config", required=True)
    p.add_argument("--task", help="task YAML used to infer learner model layout")
    p.add_argument("--tasks", nargs="+", help="task YAMLs used to infer learner model layout")
    p.add_argument("--bind", default=None, help="override config.learner.bind")
    p.add_argument("--batch-dir", help="directory of NPZ RolloutBatch files to ingest")
    p.add_argument("--checkpoint-dir", default=None, help="override config.learner.checkpoint_dir")
    p.add_argument("--max-staleness", type=int, default=None)
    p.add_argument("--publish-every-updates", type=int, default=None)
    p.add_argument("--max-entities", type=int, default=None)
    p.add_argument("--disable-macro-actions", action="store_true")
    p.add_argument("--n-macro-actions", type=int, default=None)
    p.add_argument(
        "--tier",
        default=None,
        choices=("privileged", "reduced", "human_visible"),
        help="observation tier used to size the learner model",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    bind = validate_bind_address(args.bind or cfg.learner.bind, cfg.security.bind_scope)
    checkpoint_dir = args.checkpoint_dir or cfg.learner.checkpoint_dir
    max_staleness = cfg.learner.max_staleness if args.max_staleness is None else args.max_staleness
    publish_every_updates = (
        cfg.learner.publish_every_updates
        if args.publish_every_updates is None
        else args.publish_every_updates
    )
    tasks = _load_tasks(args)
    layout = _resolve_layout(args, tasks)
    observation_space = make_observation_space(
        max_entities=layout["max_entities"],
        tier=layout["tier"],
    )
    model = _build_model(
        cfg,
        {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
        },
        max_entities=layout["max_entities"],
        enable_macro=layout["enable_macro_actions"],
        n_macros=layout["n_macro_actions"],
    )
    registry = CheckpointRegistry(str(Path(checkpoint_dir)))
    server = LearnerServer(
        model=model,
        config=cfg,
        registry=registry,
        bind=bind,
        max_staleness=max_staleness,
        publish_every_updates=publish_every_updates,
    )
    submitted_batches = _submit_batch_dir(server, args.batch_dir)
    server.serve()
    latest = registry.latest()
    return {
        "accepted_batches": server.accepted_batches,
        "algorithm": cfg.algorithm,
        "batch_dir": args.batch_dir,
        "bind": bind,
        "checkpoint_dir": registry.root,
        "enable_macro_actions": layout["enable_macro_actions"],
        "latest_checkpoint": None if latest is None else latest.version,
        "max_entities": layout["max_entities"],
        "max_staleness": max_staleness,
        "model": cfg.model.name,
        "n_macro_actions": layout["n_macro_actions"],
        "publish_every_updates": publish_every_updates,
        "policy_version": server.policy_version,
        "queued_batches": int(getattr(server.algo, "queued_batches", 0)),
        "rejected_batches": server.rejected_batches,
        "submitted_batches": submitted_batches,
        "task_ids": [task.task_id for task in tasks],
        "tier": layout["tier"],
    }


def _build_model(
    cfg: Any,
    obs_dims: dict[str, tuple[int, ...]],
    *,
    max_entities: int,
    enable_macro: bool,
    n_macros: int,
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


def _submit_batch_dir(server: LearnerServer, batch_dir: str | None) -> int:
    if batch_dir is None:
        return 0

    directory = Path(batch_dir)
    if not directory.exists():
        raise FileNotFoundError(f"batch directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"batch path is not a directory: {directory}")

    submitted = 0
    for path in sorted(directory.glob("*.npz")):
        server.submit(load_rollout_batch(path))
        submitted += 1
    return submitted


def _load_tasks(args: argparse.Namespace) -> list[TaskConfig]:
    task_paths = getattr(args, "tasks", None)
    task_path = getattr(args, "task", None)
    paths = task_paths if task_paths else ([task_path] if task_path else [])
    tasks = [load_task_config(path) for path in paths]
    if tasks:
        _validate_task_layouts(tasks)
    return tasks


def _resolve_layout(args: argparse.Namespace, tasks: list[TaskConfig]) -> dict[str, Any]:
    task = tasks[0] if tasks else None
    max_entities = args.max_entities
    if max_entities is None:
        max_entities = task.observation.max_entities if task is not None else 64

    tier = args.tier
    if tier is None:
        tier = task.observation.tier if task is not None else "privileged"

    enable_macro = task.action.enable_macro_actions if task is not None else True
    if args.disable_macro_actions:
        enable_macro = False

    n_macros = args.n_macro_actions
    if n_macros is None:
        n_macros = task.action.n_macro_actions if task is not None else DEFAULT_N_MACROS

    return {
        "enable_macro_actions": enable_macro,
        "max_entities": max_entities,
        "n_macro_actions": n_macros,
        "tier": tier,
    }


def _validate_task_layouts(tasks: list[TaskConfig]) -> None:
    base = tasks[0]
    for task in tasks[1:]:
        if task.observation.max_entities != base.observation.max_entities:
            raise ValueError("all learner tasks must share observation.max_entities")
        if task.observation.tier != base.observation.tier:
            raise ValueError("all learner tasks must share observation.tier")
        if task.action.enable_macro_actions != base.action.enable_macro_actions:
            raise ValueError("all learner tasks must share action.enable_macro_actions")
        if task.action.n_macro_actions != base.action.n_macro_actions:
            raise ValueError("all learner tasks must share action.n_macro_actions")


if __name__ == "__main__":
    raise SystemExit(main())
