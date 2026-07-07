#!/usr/bin/env python3
"""Run the remote Learner server (GPU): batch training + checkpoint publish.

Usage:
    python scripts/run_learner.py --config configs/train/remote_learner.yaml

Builds the learner core and checkpoint registry. For filesystem smoke tests,
``--batch-dir`` ingests NPZ RolloutBatch files through ``LearnerServer.submit``;
``--intake-count`` accepts that many authenticated TCP RolloutBatch uploads
through the same server method before running one update cycle. ``--serve-forever``
keeps accepting TCP RolloutBatch uploads and updates after each accepted batch
until interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from hkrl.learner.batch_intake import BatchIntakeServer
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
    resolve_auth_token,
    validate_bind_address,
    validate_service_auth,
)
from hkrl.utils.registry import get


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Learner server")
    p.add_argument("--config", required=True)
    p.add_argument("--task", help="task YAML used to infer learner model layout")
    p.add_argument(
        "--tasks", nargs="+", help="task YAMLs used to infer learner model layout"
    )
    p.add_argument("--bind", default=None, help="override config.learner.bind")
    p.add_argument("--batch-dir", help="directory of NPZ RolloutBatch files to ingest")
    p.add_argument(
        "--intake-count",
        type=int,
        default=0,
        help="number of TCP RolloutBatch uploads to accept before updating",
    )
    p.add_argument(
        "--serve-forever",
        action="store_true",
        help="keep accepting TCP rollout batches and updating until interrupted",
    )
    p.add_argument("--intake-timeout-s", type=float, default=10.0)
    p.add_argument(
        "--checkpoint-dir", default=None, help="override config.learner.checkpoint_dir"
    )
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
    _validate_learner_args(args)
    cfg = load_train_config(args.config)
    bind = validate_bind_address(args.bind or cfg.learner.bind, cfg.security.bind_scope)
    checkpoint_dir = args.checkpoint_dir or cfg.learner.checkpoint_dir
    max_staleness = (
        cfg.learner.max_staleness if args.max_staleness is None else args.max_staleness
    )
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
    intake_count = int(getattr(args, "intake_count", 0) or 0)
    serve_forever = bool(getattr(args, "serve_forever", False))
    if serve_forever and intake_count:
        raise ValueError("--serve-forever cannot be combined with --intake-count")
    if serve_forever:
        network_batches, network_accepted = _serve_network_forever(
            server,
            bind,
            cfg,
            timeout_s=float(getattr(args, "intake_timeout_s", 10.0)),
        )
    else:
        network_batches, network_accepted = _serve_network_intake(
            server,
            bind,
            cfg,
            intake_count=intake_count,
            timeout_s=float(getattr(args, "intake_timeout_s", 10.0)),
        )
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
        "network_accepted_batches": network_accepted,
        "network_submitted_batches": network_batches,
        "publish_every_updates": publish_every_updates,
        "policy_version": server.policy_version,
        "queued_batches": int(getattr(server.algo, "queued_batches", 0)),
        "rejected_batches": server.rejected_batches,
        "serve_forever": serve_forever,
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

    directory = Path(_non_empty_path_like(batch_dir, name="batch_dir"))
    if not directory.exists():
        raise FileNotFoundError(f"batch directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"batch path is not a directory: {directory}")

    submitted = 0
    for path in sorted(directory.glob("*.npz")):
        server.submit(load_rollout_batch(path))
        submitted += 1
    return submitted


def _serve_network_intake(
    server: LearnerServer,
    bind: str,
    cfg: Any,
    *,
    intake_count: int,
    timeout_s: float,
) -> tuple[int, int]:
    if intake_count < 0:
        raise ValueError("intake_count must be non-negative")
    if intake_count == 0:
        return 0, 0

    validate_service_auth(bind, cfg)
    auth_token = resolve_auth_token(cfg)
    accepted = 0
    with BatchIntakeServer(
        server, bind, auth_token=auth_token, timeout_s=timeout_s
    ) as intake:
        for _ in range(intake_count):
            result = intake.serve_once()
            accepted += int(result.accepted)
    return intake_count, accepted


def _serve_network_forever(
    server: LearnerServer,
    bind: str,
    cfg: Any,
    *,
    timeout_s: float,
) -> tuple[int, int]:
    validate_service_auth(bind, cfg)
    auth_token = resolve_auth_token(cfg)
    submitted = 0
    accepted = 0
    with BatchIntakeServer(
        server, bind, auth_token=auth_token, timeout_s=timeout_s
    ) as intake:
        while True:
            try:
                result = intake.serve_once()
            except KeyboardInterrupt:
                break
            submitted += 1
            if result.accepted:
                accepted += 1
                server.serve()
    return submitted, accepted


def _load_tasks(args: argparse.Namespace) -> list[TaskConfig]:
    task_paths = getattr(args, "tasks", None)
    task_path = getattr(args, "task", None)
    paths = task_paths if task_paths else ([task_path] if task_path else [])
    tasks = [load_task_config(path) for path in paths]
    if tasks:
        _validate_task_layouts(tasks)
    return tasks


def _resolve_layout(
    args: argparse.Namespace, tasks: list[TaskConfig]
) -> dict[str, Any]:
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


def _validate_learner_args(args: argparse.Namespace) -> None:
    _non_empty_path_like(getattr(args, "config", None), name="config")
    _optional_non_empty_path_like(getattr(args, "task", None), name="task")
    task_paths = getattr(args, "tasks", None)
    if task_paths is not None:
        if not isinstance(task_paths, Sequence) or isinstance(task_paths, (str, bytes)):
            raise ValueError("tasks must be a sequence of paths")
        if not task_paths:
            raise ValueError("tasks must contain at least one path")
        for index, task_path in enumerate(task_paths):
            _non_empty_path_like(task_path, name=f"tasks[{index}]")
    _optional_non_empty_string(getattr(args, "bind", None), name="bind")
    _optional_non_empty_path_like(getattr(args, "batch_dir", None), name="batch_dir")
    _optional_non_empty_path_like(
        getattr(args, "checkpoint_dir", None),
        name="checkpoint_dir",
    )
    intake_count = getattr(args, "intake_count", 0)
    if intake_count is None:
        intake_count = 0
    _non_negative_int(
        intake_count,
        name="intake_count",
    )
    _positive_number(
        getattr(args, "intake_timeout_s", 10.0),
        name="intake_timeout_s",
    )
    _optional_non_negative_int(
        getattr(args, "max_staleness", None), name="max_staleness"
    )
    _optional_positive_int(
        getattr(args, "publish_every_updates", None),
        name="publish_every_updates",
    )
    _optional_positive_int(getattr(args, "max_entities", None), name="max_entities")
    _optional_non_negative_int(
        getattr(args, "n_macro_actions", None),
        name="n_macro_actions",
    )
    if bool(getattr(args, "serve_forever", False)) and intake_count:
        raise ValueError("--serve-forever cannot be combined with --intake-count")


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


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name=name)


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)


def _positive_int(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


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
