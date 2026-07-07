#!/usr/bin/env python3
"""Run fixed-seed, shaping-free per-boss evaluation (PRD §13, docs/metrics.md).

Usage:
    python scripts/run_eval.py --policy scripted \
        --tasks configs/tasks/hornet_protector.yaml configs/tasks/mantis_lords.yaml \
        --episodes 20

    python scripts/run_eval.py --policy mlp --checkpoint checkpoints/v123.pt \
        --tasks configs/tasks/gruz_mother.yaml

    python scripts/run_eval.py --policy model --checkpoint-dir checkpoints_gru \
        --train-config configs/train/ppo_attention_gru.yaml \
        --tasks configs/tasks/gruz_mother.yaml

Reports per-boss win rate / damage taken / time-to-kill and (optionally) a
regression diff vs a baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from hkrl import spaces
from hkrl.env import HKRLEnv
from hkrl.eval.evaluator import Evaluator
from hkrl.eval.scripted_policies import ScriptedAggroPolicy
from hkrl.learner.checkpoint_payload import (
    validate_checkpoint_payload,
    validate_model_state_dict,
)
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.models import mlp as _mlp  # noqa: F401
from hkrl.models import recurrent_policy as _recurrent_policy  # noqa: F401
from hkrl.models.mlp import MlpActorCritic
from hkrl.transport.base import Transport
from hkrl.transport.factory import make_transport
from hkrl.utils.config import (
    TaskConfig,
    TrainConfig,
    load_task_config,
    load_train_config,
)
from hkrl.utils.registry import get
from hkrl.wrappers import NormalizeObservation


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Evaluator")
    p.add_argument("--policy", choices=("scripted", "mlp", "model"), default="scripted")
    p.add_argument(
        "--checkpoint", help="checkpoint .pt file or CheckpointRegistry directory"
    )
    p.add_argument(
        "--checkpoint-dir", help="CheckpointRegistry directory; loads latest version"
    )
    p.add_argument("--train-config", default="configs/train/ppo_mlp.yaml")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--ports",
        nargs="+",
        type=int,
        help="optional TCP port pool assigned round-robin to evaluator workers",
    )
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="number of task-level evaluator workers",
    )
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument(
        "--baseline", help="optional baseline metrics JSON for regression diff"
    )
    p.add_argument(
        "--replay-jsonl", help="optional path to write per-step replay JSONL"
    )
    p.add_argument("--output", help="optional path to write metrics JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output = run_from_args(args)
    if args.output:
        _write_output(output, Path(args.output))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    _validate_eval_args(args)
    baseline_metrics = (
        _load_baseline_metrics(Path(args.baseline)) if getattr(args, "baseline", None) else None
    )
    tasks = [load_task_config(path) for path in args.tasks]
    if args.policy in {"mlp", "model"}:
        _validate_model_task_layouts(tasks)
    train_cfg = load_train_config(args.train_config)
    policy = _build_policy(args, tasks[0], train_cfg)
    next_port = _make_port_provider(args)

    def env_factory(task: TaskConfig) -> Any:
        env = HKRLEnv(transport=_build_transport(args, train_cfg, port=next_port()), task=task)
        return env if args.no_normalize else NormalizeObservation(env)

    evaluator = Evaluator(
        policy,
        tasks=tasks,
        seeds=args.seeds,
        env_factory=env_factory,
        max_steps_per_episode=args.max_steps,
        replay_sink=_make_replay_sink(getattr(args, "replay_jsonl", None)),
        num_workers=int(getattr(args, "eval_workers", 1)),
    )
    metrics = evaluator.evaluate(episodes_per_task=args.episodes)
    output: dict[str, Any] = {
        "metadata": _build_metadata(args, tasks, train_cfg),
        "metrics": metrics,
    }

    if baseline_metrics is not None:
        output["regression"] = evaluator.regression_report(baseline_metrics, metrics)

    return output


def _build_metadata(
    args: argparse.Namespace,
    tasks: list[TaskConfig],
    train_cfg: TrainConfig,
) -> dict[str, Any]:
    return {
        "algorithm": train_cfg.algorithm,
        "checkpoint": getattr(args, "checkpoint", None),
        "checkpoint_dir": getattr(args, "checkpoint_dir", None),
        "episodes": int(args.episodes),
        "eval_workers": int(getattr(args, "eval_workers", 1)),
        "max_steps": int(args.max_steps),
        "model": train_cfg.model.name,
        "normalize": not bool(args.no_normalize),
        "policy": args.policy,
        "ports": _ports(args),
        "replay_jsonl": getattr(args, "replay_jsonl", None),
        "seeds": [int(seed) for seed in args.seeds],
        "task_ids": [task.task_id for task in tasks],
        "task_wire_ids": {task.task_id: task.wire_id for task in tasks},
        "train_config": args.train_config,
        "transport": train_cfg.transport.name,
    }


def _validate_model_task_layouts(tasks: list[TaskConfig]) -> None:
    base = tasks[0]
    for task in tasks[1:]:
        if task.observation.max_entities != base.observation.max_entities:
            raise ValueError(
                "all evaluator model tasks must share observation.max_entities"
            )
        if task.observation.tier != base.observation.tier:
            raise ValueError("all evaluator model tasks must share observation.tier")
        if task.action.enable_macro_actions != base.action.enable_macro_actions:
            raise ValueError(
                "all evaluator model tasks must share action.enable_macro_actions"
            )
        if task.action.n_macro_actions != base.action.n_macro_actions:
            raise ValueError(
                "all evaluator model tasks must share action.n_macro_actions"
            )


def _build_policy(
    args: argparse.Namespace, task: TaskConfig, train_cfg: TrainConfig | None = None
) -> Any:
    if args.policy == "scripted":
        return ScriptedAggroPolicy(
            spaces.make_action_space(
                enable_macro=task.action.enable_macro_actions,
                n_macros=task.action.n_macro_actions,
            )
        )

    train_cfg = train_cfg or load_train_config(args.train_config)
    checkpoint_path = _resolve_checkpoint_path(args)
    model = (
        _build_mlp_policy(task, train_cfg)
        if args.policy == "mlp"
        else _build_configured_policy(task, train_cfg)
    )
    state = _extract_model_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.load_state_dict(state)
    model.eval()
    return model


def _extract_model_state_dict(payload: object) -> Any:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return validate_checkpoint_payload(payload)["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return validate_model_state_dict(payload["state_dict"], name="state_dict")
    return validate_model_state_dict(payload, name="state_dict")


def _build_mlp_policy(task: TaskConfig, train_cfg: TrainConfig) -> MlpActorCritic:
    observation_space = spaces.make_observation_space(
        max_entities=task.observation.max_entities,
        tier=task.observation.tier,
    )
    model = MlpActorCritic(
        {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
        },
        hidden=train_cfg.model.rnn_hidden or 256,
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
    )
    return model


def _build_configured_policy(task: TaskConfig, train_cfg: TrainConfig) -> Any:
    observation_space = spaces.make_observation_space(
        max_entities=task.observation.max_entities,
        tier=task.observation.tier,
    )
    obs_dims = {
        "global": observation_space["global"].shape,
        "player": observation_space["player"].shape,
        "entities": observation_space["entities"].shape,
        "entity_mask": observation_space["entity_mask"].shape,
    }
    model_cls = get("model", train_cfg.model.name)
    if train_cfg.model.name == "mlp":
        return model_cls(
            obs_dims,
            hidden=train_cfg.model.rnn_hidden or 256,
            enable_macro=task.action.enable_macro_actions,
            n_macros=task.action.n_macro_actions,
        )
    return model_cls(
        obs_dims,
        entity_hidden=train_cfg.model.entity_hidden,
        attention_layers=train_cfg.model.attention_layers,
        attention_heads=train_cfg.model.attention_heads,
        rnn_type=train_cfg.model.rnn_type,
        rnn_hidden=train_cfg.model.rnn_hidden,
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )


def _build_transport(
    args: argparse.Namespace,
    train_cfg: TrainConfig,
    *,
    port: int | None = None,
) -> Transport:
    return make_transport(train_cfg, host=args.host, port=args.port if port is None else port)


def _make_port_provider(args: argparse.Namespace) -> Any:
    ports = _ports(args)
    lock = threading.Lock()
    index = 0

    def next_port() -> int:
        nonlocal index
        with lock:
            port = ports[index % len(ports)]
            index += 1
        return port

    return next_port


def _ports(args: argparse.Namespace) -> list[int]:
    ports = getattr(args, "ports", None)
    if ports:
        return [_validate_port(port, name="ports") for port in ports]
    return [_validate_port(getattr(args, "port", 5555), name="port")]


def _validate_eval_args(args: argparse.Namespace) -> None:
    tasks = getattr(args, "tasks", None)
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        raise ValueError("at least one task is required")
    for index, task in enumerate(tasks):
        _non_empty_path_like(task, name=f"tasks[{index}]")
    _positive_int(getattr(args, "episodes", None), name="episodes")
    _positive_int(getattr(args, "max_steps", None), name="max_steps")
    eval_workers = _positive_int(getattr(args, "eval_workers", None), name="eval_workers")
    _non_empty_string(getattr(args, "host", "127.0.0.1"), name="host")
    _optional_path_arg(getattr(args, "baseline", None), name="baseline")
    _optional_path_arg(getattr(args, "checkpoint", None), name="checkpoint")
    _optional_path_arg(getattr(args, "checkpoint_dir", None), name="checkpoint_dir")
    _optional_path_arg(getattr(args, "output", None), name="output")
    _optional_path_arg(getattr(args, "replay_jsonl", None), name="replay_jsonl")
    _optional_path_arg(getattr(args, "train_config", None), name="train_config")
    seeds = getattr(args, "seeds", None)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must contain at least one integer")
    for seed in seeds:
        _integer(seed, name="seeds")
    ports = _ports(args)
    if len(set(ports)) != len(ports):
        raise ValueError("ports must be unique")
    active_workers = min(eval_workers, len(tasks))
    if len(ports) < active_workers:
        raise ValueError("ports must include at least one port per active eval worker")


def _positive_int(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_port(value: Any, *, name: str) -> int:
    port = _integer(value, name=name)
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be in [1, 65535]")
    return port


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _non_empty_path_like(value: Any, *, name: str) -> str | os.PathLike[str]:
    if not isinstance(value, str | os.PathLike) or not str(value).strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_path_arg(value: Any, *, name: str) -> str | os.PathLike[str] | None:
    if value is None:
        return None
    return _non_empty_path_like(value, name=name)


def _write_output(output: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _make_replay_sink(path: str | None) -> Any:
    if path is None:
        return None

    target = Path(_non_empty_path_like(path, name="replay_jsonl"))

    def sink(record: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True))
            fh.write("\n")

    return sink


def _load_baseline_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("baseline metrics JSON must be an object")
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError("baseline metrics must be a task metrics object")
    return metrics


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    checkpoint = getattr(args, "checkpoint", None)
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint is None and checkpoint_dir is None:
        raise SystemExit(
            "--checkpoint or --checkpoint-dir is required with --policy mlp/model"
        )

    if checkpoint_dir is not None:
        return _latest_registry_checkpoint(
            Path(_non_empty_path_like(checkpoint_dir, name="checkpoint_dir"))
        )

    path = Path(_non_empty_path_like(checkpoint, name="checkpoint"))
    if path.is_dir():
        return _latest_registry_checkpoint(path)
    return path


def _latest_registry_checkpoint(root: Path) -> Path:
    registry = CheckpointRegistry(str(root))
    latest = registry.latest()
    if latest is None:
        raise SystemExit(f"checkpoint registry is empty: {root}")
    path = registry.resolve_path(latest)
    actual_hash = _sha256_file(path)
    if actual_hash != latest.sha256:
        raise ValueError(
            f"checkpoint sha256 mismatch for version {latest.version}: "
            f"expected {latest.sha256}, got {actual_hash}"
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
