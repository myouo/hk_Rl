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
from typing import Any

# Import model modules for registry side effects.
from hkrl.models import mlp as _mlp  # noqa: F401
from hkrl.models import recurrent_policy as _recurrent_policy  # noqa: F401
from hkrl.spaces import make_observation_space
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.registry import get
from hkrl.worker.checkpoint_client import CheckpointClient
from hkrl.worker.game_worker import GameWorker


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL GameWorker")
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--learner", help="learner endpoint host:port")
    p.add_argument("--registry", help="checkpoint registry endpoint")
    p.add_argument("--steps", type=int, default=None, help="optional finite rollout sample count")
    p.add_argument("--dry-run", action="store_true", help="validate wiring without env connection")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
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
        latest_checkpoint = (
            None if checkpoint_client is None else checkpoint_client.latest_version()
        )
        return {
            "algorithm": cfg.algorithm,
            "dry_run": True,
            "learner": args.learner,
            "latest_checkpoint": latest_checkpoint,
            "model": cfg.model.name,
            "registry": args.registry,
            "task_id": task.task_id,
        }

    if cfg.transport.name != "tcp":
        raise ValueError(f"worker currently supports tcp transport, got {cfg.transport.name!r}")

    from hkrl.env import HKRLEnv
    from hkrl.transport.tcp import TcpTransport
    from hkrl.wrappers import NormalizeObservation

    transport = TcpTransport(host=cfg.transport.host, port=cfg.transport.port)
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    worker = GameWorker(
        env=env,
        model=model,
        config=cfg,
        checkpoint_client=checkpoint_client,
        learner_endpoint=args.learner,
    )
    try:
        worker.run(total_steps=args.steps)
        last_batch = worker.last_batch
        return {
            "algorithm": cfg.algorithm,
            "dry_run": False,
            "learner": args.learner,
            "model": cfg.model.name,
            "policy_version": worker.policy_version,
            "rollout_steps": 0 if last_batch is None else int(last_batch.rewards.size),
            "task_id": task.task_id,
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


if __name__ == "__main__":
    raise SystemExit(main())
