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
from hkrl.spaces import make_observation_space
from hkrl.training.batch_io import load_rollout_batch
from hkrl.utils.config import load_train_config
from hkrl.utils.registry import get


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Learner server")
    p.add_argument("--config", required=True)
    p.add_argument("--bind", default="0.0.0.0:5600")
    p.add_argument("--batch-dir", help="directory of NPZ RolloutBatch files to ingest")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--publish-every-updates", type=int, default=1)
    p.add_argument("--max-entities", type=int, default=64)
    p.add_argument(
        "--tier",
        default="privileged",
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
    observation_space = make_observation_space(max_entities=args.max_entities, tier=args.tier)
    model = _build_model(
        cfg,
        {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
        },
        max_entities=args.max_entities,
    )
    registry = CheckpointRegistry(str(Path(args.checkpoint_dir)))
    server = LearnerServer(
        model=model,
        config=cfg,
        registry=registry,
        bind=args.bind,
        max_staleness=args.max_staleness,
        publish_every_updates=args.publish_every_updates,
    )
    submitted_batches = _submit_batch_dir(server, args.batch_dir)
    server.serve()
    latest = registry.latest()
    return {
        "accepted_batches": server.accepted_batches,
        "algorithm": cfg.algorithm,
        "batch_dir": args.batch_dir,
        "bind": args.bind,
        "checkpoint_dir": registry.root,
        "latest_checkpoint": None if latest is None else latest.version,
        "model": cfg.model.name,
        "policy_version": server.policy_version,
        "queued_batches": int(getattr(server.algo, "queued_batches", 0)),
        "rejected_batches": server.rejected_batches,
        "submitted_batches": submitted_batches,
    }


def _build_model(cfg: Any, obs_dims: dict[str, tuple[int, ...]], *, max_entities: int) -> Any:
    model_cls = get("model", cfg.model.name)
    if cfg.model.name == "mlp":
        return model_cls(
            obs_dims,
            hidden=cfg.model.rnn_hidden,
            enable_macro=True,
        )
    return model_cls(
        obs_dims,
        entity_hidden=cfg.model.entity_hidden,
        attention_layers=cfg.model.attention_layers,
        attention_heads=cfg.model.attention_heads,
        rnn_type=cfg.model.rnn_type,
        rnn_hidden=cfg.model.rnn_hidden,
        enable_macro=True,
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


if __name__ == "__main__":
    raise SystemExit(main())
