#!/usr/bin/env python3
"""Run fixed-seed, shaping-free per-boss evaluation (PRD §13, docs/metrics.md).

Usage:
    python scripts/run_eval.py --policy scripted \
        --tasks configs/tasks/hornet_protector.yaml configs/tasks/mantis_lords.yaml \
        --episodes 20

    python scripts/run_eval.py --policy mlp --checkpoint checkpoints/v123.pt \
        --tasks configs/tasks/gruz_mother.yaml

Reports per-boss win rate / damage taken / time-to-kill and (optionally) a
regression diff vs a baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hkrl import spaces
from hkrl.env import HKRLEnv
from hkrl.eval.evaluator import Evaluator
from hkrl.eval.scripted_policies import ScriptedAggroPolicy
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.models.mlp import MlpActorCritic
from hkrl.transport.tcp import TcpTransport
from hkrl.utils.config import TaskConfig, TrainConfig, load_task_config, load_train_config, resolve_auth_token
from hkrl.wrappers import NormalizeObservation


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Evaluator")
    p.add_argument("--policy", choices=("scripted", "mlp"), default="scripted")
    p.add_argument("--checkpoint", help="checkpoint .pt file or CheckpointRegistry directory")
    p.add_argument("--checkpoint-dir", help="CheckpointRegistry directory; loads latest version")
    p.add_argument("--train-config", default="configs/train/ppo_mlp.yaml")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--baseline", help="optional baseline metrics JSON for regression diff")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output = run_from_args(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    tasks = [load_task_config(path) for path in args.tasks]
    train_cfg = load_train_config(args.train_config)
    policy = _build_policy(args, tasks[0], train_cfg)

    def env_factory(task: TaskConfig) -> Any:
        env = HKRLEnv(transport=_build_transport(args, train_cfg), task=task)
        return env if args.no_normalize else NormalizeObservation(env)

    evaluator = Evaluator(
        policy,
        tasks=tasks,
        seeds=args.seeds,
        env_factory=env_factory,
        max_steps_per_episode=args.max_steps,
    )
    metrics = evaluator.evaluate(episodes_per_task=args.episodes)
    output: dict[str, Any] = {"metrics": metrics}

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as fh:
            baseline = json.load(fh)
        output["regression"] = evaluator.regression_report(baseline, metrics)

    return output


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

    checkpoint_path = _resolve_checkpoint_path(args)

    train_cfg = train_cfg or load_train_config(args.train_config)
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
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state)
    model.eval()
    return model


def _build_transport(args: argparse.Namespace, train_cfg: TrainConfig) -> TcpTransport:
    return TcpTransport(
        host=args.host,
        port=args.port,
        auth_token=resolve_auth_token(train_cfg),
    )


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    checkpoint = getattr(args, "checkpoint", None)
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint is None and checkpoint_dir is None:
        raise SystemExit("--checkpoint or --checkpoint-dir is required with --policy mlp")

    if checkpoint_dir is not None:
        return _latest_registry_checkpoint(Path(checkpoint_dir))

    path = Path(checkpoint)
    if path.is_dir():
        return _latest_registry_checkpoint(path)
    return path


def _latest_registry_checkpoint(root: Path) -> Path:
    latest = CheckpointRegistry(str(root)).latest()
    if latest is None:
        raise SystemExit(f"checkpoint registry is empty: {root}")
    return Path(latest.path)


if __name__ == "__main__":
    raise SystemExit(main())
