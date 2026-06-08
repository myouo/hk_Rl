"""Command-line entry points for local training and smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hkrl.eval.scripted_policies import RandomPolicy
from hkrl.models.mlp import MlpActorCritic
from hkrl.training.ppo import PPO
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.logging import MetricSink, make_sink
from hkrl.worker.game_worker import GameWorker


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKRL training entry point")
    parser.add_argument("--config", required=True, help="path to a train config YAML")
    parser.add_argument(
        "--task",
        default="configs/tasks/gruz_mother.yaml",
        help="path to a task config YAML",
    )
    parser.add_argument("--smoke", action="store_true", help="random-policy wiring check")
    parser.add_argument("--steps", type=int, default=None, help="override total smoke steps")
    parser.add_argument("--updates", type=int, default=1, help="number of PPO updates to run")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory for PPO checkpoints",
    )
    parser.add_argument(
        "--metrics",
        default="runs/smoke.jsonl",
        help="JSONL metrics path for smoke records",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the clean reset lifecycle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    summary = run_smoke_from_args(args) if args.smoke else run_training_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_training_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)

    if cfg.algorithm != "ppo":
        raise ValueError(f"training CLI currently supports ppo, got {cfg.algorithm!r}")
    if cfg.model.name != "mlp":
        raise ValueError(f"training CLI currently supports mlp model, got {cfg.model.name!r}")
    if cfg.transport.name != "tcp":
        raise ValueError(
            f"training CLI currently supports tcp transport, got {cfg.transport.name!r}"
        )

    from hkrl.env import HKRLEnv
    from hkrl.transport.tcp import TcpTransport
    from hkrl.wrappers import NormalizeObservation

    transport = TcpTransport(host=cfg.transport.host, port=cfg.transport.port)
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    observation_space: Any = env.observation_space
    model = MlpActorCritic(
        {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
        },
        hidden=cfg.model.rnn_hidden or 256,
        enable_macro=task.action.enable_macro_actions,
    )
    worker = GameWorker(env=env, model=model, config=cfg)
    algo = PPO(model=model, config=cfg)
    sink = make_sink("jsonl", path=Path(args.metrics))

    try:
        return run_ppo_training_loop(
            worker=worker,
            algo=algo,
            sink=sink,
            updates=int(args.updates),
            checkpoint_dir=Path(args.checkpoint_dir),
            model=model,
        )
    finally:
        sink.close()
        env.close()


def run_smoke_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)

    from hkrl.env import HKRLEnv
    from hkrl.transport.tcp import TcpTransport
    from hkrl.wrappers import NormalizeObservation

    if cfg.transport.name != "tcp":
        raise ValueError(f"smoke currently supports tcp transport, got {cfg.transport.name!r}")

    transport = TcpTransport(host=cfg.transport.host, port=cfg.transport.port)
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    policy = RandomPolicy(env.action_space, seed=cfg.seed)
    sink = make_sink("jsonl", path=Path(args.metrics))

    try:
        return run_random_policy_smoke(
            env=env,
            policy=policy,
            sink=sink,
            task_id=task.task_id,
            max_steps=args.steps or min(cfg.rollout_steps, 256),
            reset_timeout_s=float(args.reset_timeout),
        )
    finally:
        sink.close()
        env.close()


def run_random_policy_smoke(
    *,
    env: Any,
    policy: Any,
    sink: MetricSink,
    task_id: str,
    max_steps: int,
    reset_timeout_s: float = 30.0,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    obs, info = env.reset(options={"reset_timeout_s": reset_timeout_s})
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    for step in range(max_steps):
        action = policy.act(obs, info.get("action_mask"))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps = step + 1
        sink.log_scalar("reward", float(reward), step=steps)

        if terminated or truncated:
            break

    record = {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    sink.log_episode(record)
    sink.flush()
    return record


def run_ppo_training_loop(
    *,
    worker: Any,
    algo: Any,
    sink: MetricSink,
    updates: int,
    checkpoint_dir: Path | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("updates must be positive")

    last_metrics: dict[str, float] = {}
    last_checkpoint: str | None = None
    total_steps = 0
    for update in range(1, updates + 1):
        batch = worker.collect_rollout()
        metrics = algo.update(worker.buffer)
        last_metrics = {key: float(value) for key, value in metrics.items()}
        total_steps += int(np.asarray(batch.rewards).size)

        for key, value in last_metrics.items():
            sink.log_scalar(key, value, step=update)
        rollout_reward = float(np.asarray(batch.rewards, dtype=np.float32).sum())
        sink.log_episode(
            {
                "update": update,
                "rollout_steps": int(np.asarray(batch.rewards).size),
                "rollout_reward": rollout_reward,
                "policy_version": int(getattr(batch, "policy_version", 0)),
            }
        )

        if checkpoint_dir is not None and model is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"ppo_update_{update:06d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "update": update,
                    "metrics": last_metrics,
                },
                path,
            )
            last_checkpoint = str(path)

    sink.flush()
    return {
        "updates": updates,
        "total_steps": total_steps,
        "last_metrics": last_metrics,
        "last_checkpoint": last_checkpoint,
    }
