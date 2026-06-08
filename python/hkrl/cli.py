"""Command-line entry points for local training and smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hkrl.eval.scripted_policies import RandomPolicy
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.logging import MetricSink, make_sink


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

    if not args.smoke:
        parser.error("full PPO training lands in phase 3; pass --smoke for a wiring check")

    summary = run_smoke_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


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
