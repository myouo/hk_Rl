#!/usr/bin/env python3
"""Measure live HKRLEnvMod STEP latency and effective physics throughput."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from hkrl.env import HKRLEnv
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--task", default="configs/tasks/gruz_mother.yaml")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.steps <= 0 or args.warmup < 0:
        raise ValueError("steps must be positive and warmup must be non-negative")
    if not 1 <= args.action_repeat <= 200:
        raise ValueError("action-repeat must be in [1, 200]")
    if args.time_scale <= 0.0:
        raise ValueError("time-scale must be positive")

    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    task.action.action_repeat = args.action_repeat
    env = HKRLEnv(transport=make_transport(cfg), task=task)
    noop: dict[str, Any] = {
        "movement_x": 1,
        "aim_y": 1,
        "buttons": {},
        "duration": 0,
        "macro": 0,
    }

    try:
        observation, info = env.reset(
            options={
                "reset_timeout_s": args.reset_timeout,
                "recv_timeout_s": 10.0,
            }
        )
        env.set_timescale(args.time_scale)

        for _ in range(args.warmup):
            observation, _, terminated, truncated, info = env.step(noop)
            if terminated or truncated:
                raise RuntimeError("episode ended during benchmark warmup")

        print(
            f"BENCHMARK_READY steps={args.steps} repeat={args.action_repeat}",
            flush=True,
        )
        first_tick = int(info["server_tick"])
        latencies_ms: list[float] = []
        tick_deltas: list[int] = []
        hp_deltas: list[int] = []
        event_kinds: list[list[str]] = []
        started = time.perf_counter()
        for _ in range(args.steps):
            previous_tick = int(info["server_tick"])
            previous_hp = int(np.asarray(observation["player"])[4])
            step_started = time.perf_counter()
            observation, _, terminated, truncated, info = env.step(noop)
            latencies_ms.append((time.perf_counter() - step_started) * 1000.0)
            tick_deltas.append(int(info["server_tick"]) - previous_tick)
            hp_deltas.append(int(np.asarray(observation["player"])[4]) - previous_hp)
            event_kinds.append(
                [event.kind.name for event in info.get("reward_events", [])]
            )
            if terminated or truncated:
                raise RuntimeError("episode ended during benchmark measurement")
        elapsed = time.perf_counter() - started
        last_tick = int(info["server_tick"])

        latency = np.asarray(latencies_ms, dtype=np.float64)
        steady_latency = np.asarray(
            [
                sample
                for sample, kinds in zip(latencies_ms, event_kinds, strict=True)
                if "DAMAGE_TAKEN" not in kinds
            ],
            dtype=np.float64,
        )
        global_state = np.asarray(observation["global"], dtype=np.float64)
        slowest_indices = sorted(
            range(len(latencies_ms)),
            key=latencies_ms.__getitem__,
            reverse=True,
        )[:5]
        result = {
            "steps": len(latencies_ms),
            "action_repeat": args.action_repeat,
            "requested_time_scale": args.time_scale,
            "observed_time_scale": round(float(global_state[5]), 6),
            "fixed_delta_time_s": round(float(global_state[6]), 6),
            "elapsed_s": round(elapsed, 6),
            "requests_per_s": round(len(latencies_ms) / elapsed, 3),
            "server_tick_delta": last_tick - first_tick,
            "fixed_updates_per_s": round((last_tick - first_tick) / elapsed, 3),
            "latency_ms": {
                "mean": round(float(latency.mean()), 3),
                "p50": round(float(np.percentile(latency, 50)), 3),
                "p95": round(float(np.percentile(latency, 95)), 3),
                "p99": round(float(np.percentile(latency, 99)), 3),
                "max": round(float(latency.max()), 3),
            },
            "non_damage_latency_ms": {
                "mean": round(float(steady_latency.mean()), 3),
                "p50": round(float(np.percentile(steady_latency, 50)), 3),
                "p95": round(float(np.percentile(steady_latency, 95)), 3),
                "p99": round(float(np.percentile(steady_latency, 99)), 3),
                "max": round(float(steady_latency.max()), 3),
            },
            "slow_step_count_over_100ms": sum(
                sample > 100.0 for sample in latencies_ms
            ),
            "damage_taken_step_count": sum(
                "DAMAGE_TAKEN" in kinds for kinds in event_kinds
            ),
            "slowest_steps": [
                {
                    "index": index,
                    "latency_ms": round(latencies_ms[index], 3),
                    "server_tick_delta": tick_deltas[index],
                    "hp_delta": hp_deltas[index],
                    "event_kinds": event_kinds[index],
                }
                for index in slowest_indices
            ],
        }
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        try:
            env.set_timescale(1.0)
        except Exception:
            pass
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
