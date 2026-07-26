#!/usr/bin/env python3
"""Run a continuous live Boss arena and clean-reset after every outcome."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from hkrl import protocol
from hkrl.arena import BossArenaSupervisor
from hkrl.env import HKRLEnv
from hkrl.eval.scripted_policies import ScriptedAggroPolicy
from hkrl.spaces import PLAYER_FEATURE_INDEX
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


class ContactPolicy:
    """Walk toward the observed Boss so natural-death auto-reset is easy to test."""

    def act(
        self,
        obs: dict[str, np.ndarray],
        _info: dict[str, Any],
    ) -> dict[str, Any]:
        player = np.asarray(obs["player"], dtype=np.float32)
        entities = np.asarray(obs["entities"], dtype=np.float32)
        mask = np.asarray(obs["entity_mask"], dtype=bool)
        boss = next(
            (
                entity
                for entity in entities[mask]
                if int(entity[1]) == int(protocol.EntityType.BOSS)
            ),
            None,
        )
        movement = 1
        if boss is not None:
            dx = float(boss[6]) - float(player[PLAYER_FEATURE_INDEX["pos_x"]])
            movement = 2 if dx > 0.1 else 0 if dx < -0.1 else 1
        return {
            "movement_x": movement,
            "aim_y": 1,
            "buttons": {},
            "duration": 0,
            "macro": 0,
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train/arena_hitless_gru.yaml",
    )
    parser.add_argument(
        "--task",
        default="configs/tasks/gruz_mother_hitless_speed.yaml",
    )
    parser.add_argument("--policy", choices=("scripted", "contact"), default="scripted")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--output", help="write JSON evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.reset_timeout <= 0.0:
        raise ValueError("--reset-timeout must be positive")

    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    env = HKRLEnv(transport=make_transport(cfg), task=task)
    if args.policy == "scripted":
        scripted = ScriptedAggroPolicy(env.action_space)

        def policy(
            obs: dict[str, np.ndarray],
            info: dict[str, Any],
        ) -> dict[str, Any]:
            return scripted.act(obs, info.get("action_mask"))

    else:
        contact = ContactPolicy()
        policy = contact.act

    supervisor = BossArenaSupervisor(
        env,
        policy,
        reset_timeout_s=args.reset_timeout,
        max_steps_per_episode=args.max_steps,
    )
    started = time.time()
    try:
        results = supervisor.run(args.episodes)
    finally:
        env.close()

    target_successes = sum(result.target_met for result in results)
    hitless_wins = sum(result.won and result.hitless for result in results)
    bundle = {
        "schema_version": protocol.SCHEMA_VERSION,
        "task_id": task.task_id,
        "scene": task.scene,
        "objective": task.arena.objective,
        "target_kill_time_seconds": task.arena.target_kill_time_seconds,
        "boss_mutation_allowed": False,
        "auto_reset_on_terminal": task.arena.auto_reset_on_terminal,
        "elapsed_wall_seconds": round(time.time() - started, 3),
        "episodes": [asdict(result) for result in results],
        "summary": {
            "episodes": len(results),
            "deaths": sum(result.died for result in results),
            "wins": sum(result.won for result in results),
            "hitless_wins": hitless_wins,
            "hitless_win_rate": hitless_wins / len(results),
            "target_successes": target_successes,
            "target_success_rate": target_successes / len(results),
            "successful_auto_resets": sum(result.reset_succeeded for result in results),
        },
    }
    print(json.dumps(bundle, indent=2, sort_keys=True), flush=True)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"EVIDENCE {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
