#!/usr/bin/env python3
"""Drive deterministic actions against a live HKRLEnvMod instance.

The game must already be running and ``HKRL_AUTH_TOKEN`` must match the mod's
runtime configuration.  Each command prints a JSON observation snapshot so the
same action sequence can be compared before and after a mod change.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from hkrl import protocol
from hkrl.env import HKRLEnv
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config

MAX_SAFE_TICKS = 200


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--task", default="configs/tasks/gruz_mother.yaml")
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="action command; repeat to run a deterministic non-interactive sequence",
    )
    return parser


class LiveActionDemo:
    """Small stateful command driver for one live Gym environment."""

    def __init__(self, env: HKRLEnv, *, reset_timeout_s: float) -> None:
        self.env = env
        self.reset_timeout_s = reset_timeout_s
        self.obs: dict[str, np.ndarray] | None = None
        self.info: dict[str, Any] = {}

    def reset(self) -> None:
        self.obs, self.info = self.env.reset(
            options={
                "reset_timeout_s": self.reset_timeout_s,
                "recv_timeout_s": 10.0,
            }
        )
        self.print_status("reset")

    def execute(self, command_line: str) -> bool:
        pieces = shlex.split(command_line)
        if not pieces:
            return True
        name = pieces[0].lower()
        ticks = _positive_ticks(pieces[1] if len(pieces) > 1 else None)

        if name == "status":
            self.print_status("status")
        elif name == "reset":
            self.reset()
        elif name == "left":
            self.step("move_left", movement=0, repeat=ticks or 35)
        elif name == "right":
            self.step("move_right", movement=2, repeat=ticks or 35)
        elif name in {"noop", "wait"}:
            self.step("noop", repeat=ticks or 2)
        elif name in {"jump", "long_jump"}:
            self.step(
                "long_jump",
                movement=2,
                buttons={"jump_hold": True},
                duration=3,
                repeat=ticks or 16,
            )
        elif name == "short_jump":
            self.step(
                "short_jump",
                buttons={"jump_tap": True},
                repeat=ticks or 2,
            )
        elif name == "attack":
            self.step(
                "attack_forward",
                buttons={"attack": True},
                repeat=ticks or 2,
            )
        elif name == "up_attack":
            self.step(
                "attack_up",
                aim=2,
                buttons={"attack": True},
                repeat=ticks or 2,
            )
        elif name in {"down_attack", "pogo"}:
            self.step(
                "attack_down",
                aim=0,
                buttons={"attack": True},
                repeat=ticks or 2,
            )
        elif name == "cast":
            self.step(
                "cast_forward",
                buttons={"cast": True},
                repeat=ticks or 3,
            )
        elif name == "focus":
            self.step(
                "focus_hold",
                buttons={"focus_hold": True},
                duration=3,
                repeat=ticks or 80,
            )
        elif name == "nail_art_hold":
            self.step(
                "nail_art_hold",
                buttons={"nail_art_hold": True},
                duration=3,
                repeat=ticks or 55,
            )
        elif name == "nail_art_release":
            self.step(
                "nail_art_release",
                buttons={"nail_art_release": True},
                repeat=ticks or 2,
            )
        elif name in {"quit", "close", "exit"}:
            return False
        else:
            raise ValueError(f"unknown command: {name}")
        return True

    def step(
        self,
        label: str,
        *,
        movement: int = 1,
        aim: int = 1,
        buttons: dict[str, bool] | None = None,
        duration: int = 0,
        repeat: int,
    ) -> None:
        if self.obs is None:
            raise RuntimeError("reset must run before an action")
        action: dict[str, Any] = {
            "movement_x": movement,
            "aim_y": aim,
            "buttons": buttons or {},
            "duration": duration,
            "macro": 0,
        }
        previous_repeat = self.env.task.action.action_repeat
        self.env.task.action.action_repeat = repeat
        try:
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        finally:
            self.env.task.action.action_repeat = previous_repeat
        self.print_status(label, reward=reward)
        if terminated or truncated:
            print("EPISODE_ENDED", flush=True)

    def print_status(self, label: str, *, reward: float = 0.0) -> None:
        if self.obs is None:
            raise RuntimeError("no observation is available")
        player = self.obs["player"]
        boss = self._boss()
        payload: dict[str, Any] = {
            "label": label,
            "episode": int(self.obs["global"][8]),
            "server_tick": int(self.info["server_tick"]),
            "lifecycle": self.info["lifecycle_state"].name,
            "player": {
                "x": _round(player[0]),
                "y": _round(player[1]),
                "vx": _round(player[2]),
                "vy": _round(player[3]),
                "hp": int(player[4]),
                "max_hp": int(player[5]),
                "soul": int(player[6]),
                "on_ground": bool(player[9]),
                "jumping": bool(player[11]),
                "falling": bool(player[12]),
                "focus_state": int(player[19]),
            },
            "events": [
                {
                    "kind": event.kind.name,
                    "amount": _round(event.amount),
                    "entity_id": int(event.entity_id),
                }
                for event in self.info.get("reward_events", [])
            ],
            "reward": _round(reward),
        }
        if boss is not None:
            payload["boss"] = {
                "x": _round(boss[6]),
                "y": _round(boss[7]),
                "rel_x": _round(boss[8]),
                "rel_y": _round(boss[9]),
                "vx": _round(boss[10]),
                "vy": _round(boss[11]),
                "hp": int(boss[12]),
                "max_hp": int(boss[13]),
                "fsm_state_hash": int(boss[5]),
            }
        print("STATUS " + json.dumps(payload, sort_keys=True), flush=True)

    def _boss(self) -> np.ndarray | None:
        if self.obs is None:
            return None
        mask = self.obs["entity_mask"].astype(bool)
        for entity in self.obs["entities"][mask]:
            if int(entity[1]) == int(protocol.EntityType.BOSS):
                return entity
        return None


def run_commands(driver: LiveActionDemo, commands: Sequence[str]) -> None:
    for command in commands:
        print(f"COMMAND {command}", flush=True)
        if not driver.execute(command):
            return


def run_interactive(driver: LiveActionDemo) -> None:
    print(
        "READY commands: status, left [ticks], right [ticks], jump [ticks], "
        "short_jump [ticks], attack/up_attack/down_attack [ticks], cast [ticks], "
        "focus [ticks], nail_art_hold/release [ticks], noop [ticks], reset, close",
        flush=True,
    )
    for line in sys.stdin:
        try:
            if not driver.execute(line):
                return
        except (TypeError, ValueError) as exc:
            print(f"COMMAND_ERROR {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.reset_timeout <= 0.0:
        raise ValueError("--reset-timeout must be positive")
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    env = HKRLEnv(transport=make_transport(cfg), task=task)
    driver = LiveActionDemo(env, reset_timeout_s=args.reset_timeout)
    try:
        driver.reset()
        if args.command:
            run_commands(driver, args.command)
        else:
            run_interactive(driver)
    finally:
        env.close()
    return 0


def _positive_ticks(value: str | None) -> int | None:
    if value is None:
        return None
    ticks = int(value)
    # HKRLEnv's normal STEP receive timeout is five seconds. At the game's
    # 50 Hz fixed timestep, the protocol's raw uint8 maximum of 255 can take
    # slightly more than five seconds and make an otherwise healthy live run
    # look disconnected. Keep this diagnostic driver below that threshold.
    if not 1 <= ticks <= MAX_SAFE_TICKS:
        raise ValueError(f"ticks must be in [1, {MAX_SAFE_TICKS}]")
    return ticks


def _round(value: Any) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    raise SystemExit(main())
