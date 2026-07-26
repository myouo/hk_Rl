#!/usr/bin/env python3
"""Measure whether Hero movement stays continuous between STEP decisions.

The comparison sends the same number of commanded right/left physics ticks in
two forms:

* one long STEP request (continuous reference);
* many ordinary policy decisions using the task's normal action repeat.

Both trials start with a clean RESET and use only a movement primitive.  The
metric uses observed ``server_tick`` time, so Python/transport gaps cannot hide
an injected neutral-input tick.  Natural Boss damage invalidates a trial; the
script never pauses or mutates the Boss, Hero, physics, or clock.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hkrl.env import HKRLEnv
from hkrl.spaces import PLAYER_FEATURE_INDEX
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.mod_release import MOD_ID, fingerprint_file, load_mod_release_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
G_FIXED_DELTA = 6


@dataclass(frozen=True)
class WalkTrial:
    label: str
    direction: str
    action_repeat: int
    decisions: int
    commanded_ticks: int
    server_ticks: int
    fixed_delta_time_s: float
    wall_time_s: float
    start_x: float
    end_x: float
    signed_displacement: float
    distance: float
    game_time_s: float
    game_speed_units_s: float
    wall_speed_units_s: float
    response_velocity_x: tuple[float, ...]
    event_kinds: tuple[str, ...]
    start_hp: int
    end_hp: int


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--task", default="configs/tasks/gruz_mother.yaml")
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--active-ticks", type=int, default=24)
    parser.add_argument("--decision-repeat", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--min-speed-retention", type=float, default=0.9)
    parser.add_argument(
        "--mod-dll",
        help="installed HKRLEnvMod.dll under test",
    )
    parser.add_argument(
        "--mod-version",
        help="version claimed by --mod-dll; must match Version.props",
    )
    parser.add_argument(
        "--output",
        default="runs/live/walk-smoothness.json",
    )
    parser.add_argument(
        "--fail-on-stutter",
        action="store_true",
        help="return non-zero when stepped speed retains less than the threshold",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not 2 <= args.active_ticks <= 200:
        raise ValueError("--active-ticks must be in [2, 200]")
    if not 1 <= args.decision_repeat <= args.active_ticks:
        raise ValueError("--decision-repeat must be in [1, active-ticks]")
    if args.active_ticks % args.decision_repeat != 0:
        raise ValueError("--active-ticks must be divisible by --decision-repeat")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.reset_timeout <= 0.0:
        raise ValueError("--reset-timeout must be positive")
    if not 0.0 < args.min_speed_retention <= 1.0:
        raise ValueError("--min-speed-retention must be in (0, 1]")

    release_metadata = load_mod_release_metadata(REPO_ROOT)
    if not args.mod_dll:
        raise ValueError("--mod-dll is required so evidence is bound to the tested binary")
    requested_mod_version = args.mod_version or release_metadata.mod_version
    if requested_mod_version != release_metadata.mod_version:
        raise ValueError(
            "--mod-version does not match Version.props: "
            f"{requested_mod_version} != {release_metadata.mod_version}"
        )
    mod_fingerprint = fingerprint_file(args.mod_dll)
    if mod_fingerprint["name"] != f"{MOD_ID}.dll":
        raise ValueError(f"--mod-dll must be named {MOD_ID}.dll")
    tested_mod = {
        "id": MOD_ID,
        "version": requested_mod_version,
        "dll_name": mod_fingerprint["name"],
        "dll_size_bytes": mod_fingerprint["size_bytes"],
        "dll_sha256": mod_fingerprint["sha256"],
    }

    config = load_train_config(args.config)
    task = load_task_config(args.task)
    env = HKRLEnv(transport=make_transport(config), task=task)
    try:
        continuous = _run_clean_trial(
            env,
            label="one_long_step",
            direction=args.direction,
            action_repeat=args.active_ticks,
            decisions=1,
            max_attempts=args.max_attempts,
            reset_timeout_s=args.reset_timeout,
        )
        stepped = _run_clean_trial(
            env,
            label="ordinary_policy_steps",
            direction=args.direction,
            action_repeat=args.decision_repeat,
            decisions=args.active_ticks // args.decision_repeat,
            max_attempts=args.max_attempts,
            reset_timeout_s=args.reset_timeout,
        )
    finally:
        env.close()

    result = compare_walk_trials(
        continuous,
        stepped,
        min_speed_retention=args.min_speed_retention,
    )
    result["tested_mod"] = tested_mod
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"EVIDENCE {output.resolve()}", flush=True)
    return 1 if args.fail_on_stutter and not result["smooth"] else 0


def compare_walk_trials(
    continuous: WalkTrial,
    stepped: WalkTrial,
    *,
    min_speed_retention: float,
) -> dict[str, Any]:
    """Build the stable comparison payload used by tests and live evidence."""

    if continuous.commanded_ticks != stepped.commanded_ticks:
        raise ValueError("walk trials must command the same number of physics ticks")
    if continuous.game_speed_units_s <= 0.0:
        raise ValueError("continuous reference speed must be positive")
    retention = stepped.game_speed_units_s / continuous.game_speed_units_s
    return {
        "schema": "hkrl.walk_smoothness.v1",
        "scope": "Hero movement continuity across synchronous STEP responses",
        "boss_mutation_allowed": False,
        "simulation_control_allowed": False,
        "continuous": _trial_dict(continuous),
        "stepped": _trial_dict(stepped),
        "speed_retention": round(retention, 6),
        "min_speed_retention": min_speed_retention,
        "smooth": retention >= min_speed_retention,
    }


def _run_clean_trial(
    env: HKRLEnv,
    *,
    label: str,
    direction: str,
    action_repeat: int,
    decisions: int,
    max_attempts: int,
    reset_timeout_s: float,
) -> WalkTrial:
    last_reason = ""
    for _attempt in range(1, max_attempts + 1):
        trial = _run_trial(
            env,
            label=label,
            direction=direction,
            action_repeat=action_repeat,
            decisions=decisions,
            reset_timeout_s=reset_timeout_s,
        )
        if "DAMAGE_TAKEN" in trial.event_kinds or trial.end_hp != trial.start_hp:
            last_reason = "natural Boss damage contaminated the movement trial"
            continue
        if trial.signed_displacement <= 0.05:
            last_reason = "Hero did not move in the requested direction"
            continue
        return trial
    raise RuntimeError(f"{label} had no clean trial: {last_reason}")


def _run_trial(
    env: HKRLEnv,
    *,
    label: str,
    direction: str,
    action_repeat: int,
    decisions: int,
    reset_timeout_s: float,
) -> WalkTrial:
    env.task.action.action_repeat = action_repeat
    observation, info = env.reset(
        options={
            "reset_timeout_s": reset_timeout_s,
            "recv_timeout_s": min(10.0, reset_timeout_s),
        }
    )
    player = np.asarray(observation["player"], dtype=np.float32)
    start_x = float(player[PLAYER_FEATURE_INDEX["pos_x"]])
    start_hp = int(player[PLAYER_FEATURE_INDEX["hp"]])
    start_tick = int(info["server_tick"])
    fixed_delta = float(np.asarray(observation["global"], dtype=np.float32)[G_FIXED_DELTA])
    movement_x = 0 if direction == "left" else 2
    action: dict[str, Any] = {
        "movement_x": movement_x,
        "aim_y": 1,
        "buttons": {},
        "duration": 0,
        "macro": 0,
    }
    velocities: list[float] = []
    events: list[str] = []
    started = time.perf_counter()
    for _ in range(decisions):
        observation, _, terminated, truncated, info = env.step(action)
        player = np.asarray(observation["player"], dtype=np.float32)
        velocities.append(float(player[PLAYER_FEATURE_INDEX["vel_x"]]))
        events.extend(event.kind.name for event in info.get("reward_events", []))
        if terminated or truncated:
            raise RuntimeError(f"{label} episode ended during movement")
    wall_time = time.perf_counter() - started

    player = np.asarray(observation["player"], dtype=np.float32)
    end_x = float(player[PLAYER_FEATURE_INDEX["pos_x"]])
    end_hp = int(player[PLAYER_FEATURE_INDEX["hp"]])
    server_ticks = int(info["server_tick"]) - start_tick
    if server_ticks <= 0 or fixed_delta <= 0.0:
        raise RuntimeError(f"{label} returned invalid timing telemetry")
    displacement = end_x - start_x
    signed_displacement = displacement if direction == "right" else -displacement
    distance = abs(displacement)
    game_time = server_ticks * fixed_delta
    return WalkTrial(
        label=label,
        direction=direction,
        action_repeat=action_repeat,
        decisions=decisions,
        commanded_ticks=action_repeat * decisions,
        server_ticks=server_ticks,
        fixed_delta_time_s=round(fixed_delta, 6),
        wall_time_s=round(wall_time, 6),
        start_x=round(start_x, 6),
        end_x=round(end_x, 6),
        signed_displacement=round(signed_displacement, 6),
        distance=round(distance, 6),
        game_time_s=round(game_time, 6),
        game_speed_units_s=round(distance / game_time, 6),
        wall_speed_units_s=round(distance / wall_time, 6),
        response_velocity_x=tuple(round(value, 6) for value in velocities),
        event_kinds=tuple(events),
        start_hp=start_hp,
        end_hp=end_hp,
    )


def _trial_dict(trial: WalkTrial) -> dict[str, Any]:
    payload = asdict(trial)
    payload["response_velocity_x"] = list(trial.response_velocity_x)
    payload["event_kinds"] = list(trial.event_kinds)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
