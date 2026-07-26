#!/usr/bin/env python3
"""Measure the live Hero jump curve for every policy duration choice.

The profiler sends only ordinary policy input.  Each trial starts through the
normal RESET lifecycle, requests ``jump_hold`` with one of the action space's
1/2/4/8-tick duration values, explicitly releases, and samples position/velocity
until landing.  Trials touched by natural Boss damage are discarded and retried;
the Boss is never paused, moved, frozen, or edited.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from hkrl import protocol
from hkrl.env import HKRLEnv
from hkrl.spaces import (
    BUTTON_BITS,
    DURATION_TICKS,
    PLAYER_FEATURE_INDEX,
)
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config

G_TIME = 4
G_FIXED_DELTA = 6


@dataclass(frozen=True)
class JumpSample:
    phase: str
    server_tick: int
    time_in_episode: float
    x: float
    y: float
    vx: float
    vy: float
    on_ground: bool
    jumping: bool
    falling: bool
    applied_input_buttons: int
    event_kinds: tuple[str, ...]


@dataclass(frozen=True)
class JumpTrial:
    requested_hold_ticks: int
    status: str
    reason: str
    baseline_y: float
    apex_y: float
    height: float
    peak_vy: float
    airtime_ticks: int
    airtime_seconds: float
    observed_hold_samples: int
    samples: tuple[JumpSample, ...]


@dataclass(frozen=True)
class JumpProfile:
    requested_hold_ticks: int
    status: str
    clean_trials: int
    attempts: int
    median_height: float
    median_peak_vy: float
    median_airtime_seconds: float
    trials: tuple[JumpTrial, ...]


class LiveJumpProfiler:
    def __init__(
        self,
        env: HKRLEnv,
        *,
        reset_timeout_s: float,
        max_flight_ticks: int,
    ) -> None:
        self.env = env
        self.reset_timeout_s = reset_timeout_s
        self.max_flight_ticks = max_flight_ticks
        self.obs: dict[str, np.ndarray] | None = None
        self.info: dict[str, Any] = {}

    def run_trial(self, hold_ticks: int) -> JumpTrial:
        if hold_ticks not in DURATION_TICKS:
            raise ValueError(f"hold_ticks must be one of {DURATION_TICKS}")

        self.obs, self.info = self.env.reset(
            options={
                "reset_timeout_s": self.reset_timeout_s,
                "recv_timeout_s": 10.0,
            }
        )
        self._settle_on_ground()
        baseline = self._sample("baseline")
        samples = [baseline]

        duration_index = DURATION_TICKS.index(hold_ticks)
        samples.append(
            self._step(
                "jump_request",
                {
                    "movement_x": 1,
                    "aim_y": 1,
                    "buttons": {"jump_hold": True},
                    "duration": duration_index,
                    "macro": 0,
                },
            )
        )

        airborne_seen = not samples[-1].on_ground
        stable_ground = 0
        terminated = False
        truncated = False
        for index in range(self.max_flight_ticks):
            sample, terminated, truncated = self._step_with_terminal(
                f"release_{index + 1}",
                self._noop(),
            )
            samples.append(sample)
            airborne_seen = airborne_seen or not sample.on_ground
            if airborne_seen and sample.on_ground and abs(sample.vy) < 0.05:
                stable_ground += 1
            else:
                stable_ground = 0
            if stable_ground >= 2 or terminated or truncated:
                break

        event_kinds = {event for sample in samples for event in sample.event_kinds}
        damaged = "DAMAGE_TAKEN" in event_kinds
        landed = airborne_seen and stable_ground >= 2
        jump_bit = 1 << BUTTON_BITS["jump_hold"]
        observed_hold_samples = sum(
            bool(sample.applied_input_buttons & jump_bit) for sample in samples[1:]
        )
        apex_y = max(sample.y for sample in samples)
        peak_vy = max(sample.vy for sample in samples)
        airborne_samples = [sample for sample in samples if not sample.on_ground]
        if airborne_samples:
            takeoff_tick = airborne_samples[0].server_tick
            landing_tick = samples[-1].server_tick
            airtime_ticks = max(0, landing_tick - takeoff_tick)
        else:
            airtime_ticks = 0
        fixed_delta = float(self._global()[G_FIXED_DELTA])
        status = "verified"
        reason = "clean takeoff/apex/landing observed"
        if damaged:
            status = "discarded"
            reason = "natural Boss damage contaminated the jump curve"
        elif terminated or truncated:
            status = "discarded"
            reason = "episode ended before a clean landing"
        elif not landed:
            status = "failed"
            reason = "Hero did not complete a takeoff/landing cycle"
        elif observed_hold_samples <= 0:
            status = "failed"
            reason = "jump_hold was not observed at the input bridge"

        return JumpTrial(
            requested_hold_ticks=hold_ticks,
            status=status,
            reason=reason,
            baseline_y=_round(baseline.y),
            apex_y=_round(apex_y),
            height=_round(apex_y - baseline.y),
            peak_vy=_round(peak_vy),
            airtime_ticks=airtime_ticks,
            airtime_seconds=_round(airtime_ticks * fixed_delta),
            observed_hold_samples=observed_hold_samples,
            samples=tuple(samples),
        )

    def _settle_on_ground(self) -> None:
        stable = 0
        for _ in range(40):
            player = self._player()
            if (
                bool(player[PLAYER_FEATURE_INDEX["on_ground"]])
                and abs(float(player[PLAYER_FEATURE_INDEX["vel_y"]])) < 0.05
            ):
                stable += 1
            else:
                stable = 0
            if stable >= 2:
                return
            _, terminated, truncated = self._step_with_terminal("settle", self._noop())
            if terminated or truncated:
                raise RuntimeError(
                    "episode ended while waiting for a stable jump baseline"
                )
        raise RuntimeError("Hero did not reach a stable grounded jump baseline")

    def _step(self, phase: str, action: dict[str, Any]) -> JumpSample:
        sample, terminated, truncated = self._step_with_terminal(phase, action)
        if terminated or truncated:
            raise RuntimeError("episode ended during jump request")
        return sample

    def _step_with_terminal(
        self,
        phase: str,
        action: dict[str, Any],
    ) -> tuple[JumpSample, bool, bool]:
        previous_repeat = self.env.task.action.action_repeat
        self.env.task.action.action_repeat = 1
        try:
            self.obs, _, terminated, truncated, self.info = self.env.step(action)
        finally:
            self.env.task.action.action_repeat = previous_repeat
        return self._sample(phase), terminated, truncated

    def _sample(self, phase: str) -> JumpSample:
        player = self._player()
        global_state = self._global()
        return JumpSample(
            phase=phase,
            server_tick=int(self.info.get("server_tick", 0)),
            time_in_episode=_round(global_state[G_TIME]),
            x=_round(player[PLAYER_FEATURE_INDEX["pos_x"]]),
            y=_round(player[PLAYER_FEATURE_INDEX["pos_y"]]),
            vx=_round(player[PLAYER_FEATURE_INDEX["vel_x"]]),
            vy=_round(player[PLAYER_FEATURE_INDEX["vel_y"]]),
            on_ground=bool(player[PLAYER_FEATURE_INDEX["on_ground"]]),
            jumping=bool(player[PLAYER_FEATURE_INDEX["jumping"]]),
            falling=bool(player[PLAYER_FEATURE_INDEX["falling"]]),
            applied_input_buttons=int(
                player[PLAYER_FEATURE_INDEX["applied_input_buttons"]]
            ),
            event_kinds=tuple(
                event.kind.name for event in self.info.get("reward_events", ())
            ),
        )

    def _player(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("reset must complete before reading player state")
        player = np.asarray(self.obs["player"], dtype=np.float32)
        required = PLAYER_FEATURE_INDEX["applied_input_buttons"]
        if player.shape[0] <= required:
            raise RuntimeError("live mod lacks schema-v6 action telemetry")
        return player

    def _global(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("reset must complete before reading global state")
        return np.asarray(self.obs["global"], dtype=np.float32)

    @staticmethod
    def _noop() -> dict[str, Any]:
        return {
            "movement_x": 1,
            "aim_y": 1,
            "buttons": {},
            "duration": 0,
            "macro": 0,
        }


def collect_profiles(
    profiler: LiveJumpProfiler,
    *,
    hold_ticks: tuple[int, ...],
    clean_trials: int,
    max_attempts: int,
) -> tuple[JumpProfile, ...]:
    profiles: list[JumpProfile] = []
    for requested_ticks in hold_ticks:
        trials: list[JumpTrial] = []
        accepted: list[JumpTrial] = []
        for _ in range(max_attempts):
            trial = profiler.run_trial(requested_ticks)
            trials.append(trial)
            print("TRIAL " + json.dumps(asdict(trial), sort_keys=True), flush=True)
            if trial.status == "verified":
                accepted.append(trial)
                if len(accepted) >= clean_trials:
                    break

        profile = summarize_profile(
            requested_ticks,
            trials=tuple(trials),
            clean_trials=tuple(accepted),
            requested_clean_trials=clean_trials,
        )
        profiles.append(profile)
        print("PROFILE " + json.dumps(asdict(profile), sort_keys=True), flush=True)
    return tuple(profiles)


def summarize_profile(
    requested_hold_ticks: int,
    *,
    trials: tuple[JumpTrial, ...],
    clean_trials: tuple[JumpTrial, ...],
    requested_clean_trials: int,
) -> JumpProfile:
    status = "verified" if len(clean_trials) >= requested_clean_trials else "failed"
    return JumpProfile(
        requested_hold_ticks=requested_hold_ticks,
        status=status,
        clean_trials=len(clean_trials),
        attempts=len(trials),
        median_height=_median(trial.height for trial in clean_trials),
        median_peak_vy=_median(trial.peak_vy for trial in clean_trials),
        median_airtime_seconds=_median(trial.airtime_seconds for trial in clean_trials),
        trials=trials,
    )


def summarize_relationship(profiles: tuple[JumpProfile, ...]) -> dict[str, Any]:
    verified = [profile for profile in profiles if profile.status == "verified"]
    heights = [profile.median_height for profile in verified]
    monotonic = all(later + 0.05 >= earlier for earlier, later in pairwise(heights))
    distinct = len(heights) >= 2 and heights[-1] > heights[0] + 0.1
    return {
        "verified_profiles": len(verified),
        "monotonic_non_decreasing_height": monotonic,
        "short_long_height_distinct": distinct,
        "height_delta_long_minus_short": (
            _round(heights[-1] - heights[0]) if len(heights) >= 2 else 0.0
        ),
        "valid": len(verified) == len(profiles) and monotonic and distinct,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--task", default="configs/tasks/gruz_mother.yaml")
    parser.add_argument(
        "--hold-ticks",
        nargs="+",
        type=int,
        default=list(DURATION_TICKS),
        help=f"policy duration ticks to profile; choices: {DURATION_TICKS}",
    )
    parser.add_argument("--clean-trials", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-flight-ticks", type=int, default=120)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--output", help="write JSON evidence")
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    hold_ticks = tuple(dict.fromkeys(args.hold_ticks))
    if not hold_ticks or any(ticks not in DURATION_TICKS for ticks in hold_ticks):
        raise ValueError(f"--hold-ticks must contain values from {DURATION_TICKS}")
    if args.clean_trials <= 0:
        raise ValueError("--clean-trials must be positive")
    if args.max_attempts < args.clean_trials:
        raise ValueError("--max-attempts must be >= --clean-trials")
    if args.max_flight_ticks <= 0:
        raise ValueError("--max-flight-ticks must be positive")
    if args.reset_timeout <= 0.0:
        raise ValueError("--reset-timeout must be positive")

    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    env = HKRLEnv(transport=make_transport(cfg), task=task)
    profiler = LiveJumpProfiler(
        env,
        reset_timeout_s=args.reset_timeout,
        max_flight_ticks=args.max_flight_ticks,
    )
    started = time.time()
    try:
        profiles = collect_profiles(
            profiler,
            hold_ticks=hold_ticks,
            clean_trials=args.clean_trials,
            max_attempts=args.max_attempts,
        )
    finally:
        env.close()

    relationship = summarize_relationship(profiles)
    bundle = {
        "schema_version": protocol.SCHEMA_VERSION,
        "scope": "live jump amplitude by policy duration; ordinary input only",
        "boss_mutation_allowed": False,
        "elapsed_s": _round(time.time() - started),
        "profiles": [asdict(profile) for profile in profiles],
        "relationship": relationship,
    }
    print("SUMMARY " + json.dumps(relationship, sort_keys=True), flush=True)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"EVIDENCE {output.resolve()}", flush=True)
    return 1 if args.fail_on_invalid and not relationship["valid"] else 0


def _median(values: Any) -> float:
    items = list(values)
    return _round(statistics.median(items)) if items else 0.0


def _round(value: Any) -> float:
    return round(float(value), 4)


if __name__ == "__main__":
    raise SystemExit(main())
