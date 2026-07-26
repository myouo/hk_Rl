#!/usr/bin/env python3
"""Run a restart-safe Hall of Gods compatibility sweep against HKRLEnvMod.

For every catalogued fight the sweep verifies:

* the requested ``GG_*`` scene reaches ``RUNNING`` and reports the expected
  scene/task identity;
* one or more live Boss entities expose finite position, velocity, HP and FSM
  telemetry;
* ordinary Hero left/right, jump/gravity/landing and attack inputs work;
* a bounded right-move + jump/dash policy sequence can naturally activate any
  Boss that remains dormant after the short control probe;
* a second same-scene RESET produces a new episode with clean reward events and
  restored Boss HP.

The probe sends only policy-callable Hero primitives.  It never pauses time,
teleports, edits health, advances an FSM, or otherwise mutates Boss state.
Results are written after every Boss so ``--resume`` can safely continue a long
44-fight run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hkrl import protocol
from hkrl.env import HKRLEnv
from hkrl.godhome import GodhomeBossCatalog, GodhomeBossSpec, load_godhome_catalog
from hkrl.spaces import BUTTON_BITS, PLAYER_ACTION_FLAG_BITS, action_mask_layout
from hkrl.transport.factory import make_transport
from hkrl.utils.config import TrainConfig, load_train_config
from hkrl.utils.mod_release import (
    MOD_ID,
    fingerprint_file,
    load_mod_release_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
P_X = 0
P_Y = 1
P_VX = 2
P_VY = 3
P_HP = 4
P_MAX_HP = 5
P_ON_GROUND = 9
P_JUMPING = 11
P_FALLING = 12
P_ACTION_FLAGS = 26
P_APPLIED_INPUT_BUTTONS = 31

E_STABLE_ID = 0
E_TYPE = 1
E_FSM_STATE = 5
E_X = 6
E_Y = 7
E_VX = 10
E_VY = 11
E_HP = 12
E_MAX_HP = 13

G_SCENE_HASH = 0
G_TASK_ID = 2
G_EPISODE_ID = 8

POSITION_EPSILON = 0.02
VELOCITY_EPSILON = 0.05
SCENE_TOKEN = re.compile(rb"GG_[A-Za-z0-9_]+")


@dataclass(frozen=True)
class BossTelemetry:
    stable_id: int
    fsm_state_hash: int
    x: float
    y: float
    vx: float
    vy: float
    hp: float
    max_hp: float


@dataclass(frozen=True)
class ProbeSnapshot:
    label: str
    lifecycle: str
    server_tick: int
    episode_id: int
    scene_hash: int
    task_id: int
    player_x: float
    player_y: float
    player_vx: float
    player_vy: float
    player_hp: int
    player_max_hp: int
    on_ground: bool
    jumping: bool
    falling: bool
    action_flags: int
    applied_input_buttons: int
    boss_entities: tuple[BossTelemetry, ...]
    reward_events: tuple[str, ...]


class EpisodeEnded(RuntimeError):
    """Raised when a short control probe loses its live episode."""


class GodhomeProbe:
    """Collect one Boss's lifecycle and Hero-control evidence."""

    def __init__(
        self,
        env: HKRLEnv,
        *,
        reset_timeout_s: float,
        expected_min_bosses: int,
    ) -> None:
        self.env = env
        self.reset_timeout_s = reset_timeout_s
        self.expected_min_bosses = expected_min_bosses
        self.obs: dict[str, np.ndarray] | None = None
        self.info: dict[str, Any] = {}
        self.running = False
        self.samples: list[ProbeSnapshot] = []

    def reset(self, label: str) -> tuple[ProbeSnapshot, float]:
        started = time.monotonic()
        self.obs, self.info = self.env.reset(
            options={
                "reset_timeout_s": self.reset_timeout_s,
                "recv_timeout_s": min(5.0, self.reset_timeout_s),
            }
        )
        elapsed_s = time.monotonic() - started
        self.running = True
        sample = self.snapshot(label)
        self.samples.append(sample)
        return sample, elapsed_s

    def observe_idle(self, ticks: int) -> None:
        for index in range(ticks):
            self.step(f"idle_{index + 1}")

    def probe_hero(self) -> dict[str, Any]:
        # Exercise attack and jump before long lateral movement can walk off a
        # small platform. Attack goes first so even an aggressive Boss intro
        # cannot hide the short input/state pulse. Later probes wait for that
        # legitimate intro lock to release without modifying game state.
        attack_samples: list[ProbeSnapshot] = []
        attack_button_bit = 1 << BUTTON_BITS["attack"]
        attacking_flag = 1 << PLAYER_ACTION_FLAG_BITS["attacking"]
        for attempt in range(1, 4):
            self._wait_for_button("attack", max_ticks=400)
            attempt_samples = [
                self.step(
                    f"attack_press_{attempt}",
                    buttons=("attack",),
                ),
                *self._neutral(6, f"attack_followup_{attempt}"),
            ]
            attack_samples.extend(attempt_samples)
            if any(sample.action_flags & attacking_flag for sample in attack_samples):
                break
        attack_input_seen = any(
            sample.applied_input_buttons & attack_button_bit for sample in attack_samples
        )
        attack_state_seen = any(sample.action_flags & attacking_flag for sample in attack_samples)

        jump_samples: list[ProbeSnapshot] = []
        jump_takeoff = False
        gravity_seen = False
        jump_landed = False
        max_jump_height = 0.0
        jump_button_bit = 1 << BUTTON_BITS["jump_hold"]
        for attempt in range(1, 4):
            self._wait_for_button("attack", max_ticks=400)
            self._wait_for_ground(max_ticks=150)
            self._wait_for_button("jump_hold", max_ticks=100)
            baseline = self.snapshot(f"jump_baseline_{attempt}")
            self.samples.append(baseline)
            attempt_samples = [
                self.step(
                    f"jump_press_{attempt}",
                    buttons=("jump_hold",),
                    duration=2,
                )
            ]
            attempt_samples.extend(
                self._neutral(
                    100,
                    f"jump_arc_{attempt}",
                    stop_on_landing=True,
                )
            )
            jump_samples.extend(attempt_samples)
            airborne = any(
                sample.player_y > baseline.player_y + 0.05
                or sample.player_vy > VELOCITY_EPSILON
                or not sample.on_ground
                for sample in attempt_samples
            )
            falling = airborne and any(
                sample.player_vy < -VELOCITY_EPSILON or sample.falling for sample in attempt_samples
            )
            landed = airborne and any(
                sample.on_ground and sample.player_y <= baseline.player_y + 0.08
                for sample in attempt_samples[2:]
            )
            jump_takeoff = jump_takeoff or airborne
            gravity_seen = gravity_seen or falling
            jump_landed = jump_landed or landed
            max_jump_height = max(
                max_jump_height,
                max(sample.player_y - baseline.player_y for sample in attempt_samples),
            )
            if jump_takeoff and gravity_seen and jump_landed:
                break
            self._neutral(4, f"jump_retry_neutral_{attempt}")

        jump_input_seen = any(
            sample.applied_input_buttons & jump_button_bit for sample in jump_samples
        )

        # Some arenas briefly relinquish Hero control when the first movement
        # crosses the Boss-intro trigger (notably White Defender and Oro/Mato).
        # Probe each direction in a bounded retry loop and use attack
        # availability as a read-only proxy that the intro lock has ended.
        # Alternating the order avoids systematically favoring one direction.
        movement_samples: dict[str, list[ProbeSnapshot]] = {
            "left": [],
            "right": [],
        }
        movement_attempts: dict[str, list[dict[str, Any]]] = {
            "left": [],
            "right": [],
        }
        best_dx = {"left": 0.0, "right": 0.0}
        for round_index in range(3):
            order = (
                (("left", 0), ("right", 2)) if round_index % 2 == 0 else (("right", 2), ("left", 0))
            )
            for direction, movement in order:
                direction_passed = (
                    best_dx[direction] < -POSITION_EPSILON
                    if direction == "left"
                    else best_dx[direction] > POSITION_EPSILON
                )
                if direction_passed:
                    continue

                self._wait_for_button("attack", max_ticks=400)
                self._neutral(2, f"pre_move_{direction}_{round_index + 1}")
                start = self.snapshot(f"move_{direction}_start_{round_index + 1}")
                self.samples.append(start)
                samples = [
                    self.step(
                        f"move_{direction}_{round_index + 1}_{index + 1}",
                        movement=movement,
                    )
                    for index in range(12)
                ]
                movement_samples[direction].extend(samples)
                if direction == "left":
                    displacement = min(sample.player_x for sample in samples) - start.player_x
                    best_dx[direction] = min(best_dx[direction], displacement)
                else:
                    displacement = max(sample.player_x for sample in samples) - start.player_x
                    best_dx[direction] = max(best_dx[direction], displacement)
                movement_attempts[direction].append(
                    {
                        "round": round_index + 1,
                        "start_x": start.player_x,
                        "end_x": samples[-1].player_x,
                        "directional_dx": _round(displacement),
                    }
                )
                self._neutral(3, f"after_move_{direction}_{round_index + 1}")

            if best_dx["left"] < -POSITION_EPSILON and best_dx["right"] > POSITION_EPSILON:
                break

        left_samples = movement_samples["left"]
        right_samples = movement_samples["right"]
        left_dx = best_dx["left"]
        right_dx = best_dx["right"]
        left_velocity_seen = any(sample.player_vx < -VELOCITY_EPSILON for sample in left_samples)
        right_velocity_seen = any(sample.player_vx > VELOCITY_EPSILON for sample in right_samples)
        # STEP responses are sampled after the committed input tick and the
        # injector is neutralized while Python is thinking. Position therefore
        # remains the authoritative controllability signal; velocity is useful
        # auxiliary telemetry but may already be zero at the response boundary.
        movement_left = left_dx < -POSITION_EPSILON
        movement_right = right_dx > POSITION_EPSILON

        invalid_action_seen = any(
            "INVALID_ACTION" in sample.reward_events
            for sample in (*left_samples, *right_samples, *jump_samples, *attack_samples)
        )

        return {
            "movement_left": movement_left,
            "movement_right": movement_right,
            "left_dx": _round(left_dx),
            "right_dx": _round(right_dx),
            "movement_attempts": movement_attempts,
            "left_velocity_seen": left_velocity_seen,
            "right_velocity_seen": right_velocity_seen,
            "jump_input_seen": jump_input_seen,
            "jump_takeoff": jump_takeoff,
            "gravity_seen": gravity_seen,
            "jump_landed": jump_landed,
            "max_jump_height": _round(max_jump_height),
            "attack_input_seen": attack_input_seen,
            "attack_state_seen": attack_state_seen,
            "invalid_action_seen": invalid_action_seen,
        }

    def probe_boss_activation(
        self,
        *,
        max_steps: int = 320,
        sample_start_index: int = 0,
    ) -> dict[str, Any]:
        """Explore a bounded primitive combination until natural Boss activity.

        Hall of Gods entry gates are not all reachable with a short horizontal
        walk. Winged Nosk, for example, needs enough rightward traversal plus
        platform movement before its Battle Scene emits ENTER/START. The pattern
        stays inside the public action space and checks the action mask before
        adding jump or dash.
        """

        if sample_start_index < 0 or sample_start_index >= len(self.samples):
            raise ValueError("sample_start_index must select an existing sample")

        segment = self.samples[sample_start_index:]
        activity = summarize_boss_activity(segment)
        activity_seen = bool(activity["post_ack_activity_observed"])
        health_ready = bool(activity["full_health_observed"])
        activation_required = not (activity_seen and health_ready)
        activation_steps = 0
        if activation_required:
            baseline = {boss.stable_id: boss for boss in segment[0].boss_entities}
            for index in range(max_steps):
                phase = index % 24
                buttons: list[str] = []
                if phase < 5 and self._button_available("jump_hold"):
                    buttons.append("jump_hold")
                if phase == 10 and self._button_available("dash"):
                    buttons.append("dash")
                # Make horizontal duty explicit instead of accidentally relying
                # on the old response-boundary neutral pulse. Alternating
                # right/neutral keeps enough vertical authority for Winged
                # Nosk's platforms while the production input bridge remains
                # smooth for ordinary consecutive movement decisions.
                movement = 2 if index % 2 == 0 else 1
                sample = self.step(
                    f"boss_activation_{index + 1}",
                    movement=movement,
                    buttons=tuple(buttons),
                    duration=0,
                )
                activation_steps = index + 1
                if _bosses_demonstrate_activity(sample.boss_entities, baseline):
                    activity_seen = True
                if _boss_health_ready(sample):
                    health_ready = True
                if activity_seen and health_ready:
                    break

            segment = self.samples[sample_start_index:]
            activity = summarize_boss_activity(segment)

        activity.update(
            {
                "activation_required": activation_required,
                "activation_steps": activation_steps,
                "activation_strategy": "paced_right_neutral_jump_dash",
                "activation_complete": bool(
                    activity["post_ack_activity_observed"] and activity["full_health_observed"]
                ),
            }
        )
        return activity

    def step(
        self,
        label: str,
        *,
        movement: int = 1,
        aim: int = 1,
        buttons: tuple[str, ...] = (),
        duration: int = 0,
    ) -> ProbeSnapshot:
        if not self.running:
            raise EpisodeEnded(f"episode ended before {label}")
        action = {
            "movement_x": movement,
            "aim_y": aim,
            "buttons": {button: True for button in buttons},
            "duration": duration,
        }
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        self.running = not (terminated or truncated)
        sample = self.snapshot(label)
        self.samples.append(sample)
        if not self.running:
            raise EpisodeEnded(f"episode ended during {label}")
        return sample

    def snapshot(self, label: str) -> ProbeSnapshot:
        if self.obs is None:
            raise RuntimeError("reset must complete before taking a snapshot")
        global_state = np.asarray(self.obs["global"], dtype=np.float32)
        player = np.asarray(self.obs["player"], dtype=np.float32)
        if player.shape[0] <= P_APPLIED_INPUT_BUTTONS:
            raise RuntimeError(
                "live mod lacks privileged schema-v6 Hero telemetry; "
                "rebuild/install/restart HKRLEnvMod"
            )
        bosses = tuple(_boss_telemetry(self.obs))
        if len(bosses) < self.expected_min_bosses:
            raise RuntimeError(
                f"expected at least {self.expected_min_bosses} Boss entities, "
                f"observed {len(bosses)}"
            )
        lifecycle = self.info.get("lifecycle_state", protocol.LifecycleState.IDLE)
        return ProbeSnapshot(
            label=label,
            lifecycle=getattr(lifecycle, "name", str(lifecycle)),
            server_tick=int(self.info.get("server_tick", 0)),
            episode_id=int(global_state[G_EPISODE_ID]),
            scene_hash=int(global_state[G_SCENE_HASH]),
            task_id=int(global_state[G_TASK_ID]),
            player_x=_round(player[P_X]),
            player_y=_round(player[P_Y]),
            player_vx=_round(player[P_VX]),
            player_vy=_round(player[P_VY]),
            player_hp=int(player[P_HP]),
            player_max_hp=int(player[P_MAX_HP]),
            on_ground=bool(player[P_ON_GROUND]),
            jumping=bool(player[P_JUMPING]),
            falling=bool(player[P_FALLING]),
            action_flags=int(player[P_ACTION_FLAGS]),
            applied_input_buttons=int(player[P_APPLIED_INPUT_BUTTONS]),
            boss_entities=bosses,
            reward_events=tuple(event.kind.name for event in self.info.get("reward_events", [])),
        )

    def _neutral(
        self,
        ticks: int,
        label: str,
        *,
        stop_on_landing: bool = False,
    ) -> list[ProbeSnapshot]:
        samples: list[ProbeSnapshot] = []
        airborne_seen = False
        landed_samples = 0
        for index in range(ticks):
            sample = self.step(f"{label}_{index + 1}")
            samples.append(sample)
            if not sample.on_ground:
                airborne_seen = True
                landed_samples = 0
            elif airborne_seen:
                landed_samples += 1
                if stop_on_landing and landed_samples >= 2:
                    break
        return samples

    def _wait_for_ground(self, *, max_ticks: int) -> None:
        grounded_samples = 0
        for index in range(max_ticks):
            sample = self.snapshot(f"ground_check_{index + 1}")
            if sample.on_ground and abs(sample.player_vy) < VELOCITY_EPSILON:
                grounded_samples += 1
                if grounded_samples >= 2:
                    return
            else:
                grounded_samples = 0
            self.step(f"wait_ground_{index + 1}")
        raise RuntimeError(f"Hero did not settle on ground within {max_ticks} ticks")

    def _wait_for_button(self, button: str, *, max_ticks: int) -> None:
        for index in range(max_ticks):
            if self._button_available(button):
                return
            self.step(f"wait_{button}_{index + 1}")
        raise RuntimeError(f"action mask kept button:{button} unavailable for {max_ticks} ticks")

    def _button_available(self, button: str) -> bool:
        layout = action_mask_layout(
            enable_macro=self.env.task.action.enable_macro_actions,
            n_macros=self.env.task.action.n_macro_actions,
        )
        mask = np.asarray(self.info.get("action_mask", []), dtype=bool)
        label = f"button:{button}"
        return label in layout and mask.shape == (len(layout),) and bool(mask[layout.index(label)])


def run_boss(
    *,
    config: TrainConfig,
    catalog: GodhomeBossCatalog,
    boss: GodhomeBossSpec,
    reset_timeout_s: float,
    build_scene_present: bool | None,
) -> dict[str, Any]:
    """Run one isolated Boss probe and return its serializable result."""

    started = time.monotonic()
    env = HKRLEnv(
        transport=make_transport(config),
        task=catalog.make_task(boss),
    )
    probe = GodhomeProbe(
        env,
        reset_timeout_s=reset_timeout_s,
        expected_min_bosses=boss.expected_min_boss_entities,
    )
    failures: list[str] = []
    warnings: list[str] = []
    expected_hash = wire_scene_hash(boss.scene)
    try:
        initial, initial_reset_s = probe.reset("initial_reset")
        failures.extend(
            validate_reset(
                initial,
                boss=boss,
                expected_scene_hash=expected_hash,
                label="initial",
            )
        )
        probe.observe_idle(8)
        hero = probe.probe_hero()
        failures.extend(validate_hero(hero))

        boss_activity = probe.probe_boss_activation()
        if not boss_activity["post_ack_activity_observed"]:
            failures.append(
                "Boss did not demonstrate natural position, velocity, HP, or "
                "FSM activity after bounded Hero action exploration"
            )
        if not boss_activity["full_health_observed"]:
            failures.append(
                "Boss never exposed a positive, full-health lifecycle state "
                "after bounded Hero action exploration"
            )

        previous_episode_id = initial.episode_id
        post_reset, post_reset_s = probe.reset("same_scene_reset")
        post_reset_sample_index = len(probe.samples) - 1
        failures.extend(
            validate_reset(
                post_reset,
                boss=boss,
                expected_scene_hash=expected_hash,
                label="post_reset",
            )
        )
        if post_reset.episode_id == previous_episode_id:
            failures.append(f"same-scene RESET did not advance episode_id ({previous_episode_id})")
        if post_reset.reward_events:
            failures.append(
                "same-scene RESET leaked stale reward events: "
                + ", ".join(post_reset.reward_events)
            )

        post_reset_boss_activity = probe.probe_boss_activation(
            sample_start_index=post_reset_sample_index,
        )
        if not post_reset_boss_activity["post_ack_activity_observed"]:
            failures.append("Boss did not naturally reactivate after same-scene RESET")
        if not post_reset_boss_activity["full_health_observed"]:
            failures.append(
                "Boss did not return to a positive, full-health lifecycle "
                "state after same-scene RESET"
            )
        if boss_activity["full_health_max_hp"] != post_reset_boss_activity["full_health_max_hp"]:
            failures.append(
                "same-scene RESET changed Boss health capacity: "
                + f"{boss_activity['full_health_max_hp']} -> "
                + f"{post_reset_boss_activity['full_health_max_hp']}"
            )

        initial_hp = _boss_hp_summary(initial)
        post_reset_hp = _boss_hp_summary(post_reset)

        return {
            "boss_id": boss.boss_id,
            "display_name": boss.display_name,
            "scene": boss.scene,
            "wire_id": boss.wire_id,
            "variant_of": boss.variant_of,
            "status": "verified" if not failures else "failed",
            "failures": failures,
            "warnings": warnings,
            "build_scene_present": build_scene_present,
            "expected_scene_hash": expected_hash,
            "reset": {
                "initial_duration_s": _round(initial_reset_s),
                "same_scene_duration_s": _round(post_reset_s),
                "initial_episode_id": previous_episode_id,
                "same_scene_episode_id": post_reset.episode_id,
                "initial_boss_hp": initial_hp,
                "same_scene_boss_hp": post_reset_hp,
            },
            "boss_activity": boss_activity,
            "post_reset_boss_activity": post_reset_boss_activity,
            "hero": hero,
            "initial_snapshot": _snapshot_dict(initial),
            "post_reset_snapshot": _snapshot_dict(post_reset),
            "sample_count": len(probe.samples),
            "elapsed_s": _round(time.monotonic() - started),
        }
    finally:
        env.close()


def validate_reset(
    sample: ProbeSnapshot,
    *,
    boss: GodhomeBossSpec,
    expected_scene_hash: int,
    label: str,
) -> list[str]:
    """Return concrete lifecycle/observation contract failures."""

    failures: list[str] = []
    prefix = f"{label}: "
    if sample.lifecycle != protocol.LifecycleState.RUNNING.name:
        failures.append(prefix + f"lifecycle is {sample.lifecycle}, expected RUNNING")
    if sample.scene_hash != expected_scene_hash:
        failures.append(
            prefix
            + f"scene hash {sample.scene_hash} != {expected_scene_hash} "
            + f"for {boss.scene}"
        )
    if sample.task_id != boss.wire_id:
        failures.append(prefix + f"task id {sample.task_id} != {boss.wire_id}")
    if sample.player_hp <= 0 or sample.player_max_hp < sample.player_hp:
        failures.append(prefix + f"invalid Hero HP {sample.player_hp}/{sample.player_max_hp}")
    if len(sample.boss_entities) < boss.expected_min_boss_entities:
        failures.append(
            prefix
            + f"Boss entity count {len(sample.boss_entities)} "
            + f"< {boss.expected_min_boss_entities}"
        )
    if sample.reward_events:
        failures.append(prefix + "RESET returned reward events: " + ", ".join(sample.reward_events))
    return failures


def validate_hero(hero: dict[str, Any]) -> list[str]:
    """Return failed Hero-control capabilities."""

    checks = {
        "movement_left": "Hero did not move left with left input",
        "movement_right": "Hero did not move right with right input",
        "jump_input_seen": "jump input was not applied by HKRLEnvMod",
        "jump_takeoff": "Hero did not take off after jump input",
        "gravity_seen": "Hero jump never transitioned into gravity/falling",
        "jump_landed": "Hero did not land after the jump arc",
        "attack_input_seen": "attack input was not applied by HKRLEnvMod",
        "attack_state_seen": "Hero attack state was not observed",
    }
    failures = [message for key, message in checks.items() if not bool(hero.get(key))]
    if bool(hero.get("invalid_action_seen")):
        failures.append("Hero probe emitted INVALID_ACTION")
    return failures


def summarize_boss_activity(samples: list[ProbeSnapshot]) -> dict[str, Any]:
    """Summarize natural Boss telemetry without prescribing a movement pattern."""

    if not samples:
        raise ValueError("Boss activity summary requires at least one sample")

    by_id: dict[int, list[BossTelemetry]] = {}
    for sample in samples:
        for boss in sample.boss_entities:
            by_id.setdefault(boss.stable_id, []).append(boss)

    first_ids = {boss.stable_id for boss in samples[0].boss_entities}
    entity_set_changed = any(
        {boss.stable_id for boss in sample.boss_entities} != first_ids for sample in samples[1:]
    )
    position_changed = False
    velocity_seen = False
    fsm_changed = False
    hp_changed = False
    for observations in by_id.values():
        first = observations[0]
        position_changed = position_changed or any(
            abs(observation.x - first.x) > POSITION_EPSILON
            or abs(observation.y - first.y) > POSITION_EPSILON
            for observation in observations[1:]
        )
        velocity_seen = velocity_seen or any(
            abs(observation.vx) > VELOCITY_EPSILON or abs(observation.vy) > VELOCITY_EPSILON
            for observation in observations
        )
        fsm_changed = (
            fsm_changed or len({observation.fsm_state_hash for observation in observations}) > 1
        )
        hp_changed = hp_changed or any(
            observation.hp != first.hp for observation in observations[1:]
        )

    positive_hp_observed = any(
        boss.hp > 0.0 and boss.max_hp > 0.0 for sample in samples for boss in sample.boss_entities
    )
    full_health_sample = next(
        (sample for sample in samples if _boss_health_ready(sample)),
        None,
    )
    return {
        "stable_ids": sorted(by_id),
        "max_simultaneous_boss_entities": max(
            (len(sample.boss_entities) for sample in samples),
            default=0,
        ),
        "position_changed": position_changed,
        "velocity_seen": velocity_seen,
        "fsm_changed": fsm_changed,
        "hp_changed": hp_changed,
        "entity_set_changed": entity_set_changed,
        "post_ack_activity_observed": (
            position_changed or velocity_seen or fsm_changed or hp_changed or entity_set_changed
        ),
        "positive_hp_observed": positive_hp_observed,
        "full_health_observed": full_health_sample is not None,
        "full_health_sample_label": (
            None if full_health_sample is None else full_health_sample.label
        ),
        "full_health_max_hp": (
            []
            if full_health_sample is None
            else sorted(
                boss.max_hp for boss in full_health_sample.boss_entities if boss.max_hp > 0.0
            )
        ),
        "reset_object_gate_passed": True,
    }


def _boss_health_ready(sample: ProbeSnapshot) -> bool:
    combat_bosses = [boss for boss in sample.boss_entities if boss.max_hp > 0.0]
    return bool(combat_bosses) and all(
        boss.hp > 0.0 and boss.hp >= boss.max_hp - 0.5 for boss in combat_bosses
    )


def _bosses_demonstrate_activity(
    bosses: tuple[BossTelemetry, ...],
    baseline: dict[int, BossTelemetry],
) -> bool:
    for boss in bosses:
        first = baseline.get(boss.stable_id)
        if first is None:
            return True
        if (
            abs(boss.x - first.x) > POSITION_EPSILON
            or abs(boss.y - first.y) > POSITION_EPSILON
            or abs(boss.vx) > VELOCITY_EPSILON
            or abs(boss.vy) > VELOCITY_EPSILON
            or boss.fsm_state_hash != first.fsm_state_hash
            or boss.hp != first.hp
        ):
            return True
    return False


def load_build_scenes(path: str | Path) -> set[str]:
    """Extract exact ``GG_*`` tokens from Unity's build-scene table."""

    payload = Path(path).expanduser().read_bytes()
    return {match.decode("ascii") for match in SCENE_TOKEN.findall(payload)}


def select_bosses(
    catalog: GodhomeBossCatalog,
    *,
    boss_ids: list[str],
    start_at: str | None,
    max_bosses: int | None,
) -> list[GodhomeBossSpec]:
    """Apply deterministic CLI selection while preserving catalog order."""

    known = {boss.boss_id for boss in catalog.bosses}
    unknown = sorted(set(boss_ids) - known)
    if unknown:
        raise ValueError(f"unknown Boss id(s): {', '.join(unknown)}")
    if start_at is not None and start_at not in known:
        raise ValueError(f"unknown --start-at Boss id: {start_at}")
    selected = list(catalog.bosses)
    if start_at is not None:
        start_index = next(index for index, boss in enumerate(selected) if boss.boss_id == start_at)
        selected = selected[start_index:]
    if boss_ids:
        selected = [boss for boss in selected if boss.boss_id in boss_ids]
    if max_bosses is not None:
        if max_bosses <= 0:
            raise ValueError("--max-bosses must be positive")
        selected = selected[:max_bosses]
    if not selected:
        raise ValueError("Boss selection is empty")
    return selected


def fnv1a_32(text: str) -> int:
    """Match ``GlobalObserver.StableHash``'s signed FNV-1a result."""

    value = 2166136261
    for character in text:
        value ^= ord(character)
        value = (value * 16777619) & 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def wire_scene_hash(scene: str) -> int:
    """Account for the protocol's current float32 GlobalState representation."""

    return int(np.float32(fnv1a_32(scene)))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--catalog", default="configs/godhome_bosses.yaml")
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--boss",
        action="append",
        default=[],
        help="stable Boss id to test; repeat for a subset (default: all 44)",
    )
    parser.add_argument("--start-at", help="start at this catalog Boss id")
    parser.add_argument("--max-bosses", type=int)
    parser.add_argument(
        "--globalgamemanagers",
        help="optional installed game file used to preflight every GG_* scene",
    )
    parser.add_argument(
        "--mod-dll",
        help="installed HKRLEnvMod.dll under test (required unless --list)",
    )
    parser.add_argument(
        "--mod-version",
        help="version claimed by --mod-dll; must match Version.props",
    )
    parser.add_argument(
        "--output",
        default="runs/live/godhome-all-boss-sweep.json",
        help="incremental JSON evidence path",
    )
    parser.add_argument(
        "--report",
        default="runs/live/godhome-all-boss-sweep.md",
        help="incremental Markdown summary path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse verified results already present in --output",
    )
    parser.add_argument("--list", action="store_true", help="print selected catalog and exit")
    parser.add_argument(
        "--fail-on-failed",
        action="store_true",
        help="return non-zero if any selected Boss remains failed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.reset_timeout <= 0.0:
        raise ValueError("--reset-timeout must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")

    catalog = load_godhome_catalog(args.catalog)
    selected = select_bosses(
        catalog,
        boss_ids=args.boss,
        start_at=args.start_at,
        max_bosses=args.max_bosses,
    )
    if args.list:
        for boss in selected:
            print(
                json.dumps(
                    {
                        "boss_id": boss.boss_id,
                        "display_name": boss.display_name,
                        "scene": boss.scene,
                        "variant_of": boss.variant_of,
                    },
                    sort_keys=True,
                )
            )
        return 0

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
    catalog_fingerprint = fingerprint_file(args.catalog)

    build_scenes: set[str] | None = None
    installed_build: dict[str, Any] | None = None
    if args.globalgamemanagers:
        build_scenes = load_build_scenes(args.globalgamemanagers)
        installed_build = fingerprint_file(args.globalgamemanagers)
        missing = [boss.scene for boss in selected if boss.scene not in build_scenes]
        if missing:
            raise ValueError("catalog scenes absent from installed build: " + ", ".join(missing))

    output = Path(args.output).expanduser()
    report = Path(args.report).expanduser()
    existing = (
        _load_resume_results(
            output,
            expected_tested_mod=tested_mod,
            expected_catalog=catalog_fingerprint,
            expected_installed_build=installed_build,
        )
        if args.resume
        else {}
    )
    results: dict[str, dict[str, Any]] = {
        boss_id: result
        for boss_id, result in existing.items()
        if boss_id in {boss.boss_id for boss in selected} and result.get("status") == "verified"
    }
    config = load_train_config(args.config)
    started = time.time()

    for index, boss in enumerate(selected, start=1):
        if boss.boss_id in results:
            print(
                f"SKIP {index}/{len(selected)} {boss.boss_id}: already verified",
                flush=True,
            )
            continue

        result: dict[str, Any] | None = None
        for attempt in range(1, args.max_attempts + 1):
            print(
                f"PROBE {index}/{len(selected)} {boss.boss_id} "
                f"scene={boss.scene} attempt={attempt}/{args.max_attempts}",
                flush=True,
            )
            try:
                result = run_boss(
                    config=config,
                    catalog=catalog,
                    boss=boss,
                    reset_timeout_s=args.reset_timeout,
                    build_scene_present=(
                        None if build_scenes is None else boss.scene in build_scenes
                    ),
                )
            except Exception as exc:
                result = {
                    "boss_id": boss.boss_id,
                    "display_name": boss.display_name,
                    "scene": boss.scene,
                    "wire_id": boss.wire_id,
                    "variant_of": boss.variant_of,
                    "status": "failed",
                    "failures": [f"{type(exc).__name__}: {exc}"],
                    "warnings": [],
                    "build_scene_present": (
                        None if build_scenes is None else boss.scene in build_scenes
                    ),
                    "attempt": attempt,
                }
            result["attempt"] = attempt
            if result["status"] == "verified":
                break
            print(
                "RETRY_REASON "
                + json.dumps(
                    {
                        "boss_id": boss.boss_id,
                        "failures": result.get("failures", []),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        assert result is not None
        results[boss.boss_id] = result
        bundle = make_bundle(
            catalog=catalog,
            selected=selected,
            results=results,
            catalog_fingerprint=catalog_fingerprint,
            installed_build=installed_build,
            tested_mod=tested_mod,
            started=started,
        )
        write_evidence(output, report, bundle)
        print(
            "RESULT "
            + json.dumps(
                {
                    "boss_id": boss.boss_id,
                    "status": result["status"],
                    "attempt": result["attempt"],
                    "failures": result.get("failures", []),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    bundle = make_bundle(
        catalog=catalog,
        selected=selected,
        results=results,
        catalog_fingerprint=catalog_fingerprint,
        installed_build=installed_build,
        tested_mod=tested_mod,
        started=started,
    )
    write_evidence(output, report, bundle)
    print("SUMMARY " + json.dumps(bundle["counts"], sort_keys=True), flush=True)
    print(f"EVIDENCE {output.resolve()}", flush=True)
    print(f"REPORT {report.resolve()}", flush=True)
    return 1 if args.fail_on_failed and bundle["counts"]["failed"] else 0


def make_bundle(
    *,
    catalog: GodhomeBossCatalog,
    selected: list[GodhomeBossSpec],
    results: dict[str, dict[str, Any]],
    catalog_fingerprint: dict[str, Any],
    installed_build: dict[str, Any] | None,
    tested_mod: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Build the stable incremental evidence envelope."""

    ordered = [results[boss.boss_id] for boss in selected if boss.boss_id in results]
    counts = {
        "selected": len(selected),
        "completed": len(ordered),
        "verified": sum(result.get("status") == "verified" for result in ordered),
        "failed": sum(result.get("status") == "failed" for result in ordered),
        "remaining": len(selected) - len(ordered),
    }
    return {
        "schema_version": protocol.SCHEMA_VERSION,
        "catalog_version": catalog.catalog_version,
        "scope": "Hall of Gods scene, lifecycle, Boss telemetry, and Hero primitive controls",
        "boss_mutation_allowed": False,
        "simulation_control_allowed": False,
        "selected_boss_ids": [boss.boss_id for boss in selected],
        "catalog": catalog_fingerprint,
        "installed_build": installed_build,
        "tested_mod": tested_mod,
        "started_unix_s": started,
        "updated_unix_s": time.time(),
        "elapsed_s": _round(time.time() - started),
        "counts": counts,
        "results": ordered,
    }


def write_evidence(output: Path, report: Path, bundle: dict[str, Any]) -> None:
    """Atomically checkpoint JSON and its compact human-readable report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    _atomic_write(report, render_report(bundle))


def render_report(bundle: dict[str, Any]) -> str:
    counts = bundle["counts"]
    lines = [
        "# Godhome all-Boss live compatibility sweep",
        "",
        (
            f"- Result: {counts['verified']} verified, {counts['failed']} failed, "
            f"{counts['remaining']} remaining / {counts['selected']} selected"
        ),
        f"- Protocol schema: v{bundle['schema_version']}",
        f"- Catalog: v{bundle['catalog_version']}",
        (
            f"- Tested Mod: HKRLEnvMod v{bundle['tested_mod']['version']} "
            f"(`{bundle['tested_mod']['dll_sha256']}`)"
        ),
        "- Constraint: ordinary Hero policy input only; no Boss or simulation-state mutation",
        "- Lifecycle: natural activation + full combat HP on entry and same-scene reload",
        "- `@N`: extra bounded activation steps; `@0` means the Hero control probe activated it",
        "",
        (
            "| Boss | Scene | Result | Reset (first/reload) | "
            "Boss lifecycle (entry/reload) | Hero controls |"
        ),
        "|---|---|---:|---:|---|---|",
    ]
    for result in bundle["results"]:
        reset = result.get("reset", {})
        activity = result.get("boss_activity", {})
        post_reset_activity = result.get("post_reset_boss_activity", {})
        hero = result.get("hero", {})
        reset_text = (
            f"{reset.get('initial_duration_s', '—')}/{reset.get('same_scene_duration_s', '—')} s"
        )

        def activity_text(value: dict[str, Any]) -> str:
            signals = "+".join(
                name
                for name, flag in (
                    ("position", value.get("position_changed")),
                    ("velocity", value.get("velocity_seen")),
                    ("FSM", value.get("fsm_changed")),
                    ("HP", value.get("hp_changed")),
                    ("entity", value.get("entity_set_changed")),
                )
                if flag
            )
            if value.get("full_health_observed"):
                signals = f"{signals}+full-HP" if signals else "full-HP"
            return f"{signals or 'none'}@{value.get('activation_steps', '—')}"

        lifecycle_text = f"{activity_text(activity)}/{activity_text(post_reset_activity)}"
        hero_text = (
            "L/R/J/G/L/A"
            if all(
                bool(hero.get(key))
                for key in (
                    "movement_left",
                    "movement_right",
                    "jump_takeoff",
                    "gravity_seen",
                    "jump_landed",
                    "attack_state_seen",
                )
            )
            else "incomplete"
        )
        lines.append(
            f"| {_escape_table(result['display_name'])} | `{result['scene']}` | "
            f"{result['status']} | {reset_text} | {lifecycle_text} | {hero_text} |"
        )
        for failure in result.get("failures", []):
            lines.append(f"| ↳ failure |  |  |  |  | {_escape_table(failure)} |")
    lines.append("")
    return "\n".join(lines)


def _boss_telemetry(obs: dict[str, np.ndarray]) -> list[BossTelemetry]:
    mask = np.asarray(obs["entity_mask"], dtype=bool)
    bosses: list[BossTelemetry] = []
    for entity in np.asarray(obs["entities"], dtype=np.float32)[mask]:
        if int(entity[E_TYPE]) != int(protocol.EntityType.BOSS):
            continue
        bosses.append(
            BossTelemetry(
                stable_id=int(entity[E_STABLE_ID]),
                fsm_state_hash=int(entity[E_FSM_STATE]),
                x=_round(entity[E_X]),
                y=_round(entity[E_Y]),
                vx=_round(entity[E_VX]),
                vy=_round(entity[E_VY]),
                hp=_round(entity[E_HP]),
                max_hp=_round(entity[E_MAX_HP]),
            )
        )
    return bosses


def _boss_hp_summary(sample: ProbeSnapshot) -> dict[str, Any]:
    positive = [boss for boss in sample.boss_entities if boss.max_hp > 0.0]
    return {
        "boss_count": len(sample.boss_entities),
        "hp": [boss.hp for boss in sample.boss_entities],
        "max_hp": [boss.max_hp for boss in sample.boss_entities],
        "all_full_health": bool(positive)
        and all(boss.hp >= boss.max_hp - 0.5 for boss in positive),
    }


def _snapshot_dict(sample: ProbeSnapshot) -> dict[str, Any]:
    data = asdict(sample)
    data["boss_entities"] = [asdict(boss) for boss in sample.boss_entities]
    data["reward_events"] = list(sample.reward_events)
    return data


def _load_resume_results(
    path: Path,
    *,
    expected_tested_mod: dict[str, Any],
    expected_catalog: dict[str, Any],
    expected_installed_build: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"invalid resume evidence bundle: {path}")
    expected_metadata = {
        "schema_version": protocol.SCHEMA_VERSION,
        "tested_mod": expected_tested_mod,
        "catalog": expected_catalog,
        "installed_build": expected_installed_build,
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(f"resume evidence {key} does not match the current test input")
    return {
        str(result["boss_id"]): result
        for result in payload["results"]
        if isinstance(result, dict) and "boss_id" in result
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _escape_table(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _round(value: Any) -> float:
    return round(float(value), 4)


if __name__ == "__main__":
    raise SystemExit(main())
