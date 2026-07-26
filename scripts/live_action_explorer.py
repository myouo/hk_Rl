#!/usr/bin/env python3
"""Explore Hollow Knight combat-action equivalence classes in a live game.

This is deliberately an input-only diagnostic.  It may request a clean episode
RESET, then sends only the same movement/aim/button/macro actions available to a
training policy.  It never pauses time, teleports, changes health/soul, or
touches Boss FSM/physics state.

The hybrid action space has thousands of raw bit combinations, most of which
are contradictory or game-equivalent.  The catalog below covers the meaningful
combat semantics: movement, short/long/directional/double jumps, ground/air
dashes, ordinary slash variants, Dream Nail, all three directional spell
families, focus, all nail-art families, and every configured bootstrap macro.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hkrl import protocol
from hkrl.action_combinations import get_action_combination
from hkrl.env import HKRLEnv
from hkrl.spaces import BUTTON_BITS, PLAYER_ACTION_FLAG_BITS, action_mask_layout
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config

MAX_SAFE_TICKS = 200

P_X = 0
P_Y = 1
P_VX = 2
P_VY = 3
P_HP = 4
P_MAX_HP = 5
P_SOUL = 6
P_ON_GROUND = 9
P_JUMPING = 11
P_FALLING = 12
P_DASHING = 13
P_FOCUS_STATE = 19
P_ACTION_FLAGS = 26
P_SPELL_FSM = 27
P_DREAM_NAIL_FSM = 28
P_NAIL_ARTS_FSM = 29
P_NAIL_CHARGE_TIMER = 30
P_APPLIED_INPUT_BUTTONS = 31


@dataclass(frozen=True)
class ActionPhase:
    label: str
    movement: int = 1
    aim: int = 1
    buttons: tuple[str, ...] = ()
    duration: int = 0
    ticks: int = 1
    macro: int = 0

    def to_action(self) -> dict[str, Any]:
        return {
            "movement_x": self.movement,
            "aim_y": self.aim,
            "buttons": {button: True for button in self.buttons},
            "duration": self.duration,
            "macro": self.macro,
        }


@dataclass(frozen=True)
class ActionCase:
    name: str
    family: str
    description: str
    phases: tuple[ActionPhase, ...]
    expectation: str
    expected_flag: str | None = None
    availability_button: str | None = None
    soul_required: int = 0
    damage_required: bool = False
    grounded_start: bool = True
    expected_hold_steps: int = 0


@dataclass
class Snapshot:
    phase: str
    server_tick: int
    player_x: float
    player_y: float
    player_vx: float
    player_vy: float
    player_hp: int
    player_max_hp: int
    soul: int
    on_ground: bool
    jumping: bool
    falling: bool
    dashing: bool
    focus_state: int
    action_flags: int
    spell_fsm_state_hash: int
    dream_nail_fsm_state_hash: int
    nail_arts_fsm_state_hash: int
    nail_charge_timer: float
    applied_input_buttons: int
    boss_x: float | None
    boss_y: float | None
    boss_vx: float | None
    boss_vy: float | None
    boss_hp: int | None
    event_kinds: list[str]


@dataclass
class CaseResult:
    name: str
    family: str
    status: str
    reason: str
    snapshots: list[Snapshot]


def p(
    label: str,
    *,
    movement: int = 1,
    aim: int = 1,
    buttons: tuple[str, ...] = (),
    duration: int = 0,
    ticks: int = 1,
    macro: int = 0,
) -> ActionPhase:
    return ActionPhase(label, movement, aim, buttons, duration, ticks, macro)


def duration_case(duration_index: int, hold_steps: int) -> ActionCase:
    phases = (
        p(
            "hold_1",
            buttons=("dream_nail",),
            duration=duration_index,
        ),
        *tuple(p(f"hold_{step}") for step in range(2, hold_steps + 1)),
        p("release"),
        p("post_release", ticks=4),
    )
    return ActionCase(
        f"duration_{hold_steps}_ticks",
        "duration",
        f"hold input for exactly {hold_steps} applied policy ticks",
        phases,
        "duration_hold",
        availability_button="dream_nail",
        expected_hold_steps=hold_steps,
    )


def macro_case(index: int, name: str) -> ActionCase:
    ticks = {
        "approach": 4,
        "retreat": 4,
        "jump_attack": 2,
        "pogo": 5,
        "dash_away": 2,
        "dash_through": 2,
        "cast_forward": 2,
        "cast_up": 2,
        "focus_when_safe": 120,
        "short_hop": 4,
        "long_jump": 4,
    }[name]
    availability = {
        "jump_attack": "attack",
        "pogo": "attack",
        "dash_away": "dash",
        "dash_through": "dash",
        "cast_forward": "cast",
        "cast_up": "cast",
        "focus_when_safe": "focus_hold",
        "short_hop": "jump_tap",
        "long_jump": "jump_hold",
    }.get(name)
    phases = (p("macro", macro=index + 1, ticks=ticks),)
    if name == "focus_when_safe":
        phases += (p("release", ticks=2),)
    return ActionCase(
        f"macro_{name}",
        "macro",
        f"configured bootstrap macro: {name}",
        phases,
        "macro",
        availability_button=availability,
        soul_required=33
        if name in {"cast_forward", "cast_up", "focus_when_safe"}
        else 0,
        damage_required=name == "focus_when_safe",
    )


def combination_case(
    name: str,
    *,
    expectation: str,
    expected_flag: str | None = None,
    availability_button: str,
) -> ActionCase:
    combination = get_action_combination(name)
    phases = tuple(
        p(
            f"combo_{index + 1}",
            movement=phase.movement_x,
            aim=phase.aim_y,
            buttons=phase.buttons,
            duration=phase.duration,
            ticks=phase.ticks,
        )
        for index, phase in enumerate(combination.phases)
    )
    return ActionCase(
        f"combo_{name}",
        "combination",
        combination.description,
        phases,
        expectation,
        expected_flag=expected_flag,
        availability_button=availability_button,
        soul_required=combination.min_soul,
    )


ACTION_CASES: tuple[ActionCase, ...] = (
    ActionCase(
        "move_left",
        "movement",
        "ground movement toward the left arena boundary",
        (p("left", movement=0, ticks=10),),
        "move_left",
    ),
    ActionCase(
        "move_right",
        "movement",
        "ground movement toward the right arena boundary",
        (p("right", movement=2, ticks=10),),
        "move_right",
    ),
    ActionCase(
        "short_jump_neutral",
        "jump",
        "neutral short hop",
        (p("jump", buttons=("jump_tap",)), p("coast")),
        "jump",
    ),
    ActionCase(
        "short_jump_left",
        "jump",
        "left-moving short hop",
        (p("jump", movement=0, buttons=("jump_tap",)), p("coast", movement=0)),
        "jump",
    ),
    ActionCase(
        "short_jump_right",
        "jump",
        "right-moving short hop",
        (p("jump", movement=2, buttons=("jump_tap",)), p("coast", movement=2)),
        "jump",
    ),
    ActionCase(
        "long_jump_neutral",
        "jump",
        "neutral full-height jump",
        (p("jump_hold", buttons=("jump_hold",), ticks=8),),
        "jump",
    ),
    ActionCase(
        "long_jump_left",
        "jump",
        "left-moving full-height jump",
        (p("jump_hold", movement=0, buttons=("jump_hold",), ticks=8),),
        "jump",
    ),
    ActionCase(
        "long_jump_right",
        "jump",
        "right-moving full-height jump",
        (p("jump_hold", movement=2, buttons=("jump_hold",), ticks=8),),
        "jump",
    ),
    ActionCase(
        "double_jump",
        "jump",
        "ground jump followed by Monarch Wings",
        (
            p("first_jump", buttons=("jump_hold",), ticks=6),
            p("release", ticks=2),
            p("second_jump", buttons=("jump_tap",)),
        ),
        "double_jump",
        availability_button="jump_tap",
    ),
    ActionCase(
        "ground_dash_left",
        "dash",
        "ground dash left",
        (p("dash", movement=0, buttons=("dash",), ticks=2),),
        "dash",
        availability_button="dash",
    ),
    ActionCase(
        "ground_dash_right",
        "dash",
        "ground dash right",
        (p("dash", movement=2, buttons=("dash",), ticks=2),),
        "dash",
        availability_button="dash",
    ),
    ActionCase(
        "air_dash_left",
        "dash",
        "jump then air dash left",
        (
            p("jump", movement=0, buttons=("jump_hold",), ticks=6),
            p("dash", movement=0, buttons=("dash",), ticks=2),
        ),
        "dash",
        availability_button="dash",
    ),
    ActionCase(
        "air_dash_right",
        "dash",
        "jump then air dash right",
        (
            p("jump", movement=2, buttons=("jump_hold",), ticks=6),
            p("dash", movement=2, buttons=("dash",), ticks=2),
        ),
        "dash",
        availability_button="dash",
    ),
    ActionCase(
        "ground_side_slash_left",
        "ordinary_attack",
        "stationary ground slash facing left",
        (
            p("face", movement=0, ticks=2),
            p("slash", buttons=("attack",)),
        ),
        "action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    ActionCase(
        "ground_side_slash_right",
        "ordinary_attack",
        "stationary ground slash facing right",
        (
            p("face", movement=2, ticks=2),
            p("slash", buttons=("attack",)),
        ),
        "action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    ActionCase(
        "running_slash_right",
        "ordinary_attack",
        "right-moving ground slash",
        (p("run_slash", movement=2, buttons=("attack",)),),
        "action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    ActionCase(
        "aerial_side_slash_left",
        "ordinary_attack",
        "left-facing horizontal aerial slash",
        (
            p("jump", movement=0, buttons=("jump_hold",), ticks=6),
            p("slash", movement=0, buttons=("attack",)),
        ),
        "action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    ActionCase(
        "aerial_side_slash_right",
        "ordinary_attack",
        "right-facing horizontal aerial slash",
        (
            p("jump", movement=2, buttons=("jump_hold",), ticks=6),
            p("slash", movement=2, buttons=("attack",)),
        ),
        "action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    ActionCase(
        "ground_up_slash",
        "ordinary_attack",
        "grounded up-slash",
        (p("up_slash", aim=2, buttons=("attack",)),),
        "action_flag",
        expected_flag="up_attacking",
        availability_button="attack",
    ),
    ActionCase(
        "aerial_up_slash",
        "ordinary_attack",
        "airborne up-slash",
        (
            p("jump", buttons=("jump_hold",), ticks=6),
            p("up_slash", aim=2, buttons=("attack",)),
        ),
        "action_flag",
        expected_flag="up_attacking",
        availability_button="attack",
    ),
    ActionCase(
        "aerial_down_slash",
        "ordinary_attack",
        "airborne down-slash / pogo input",
        (
            p("jump", buttons=("jump_hold",), ticks=7),
            p("down_slash", aim=0, buttons=("attack",)),
        ),
        "action_flag",
        expected_flag="down_attacking",
        availability_button="attack",
    ),
    ActionCase(
        "dream_nail",
        "dream_nail",
        "normal grounded Dream Nail charge and release",
        (
            p("charge", buttons=("dream_nail",), ticks=80),
            p("release"),
        ),
        "dream_nail",
        availability_button="dream_nail",
    ),
    ActionCase(
        "fireball_left",
        "spell",
        "left-facing Vengeful Spirit / Shade Soul",
        (
            p("face", movement=0, ticks=2),
            p("cast", buttons=("cast",), ticks=2),
        ),
        "spell",
        availability_button="cast",
        soul_required=33,
    ),
    ActionCase(
        "fireball_right",
        "spell",
        "right-facing Vengeful Spirit / Shade Soul",
        (
            p("face", movement=2, ticks=2),
            p("cast", buttons=("cast",), ticks=2),
        ),
        "spell",
        availability_button="cast",
        soul_required=33,
    ),
    ActionCase(
        "scream_up",
        "spell",
        "up-directed Howling Wraiths / Abyss Shriek",
        (p("cast_up", aim=2, buttons=("cast",), ticks=2),),
        "spell",
        availability_button="cast",
        soul_required=33,
    ),
    ActionCase(
        "quake_down",
        "spell",
        "airborne down-directed Desolate Dive / Descending Dark",
        (
            p("jump", buttons=("jump_hold",), ticks=7),
            p("cast_down", aim=0, buttons=("cast",), ticks=2),
        ),
        "quake",
        availability_button="cast",
        soul_required=33,
    ),
    ActionCase(
        "focus_heal",
        "focus",
        "grounded focus hold through one completed heal",
        (
            p("focus", buttons=("focus_hold",), ticks=120),
            p("release", ticks=2),
        ),
        "focus",
        availability_button="focus_hold",
        soul_required=33,
        damage_required=True,
    ),
    ActionCase(
        "great_slash_left",
        "nail_art",
        "charge then release Great Slash facing left",
        (
            p("face", movement=0, ticks=2),
            p("charge", buttons=("nail_art_hold",), ticks=110),
            p("release", buttons=("nail_art_release",), ticks=2),
        ),
        "nail_art",
        availability_button="nail_art_hold",
    ),
    ActionCase(
        "great_slash_right",
        "nail_art",
        "charge then release Great Slash facing right",
        (
            p("face", movement=2, ticks=2),
            p("charge", buttons=("nail_art_hold",), ticks=110),
            p("release", buttons=("nail_art_release",), ticks=2),
        ),
        "nail_art",
        availability_button="nail_art_hold",
    ),
    ActionCase(
        "cyclone_slash_up",
        "nail_art",
        "charge then release Cyclone Slash with up held",
        (
            p("charge", buttons=("nail_art_hold",), ticks=110),
            p("select_up", aim=2, buttons=("nail_art_hold",)),
            p("release_up", aim=2, buttons=("nail_art_release",)),
            p("extend", aim=2, buttons=("attack",), ticks=4),
        ),
        "cyclone",
        availability_button="nail_art_hold",
    ),
    ActionCase(
        "cyclone_slash_down",
        "nail_art",
        "charge then release Cyclone Slash with down held",
        (
            p("charge", buttons=("nail_art_hold",), ticks=110),
            p("select_down", aim=0, buttons=("nail_art_hold",)),
            p("release_down", aim=0, buttons=("nail_art_release",)),
            p("extend", aim=0, buttons=("attack",), ticks=4),
        ),
        "cyclone",
        availability_button="nail_art_hold",
    ),
    ActionCase(
        "dash_slash_left",
        "nail_art",
        "charge then combine left dash with attack release",
        (
            p("charge", buttons=("nail_art_hold",), ticks=110),
            p(
                "dash_start",
                movement=0,
                buttons=("dash", "nail_art_hold"),
            ),
            p(
                "dash_release",
                movement=0,
                buttons=("nail_art_release",),
                ticks=2,
            ),
        ),
        "dash_nail_art",
        availability_button="nail_art_hold",
    ),
    ActionCase(
        "dash_slash_right",
        "nail_art",
        "charge then combine right dash with attack release",
        (
            p("charge", buttons=("nail_art_hold",), ticks=110),
            p(
                "dash_start",
                movement=2,
                buttons=("dash", "nail_art_hold"),
            ),
            p(
                "dash_release",
                movement=2,
                buttons=("nail_art_release",),
                ticks=2,
            ),
        ),
        "dash_nail_art",
        availability_button="nail_art_hold",
    ),
    combination_case(
        "jump_side_slash_left",
        expectation="jump_action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    combination_case(
        "jump_side_slash_right",
        expectation="jump_action_flag",
        expected_flag="attacking",
        availability_button="attack",
    ),
    combination_case(
        "jump_up_slash",
        expectation="jump_action_flag",
        expected_flag="up_attacking",
        availability_button="attack",
    ),
    combination_case(
        "jump_down_slash",
        expectation="jump_action_flag",
        expected_flag="down_attacking",
        availability_button="attack",
    ),
    combination_case(
        "jump_cyclone_up",
        expectation="jump_cyclone",
        availability_button="nail_art_hold",
    ),
    combination_case(
        "jump_cyclone_down",
        expectation="jump_cyclone",
        availability_button="nail_art_hold",
    ),
    duration_case(0, 1),
    duration_case(1, 2),
    duration_case(2, 4),
    duration_case(3, 8),
    *tuple(
        macro_case(index, name)
        for index, name in enumerate(
            (
                "approach",
                "retreat",
                "jump_attack",
                "pogo",
                "dash_away",
                "dash_through",
                "cast_forward",
                "cast_up",
                "focus_when_safe",
                "short_hop",
                "long_jump",
            )
        )
    ),
)


class ResourceUnavailable(RuntimeError):
    """A live-game prerequisite could not be obtained through ordinary input."""


class LiveActionExplorer:
    def __init__(
        self,
        env: HKRLEnv,
        *,
        reset_timeout_s: float,
        max_resource_steps: int,
    ) -> None:
        self.env = env
        self.reset_timeout_s = reset_timeout_s
        self.max_resource_steps = max_resource_steps
        self.obs: dict[str, np.ndarray] | None = None
        self.info: dict[str, Any] = {}
        self.running = False

    def reset(self) -> Snapshot:
        self.obs, self.info = self.env.reset(
            options={
                "reset_timeout_s": self.reset_timeout_s,
                "recv_timeout_s": 10.0,
            }
        )
        self.running = True
        snapshot = self.snapshot("reset")
        print("RESET " + json.dumps(asdict(snapshot), sort_keys=True), flush=True)
        return snapshot

    def run_case(self, case: ActionCase) -> CaseResult:
        snapshots: list[Snapshot] = []
        try:
            if not self.running:
                snapshots.append(self.reset())
            if case.grounded_start:
                self._settle_on_ground()
            self._ensure_resources(case.soul_required, case.damage_required)
            if case.expectation == "focus" or case.name == "macro_focus_when_safe":
                self._retreat_for_focus()
            if case.availability_button is not None:
                self._wait_for_button(case.availability_button)

            baseline = self.snapshot("baseline")
            snapshots.append(baseline)
            for phase in case.phases:
                snapshots.append(self._step_phase(phase))
                if not self.running:
                    break

            status, reason = verify_case(case, snapshots)
        except ResourceUnavailable as exc:
            status, reason = "skipped", str(exc)
        except Exception as exc:
            status, reason = "failed", f"{type(exc).__name__}: {exc}"

        result = CaseResult(case.name, case.family, status, reason, snapshots)
        print("CASE " + json.dumps(_result_dict(result), sort_keys=True), flush=True)
        return result

    def _step_phase(self, phase: ActionPhase) -> Snapshot:
        if not self.running:
            raise ResourceUnavailable("episode ended before phase execution")
        if not 1 <= phase.ticks <= MAX_SAFE_TICKS:
            raise ValueError(f"phase ticks must be in [1, {MAX_SAFE_TICKS}]")

        previous_repeat = self.env.task.action.action_repeat
        self.env.task.action.action_repeat = phase.ticks
        try:
            self.obs, _, terminated, truncated, self.info = self.env.step(
                phase.to_action()
            )
        finally:
            self.env.task.action.action_repeat = previous_repeat
        self.running = not (terminated or truncated)
        return self.snapshot(phase.label)

    def _step_raw(
        self,
        *,
        movement: int = 1,
        aim: int = 1,
        buttons: tuple[str, ...] = (),
        ticks: int = 1,
    ) -> Snapshot:
        return self._step_phase(
            p(
                "precondition",
                movement=movement,
                aim=aim,
                buttons=buttons,
                duration=0,
                ticks=ticks,
            )
        )

    def _settle_on_ground(self) -> None:
        stable_samples = 0
        for _ in range(50):
            player = self._player()
            if (
                bool(player[P_ON_GROUND])
                and not bool(player[P_DASHING])
                and abs(float(player[P_VY])) < 0.05
            ):
                stable_samples += 1
            else:
                stable_samples = 0
            if stable_samples >= 2:
                return
            self._step_raw(ticks=2)
            if not self.running:
                raise ResourceUnavailable("episode ended while waiting to land")
        raise ResourceUnavailable("Hero did not return to a stable grounded state")

    def _wait_for_button(self, name: str) -> None:
        # Attack and dash masks are expected to close briefly after an earlier
        # case. Wait through that ordinary game cooldown before deciding that an
        # ability is unavailable in the loaded save.
        for _ in range(30):
            if self._button_available(name):
                return
            self._step_raw(ticks=2)
            if not self.running:
                raise ResourceUnavailable(
                    f"episode ended while waiting for button:{name}"
                )
        raise ResourceUnavailable(
            f"action mask keeps button:{name} unavailable after cooldown wait"
        )

    def _ensure_resources(self, soul_required: int, damage_required: bool) -> None:
        if soul_required > 0 and int(self._player()[P_SOUL]) < soul_required:
            self._farm_soul(soul_required)
        if damage_required and int(self._player()[P_HP]) >= int(
            self._player()[P_MAX_HP]
        ):
            self._invite_natural_damage()
        if soul_required > 0 and int(self._player()[P_SOUL]) < soul_required:
            raise ResourceUnavailable(
                f"only {int(self._player()[P_SOUL])} soul; need {soul_required}"
            )
        if damage_required and int(self._player()[P_HP]) >= int(
            self._player()[P_MAX_HP]
        ):
            raise ResourceUnavailable("focus needs naturally received damage")

    def _farm_soul(self, target: int) -> None:
        for _ in range(self.max_resource_steps):
            player = self._player()
            if int(player[P_SOUL]) >= target:
                return
            boss = self._boss()
            if boss is None:
                raise ResourceUnavailable(
                    "Boss observation disappeared while farming soul"
                )
            if not self.running:
                raise ResourceUnavailable("episode ended while farming soul")

            dx = float(boss[6] - player[P_X])
            dy = float(boss[7] - player[P_Y])
            movement = 2 if dx > 0.4 else 0 if dx < -0.4 else 1
            if bool(player[P_ON_GROUND]):
                self._step_raw(
                    movement=movement,
                    buttons=("jump_hold",),
                    ticks=5,
                )
                continue

            aim = 2 if dy > 1.0 else 0 if dy < -1.0 else 1
            self._step_raw(
                movement=movement,
                aim=aim,
                buttons=("attack",),
                ticks=1,
            )
            if self.running:
                self._step_raw(movement=movement, ticks=1)

        raise ResourceUnavailable(
            f"could not earn {target} soul through ordinary attacks in "
            f"{self.max_resource_steps} control decisions"
        )

    def _invite_natural_damage(self) -> None:
        for _ in range(self.max_resource_steps):
            player = self._player()
            if int(player[P_HP]) < int(player[P_MAX_HP]):
                return
            boss = self._boss()
            if boss is None:
                raise ResourceUnavailable(
                    "Boss observation disappeared before focus test"
                )
            dx = float(boss[6] - player[P_X])
            movement = 2 if dx > 0.3 else 0 if dx < -0.3 else 1
            self._step_raw(movement=movement, ticks=3)
            if not self.running:
                raise ResourceUnavailable(
                    "episode ended before natural damage was observed"
                )
        raise ResourceUnavailable("Boss did not naturally damage the Hero in time")

    def _retreat_for_focus(self) -> None:
        """Create healing distance with ordinary movement/dash input."""
        boss = self._boss()
        if boss is None:
            raise ResourceUnavailable(
                "Boss observation disappeared before focus retreat"
            )
        player = self._player()
        movement = 0 if float(boss[6]) >= float(player[P_X]) else 2
        if self._button_available("dash"):
            self._step_raw(movement=movement, buttons=("dash",), ticks=2)
        if self.running:
            self._step_raw(movement=movement, ticks=18)
        if not self.running:
            raise ResourceUnavailable("episode ended during focus retreat")
        self._settle_on_ground()

    def _button_available(self, name: str) -> bool:
        mask = np.asarray(self.info.get("action_mask", []), dtype=bool)
        layout = action_mask_layout(
            enable_macro=self.env.task.action.enable_macro_actions,
            n_macros=self.env.task.action.n_macro_actions,
        )
        label = f"button:{name}"
        if label not in layout or mask.shape != (len(layout),):
            return False
        return bool(mask[layout.index(label)])

    def snapshot(self, phase: str) -> Snapshot:
        player = self._player()
        boss = self._boss()
        return Snapshot(
            phase=phase,
            server_tick=int(self.info.get("server_tick", 0)),
            player_x=_round(player[P_X]),
            player_y=_round(player[P_Y]),
            player_vx=_round(player[P_VX]),
            player_vy=_round(player[P_VY]),
            player_hp=int(player[P_HP]),
            player_max_hp=int(player[P_MAX_HP]),
            soul=int(player[P_SOUL]),
            on_ground=bool(player[P_ON_GROUND]),
            jumping=bool(player[P_JUMPING]),
            falling=bool(player[P_FALLING]),
            dashing=bool(player[P_DASHING]),
            focus_state=int(player[P_FOCUS_STATE]),
            action_flags=int(player[P_ACTION_FLAGS]),
            spell_fsm_state_hash=int(player[P_SPELL_FSM]),
            dream_nail_fsm_state_hash=int(player[P_DREAM_NAIL_FSM]),
            nail_arts_fsm_state_hash=int(player[P_NAIL_ARTS_FSM]),
            nail_charge_timer=_round(player[P_NAIL_CHARGE_TIMER]),
            applied_input_buttons=int(player[P_APPLIED_INPUT_BUTTONS]),
            boss_x=None if boss is None else _round(boss[6]),
            boss_y=None if boss is None else _round(boss[7]),
            boss_vx=None if boss is None else _round(boss[10]),
            boss_vy=None if boss is None else _round(boss[11]),
            boss_hp=None if boss is None else int(boss[12]),
            event_kinds=[
                event.kind.name for event in self.info.get("reward_events", [])
            ],
        )

    def _player(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("reset must complete before reading the player")
        player = np.asarray(self.obs["player"], dtype=np.float32)
        if player.shape[0] <= P_APPLIED_INPUT_BUTTONS:
            raise RuntimeError(
                "live mod lacks schema-v6 action telemetry; rebuild/install/restart"
            )
        return player

    def _boss(self) -> np.ndarray | None:
        if self.obs is None:
            return None
        mask = np.asarray(self.obs["entity_mask"], dtype=bool)
        for entity in np.asarray(self.obs["entities"])[mask]:
            if int(entity[1]) == int(protocol.EntityType.BOSS):
                return entity
        return None


def verify_case(
    case: ActionCase,
    snapshots: list[Snapshot],
) -> tuple[str, str]:
    if len(snapshots) < 2:
        return "failed", "no action phase produced an observation"
    if any("INVALID_ACTION" in snapshot.event_kinds for snapshot in snapshots):
        return "failed", "mod reported InvalidAction"
    baseline = snapshots[0]
    action_samples = snapshots[1:]

    if case.expectation == "move_left":
        delta = action_samples[-1].player_x - baseline.player_x
        return _threshold(delta < -0.05, f"dx={delta:.3f}")
    if case.expectation == "move_right":
        delta = action_samples[-1].player_x - baseline.player_x
        return _threshold(delta > 0.05, f"dx={delta:.3f}")
    if case.expectation == "jump":
        observed = any(
            sample.player_y > baseline.player_y + 0.05
            or sample.player_vy > 0.05
            or sample.jumping
            or sample.falling
            for sample in action_samples
        )
        return _threshold(observed, "airborne/jump state observed")
    if case.expectation == "double_jump":
        bit = 1 << PLAYER_ACTION_FLAG_BITS["double_jumping"]
        flagged = any(sample.action_flags & bit for sample in action_samples)
        second_rise = (
            len(action_samples) >= 3
            and action_samples[-1].player_vy > action_samples[-2].player_vy + 0.1
        )
        return _threshold(flagged or second_rise, "second upward impulse observed")
    if case.expectation == "dash":
        peak_speed = max(abs(sample.player_vx) for sample in action_samples)
        dash_state_seen = any(sample.dashing for sample in action_samples)
        # Ordinary aerial steering reaches about 8.3 units/s in the live game,
        # while a real dash reaches about 20. Requiring dash state or a clearly
        # dash-scale velocity prevents a jumping horizontal drift from passing.
        observed = dash_state_seen or peak_speed >= 15.0
        return _threshold(
            observed,
            f"dash_state={dash_state_seen}, peak_abs_vx={peak_speed:.3f}",
        )
    if case.expectation == "duration_hold":
        hold_count = case.expected_hold_steps
        if len(action_samples) != hold_count + 2:
            return (
                "failed",
                f"expected {hold_count} hold samples plus release/settle, "
                f"got {len(action_samples)} samples",
            )
        held_samples = action_samples[:hold_count]
        dream_nail_bit = 1 << BUTTON_BITS["dream_nail"]
        held = all(
            sample.applied_input_buttons & dream_nail_bit for sample in held_samples
        )
        released = all(
            not sample.applied_input_buttons & dream_nail_bit
            for sample in action_samples[hold_count:]
        )
        return _threshold(
            held and released,
            f"held for {hold_count} policy ticks, then explicit release",
        )
    if case.expectation == "action_flag":
        assert case.expected_flag is not None
        bit = 1 << PLAYER_ACTION_FLAG_BITS[case.expected_flag]
        observed = any(sample.action_flags & bit for sample in action_samples)
        return _threshold(observed, f"{case.expected_flag} flag observed")
    if case.expectation == "jump_action_flag":
        assert case.expected_flag is not None
        bit = 1 << PLAYER_ACTION_FLAG_BITS[case.expected_flag]
        airborne = any(
            not sample.on_ground or sample.jumping or sample.falling
            for sample in action_samples
        )
        action_seen = any(sample.action_flags & bit for sample in action_samples)
        return _threshold(
            airborne and action_seen,
            f"airborne={airborne}, {case.expected_flag}={action_seen}",
        )
    if case.expectation == "dream_nail":
        observed = any(
            sample.dream_nail_fsm_state_hash != baseline.dream_nail_fsm_state_hash
            for sample in action_samples
        )
        soul_gain = action_samples[-1].soul > baseline.soul
        return _threshold(
            observed or soul_gain, "Dream Nail FSM/soul response observed"
        )
    if case.expectation in {"spell", "quake"}:
        fsm_changed = any(
            sample.spell_fsm_state_hash != baseline.spell_fsm_state_hash
            for sample in action_samples
        )
        soul_spent = action_samples[-1].soul < baseline.soul
        quake_bit = 1 << PLAYER_ACTION_FLAG_BITS["spell_quake"]
        quake_seen = any(sample.action_flags & quake_bit for sample in action_samples)
        observed = (
            fsm_changed or soul_spent or (case.expectation == "quake" and quake_seen)
        )
        return _threshold(observed, "spell FSM/resource response observed")
    if case.expectation == "focus":
        fsm_changed = any(
            sample.spell_fsm_state_hash != baseline.spell_fsm_state_hash
            for sample in action_samples
        )
        focus_seen = any(sample.focus_state > 0 for sample in action_samples)
        healed = action_samples[-1].player_hp > baseline.player_hp
        heal_event = any("HEAL" in sample.event_kinds for sample in action_samples)
        soul_spent = action_samples[-1].soul < baseline.soul
        return _threshold(
            healed or heal_event,
            "completed heal observed "
            f"(fsm={fsm_changed}, focus={focus_seen}, soul_spent={soul_spent})",
        )
    if case.expectation in {"nail_art", "cyclone", "dash_nail_art"}:
        charge_bit = 1 << PLAYER_ACTION_FLAG_BITS["nail_charging"]
        cyclone_bit = 1 << PLAYER_ACTION_FLAG_BITS["nail_art_cyclone"]
        charged = any(
            sample.action_flags & charge_bit or sample.nail_charge_timer >= 0.5
            for sample in action_samples
        )
        art_changed = any(
            sample.nail_arts_fsm_state_hash != baseline.nail_arts_fsm_state_hash
            for sample in action_samples
        )
        cyclone_seen = any(
            sample.action_flags & cyclone_bit for sample in action_samples
        )
        released = art_changed
        if case.expectation == "cyclone":
            return _threshold(
                charged and released and cyclone_seen,
                "charge, Nail Arts FSM release, and cyclone flag observed",
            )
        if case.expectation == "dash_nail_art":
            peak_speed = max(abs(sample.player_vx) for sample in action_samples)
            dash_seen = (
                any(sample.dashing for sample in action_samples) or peak_speed >= 15.0
            )
            return _threshold(
                charged and released and dash_seen,
                "charge, dash state, and Nail Arts FSM release observed",
            )
        return _threshold(charged and released, "charge and nail-art release observed")
    if case.expectation == "jump_cyclone":
        charge_bit = 1 << PLAYER_ACTION_FLAG_BITS["nail_charging"]
        cyclone_bit = 1 << PLAYER_ACTION_FLAG_BITS["nail_art_cyclone"]
        jump_hold_bit = 1 << BUTTON_BITS["jump_hold"]
        nail_hold_bit = 1 << BUTTON_BITS["nail_art_hold"]
        charged = any(
            sample.action_flags & charge_bit or sample.nail_charge_timer >= 0.5
            for sample in action_samples
        )
        airborne = any(
            not sample.on_ground or sample.jumping or sample.falling
            for sample in action_samples
        )
        combined_hold = any(
            sample.applied_input_buttons & jump_hold_bit
            and sample.applied_input_buttons & nail_hold_bit
            for sample in action_samples
        )
        released = any(
            sample.nail_arts_fsm_state_hash != baseline.nail_arts_fsm_state_hash
            for sample in action_samples
        )
        cyclone_seen = any(
            sample.action_flags & cyclone_bit for sample in action_samples
        )
        return _threshold(
            charged and airborne and combined_hold and released and cyclone_seen,
            "charge, simultaneous jump+nail hold, airborne Cyclone release observed",
        )
    if case.expectation == "macro":
        macro_name = case.name.removeprefix("macro_")
        final = action_samples[-1]
        if macro_name in {"approach", "retreat"}:
            delta = final.player_x - baseline.player_x
            expected = delta > 0.05 if macro_name == "approach" else delta < -0.05
            return _threshold(expected, f"macro ground displacement dx={delta:.3f}")
        if macro_name == "jump_attack":
            attack_bit = 1 << PLAYER_ACTION_FLAG_BITS["attacking"]
            airborne = any(
                not sample.on_ground or sample.jumping or sample.falling
                for sample in action_samples
            )
            attacking = any(
                sample.action_flags & attack_bit for sample in action_samples
            )
            return _threshold(
                airborne and attacking,
                f"airborne={airborne}, attacking={attacking}",
            )
        if macro_name == "pogo":
            down_bit = 1 << PLAYER_ACTION_FLAG_BITS["down_attacking"]
            airborne = any(not sample.on_ground for sample in action_samples)
            down_attacking = any(
                sample.action_flags & down_bit for sample in action_samples
            )
            return _threshold(
                airborne and down_attacking,
                f"airborne={airborne}, down_attacking={down_attacking}",
            )
        if macro_name in {"dash_away", "dash_through"}:
            peak_speed = max(abs(sample.player_vx) for sample in action_samples)
            dashing = any(sample.dashing for sample in action_samples)
            expected_sign = (
                final.player_vx < 0
                if macro_name == "dash_away"
                else final.player_vx > 0
            )
            return _threshold(
                (dashing or peak_speed >= 15.0) and expected_sign,
                f"dashing={dashing}, peak_abs_vx={peak_speed:.3f}, final_vx={final.player_vx:.3f}",
            )
        if macro_name in {"cast_forward", "cast_up"}:
            fsm_changed = any(
                sample.spell_fsm_state_hash != baseline.spell_fsm_state_hash
                for sample in action_samples
            )
            soul_spent = final.soul < baseline.soul
            return _threshold(
                fsm_changed or soul_spent,
                f"spell_fsm_changed={fsm_changed}, soul_spent={soul_spent}",
            )
        if macro_name == "focus_when_safe":
            healed = final.player_hp > baseline.player_hp
            heal_event = any("HEAL" in sample.event_kinds for sample in action_samples)
            return _threshold(
                healed or heal_event,
                f"healed={healed}, heal_event={heal_event}",
            )
        if macro_name in {"short_hop", "long_jump"}:
            jump_bits = (1 << BUTTON_BITS["jump_tap"]) | (1 << BUTTON_BITS["jump_hold"])
            airborne = any(
                sample.player_y > baseline.player_y + 0.05 or not sample.on_ground
                for sample in action_samples
            )
            final_holds_jump = bool(final.applied_input_buttons & jump_bits)
            distinct_hold = (
                not final_holds_jump if macro_name == "short_hop" else final_holds_jump
            )
            return _threshold(
                airborne and distinct_hold,
                f"airborne={airborne}, final_holds_jump={final_holds_jump}",
            )
        return "failed", f"unknown macro semantic {macro_name}"
    return "failed", f"unknown expectation {case.expectation}"


def _threshold(passed: bool, reason: str) -> tuple[str, str]:
    return ("verified" if passed else "failed"), reason


def _result_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "family": result.family,
        "status": result.status,
        "reason": result.reason,
        "snapshots": [asdict(snapshot) for snapshot in result.snapshots],
    }


def _round(value: Any) -> float:
    return round(float(value), 4)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--task", default="configs/tasks/gruz_mother.yaml")
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--max-resource-steps", type=int, default=240)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case name to run; repeat to select a subset (default: all)",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="family to run; repeat to select several",
    )
    parser.add_argument(
        "--exclude-macros",
        action="store_true",
        help="omit the 11 configured bootstrap-macro cases",
    )
    parser.add_argument(
        "--reset-between-families",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reset-between-cases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="request a clean episode before each selected case",
    )
    parser.add_argument("--list", action="store_true", help="print catalog and exit")
    parser.add_argument("--output", help="write the complete JSON evidence bundle")
    parser.add_argument(
        "--fail-on-failed",
        action="store_true",
        help="return non-zero when a selected case fails (skips remain reported)",
    )
    return parser


def select_cases(
    *,
    names: list[str],
    families: list[str],
    exclude_macros: bool,
) -> list[ActionCase]:
    known_names = {case.name for case in ACTION_CASES}
    unknown = sorted(set(names) - known_names)
    if unknown:
        raise ValueError(f"unknown action cases: {', '.join(unknown)}")
    selected = [
        case
        for case in ACTION_CASES
        if (not names or case.name in names)
        and (not families or case.family in families)
        and not (exclude_macros and case.family == "macro")
    ]
    if not selected:
        raise ValueError("action-case selection is empty")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.reset_timeout <= 0.0:
        raise ValueError("--reset-timeout must be positive")
    if args.max_resource_steps <= 0:
        raise ValueError("--max-resource-steps must be positive")

    selected = select_cases(
        names=args.case,
        families=args.family,
        exclude_macros=args.exclude_macros,
    )
    if args.list:
        for case in selected:
            print(
                json.dumps(
                    {
                        "name": case.name,
                        "family": case.family,
                        "description": case.description,
                    },
                    sort_keys=True,
                )
            )
        return 0

    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    env = HKRLEnv(transport=make_transport(cfg), task=task)
    explorer = LiveActionExplorer(
        env,
        reset_timeout_s=args.reset_timeout,
        max_resource_steps=args.max_resource_steps,
    )
    results: list[CaseResult] = []
    current_family: str | None = None
    started = time.time()
    try:
        for case in selected:
            if not explorer.running or (
                results
                and (
                    args.reset_between_cases
                    or (
                        args.reset_between_families
                        and current_family is not None
                        and case.family != current_family
                    )
                )
            ):
                explorer.reset()
            current_family = case.family
            results.append(explorer.run_case(case))
    finally:
        env.close()

    counts = {
        status: sum(result.status == status for result in results)
        for status in ("verified", "observed", "skipped", "failed")
    }
    bundle = {
        "schema_version": protocol.SCHEMA_VERSION,
        "scope": (
            "semantic combat-action equivalence classes; ordinary player input only"
        ),
        "boss_mutation_allowed": False,
        "selected_cases": [case.name for case in selected],
        "elapsed_s": round(time.time() - started, 3),
        "counts": counts,
        "results": [_result_dict(result) for result in results],
    }
    print("SUMMARY " + json.dumps(bundle["counts"], sort_keys=True), flush=True)
    if args.output:
        output = Path(args.output).expanduser()
        if not str(output):
            raise ValueError("--output must not be empty")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"EVIDENCE {output.resolve()}", flush=True)

    return 1 if args.fail_on_failed and counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
