"""Versioned semantic combinations over the existing factorized action space.

The policy still predicts movement, aim, buttons, duration, and optional macro
components independently.  This module is a side catalog for live validation,
coverage logging, and optional curriculum data collection; it is deliberately
not another policy action dimension.

Availability is represented as a Python integer bitset.  The catalog is small
and immutable, so deriving the bitset is O(number of catalog entries) with no
wire payload or per-step list/string allocation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol

from hkrl.spaces import (
    BUTTON_BITS,
    DURATION_TICKS,
    N_AIM_Y,
    N_BUTTONS,
    N_DURATION,
    N_MOVEMENT_X,
    PLAYER_FEATURE_INDEX,
    action_mask_layout,
)

COMBINATION_CATALOG_VERSION = 1
StartCondition = Literal["any", "grounded", "airborne"]


class _Indexable(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


@dataclass(frozen=True)
class CombinationPhase:
    """One primitive phase in a semantic, potentially temporal combination."""

    movement_x: int = 1
    aim_y: int = 1
    buttons: tuple[str, ...] = ()
    duration: int = 0
    ticks: int = 1

    def to_action(self, *, enable_macro: bool) -> dict[str, object]:
        action: dict[str, object] = {
            "movement_x": self.movement_x,
            "aim_y": self.aim_y,
            "buttons": {name: True for name in self.buttons},
            "duration": self.duration,
        }
        if enable_macro:
            action["macro"] = 0
        return action


@dataclass(frozen=True)
class ActionCombination:
    """Stable catalog entry describing a meaningful primitive composition."""

    combo_id: int
    name: str
    family: str
    description: str
    phases: tuple[CombinationPhase, ...]
    start_condition: StartCondition = "any"
    min_soul: int = 0


def _phase(
    *,
    movement_x: int = 1,
    aim_y: int = 1,
    buttons: tuple[str, ...] = (),
    duration: int = 0,
    ticks: int = 1,
) -> CombinationPhase:
    return CombinationPhase(movement_x, aim_y, buttons, duration, ticks)


ACTION_COMBINATIONS: tuple[ActionCombination, ...] = (
    ActionCombination(
        0,
        "moving_slash_left",
        "movement_attack",
        "move left while performing an ordinary side slash",
        (_phase(movement_x=0, buttons=("attack",)),),
        "grounded",
    ),
    ActionCombination(
        1,
        "moving_slash_right",
        "movement_attack",
        "move right while performing an ordinary side slash",
        (_phase(movement_x=2, buttons=("attack",)),),
        "grounded",
    ),
    ActionCombination(
        2,
        "jump_side_slash_left",
        "jump_attack",
        "jump left, then perform an aerial side slash",
        (
            _phase(movement_x=0, buttons=("jump_hold",), ticks=4),
            _phase(movement_x=0, buttons=("attack",)),
        ),
        "grounded",
    ),
    ActionCombination(
        3,
        "jump_side_slash_right",
        "jump_attack",
        "jump right, then perform an aerial side slash",
        (
            _phase(movement_x=2, buttons=("jump_hold",), ticks=4),
            _phase(movement_x=2, buttons=("attack",)),
        ),
        "grounded",
    ),
    ActionCombination(
        4,
        "jump_up_slash",
        "jump_attack",
        "jump, then combine up aim with an aerial slash",
        (
            _phase(buttons=("jump_hold",), ticks=4),
            _phase(aim_y=2, buttons=("attack",)),
        ),
        "grounded",
    ),
    ActionCombination(
        5,
        "jump_down_slash",
        "jump_attack",
        "jump, then combine down aim with an aerial slash/pogo input",
        (
            _phase(buttons=("jump_hold",), ticks=6),
            _phase(aim_y=0, buttons=("attack",)),
        ),
        "grounded",
    ),
    ActionCombination(
        6,
        "air_dash_left",
        "jump_dash",
        "jump left, then air dash left",
        (
            _phase(movement_x=0, buttons=("jump_hold",), ticks=4),
            _phase(movement_x=0, buttons=("dash",), ticks=2),
        ),
        "grounded",
    ),
    ActionCombination(
        7,
        "air_dash_right",
        "jump_dash",
        "jump right, then air dash right",
        (
            _phase(movement_x=2, buttons=("jump_hold",), ticks=4),
            _phase(movement_x=2, buttons=("dash",), ticks=2),
        ),
        "grounded",
    ),
    ActionCombination(
        8,
        "jump_fireball_left",
        "jump_spell",
        "jump left, face left, and cast a horizontal spell",
        (
            _phase(movement_x=0, buttons=("jump_hold",), ticks=4),
            _phase(movement_x=0, buttons=("cast",), ticks=2),
        ),
        "grounded",
        33,
    ),
    ActionCombination(
        9,
        "jump_fireball_right",
        "jump_spell",
        "jump right, face right, and cast a horizontal spell",
        (
            _phase(movement_x=2, buttons=("jump_hold",), ticks=4),
            _phase(movement_x=2, buttons=("cast",), ticks=2),
        ),
        "grounded",
        33,
    ),
    ActionCombination(
        10,
        "jump_scream_up",
        "jump_spell",
        "jump, then combine up aim with Howling Wraiths/Abyss Shriek",
        (
            _phase(buttons=("jump_hold",), ticks=4),
            _phase(aim_y=2, buttons=("cast",), ticks=2),
        ),
        "grounded",
        33,
    ),
    ActionCombination(
        11,
        "jump_quake_down",
        "jump_spell",
        "jump, then combine down aim with Desolate Dive/Descending Dark",
        (
            _phase(buttons=("jump_hold",), ticks=6),
            _phase(aim_y=0, buttons=("cast",), ticks=2),
        ),
        "grounded",
        33,
    ),
    ActionCombination(
        12,
        "jump_great_slash_left",
        "jump_nail_art",
        "charge, jump left while holding, then release Great Slash",
        (
            _phase(buttons=("nail_art_hold",), ticks=110),
            _phase(movement_x=0, buttons=("jump_hold", "nail_art_hold"), ticks=4),
            _phase(movement_x=0, buttons=("nail_art_release",), ticks=2),
        ),
        "grounded",
    ),
    ActionCombination(
        13,
        "jump_great_slash_right",
        "jump_nail_art",
        "charge, jump right while holding, then release Great Slash",
        (
            _phase(buttons=("nail_art_hold",), ticks=110),
            _phase(movement_x=2, buttons=("jump_hold", "nail_art_hold"), ticks=4),
            _phase(movement_x=2, buttons=("nail_art_release",), ticks=2),
        ),
        "grounded",
    ),
    ActionCombination(
        14,
        "jump_cyclone_up",
        "jump_nail_art",
        "charge, jump while holding, then release and extend Cyclone Slash with up",
        (
            _phase(buttons=("nail_art_hold",), ticks=110),
            _phase(buttons=("jump_hold", "nail_art_hold"), ticks=6),
            _phase(aim_y=2, buttons=("nail_art_release",)),
            _phase(aim_y=2, buttons=("attack",), ticks=4),
        ),
        "grounded",
    ),
    ActionCombination(
        15,
        "jump_cyclone_down",
        "jump_nail_art",
        "charge, jump while holding, then release and extend Cyclone Slash with down",
        (
            _phase(buttons=("nail_art_hold",), ticks=110),
            _phase(buttons=("jump_hold", "nail_art_hold"), ticks=6),
            _phase(aim_y=0, buttons=("nail_art_release",)),
            _phase(aim_y=0, buttons=("attack",), ticks=4),
        ),
        "grounded",
    ),
    ActionCombination(
        16,
        "dash_slash_left",
        "dash_nail_art",
        "charge, dash left while holding, then release Dash Slash",
        (
            _phase(buttons=("nail_art_hold",), ticks=110),
            _phase(movement_x=0, buttons=("dash", "nail_art_hold")),
            _phase(movement_x=0, buttons=("nail_art_release",), ticks=2),
        ),
        "grounded",
    ),
    ActionCombination(
        17,
        "dash_slash_right",
        "dash_nail_art",
        "charge, dash right while holding, then release Dash Slash",
        (
            _phase(buttons=("nail_art_hold",), ticks=110),
            _phase(movement_x=2, buttons=("dash", "nail_art_hold")),
            _phase(movement_x=2, buttons=("nail_art_release",), ticks=2),
        ),
        "grounded",
    ),
)

_COMBINATION_BY_NAME = {combo.name: combo for combo in ACTION_COMBINATIONS}


def get_action_combination(name: str) -> ActionCombination:
    """Return one immutable catalog entry by name."""

    try:
        return _COMBINATION_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown action combination {name!r}") from exc


def startable_combination_bits(
    action_mask: _Indexable,
    *,
    enable_macro: bool,
    n_macros: int,
    player_state: _Indexable | None = None,
) -> int:
    """Return a compact bitset of combinations startable in the current state.

    This is a conservative start check, not a promise that a multi-tick option
    will remain legal: every later phase must still obey the live action mask.
    """

    expected_mask_length = N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION
    if enable_macro:
        expected_mask_length += n_macros + 1
    if len(action_mask) != expected_mask_length:
        raise ValueError(
            f"action_mask length {len(action_mask)} != expected {expected_mask_length}"
        )

    grounded: bool | None = None
    soul: int | None = None
    if player_state is not None:
        on_ground_index = PLAYER_FEATURE_INDEX["on_ground"]
        soul_index = PLAYER_FEATURE_INDEX["soul"]
        if len(player_state) <= max(on_ground_index, soul_index):
            raise ValueError("player_state is too short for combination availability")
        grounded = bool(player_state[on_ground_index])
        soul = int(float(player_state[soul_index]))

    bits = 0
    requirements = _required_mask_indices(enable_macro, n_macros)
    for combo, indices in zip(ACTION_COMBINATIONS, requirements, strict=True):
        if grounded is not None:
            if combo.start_condition == "grounded" and not grounded:
                continue
            if combo.start_condition == "airborne" and grounded:
                continue
        if soul is not None and soul < combo.min_soul:
            continue
        if all(bool(action_mask[index]) for index in indices):
            bits |= 1 << combo.combo_id
    return bits


def decode_combination_bits(bits: int) -> tuple[ActionCombination, ...]:
    """Decode a non-negative availability bitset without allocating names."""

    if bits < 0:
        raise ValueError("combination bits must be non-negative")
    known_mask = (1 << len(ACTION_COMBINATIONS)) - 1
    if bits & ~known_mask:
        raise ValueError("combination bits contain unknown catalog ids")
    return tuple(combo for combo in ACTION_COMBINATIONS if bits & (1 << combo.combo_id))


def catalog_records() -> tuple[dict[str, object], ...]:
    """Return a JSON-ready catalog for diagnostics and UI discovery."""

    return tuple(
        {
            "combo_id": combo.combo_id,
            "name": combo.name,
            "family": combo.family,
            "description": combo.description,
            "start_condition": combo.start_condition,
            "min_soul": combo.min_soul,
            "phases": tuple(
                {
                    "movement_x": phase.movement_x,
                    "aim_y": phase.aim_y,
                    "buttons": phase.buttons,
                    "duration": phase.duration,
                    "ticks": phase.ticks,
                }
                for phase in combo.phases
            ),
        }
        for combo in ACTION_COMBINATIONS
    )


class CombinationCoverageBandit:
    """Tiny UCB1 selector for optional smoke/curriculum coverage collection.

    PPO rollouts must continue sampling from their own policy distribution.
    This helper is intended only for separately labelled scripted collection or
    live validation, where it avoids repeatedly testing already-covered motifs.
    """

    def __init__(self) -> None:
        self._counts = [0] * len(ACTION_COMBINATIONS)
        self._score_sums = [0.0] * len(ACTION_COMBINATIONS)
        self._total = 0

    def observe(self, combo_id: int, score: float) -> None:
        if not 0 <= combo_id < len(ACTION_COMBINATIONS):
            raise ValueError(f"combo_id must be in [0, {len(ACTION_COMBINATIONS)})")
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        self._counts[combo_id] += 1
        self._score_sums[combo_id] += float(score)
        self._total += 1

    def select(
        self,
        startable_bits: int,
        *,
        rng: random.Random,
        exploration: float = 1.0,
    ) -> ActionCombination:
        if exploration < 0.0 or not math.isfinite(exploration):
            raise ValueError("exploration must be finite and non-negative")
        candidates = decode_combination_bits(startable_bits)
        if not candidates:
            raise ValueError("no startable action combinations")

        unseen = [combo for combo in candidates if self._counts[combo.combo_id] == 0]
        if unseen:
            return rng.choice(unseen)

        log_total = math.log(max(self._total, 1))
        scored: list[tuple[float, ActionCombination]] = []
        for combo in candidates:
            count = self._counts[combo.combo_id]
            mean = self._score_sums[combo.combo_id] / count
            bonus = exploration * math.sqrt(log_total / count)
            scored.append((mean + bonus, combo))
        best_score = max(score for score, _ in scored)
        tied = [combo for score, combo in scored if abs(score - best_score) <= 1.0e-12]
        return rng.choice(tied)

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(self._counts)


@lru_cache(maxsize=4)
def _required_mask_indices(
    enable_macro: bool,
    n_macros: int,
) -> tuple[tuple[int, ...], ...]:
    layout = action_mask_layout(enable_macro=enable_macro, n_macros=n_macros)
    index_by_label = {label: index for index, label in enumerate(layout)}
    requirements: list[tuple[int, ...]] = []
    for combo in ACTION_COMBINATIONS:
        labels: set[str] = set()
        for phase in combo.phases:
            labels.add(f"movement_x:{phase.movement_x}")
            labels.add(f"aim_y:{phase.aim_y}")
            labels.add(f"duration:{DURATION_TICKS[phase.duration]}")
            labels.update(f"button:{name}" for name in phase.buttons)
        if enable_macro:
            labels.add("macro:0")
        requirements.append(tuple(sorted(index_by_label[label] for label in labels)))
    return tuple(requirements)


def _validate_catalog() -> None:
    ids = [combo.combo_id for combo in ACTION_COMBINATIONS]
    names = [combo.name for combo in ACTION_COMBINATIONS]
    if ids != list(range(len(ACTION_COMBINATIONS))):
        raise RuntimeError("action combination ids must be contiguous and append-only")
    if len(names) != len(set(names)):
        raise RuntimeError("action combination names must be unique")
    for combo in ACTION_COMBINATIONS:
        if not combo.phases:
            raise RuntimeError(f"action combination {combo.name!r} has no phases")
        for phase in combo.phases:
            if phase.movement_x not in (0, 1, 2):
                raise RuntimeError(f"{combo.name}: invalid movement_x")
            if phase.aim_y not in (0, 1, 2):
                raise RuntimeError(f"{combo.name}: invalid aim_y")
            if not 0 <= phase.duration < len(DURATION_TICKS):
                raise RuntimeError(f"{combo.name}: invalid duration")
            if phase.ticks <= 0:
                raise RuntimeError(f"{combo.name}: invalid phase ticks")
            if not set(phase.buttons) <= set(BUTTON_BITS):
                raise RuntimeError(f"{combo.name}: unknown button")


_validate_catalog()

__all__ = [
    "ACTION_COMBINATIONS",
    "COMBINATION_CATALOG_VERSION",
    "ActionCombination",
    "CombinationCoverageBandit",
    "CombinationPhase",
    "catalog_records",
    "decode_combination_bits",
    "get_action_combination",
    "startable_combination_bits",
]
