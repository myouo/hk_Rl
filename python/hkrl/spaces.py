"""Action/observation spaces + action-mask layout.

Implements docs/action_space.md and docs/observation_schema.md. The hybrid action
space and the canonical action-mask index order defined here MUST match the mod's
``ActionMasker`` / ``InputInjector`` (button bit layout below). Mismatch is the #1
cause of high invalid_action_ratio (docs/troubleshooting.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

# --- Button bit layout (mirror mod InputInjector + schema Action.buttons) -----
BUTTON_BITS: dict[str, int] = {
    "jump_tap": 0,
    "jump_hold": 1,
    "dash": 2,
    "attack": 3,
    "cast": 4,
    "focus_hold": 5,
    "dream_nail": 6,
    "nail_art_hold": 7,
    "nail_art_release": 8,
}
N_BUTTONS = len(BUTTON_BITS)

# Discrete component sizes.
N_MOVEMENT_X = 3  # left / neutral / right
N_AIM_Y = 3  # down / neutral / up
DURATION_TICKS: tuple[int, ...] = (1, 2, 4, 8)
N_DURATION = len(DURATION_TICKS)
DEFAULT_N_MACROS = 11
# Primitive ticks emitted by MacroActionScheduler for macro ids 0..10. Python's
# Gym action uses 0=no macro and 1..11 for these entries.
MACRO_DURATION_TICKS: tuple[int, ...] = (4, 4, 2, 5, 1, 1, 1, 1, 120, 1, 4)

# Normalization constants (docs/observation_schema.md §2). Tune per-arena.
ARENA_SCALE = 30.0
VEL_SCALE = 20.0
T_MAX = 2.0  # seconds, for clamping cooldown/lock/ttl/invuln timers

# Feature vector sizes used by decoded observations before model-specific
# embedding/splitting. Keep these aligned with docs/observation_schema.md.
GLOBAL_FEATURE_DIM = 9
GLOBAL_FEATURE_INDEX: dict[str, int] = {
    "scene_hash": 0,
    "arena_id": 1,
    "task_id": 2,
    "difficulty": 3,
    "time_in_episode": 4,
    "time_scale": 5,
    "fixed_delta_time": 6,
    "stage_index": 7,
    "episode_id": 8,
}
PLAYER_FEATURE_DIMS: dict[str, int] = {
    "privileged": 32,
    "reduced": 21,
    "human_visible": 15,
}
PLAYER_ACTION_FLAG_BITS: dict[str, int] = {
    "attacking": 0,
    "up_attacking": 1,
    "down_attacking": 2,
    "nail_charging": 3,
    "nail_art_cyclone": 4,
    "spell_quake": 5,
    "double_jumping": 6,
}
PLAYER_FEATURE_INDEX: dict[str, int] = {
    "pos_x": 0,
    "pos_y": 1,
    "vel_x": 2,
    "vel_y": 3,
    "hp": 4,
    "max_hp": 5,
    "soul": 6,
    "max_soul": 7,
    "facing": 8,
    "on_ground": 9,
    "wall_sliding": 10,
    "jumping": 11,
    "falling": 12,
    "dashing": 13,
    "shadow_dashing": 14,
    "invulnerable": 15,
    "invuln_timer": 16,
    "attack_lock_timer": 17,
    "cast_lock_timer": 18,
    "focus_state": 19,
    "dash_cooldown": 20,
    "double_jump_available": 21,
    "can_attack": 22,
    "can_cast": 23,
    "can_focus": 24,
    "actor_state_hash": 25,
    "action_flags": 26,
    "spell_fsm_state_hash": 27,
    "dream_nail_fsm_state_hash": 28,
    "nail_arts_fsm_state_hash": 29,
    "nail_charge_timer": 30,
    "applied_input_buttons": 31,
}
ENTITY_FEATURE_DIMS: dict[str, int] = {
    "privileged": 24,
    "reduced": 18,
    "human_visible": 12,
}
ENTITY_FEATURE_INDEX: dict[str, int] = {
    "entity_id": 0,
    "entity_type": 1,
    "team": 2,
    "prefab_hash": 3,
    "fsm_name_hash": 4,
    "fsm_state_hash": 5,
    "pos_x": 6,
    "pos_y": 7,
    "rel_x": 8,
    "rel_y": 9,
    "vel_x": 10,
    "vel_y": 11,
    "hp": 12,
    "max_hp": 13,
    "hurtbox_center_x": 14,
    "hurtbox_center_y": 15,
    "hurtbox_size_x": 16,
    "hurtbox_size_y": 17,
    "hitbox_active": 18,
    "damage": 19,
    "ttl": 20,
    "phase": 21,
    "threat_score": 22,
    "flags": 23,
}


def canonical_noop_action_values(*, enable_macro: bool) -> tuple[int, ...]:
    """Return the packed no-op used for recurrent context at episode start."""
    values = (1, 1, *(0 for _ in range(N_BUTTONS)), 0)
    return (*values, 0) if enable_macro else values


def make_action_space(
    enable_macro: bool = True, n_macros: int = DEFAULT_N_MACROS
) -> gym.spaces.Dict:
    """Build the hybrid action space (docs/action_space.md §2).

    Components: movement_x Discrete(3), aim_y Discrete(3), buttons MultiBinary(9),
    duration Discrete(4), macro Discrete(n_macros+1) when enabled.
    """
    if n_macros < 0:
        raise ValueError("n_macros must be non-negative")

    from gymnasium import spaces as gym_spaces

    space_dict: dict[str, gym_spaces.Space] = {
        "movement_x": gym_spaces.Discrete(N_MOVEMENT_X),
        "aim_y": gym_spaces.Discrete(N_AIM_Y),
        "buttons": gym_spaces.MultiBinary(N_BUTTONS),
        "duration": gym_spaces.Discrete(N_DURATION),
    }
    if enable_macro:
        space_dict["macro"] = gym_spaces.Discrete(n_macros + 1)

    return gym_spaces.Dict(space_dict)


def make_observation_space(max_entities: int = 64, tier: str = "privileged") -> gym.spaces.Dict:
    """Build the entity-list observation space for the given ablation tier.

    Keys: global, player, entities (max_entities x feat), entity_mask. Feature
    dims depend on ``tier`` (privileged/reduced/human_visible).
    """
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    if tier not in PLAYER_FEATURE_DIMS:
        known = ", ".join(sorted(PLAYER_FEATURE_DIMS))
        raise ValueError(f"unknown observation tier {tier!r}; expected one of: {known}")

    import numpy as np
    from gymnasium import spaces as gym_spaces

    return gym_spaces.Dict(
        {
            "global": gym_spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(GLOBAL_FEATURE_DIM,),
                dtype=np.float32,
            ),
            "player": gym_spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(PLAYER_FEATURE_DIMS[tier],),
                dtype=np.float32,
            ),
            "entities": gym_spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_entities, ENTITY_FEATURE_DIMS[tier]),
                dtype=np.float32,
            ),
            "entity_mask": gym_spaces.MultiBinary(max_entities),
        }
    )


def action_mask_layout(enable_macro: bool = True, n_macros: int = DEFAULT_N_MACROS) -> list[str]:
    """Canonical flat index order for ``StepResponse.action_mask`` (and policy heads).

    Order: movement_x(3), aim_y(3), buttons(9), duration(4), [macro(n_macros+1)].
    The mod's ``ActionMasker`` writes the mask in exactly this order.
    """
    if n_macros < 0:
        raise ValueError("n_macros must be non-negative")

    layout = [f"movement_x:{i}" for i in range(N_MOVEMENT_X)]
    layout += [f"aim_y:{i}" for i in range(N_AIM_Y)]
    layout += [f"button:{name}" for name in BUTTON_BITS]
    layout += [f"duration:{t}" for t in DURATION_TICKS]
    if enable_macro:
        layout += [f"macro:{i}" for i in range(n_macros + 1)]
    return layout


def action_execution_ticks(
    action: object,
    *,
    minimum_ticks: int,
    enable_macro: bool,
    n_macros: int,
) -> int:
    """Return ticks needed to finish one sampled primitive or macro action.

    A policy decision must not return before its duration/macro plan completes;
    otherwise later sampled actions are recorded even though the mod is still
    executing the previous one. The protocol field remains ``action_repeat``:
    this helper only chooses its per-decision value.
    """
    if minimum_ticks <= 0:
        raise ValueError("minimum_ticks must be positive")
    if n_macros < 0:
        raise ValueError("n_macros must be non-negative")
    if not isinstance(action, Mapping):
        return minimum_ticks

    if enable_macro:
        macro_value = action.get("macro")
        if macro_value is None and "macro_id" in action:
            macro_value = int(action["macro_id"]) + 1
        macro = 0 if macro_value is None else int(macro_value)
        if not 0 <= macro <= n_macros:
            raise ValueError(f"macro must be in [0, {n_macros}], got {macro}")
        if macro > 0:
            if macro > len(MACRO_DURATION_TICKS):
                raise ValueError(f"macro must be in [0, {n_macros}], got {macro}")
            return max(minimum_ticks, MACRO_DURATION_TICKS[macro - 1])

    duration_value = action.get("duration_idx", action.get("duration", 0))
    duration = 0 if duration_value is None else int(duration_value)
    if not 0 <= duration < N_DURATION:
        raise ValueError(f"duration must be in [0, {N_DURATION}), got {duration}")
    return max(minimum_ticks, DURATION_TICKS[duration])
