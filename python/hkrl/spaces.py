"""Action/observation spaces + action-mask layout.

Implements docs/action_space.md and docs/observation_schema.md. The hybrid action
space and the canonical action-mask index order defined here MUST match the mod's
``ActionMasker`` / ``InputInjector`` (button bit layout below). Mismatch is the #1
cause of high invalid_action_ratio (docs/troubleshooting.md).
"""

from __future__ import annotations

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

# Normalization constants (docs/observation_schema.md §2). Tune per-arena.
ARENA_SCALE = 30.0
VEL_SCALE = 20.0
T_MAX = 2.0  # seconds, for clamping cooldown/lock/ttl/invuln timers

# Feature vector sizes used by decoded observations before model-specific
# embedding/splitting. Keep these aligned with docs/observation_schema.md.
GLOBAL_FEATURE_DIM = 9
PLAYER_FEATURE_DIMS: dict[str, int] = {
    "privileged": 25,
    "reduced": 21,
    "human_visible": 15,
}
ENTITY_FEATURE_DIMS: dict[str, int] = {
    "privileged": 24,
    "reduced": 18,
    "human_visible": 12,
}


def make_action_space(enable_macro: bool = True, n_macros: int = 11) -> gym.spaces.Dict:
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


def action_mask_layout(enable_macro: bool = True, n_macros: int = 11) -> list[str]:
    """Canonical flat index order for ``StepResponse.action_mask`` (and policy heads).

    Order: movement_x(3), aim_y(3), buttons(9), duration(4), [macro(n_macros+1)].
    The mod's ``ActionMasker`` writes the mask in exactly this order.
    """
    layout = [f"movement_x:{i}" for i in range(N_MOVEMENT_X)]
    layout += [f"aim_y:{i}" for i in range(N_AIM_Y)]
    layout += [f"button:{name}" for name in BUTTON_BITS]
    layout += [f"duration:{t}" for t in DURATION_TICKS]
    if enable_macro:
        layout += [f"macro:{i}" for i in range(n_macros + 1)]
    return layout
