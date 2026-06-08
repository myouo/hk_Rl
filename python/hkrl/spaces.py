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


def make_action_space(enable_macro: bool = True, n_macros: int = 11) -> gym.spaces.Dict:
    """Build the hybrid action space (docs/action_space.md §2).

    Components: movement_x Discrete(3), aim_y Discrete(3), buttons MultiBinary(9),
    duration Discrete(4), macro Discrete(n_macros+1) when enabled.

    TODO(phase-2): construct gymnasium.spaces.Dict; import gymnasium lazily.
    """
    raise NotImplementedError


def make_observation_space(max_entities: int = 64, tier: str = "privileged") -> gym.spaces.Dict:
    """Build the entity-list observation space for the given ablation tier.

    Keys: global, player, entities (max_entities x feat), entity_mask. Feature
    dims depend on ``tier`` (privileged/reduced/human_visible).

    TODO(phase-2/phase-4): construct spaces; feat dims from schema field sets.
    """
    raise NotImplementedError


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
