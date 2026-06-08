"""Scripted / random policies for smoke tests and baselines (PRD Phase 2).

A random policy (respecting the action mask) is the first thing that must run
1000 episodes without crashing (PRD §2.1 MVP). Scripted policies provide a
sanity baseline above random.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hkrl import spaces


class RandomPolicy:
    """Samples valid actions uniformly under the action mask."""

    def __init__(self, action_space: Any, seed: int = 0) -> None:
        self.action_space = action_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if hasattr(action_space, "seed"):
            action_space.seed(seed)

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        """Return a masked-uniform random action.

        ``action_mask`` follows ``hkrl.spaces.action_mask_layout``. Button mask
        entries mean "pressing this button is allowed"; unpressed is always valid.
        """
        del obs

        if action_mask is None:
            return self.action_space.sample()

        enable_macro = "macro" in self.action_space.spaces
        n_macros = int(self.action_space["macro"].n - 1) if enable_macro else 0
        layout = spaces.action_mask_layout(enable_macro=enable_macro, n_macros=n_macros)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)

        if len(mask) != len(layout):
            raise ValueError(
                f"action_mask length {len(mask)} does not match layout length {len(layout)}"
            )

        offset = 0
        movement_x = self._sample_discrete(
            mask[offset : offset + spaces.N_MOVEMENT_X], "movement_x"
        )
        offset += spaces.N_MOVEMENT_X

        aim_y = self._sample_discrete(mask[offset : offset + spaces.N_AIM_Y], "aim_y")
        offset += spaces.N_AIM_Y

        button_mask = mask[offset : offset + spaces.N_BUTTONS]
        buttons = self.rng.integers(0, 2, size=spaces.N_BUTTONS, dtype=np.int8)
        buttons = np.where(button_mask, buttons, 0).astype(np.int8, copy=False)
        offset += spaces.N_BUTTONS

        duration = self._sample_discrete(mask[offset : offset + spaces.N_DURATION], "duration")
        offset += spaces.N_DURATION

        action: dict[str, Any] = {
            "movement_x": movement_x,
            "aim_y": aim_y,
            "buttons": buttons,
            "duration": duration,
        }

        if enable_macro:
            action["macro"] = self._sample_discrete(mask[offset:], "macro")

        return action

    def _sample_discrete(self, mask: np.ndarray, name: str) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise ValueError(f"action_mask has no valid entries for {name}")
        return int(self.rng.choice(valid))


class ScriptedAggroPolicy:
    """Simple heuristic: approach + attack, dodge on incoming hit. Sanity baseline."""

    def __init__(
        self,
        action_space: Any,
        *,
        attack_range: float = 0.25,
        approach_deadzone: float = 0.10,
        vertical_deadzone: float = 0.20,
    ) -> None:
        self.action_space = action_space
        self.attack_range = attack_range
        self.approach_deadzone = approach_deadzone
        self.vertical_deadzone = vertical_deadzone

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        mask = _validated_action_mask(self.action_space, action_mask)
        rel_x, rel_y = _target_relative_position(obs)

        offset = 0
        movement_x = _choose_discrete(
            mask[offset : offset + spaces.N_MOVEMENT_X],
            preferred=_movement_from_rel_x(rel_x, self.approach_deadzone),
            fallback=1,
            name="movement_x",
        )
        offset += spaces.N_MOVEMENT_X

        aim_y = _choose_discrete(
            mask[offset : offset + spaces.N_AIM_Y],
            preferred=_aim_from_rel_y(rel_y, self.vertical_deadzone),
            fallback=1,
            name="aim_y",
        )
        offset += spaces.N_AIM_Y

        button_mask = mask[offset : offset + spaces.N_BUTTONS]
        buttons = np.zeros((spaces.N_BUTTONS,), dtype=np.int8)
        attack_idx = spaces.BUTTON_BITS["attack"]
        if abs(rel_x) <= self.attack_range and button_mask[attack_idx]:
            buttons[attack_idx] = 1
        offset += spaces.N_BUTTONS

        duration = _choose_discrete(
            mask[offset : offset + spaces.N_DURATION],
            preferred=1,
            fallback=0,
            name="duration",
        )
        offset += spaces.N_DURATION

        action: dict[str, Any] = {
            "movement_x": movement_x,
            "aim_y": aim_y,
            "buttons": buttons,
            "duration": duration,
        }
        if "macro" in self.action_space.spaces:
            action["macro"] = _choose_discrete(mask[offset:], preferred=0, fallback=0, name="macro")
        return action


def _validated_action_mask(action_space: Any, action_mask: Any | None) -> np.ndarray:
    enable_macro = "macro" in action_space.spaces
    n_macros = int(action_space["macro"].n - 1) if enable_macro else 0
    layout = spaces.action_mask_layout(enable_macro=enable_macro, n_macros=n_macros)

    if action_mask is None:
        return np.ones((len(layout),), dtype=bool)

    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    if len(mask) != len(layout):
        raise ValueError(
            f"action_mask length {len(mask)} does not match layout length {len(layout)}"
        )
    return mask


def _choose_discrete(mask: np.ndarray, *, preferred: int, fallback: int, name: str) -> int:
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        raise ValueError(f"action_mask has no valid entries for {name}")
    if preferred in valid:
        return int(preferred)
    if fallback in valid:
        return int(fallback)
    return int(valid[0])


def _target_relative_position(obs: Any) -> tuple[float, float]:
    if not isinstance(obs, dict) or "entities" not in obs:
        return 0.0, 0.0

    entities = np.asarray(obs["entities"], dtype=np.float32)
    if entities.ndim != 2 or entities.shape[0] == 0:
        return 0.0, 0.0

    if "entity_mask" in obs:
        mask = np.asarray(obs["entity_mask"], dtype=bool).reshape(-1)
        if len(mask) != entities.shape[0]:
            raise ValueError(
                f"entity_mask length {len(mask)} != entities length {entities.shape[0]}"
            )
    else:
        mask = np.ones((entities.shape[0],), dtype=bool)

    valid = entities[mask]
    if valid.size == 0:
        return 0.0, 0.0

    target_pool = valid
    if valid.shape[1] > 1:
        bosses = valid[valid[:, 1] == 1.0]
        if len(bosses) > 0:
            target_pool = bosses

    rel_x_idx, rel_y_idx = (8, 9) if target_pool.shape[1] > 9 else (6, 7)
    rel = target_pool[:, [rel_x_idx, rel_y_idx]]
    target = rel[np.argmin(np.linalg.norm(rel, axis=1))]
    return float(target[0]), float(target[1])


def _movement_from_rel_x(rel_x: float, deadzone: float) -> int:
    if rel_x < -deadzone:
        return 0
    if rel_x > deadzone:
        return 2
    return 1


def _aim_from_rel_y(rel_y: float, deadzone: float) -> int:
    if rel_y < -deadzone:
        return 0
    if rel_y > deadzone:
        return 2
    return 1
