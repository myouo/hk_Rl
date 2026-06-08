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
    """Simple heuristic: approach + attack, dodge on incoming hit. Sanity baseline.

    TODO(phase-3): basic rule set over decoded observation fields.
    """

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        raise NotImplementedError
