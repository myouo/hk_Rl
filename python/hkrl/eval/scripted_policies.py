"""Scripted / random policies for smoke tests and baselines (PRD Phase 2).

A random policy (respecting the action mask) is the first thing that must run
1000 episodes without crashing (PRD §2.1 MVP). Scripted policies provide a
sanity baseline above random.
"""

from __future__ import annotations

from typing import Any


class RandomPolicy:
    """Samples valid actions uniformly under the action mask."""

    def __init__(self, action_space: Any, seed: int = 0) -> None:
        self.action_space = action_space
        self.seed = seed
        # TODO(phase-2): seeded RNG.

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        """Return a masked-uniform random action.

        TODO(phase-2): sample each component, respecting the mask layout.
        """
        raise NotImplementedError


class ScriptedAggroPolicy:
    """Simple heuristic: approach + attack, dodge on incoming hit. Sanity baseline.

    TODO(phase-3): basic rule set over decoded observation fields.
    """

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        raise NotImplementedError
