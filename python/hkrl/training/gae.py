"""Generalized Advantage Estimation (Schulman et al., 2016).

Shared by all algorithms. Handles ``truncated`` (bootstrap) vs ``terminated``
(no bootstrap) correctly, and per-sequence resets for the recurrent case.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    truncateds: np.ndarray,
    last_value: np.ndarray,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` along the time axis.

    Bootstrap value is used on ``truncated`` steps but NOT on ``terminated`` ones
    (docs/distributed_training.md). Shapes are (T, N) with N parallel envs.

    TODO(phase-3): implement the standard reverse recursion.
    """
    raise NotImplementedError
