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
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching shapes")
    if rewards.shape != truncateds.shape:
        raise ValueError("truncateds must match rewards shape")
    if rewards.ndim != 2:
        raise ValueError("GAE inputs must have shape (T, N)")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError("last_value must have shape (N,)")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = last_value.astype(np.float32, copy=False)

    for step in range(rewards.shape[0] - 1, -1, -1):
        terminated = np.logical_and(dones[step], np.logical_not(truncateds[step]))
        next_non_terminal = 1.0 - terminated.astype(np.float32)
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
        next_value = values[step]

    returns = advantages + values
    return advantages, returns
