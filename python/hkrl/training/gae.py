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
    discount_exponents: np.ndarray | None = None,
    truncation_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` along the time axis.

    Bootstrap value is used on ``truncated`` steps but NOT on ``terminated`` ones
    (docs/distributed_training.md). Shapes are (T, N) with N parallel envs.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching shapes")
    if rewards.shape != truncateds.shape:
        raise ValueError("truncateds must match rewards shape")
    if discount_exponents is None:
        discount_exponents = np.ones_like(rewards, dtype=np.float32)
    if discount_exponents.shape != rewards.shape:
        raise ValueError("discount_exponents must match rewards shape")
    if truncation_values is not None and truncation_values.shape != rewards.shape:
        raise ValueError("truncation_values must match rewards shape")
    truncated_mask = np.asarray(truncateds, dtype=bool)
    if rewards.ndim != 2:
        raise ValueError("GAE inputs must have shape (T, N)")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError("last_value must have shape (N,)")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    for name, array in (
        ("rewards", rewards),
        ("values", values),
        ("last_value", last_value),
        ("discount_exponents", discount_exponents),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    if not (discount_exponents > 0.0).all():
        raise ValueError("discount_exponents must be positive")
    if truncation_values is not None and not np.isfinite(truncation_values[truncated_mask]).all():
        raise ValueError("truncation_values must be finite on truncated steps")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = last_value.astype(np.float32, copy=False)

    for step in range(rewards.shape[0] - 1, -1, -1):
        truncated = truncated_mask[step]
        terminated = np.logical_and(dones[step], np.logical_not(truncated))
        bootstrap_value = next_value
        if truncation_values is not None:
            bootstrap_value = np.where(
                truncated,
                truncation_values[step],
                bootstrap_value,
            )
        bootstrap_mask = 1.0 - terminated.astype(np.float32)
        trace_continuation = 1.0 - np.logical_or(terminated, truncated).astype(np.float32)
        step_gamma = np.power(gamma, discount_exponents[step]).astype(np.float32)
        step_lambda = np.power(gae_lambda, discount_exponents[step]).astype(np.float32)
        delta = rewards[step] + step_gamma * bootstrap_value * bootstrap_mask - values[step]
        last_gae = delta + step_gamma * step_lambda * trace_continuation * last_gae
        advantages[step] = last_gae
        next_value = values[step]

    returns = advantages + values
    return advantages, returns
