"""Numerical safety checks for learner update paths."""

from __future__ import annotations

import torch
from torch import Tensor


def require_finite_tensor(name: str, value: Tensor) -> None:
    """Reject NaN/Inf tensors before they can reach optimizer state."""
    require_finite_tensors(((name, value),))


def require_finite_tensors(values: tuple[tuple[str, Tensor], ...]) -> None:
    """Check several tensors with one device-to-host synchronization.

    The common success path transfers only one aggregate boolean. Individual
    results are inspected only after a failure so the exception still names
    every offending tensor.
    """
    if not values:
        return
    checks = torch.stack(tuple(torch.isfinite(value.detach()).all() for _, value in values))
    if bool(checks.all().item()):
        return
    failed = [
        name
        for (name, _), is_finite in zip(values, checks.unbind(), strict=True)
        if not bool(is_finite.item())
    ]
    raise ValueError(f"{', '.join(failed)} contains non-finite values")


__all__ = ["require_finite_tensor", "require_finite_tensors"]
