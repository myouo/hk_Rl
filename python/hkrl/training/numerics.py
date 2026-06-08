"""Numerical safety checks for learner update paths."""

from __future__ import annotations

import torch
from torch import Tensor


def require_finite_tensor(name: str, value: Tensor) -> None:
    """Reject NaN/Inf tensors before they can reach optimizer state."""
    if not bool(torch.isfinite(value.detach()).all().cpu().item()):
        raise ValueError(f"{name} contains non-finite values")
