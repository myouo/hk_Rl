"""Checkpoint payload validation shared by registry and workers."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch


def validate_checkpoint_payload(payload: object) -> dict[str, object]:
    """Return a loadable checkpoint payload after structural/numeric checks."""
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")

    validate_model_state_dict(payload.get("model_state_dict"), name="model_state_dict")

    policy_version = payload.get("policy_version")
    if policy_version is not None:
        _validate_non_negative_int("policy_version", policy_version)

    return payload


def validate_model_state_dict(
    state: object,
    *,
    name: str = "model_state_dict",
) -> Mapping[Any, Any]:
    """Return a state-dict mapping after checking tensor structure and finiteness."""
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint missing {name}")
    _validate_state_mapping(name, state)
    return state


def _validate_state_mapping(name: str, state: Mapping[Any, Any]) -> None:
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        field = f"{name}.{key}"
        if isinstance(value, torch.Tensor):
            _validate_tensor(field, value)
        elif isinstance(value, Mapping):
            _validate_state_mapping(field, value)
        else:
            raise ValueError(f"{field} must be a tensor")


def _validate_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all().cpu().item()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"checkpoint {name} must be an integer")
    if int(value) < 0:
        raise ValueError(f"checkpoint {name} must be non-negative")
