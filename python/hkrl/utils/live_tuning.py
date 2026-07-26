"""Validated runtime-tuning snapshots shared by learner, registry, and workers.

Live tuning is deliberately narrower than ``TrainConfig``. Only parameters
whose implementation can change at a rollout/update boundary without changing
tensor, action, observation, or protocol layouts are accepted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hkrl.utils.config import RewardWeights, TrainConfig

LIVE_TUNING_REQUEST = "live_tuning.json"
LIVE_TUNING_STATUS = "live_tuning_status.json"


class _StrictTuningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RewardTuning(_StrictTuningModel):
    """Optional overrides for Python-side scalar reward composition."""

    boss_damage: float | None = None
    player_damage: float | None = None
    soul_gained: float | None = None
    heal_amount: float | None = None
    boss_kill: float | None = None
    player_death: float | None = None
    time_penalty: float | None = None
    invalid_action: float | None = None


class LearnerTuning(_StrictTuningModel):
    """Optimizer/loss knobs safe to replace between learner updates."""

    learning_rate: float | None = Field(default=None, gt=0.0)
    entropy_coef: float | None = Field(default=None, ge=0.0)
    value_coef: float | None = Field(default=None, ge=0.0)
    clip_range: float | None = Field(default=None, gt=0.0)
    max_grad_norm: float | None = Field(default=None, gt=0.0)
    target_kl: float | Literal["off"] | None = None

    @field_validator("target_kl")
    @classmethod
    def _validate_target_kl(cls, value: float | str | None) -> float | str | None:
        if value is None or value == "off":
            return value
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError("target_kl must be positive, finite, or 'off'")
        return result


class WorkerTuning(_StrictTuningModel):
    """Game-host knobs safe to apply between rollout batches."""

    time_scale: float | None = Field(default=None, gt=0.0)


class LiveTuning(_StrictTuningModel):
    """Complete, monotonically versioned snapshot of runtime overrides."""

    version: int = Field(ge=1)
    reward: RewardTuning = Field(default_factory=RewardTuning)
    learner: LearnerTuning = Field(default_factory=LearnerTuning)
    worker: WorkerTuning = Field(default_factory=WorkerTuning)
    reset_to_base: bool = False
    note: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _require_override(self) -> LiveTuning:
        sections = (self.reward, self.learner, self.worker)
        has_override = any(section.model_dump(exclude_none=True) for section in sections)
        if self.reset_to_base and has_override:
            raise ValueError("reset_to_base cannot be combined with parameter overrides")
        if not self.reset_to_base and not has_override:
            raise ValueError("live tuning must contain at least one parameter override")
        return self

    def checkpoint_payload(self) -> dict[str, Any]:
        """Return the full snapshot stored in a signed learner checkpoint."""
        return self.model_dump(exclude_none=True)

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.checkpoint_payload())).hexdigest()


def load_live_tuning(path: str | Path) -> LiveTuning | None:
    """Load one atomically-written request file, or return ``None`` if absent."""
    source = Path(path)
    if not source.exists():
        return None
    with source.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return LiveTuning.model_validate(payload)


def effective_reward_weights(
    base: RewardWeights,
    tuning: LiveTuning | None,
) -> RewardWeights:
    """Resolve a full reward vector from immutable startup values + snapshot."""
    if tuning is None:
        return base.model_copy(deep=True)
    return base.model_copy(
        update=tuning.reward.model_dump(exclude_none=True),
        deep=True,
    )


def effective_train_config(base: TrainConfig, tuning: LiveTuning | None) -> TrainConfig:
    """Resolve learner-safe fields from immutable startup config + snapshot."""
    if tuning is None:
        return base.model_copy(deep=True)

    learner_updates = tuning.learner.model_dump(
        exclude={"learning_rate", "entropy_coef", "value_coef", "clip_range", "max_grad_norm"},
        exclude_none=True,
    )
    if learner_updates.get("target_kl") == "off":
        learner_updates["target_kl"] = None

    top_level: dict[str, Any] = {}
    for name in (
        "learning_rate",
        "entropy_coef",
        "value_coef",
        "clip_range",
        "max_grad_norm",
    ):
        value = getattr(tuning.learner, name)
        if value is not None:
            top_level[name] = value

    if learner_updates:
        top_level["learner"] = base.learner.model_copy(
            update=learner_updates,
            deep=True,
        )
    return base.model_copy(update=top_level, deep=True)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write compact JSON with fsync + replace so readers never see a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload) + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        temporary.replace(target)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return target


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "LIVE_TUNING_REQUEST",
    "LIVE_TUNING_STATUS",
    "LearnerTuning",
    "LiveTuning",
    "RewardTuning",
    "WorkerTuning",
    "atomic_write_json",
    "effective_reward_weights",
    "effective_train_config",
    "load_live_tuning",
]
