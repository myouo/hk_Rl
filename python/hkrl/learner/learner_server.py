"""Learner server (Remote GPU): collect batches, update, publish (PRD §8.1).

Receives RolloutBatches from workers, filters by ``policy_version``, runs the
configured algorithm's update, and publishes new checkpoints to the registry.
Only large-batch training happens here — never real-time inference (ADR-0004).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any

from hkrl.learner.checkpoint_registry import CheckpointMeta, CheckpointRegistry
from hkrl.models.base import ActorCritic
from hkrl.training import appo as _appo  # noqa: F401
from hkrl.training import ppo as _ppo  # noqa: F401
from hkrl.training import recurrent_ppo as _recurrent_ppo  # noqa: F401
from hkrl.training.rollout_buffer import RolloutBatch
from hkrl.utils.config import TrainConfig
from hkrl.utils.live_tuning import (
    LIVE_TUNING_REQUEST,
    LIVE_TUNING_STATUS,
    LiveTuning,
    atomic_write_json,
    effective_train_config,
    load_live_tuning,
)
from hkrl.utils.registry import get


class LearnerServer:
    """Hosts the training loop and the inbound rollout endpoint."""

    def __init__(
        self,
        model: ActorCritic,
        config: TrainConfig,
        registry: CheckpointRegistry,
        bind: str = "127.0.0.1:5600",
        batches_per_update: int = 1,
        max_staleness: int = 4,
        publish_every_updates: int = 1,
    ) -> None:
        if batches_per_update <= 0:
            raise ValueError("batches_per_update must be positive")
        if publish_every_updates <= 0:
            raise ValueError("publish_every_updates must be positive")

        self.model = model
        self._base_cfg = config.model_copy(deep=True)
        self.cfg = config.model_copy(deep=True)
        self.registry = registry
        self.bind = bind
        self.batches_per_update = batches_per_update
        self.publish_every_updates = publish_every_updates
        self.algo = _build_algorithm(model, config, max_staleness=max_staleness)
        self.policy_version = int(getattr(self.algo, "current_version", 0))
        self.update_count = 0
        self.accepted_batches = 0
        self.rejected_batches = 0
        self.last_metrics: dict[str, float] = {}
        self.last_checkpoint: CheckpointMeta | None = None
        self.live_tuning: LiveTuning | None = None
        self.pending_live_tuning: LiveTuning | None = None
        self.tuning_version = 0
        self.live_tuning_path = Path(self.registry.root) / LIVE_TUNING_REQUEST
        self.live_tuning_status_path = Path(self.registry.root) / LIVE_TUNING_STATUS

    def submit(self, batch: RolloutBatch) -> bool:
        """Submit one worker rollout batch for the next learner update."""
        if batch.tuning_version != self.tuning_version:
            self.rejected_batches += 1
            return False
        ingest = getattr(self.algo, "ingest", None)
        if ingest is None:
            raise TypeError(f"algorithm {self.cfg.algorithm!r} does not accept RolloutBatch intake")

        accepted = bool(ingest(batch, current_version=self.policy_version))
        if accepted:
            self.accepted_batches += 1
        else:
            self.rejected_batches += 1
        return accepted

    def update_once(self, *, publish: bool = True) -> dict[str, float]:
        """Run one learner update over queued batches and publish as configured."""
        metrics = self.algo.update()
        self.update_count += 1
        self.policy_version = int(getattr(self.algo, "current_version", self.policy_version + 1))
        self.last_metrics = {
            **{key: float(value) for key, value in metrics.items()},
            "tuning_version": float(self.tuning_version),
        }

        if publish and self.update_count % self.publish_every_updates == 0:
            self._publish_checkpoint()
        return self.last_metrics

    def poll_live_tuning(self) -> bool:
        """Load a newer validated request without applying it mid-batch."""
        tuning = load_live_tuning(self.live_tuning_path)
        if tuning is None or tuning.version <= self.tuning_version:
            return False
        pending = self.pending_live_tuning
        if pending is not None:
            if tuning.version < pending.version:
                return False
            if tuning.version == pending.version:
                if tuning.digest != pending.digest:
                    raise ValueError(f"pending tuning version {tuning.version} changed content")
                return False
        self.pending_live_tuning = tuning
        return True

    def reconcile_live_tuning(self) -> dict[str, object] | None:
        """Apply the newest request at an empty-queue update boundary.

        A partial queue is trained first under its original tuning version.
        The new full snapshot is then checkpointed immediately, so workers see
        it at their next rollout boundary and stale tuning versions are rejected.
        """
        self.poll_live_tuning()
        tuning = self.pending_live_tuning
        if tuning is None:
            return None

        flushed_update = False
        if int(getattr(self.algo, "queued_batches", 0)) > 0:
            self.update_once(publish=False)
            flushed_update = True

        self._apply_live_tuning(tuning)
        self.pending_live_tuning = None
        checkpoint = self._publish_checkpoint()
        status = {
            "applied_at_unix": time(),
            "checkpoint_version": checkpoint.version,
            "digest": tuning.digest,
            "flushed_update": flushed_update,
            "policy_version": self.policy_version,
            "tuning_version": self.tuning_version,
            "update": self.update_count,
        }
        atomic_write_json(self.live_tuning_status_path, status)
        return status

    def restore_live_tuning(self, payload: object) -> bool:
        """Restore a tuning snapshot embedded in a verified checkpoint."""
        if payload is None:
            return False
        tuning = LiveTuning.model_validate(payload)
        self._apply_live_tuning(tuning)
        return True

    def ensure_live_tuning_status(self) -> dict[str, object] | None:
        """Recreate a missing apply acknowledgement after a learner restart."""
        if self.live_tuning is None or self.last_checkpoint is None:
            return None
        try:
            with self.live_tuning_status_path.open(encoding="utf-8") as fh:
                current = json.load(fh)
            current_version = (
                int(current.get("tuning_version", -1)) if isinstance(current, dict) else -1
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            current_version = -1
        if current_version >= self.tuning_version:
            return None

        status: dict[str, object] = {
            "applied_at_unix": time(),
            "checkpoint_version": self.last_checkpoint.version,
            "digest": self.live_tuning.digest,
            "flushed_update": False,
            "policy_version": self.policy_version,
            "recovered_after_restart": True,
            "tuning_version": self.tuning_version,
            "update": self.update_count,
        }
        atomic_write_json(self.live_tuning_status_path, status)
        return status

    def _apply_live_tuning(self, tuning: LiveTuning) -> None:
        if tuning.version < self.tuning_version:
            raise ValueError(
                f"tuning version regressed from {self.tuning_version} to {tuning.version}"
            )
        if tuning.version == self.tuning_version:
            if self.live_tuning is not None and tuning.digest != self.live_tuning.digest:
                raise ValueError(f"tuning version {tuning.version} changed content")
            return

        config = effective_train_config(self._base_cfg, tuning)
        self.cfg = config
        self.algo.cfg = config
        for group in self.algo.optimizer.param_groups:
            group["lr"] = config.learning_rate
        self.live_tuning = tuning
        self.tuning_version = tuning.version
        self.last_metrics["tuning_version"] = float(self.tuning_version)

    def _publish_checkpoint(self) -> CheckpointMeta:
        self.last_checkpoint = self.registry.publish(
            self.checkpoint_payload(),
            policy_version=self.policy_version,
            step=self.update_count,
        )
        return self.last_checkpoint

    def checkpoint_payload(self) -> dict[str, object]:
        """Build a resumable learner checkpoint also safe for worker loading."""
        payload: dict[str, object] = {
            "model_state_dict": self.model.state_dict(),
            "policy_version": self.policy_version,
            "update": self.update_count,
            "metrics": self.last_metrics,
        }
        optimizer = getattr(self.algo, "optimizer", None)
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["tuning_version"] = self.tuning_version
        if self.live_tuning is not None:
            payload["live_tuning"] = self.live_tuning.checkpoint_payload()
        return payload

    def restore_optimizer_state(self, state: object) -> bool:
        """Restore optimizer momentum/state when a checkpoint contains it."""
        if state is None:
            return False
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint optimizer_state_dict must be a mapping")
        optimizer = getattr(self.algo, "optimizer", None)
        if optimizer is None:
            raise ValueError(f"algorithm {self.cfg.algorithm!r} has no optimizer to restore")
        optimizer.load_state_dict(dict(state))
        return True

    @property
    def ready_to_update(self) -> bool:
        """Whether enough accepted rollouts are queued for one GPU update."""
        queued_batches = int(getattr(self.algo, "queued_batches", 0))
        return queued_batches >= self.batches_per_update

    def serve(self, *, force: bool = False) -> bool:
        """Run the receive->filter->update->publish loop.

        The network listener is intentionally separate from the training core.
        It combines multiple worker rollouts into one larger GPU update. ``force``
        flushes a partial queue during finite runs or graceful shutdown.
        """
        queued_batches = int(getattr(self.algo, "queued_batches", 0))
        if queued_batches > 0 and (force or self.ready_to_update):
            self.update_once()
            return True
        return False


def _build_algorithm(model: ActorCritic, config: TrainConfig, *, max_staleness: int) -> Any:
    algo_cls = get("algo", config.algorithm)
    if config.algorithm == "appo":
        return algo_cls(model, config, max_staleness=max_staleness)
    return algo_cls(model, config)
