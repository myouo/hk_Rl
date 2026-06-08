"""Flat on-policy rollout buffer + RolloutBatch (PRD §8.3).

Stores transitions for non-recurrent PPO and serializes to the on-the-wire
RolloutBatch the worker uploads to the learner. Fields mirror PRD §8.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RolloutBatch:
    """On-the-wire training sample bundle (docs/distributed_training.md §3)."""

    obs_global: np.ndarray
    obs_player: np.ndarray
    obs_entities: np.ndarray
    entity_mask: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray
    action_masks: np.ndarray
    prev_actions: np.ndarray
    rnn_states: np.ndarray | None
    episode_ids: np.ndarray
    task_ids: np.ndarray
    policy_version: int


class RolloutBuffer:
    """Fixed-capacity per-worker buffer; computes GAE then emits a RolloutBatch."""

    def __init__(self, capacity: int, num_envs: int, obs_spec: dict[str, Any]) -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        # TODO(phase-3): preallocate numpy arrays per field.

    def add(self, **transition: Any) -> None:
        raise NotImplementedError  # TODO(phase-3)

    def is_full(self) -> bool:
        raise NotImplementedError  # TODO(phase-3)

    def compute_returns(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        raise NotImplementedError  # TODO(phase-3): call gae.compute_gae

    def to_batch(self, policy_version: int) -> RolloutBatch:
        raise NotImplementedError  # TODO(phase-3)

    def clear(self) -> None:
        raise NotImplementedError  # TODO(phase-3)
