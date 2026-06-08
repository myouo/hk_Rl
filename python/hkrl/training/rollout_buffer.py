"""Flat on-policy rollout buffer + RolloutBatch (PRD §8.3).

Stores transitions for non-recurrent PPO and serializes to the on-the-wire
RolloutBatch the worker uploads to the learner. Fields mirror PRD §8.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hkrl.training.gae import compute_gae


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
    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray
    action_masks: np.ndarray
    prev_actions: np.ndarray
    prev_rewards: np.ndarray
    rnn_states: np.ndarray | None
    episode_ids: np.ndarray
    task_ids: np.ndarray
    policy_version: int


class RolloutBuffer:
    """Fixed-capacity per-worker buffer; computes GAE then emits a RolloutBatch."""

    def __init__(self, capacity: int, num_envs: int, obs_spec: dict[str, Any]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.capacity = capacity
        self.num_envs = num_envs
        self.pos = 0
        self._full = False

        global_shape = _shape_from_spec(obs_spec, "global")
        player_shape = _shape_from_spec(obs_spec, "player")
        entities_shape = _shape_from_spec(obs_spec, "entities")
        entity_mask_shape = _shape_from_spec(obs_spec, "entity_mask")
        action_shape = _shape_from_spec(obs_spec, "action", default=())
        action_mask_shape = _shape_from_spec(obs_spec, "action_mask", default=())

        env_prefix = (capacity, num_envs)
        self.obs_global = np.zeros(env_prefix + global_shape, dtype=np.float32)
        self.obs_player = np.zeros(env_prefix + player_shape, dtype=np.float32)
        self.obs_entities = np.zeros(env_prefix + entities_shape, dtype=np.float32)
        self.entity_mask = np.zeros(env_prefix + entity_mask_shape, dtype=bool)
        self.actions = np.zeros(env_prefix + action_shape, dtype=np.int64)
        self.log_probs = np.zeros(env_prefix, dtype=np.float32)
        self.values = np.zeros(env_prefix, dtype=np.float32)
        self.rewards = np.zeros(env_prefix, dtype=np.float32)
        self.dones = np.zeros(env_prefix, dtype=bool)
        self.truncateds = np.zeros(env_prefix, dtype=bool)
        self.action_masks = np.zeros(env_prefix + action_mask_shape, dtype=bool)
        self.prev_actions = np.zeros(env_prefix + action_shape, dtype=np.int64)
        self.prev_rewards = np.zeros(env_prefix, dtype=np.float32)
        self.advantages = np.zeros(env_prefix, dtype=np.float32)
        self.returns = np.zeros(env_prefix, dtype=np.float32)
        self.episode_ids = np.zeros(env_prefix, dtype=np.uint64)
        self.task_ids = np.zeros(env_prefix, dtype=np.int64)

    def add(self, **transition: Any) -> None:
        if self.is_full():
            raise RuntimeError("rollout buffer is full")

        obs = transition.get("obs")
        if obs is None:
            obs = {
                "global": transition["obs_global"],
                "player": transition["obs_player"],
                "entities": transition["obs_entities"],
                "entity_mask": transition["entity_mask"],
            }

        idx = self.pos
        self.obs_global[idx] = _env_array(obs["global"], self.num_envs)
        self.obs_player[idx] = _env_array(obs["player"], self.num_envs)
        self.obs_entities[idx] = _env_array(obs["entities"], self.num_envs)
        self.entity_mask[idx] = _env_array(obs["entity_mask"], self.num_envs).astype(bool)
        self.actions[idx] = _env_array(transition["action"], self.num_envs)
        self.log_probs[idx] = _flat_env_array(transition["log_prob"], self.num_envs)
        self.values[idx] = _flat_env_array(transition["value"], self.num_envs)
        self.rewards[idx] = _flat_env_array(transition["reward"], self.num_envs)
        self.dones[idx] = _flat_env_array(transition["done"], self.num_envs).astype(bool)
        self.truncateds[idx] = _flat_env_array(transition["truncated"], self.num_envs).astype(bool)
        self.action_masks[idx] = _env_array(transition["action_mask"], self.num_envs).astype(bool)
        self.prev_actions[idx] = _env_array(
            transition.get("prev_action", np.zeros_like(self.actions[idx])),
            self.num_envs,
        )
        self.prev_rewards[idx] = _flat_env_array(
            transition.get("prev_reward", np.zeros((self.num_envs,), dtype=np.float32)),
            self.num_envs,
        )
        self.episode_ids[idx] = _flat_env_array(
            transition.get("episode_id", np.zeros((self.num_envs,), dtype=np.uint64)),
            self.num_envs,
        ).astype(np.uint64)
        self.task_ids[idx] = _flat_env_array(
            transition.get("task_id", np.zeros((self.num_envs,), dtype=np.int64)),
            self.num_envs,
        ).astype(np.int64)
        _require_finite_stored_transition(self, idx)

        self.pos += 1
        self._full = self.pos >= self.capacity

    def is_full(self) -> bool:
        return self._full

    def compute_returns(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        length = self._length()
        self.advantages[:length], self.returns[:length] = compute_gae(
            self.rewards[:length],
            self.values[:length],
            self.dones[:length],
            self.truncateds[:length],
            np.asarray(last_value, dtype=np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        _require_finite("advantages", self.advantages[:length])
        _require_finite("returns", self.returns[:length])

    def to_batch(self, policy_version: int) -> RolloutBatch:
        length = self._length()
        _require_finite_stored_window(self, length)
        return RolloutBatch(
            obs_global=self.obs_global[:length].copy(),
            obs_player=self.obs_player[:length].copy(),
            obs_entities=self.obs_entities[:length].copy(),
            entity_mask=self.entity_mask[:length].copy(),
            actions=self.actions[:length].copy(),
            log_probs=self.log_probs[:length].copy(),
            values=self.values[:length].copy(),
            advantages=self.advantages[:length].copy(),
            returns=self.returns[:length].copy(),
            rewards=self.rewards[:length].copy(),
            dones=self.dones[:length].copy(),
            truncateds=self.truncateds[:length].copy(),
            action_masks=self.action_masks[:length].copy(),
            prev_actions=self.prev_actions[:length].copy(),
            prev_rewards=self.prev_rewards[:length].copy(),
            rnn_states=None,
            episode_ids=self.episode_ids[:length].copy(),
            task_ids=self.task_ids[:length].copy(),
            policy_version=policy_version,
        )

    def clear(self) -> None:
        self.pos = 0
        self._full = False

    def _length(self) -> int:
        return self.capacity if self._full else self.pos


def _shape_from_spec(
    obs_spec: dict[str, Any],
    key: str,
    *,
    default: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if key not in obs_spec:
        if default is not None:
            return default
        raise KeyError(f"obs_spec missing {key!r}")

    value = obs_spec[key]
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, int):
        return (value,)
    return tuple(int(dim) for dim in value)


def _env_array(value: Any, num_envs: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:1] == (num_envs,):
        return array
    if num_envs == 1:
        return array.reshape((1, *array.shape))
    raise ValueError(f"value must have leading num_envs dimension {num_envs}, got {array.shape}")


def _flat_env_array(value: Any, num_envs: int) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if array.shape != (num_envs,):
        raise ValueError(f"value must have shape ({num_envs},), got {array.shape}")
    return array


def _require_finite(field: str, array: Any) -> None:
    if not np.isfinite(np.asarray(array)).all():
        raise ValueError(f"{field} contains non-finite values")


def _require_finite_stored_transition(buffer: RolloutBuffer, idx: int) -> None:
    for field in _FINITE_STORED_FIELDS:
        _require_finite(field, getattr(buffer, field)[idx])


def _require_finite_stored_window(buffer: RolloutBuffer, length: int) -> None:
    for field in (*_FINITE_STORED_FIELDS, "advantages", "returns"):
        _require_finite(field, getattr(buffer, field)[:length])


_FINITE_STORED_FIELDS: tuple[str, ...] = (
    "obs_global",
    "obs_player",
    "obs_entities",
    "log_probs",
    "values",
    "rewards",
    "prev_rewards",
)
