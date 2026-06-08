"""Sequence buffer for recurrent PPO (docs/model_architecture.md §4).

Chunks trajectories into fixed-length sequences for truncated BPTT, stores the
hidden state at each sequence boundary, masks padded timesteps, and supports an
optional burn-in prefix used only to warm the RNN (no loss on burn-in steps).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from hkrl.training.gae import compute_gae
from hkrl.training.rollout_buffer import (
    RolloutBatch,
    _env_array,
    _flat_env_array,
    _shape_from_spec,
)


@dataclass(frozen=True)
class RecurrentSequenceBatch:
    """Batch-first sequence chunk emitted by :class:`RecurrentRolloutBuffer`."""

    obs: dict[str, np.ndarray]
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray
    action_masks: np.ndarray
    prev_actions: np.ndarray
    rnn_state: Any
    loss_mask: np.ndarray
    episode_ids: np.ndarray
    task_ids: np.ndarray


class RecurrentRolloutBuffer:
    """Stores transitions + per-step rnn_state and yields (seq_len, batch) chunks.

    Critical correctness points (docs/troubleshooting.md): hidden state reset at
    episode boundaries, padded-timestep masking in the loss, and burn-in steps
    excluded from the policy/value loss.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        sequence_length: int = 32,
        burn_in: int = 0,
        obs_spec: dict[str, Any] | None = None,
    ) -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if obs_spec is None:
            raise ValueError("obs_spec is required")

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
        self.advantages = np.zeros(env_prefix, dtype=np.float32)
        self.returns = np.zeros(env_prefix, dtype=np.float32)
        self.episode_ids = np.zeros(env_prefix, dtype=np.uint64)
        self.task_ids = np.zeros(env_prefix, dtype=np.int64)
        self.rnn_states: list[Any] = [None for _ in range(capacity)]

    def add(self, **transition: Any) -> None:
        if self.is_full():
            raise RuntimeError("recurrent rollout buffer is full")

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
        self.episode_ids[idx] = _flat_env_array(
            transition.get("episode_id", np.zeros((self.num_envs,), dtype=np.uint64)),
            self.num_envs,
        ).astype(np.uint64)
        self.task_ids[idx] = _flat_env_array(
            transition.get("task_id", np.zeros((self.num_envs,), dtype=np.int64)),
            self.num_envs,
        ).astype(np.int64)
        self.rnn_states[idx] = _copy_rnn_state(transition.get("rnn_state"))

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

    def to_batch(self, policy_version: int) -> RolloutBatch:
        """Return a flat RolloutBatch view for logging/upload boundaries."""
        length = self._length()
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
            rnn_states=_time_rnn_states(self.rnn_states[:length]),
            episode_ids=self.episode_ids[:length].copy(),
            task_ids=self.task_ids[:length].copy(),
            policy_version=policy_version,
        )

    def iter_sequences(
        self,
        minibatch_size: int | None = None,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[RecurrentSequenceBatch]:
        """Yield (obs, actions, ..., rnn_state0, loss_mask) sequence minibatches.

        ``loss_mask`` is False on padding and burn-in steps.
        """
        descriptors = self._sequence_descriptors()
        if not descriptors:
            return

        if minibatch_size is None:
            minibatch_size = len(descriptors)
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")

        order = np.arange(len(descriptors))
        if shuffle:
            generator = rng if rng is not None else np.random.default_rng()
            generator.shuffle(order)

        for start in range(0, len(order), minibatch_size):
            batch_indices = order[start : start + minibatch_size]
            yield self._build_sequence_batch([descriptors[int(idx)] for idx in batch_indices])

    def clear(self) -> None:
        self.pos = 0
        self._full = False
        self.rnn_states = [None for _ in range(self.capacity)]

    def _length(self) -> int:
        return self.capacity if self._full else self.pos

    def _sequence_descriptors(self) -> list[tuple[int, int, int, int, int]]:
        """Return ``(env, input_start, valid_len, loss_offset, loss_len)`` chunks."""
        length = self._length()
        descriptors: list[tuple[int, int, int, int, int]] = []

        for env_idx in range(self.num_envs):
            segment_start = 0
            while segment_start < length:
                segment_end = segment_start
                while segment_end < length:
                    segment_end += 1
                    if self.dones[segment_end - 1, env_idx]:
                        break

                loss_start = segment_start
                while loss_start < segment_end:
                    loss_end = min(loss_start + self.sequence_length, segment_end)
                    input_start = max(segment_start, loss_start - self.burn_in)
                    loss_offset = loss_start - input_start
                    valid_len = (loss_start - input_start) + (loss_end - loss_start)
                    loss_len = loss_end - loss_start
                    descriptors.append((env_idx, input_start, valid_len, loss_offset, loss_len))
                    loss_start = loss_end

                segment_start = segment_end

        return descriptors

    def _build_sequence_batch(
        self, descriptors: list[tuple[int, int, int, int, int]]
    ) -> RecurrentSequenceBatch:
        batch_size = len(descriptors)
        max_len = self.sequence_length + self.burn_in
        obs = {
            "global": np.zeros((batch_size, max_len, *self.obs_global.shape[2:]), dtype=np.float32),
            "player": np.zeros((batch_size, max_len, *self.obs_player.shape[2:]), dtype=np.float32),
            "entities": np.zeros(
                (batch_size, max_len, *self.obs_entities.shape[2:]), dtype=np.float32
            ),
            "entity_mask": np.zeros((batch_size, max_len, *self.entity_mask.shape[2:]), dtype=bool),
        }
        actions = np.zeros((batch_size, max_len, *self.actions.shape[2:]), dtype=np.int64)
        old_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
        old_values = np.zeros((batch_size, max_len), dtype=np.float32)
        advantages = np.zeros((batch_size, max_len), dtype=np.float32)
        returns = np.zeros((batch_size, max_len), dtype=np.float32)
        rewards = np.zeros((batch_size, max_len), dtype=np.float32)
        dones = np.zeros((batch_size, max_len), dtype=bool)
        truncateds = np.zeros((batch_size, max_len), dtype=bool)
        action_masks = np.ones(
            (batch_size, max_len, *self.action_masks.shape[2:]),
            dtype=bool,
        )
        prev_actions = np.zeros((batch_size, max_len, *self.prev_actions.shape[2:]), dtype=np.int64)
        loss_mask = np.zeros((batch_size, max_len), dtype=bool)
        episode_ids = np.zeros((batch_size, max_len), dtype=np.uint64)
        task_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        rnn_states: list[Any] = []

        for batch_idx, (env_idx, input_start, valid_len, loss_offset, loss_len) in enumerate(
            descriptors
        ):
            source = slice(input_start, input_start + valid_len)
            target = slice(0, valid_len)
            obs["global"][batch_idx, target] = self.obs_global[source, env_idx]
            obs["player"][batch_idx, target] = self.obs_player[source, env_idx]
            obs["entities"][batch_idx, target] = self.obs_entities[source, env_idx]
            obs["entity_mask"][batch_idx, target] = self.entity_mask[source, env_idx]
            actions[batch_idx, target] = self.actions[source, env_idx]
            old_log_probs[batch_idx, target] = self.log_probs[source, env_idx]
            old_values[batch_idx, target] = self.values[source, env_idx]
            advantages[batch_idx, target] = self.advantages[source, env_idx]
            returns[batch_idx, target] = self.returns[source, env_idx]
            rewards[batch_idx, target] = self.rewards[source, env_idx]
            dones[batch_idx, target] = self.dones[source, env_idx]
            truncateds[batch_idx, target] = self.truncateds[source, env_idx]
            action_masks[batch_idx, target] = self.action_masks[source, env_idx]
            prev_actions[batch_idx, target] = self.prev_actions[source, env_idx]
            episode_ids[batch_idx, target] = self.episode_ids[source, env_idx]
            task_ids[batch_idx, target] = self.task_ids[source, env_idx]
            loss_mask[batch_idx, loss_offset : loss_offset + loss_len] = True
            rnn_states.append(
                _select_rnn_state(self.rnn_states[input_start], env_idx, self.num_envs)
            )

        return RecurrentSequenceBatch(
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            rewards=rewards,
            dones=dones,
            truncateds=truncateds,
            action_masks=action_masks,
            prev_actions=prev_actions,
            rnn_state=_stack_rnn_states(rnn_states),
            loss_mask=loss_mask,
            episode_ids=episode_ids,
            task_ids=task_ids,
        )


def _copy_rnn_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_copy_rnn_state(part) for part in state)
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    return np.asarray(state).copy()


def _select_rnn_state(state: Any, env_idx: int, num_envs: int) -> Any:
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_select_rnn_state(part, env_idx, num_envs) for part in state)

    array = np.asarray(state)
    if array.ndim >= 2 and array.shape[1] == num_envs:
        return array[:, env_idx : env_idx + 1, ...].copy()
    if num_envs == 1:
        if array.ndim == 0:
            return array.reshape((1, 1)).copy()
        if array.ndim == 1:
            return array.reshape((1, 1, *array.shape)).copy()
        if array.ndim >= 2 and array.shape[1] == 1:
            return array.copy()
    raise ValueError(
        f"rnn_state must include env batch dimension {num_envs}, got shape {array.shape}"
    )


def _stack_rnn_states(states: list[Any]) -> Any:
    if not states or all(state is None for state in states):
        return None
    if any(state is None for state in states):
        raise ValueError("cannot mix missing and present rnn_state values")

    first = states[0]
    if isinstance(first, tuple):
        return tuple(
            _stack_rnn_states([state[idx] for state in states]) for idx in range(len(first))
        )
    return np.concatenate([np.asarray(state) for state in states], axis=1)


def _time_rnn_states(states: list[Any]) -> np.ndarray | None:
    if not states or all(state is None for state in states):
        return None
    if any(state is None for state in states):
        raise ValueError("cannot mix missing and present rnn_state values")
    if isinstance(states[0], tuple):
        return None
    return np.stack([np.asarray(state) for state in states], axis=0)
