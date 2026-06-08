"""RolloutBatch field-contract tests (must match docs/distributed_training.md §3)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from hkrl.training.batch_io import (
    deserialize_rollout_batch,
    load_rollout_batch,
    save_rollout_batch,
    serialize_rollout_batch,
)
from hkrl.training.gae import compute_gae
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer


def test_rollout_batch_has_required_fields() -> None:
    names = {f.name for f in dataclasses.fields(RolloutBatch)}
    required = {
        "obs_global",
        "obs_player",
        "obs_entities",
        "entity_mask",
        "actions",
        "log_probs",
        "values",
        "advantages",
        "returns",
        "rewards",
        "dones",
        "truncateds",
        "action_masks",
        "prev_actions",
        "rnn_states",
        "episode_ids",
        "task_ids",
        "policy_version",
    }
    assert required <= names


def test_compute_gae_reverse_recursion() -> None:
    advantages, returns = compute_gae(
        rewards=np.ones((3, 1), dtype=np.float32),
        values=np.zeros((3, 1), dtype=np.float32),
        dones=np.zeros((3, 1), dtype=bool),
        truncateds=np.zeros((3, 1), dtype=bool),
        last_value=np.zeros((1,), dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages[:, 0], np.array([3.0, 2.0, 1.0]))
    np.testing.assert_allclose(returns[:, 0], np.array([3.0, 2.0, 1.0]))


def test_compute_gae_terminated_step_does_not_bootstrap() -> None:
    advantages, returns = compute_gae(
        rewards=np.array([[1.0]], dtype=np.float32),
        values=np.array([[0.0]], dtype=np.float32),
        dones=np.array([[True]]),
        truncateds=np.array([[False]]),
        last_value=np.array([10.0], dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, np.array([[1.0]], dtype=np.float32))
    np.testing.assert_allclose(returns, np.array([[1.0]], dtype=np.float32))


def test_compute_gae_truncated_step_bootstraps() -> None:
    advantages, returns = compute_gae(
        rewards=np.array([[1.0]], dtype=np.float32),
        values=np.array([[0.0]], dtype=np.float32),
        dones=np.array([[True]]),
        truncateds=np.array([[True]]),
        last_value=np.array([10.0], dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, np.array([[11.0]], dtype=np.float32))
    np.testing.assert_allclose(returns, np.array([[11.0]], dtype=np.float32))


def test_rollout_buffer_add_compute_batch_and_clear() -> None:
    buffer = RolloutBuffer(
        capacity=2,
        num_envs=1,
        obs_spec={
            "global": (2,),
            "player": (3,),
            "entities": (4, 5),
            "entity_mask": (4,),
            "action": (2,),
            "action_mask": (6,),
        },
    )

    for step in range(2):
        buffer.add(
            obs={
                "global": np.array([step, step + 1], dtype=np.float32),
                "player": np.ones((3,), dtype=np.float32) * step,
                "entities": np.ones((4, 5), dtype=np.float32) * step,
                "entity_mask": np.array([1, 1, 0, 0], dtype=bool),
            },
            action=np.array([step, step + 1], dtype=np.int64),
            log_prob=np.array([-0.1 * (step + 1)], dtype=np.float32),
            value=np.array([0.0], dtype=np.float32),
            reward=np.array([1.0], dtype=np.float32),
            done=np.array([False]),
            truncated=np.array([False]),
            action_mask=np.array([1, 0, 1, 0, 1, 0], dtype=bool),
            episode_id=np.array([7], dtype=np.uint64),
            task_id=np.array([3], dtype=np.int64),
        )

    assert buffer.is_full()
    buffer.compute_returns(last_value=np.array([0.0], dtype=np.float32), gamma=1.0, gae_lambda=1.0)
    batch = buffer.to_batch(policy_version=5)

    assert batch.policy_version == 5
    assert batch.obs_global.shape == (2, 1, 2)
    assert batch.actions.shape == (2, 1, 2)
    np.testing.assert_allclose(batch.advantages[:, 0], [2.0, 1.0])
    np.testing.assert_allclose(batch.returns[:, 0], [2.0, 1.0])
    np.testing.assert_allclose(buffer.advantages[:, 0], [2.0, 1.0])
    np.testing.assert_allclose(buffer.returns[:, 0], [2.0, 1.0])
    np.testing.assert_array_equal(batch.episode_ids[:, 0], np.array([7, 7], dtype=np.uint64))
    np.testing.assert_array_equal(batch.task_ids[:, 0], np.array([3, 3]))

    batch.obs_global[0, 0, 0] = 99.0
    assert buffer.obs_global[0, 0, 0] == 0.0

    buffer.clear()
    assert not buffer.is_full()
    assert buffer.pos == 0


def test_rollout_buffer_rejects_overfill() -> None:
    buffer = RolloutBuffer(
        capacity=1,
        num_envs=1,
        obs_spec={
            "global": (1,),
            "player": (1,),
            "entities": (1, 1),
            "entity_mask": (1,),
            "action": (),
            "action_mask": (1,),
        },
    )
    transition = {
        "obs": {
            "global": np.zeros((1,), dtype=np.float32),
            "player": np.zeros((1,), dtype=np.float32),
            "entities": np.zeros((1, 1), dtype=np.float32),
            "entity_mask": np.ones((1,), dtype=bool),
        },
        "action": np.array(0),
        "log_prob": np.array([0.0], dtype=np.float32),
        "value": np.array([0.0], dtype=np.float32),
        "reward": np.array([0.0], dtype=np.float32),
        "done": np.array([False]),
        "truncated": np.array([False]),
        "action_mask": np.array([True]),
    }

    buffer.add(**transition)
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(**transition)


def test_rollout_batch_npz_roundtrip(tmp_path: Path) -> None:
    batch = _sample_batch(policy_version=9)
    path = save_rollout_batch(tmp_path / "batch.npz", batch)

    loaded = load_rollout_batch(path)

    assert loaded.policy_version == 9
    assert loaded.rnn_states is None
    _assert_batch_arrays_equal(loaded, batch)


def test_rollout_batch_npz_roundtrip_with_rnn_state(tmp_path: Path) -> None:
    batch = _sample_batch(
        policy_version=10,
        rnn_states=np.arange(12, dtype=np.float32).reshape(2, 1, 1, 6),
    )
    path = save_rollout_batch(tmp_path / "nested" / "batch.npz", batch)

    loaded = load_rollout_batch(path)

    assert loaded.policy_version == 10
    assert loaded.rnn_states is not None
    np.testing.assert_array_equal(loaded.rnn_states, batch.rnn_states)
    _assert_batch_arrays_equal(loaded, batch)


def test_rollout_batch_memory_roundtrip_with_rnn_state() -> None:
    batch = _sample_batch(
        policy_version=11,
        rnn_states=np.arange(12, dtype=np.float32).reshape(2, 1, 1, 6),
    )

    loaded = deserialize_rollout_batch(serialize_rollout_batch(batch))

    assert loaded.policy_version == 11
    assert loaded.rnn_states is not None
    np.testing.assert_array_equal(loaded.rnn_states, batch.rnn_states)
    _assert_batch_arrays_equal(loaded, batch)


def test_rollout_batch_deserialize_rejects_mismatched_time_env_shapes() -> None:
    batch = _sample_batch(policy_version=12)
    batch.rewards = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="time/env shape"):
        deserialize_rollout_batch(serialize_rollout_batch(batch))


def test_rollout_batch_deserialize_rejects_bad_rnn_state_shape() -> None:
    batch = _sample_batch(
        policy_version=13,
        rnn_states=np.zeros((2, 1, 6), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="rnn_states"):
        deserialize_rollout_batch(serialize_rollout_batch(batch))


def test_rollout_batch_npz_rejects_unknown_format_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    with open(path, "wb") as fh:
        np.savez_compressed(
            fh,
            batch_format_version=np.array([999], dtype=np.int32),
            policy_version=np.array([1], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="unsupported RolloutBatch format version"):
        load_rollout_batch(path)


def _sample_batch(
    *,
    policy_version: int,
    rnn_states: np.ndarray | None = None,
) -> RolloutBatch:
    return RolloutBatch(
        obs_global=np.arange(4, dtype=np.float32).reshape(2, 1, 2),
        obs_player=np.arange(6, dtype=np.float32).reshape(2, 1, 3),
        obs_entities=np.arange(40, dtype=np.float32).reshape(2, 1, 4, 5),
        entity_mask=np.array([[[True, True, False, False]], [[True, False, False, False]]]),
        actions=np.array([[[0, 1]], [[1, 2]]], dtype=np.int64),
        log_probs=np.array([[-0.1], [-0.2]], dtype=np.float32),
        values=np.array([[0.3], [0.4]], dtype=np.float32),
        advantages=np.array([[1.0], [0.5]], dtype=np.float32),
        returns=np.array([[1.3], [0.9]], dtype=np.float32),
        rewards=np.array([[1.0], [0.0]], dtype=np.float32),
        dones=np.array([[False], [True]]),
        truncateds=np.array([[False], [False]]),
        action_masks=np.ones((2, 1, 6), dtype=bool),
        prev_actions=np.array([[[0, 0]], [[0, 1]]], dtype=np.int64),
        rnn_states=rnn_states,
        episode_ids=np.array([[7], [7]], dtype=np.uint64),
        task_ids=np.array([[3], [3]], dtype=np.int64),
        policy_version=policy_version,
    )


def _assert_batch_arrays_equal(left: RolloutBatch, right: RolloutBatch) -> None:
    for field in dataclasses.fields(RolloutBatch):
        if field.name in {"policy_version", "rnn_states"}:
            continue
        np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))
