"""RolloutBatch field-contract tests (must match docs/distributed_training.md §3)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
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
