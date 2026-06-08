"""RolloutBatch field-contract tests (must match docs/distributed_training.md §3)."""

from __future__ import annotations

import dataclasses

import numpy as np
from hkrl.training.gae import compute_gae
from hkrl.training.rollout_buffer import RolloutBatch


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
