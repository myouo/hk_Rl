"""Recurrent rollout buffer tests."""

from __future__ import annotations

import numpy as np
import pytest
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer


def test_recurrent_buffer_chunks_by_episode_and_burn_in() -> None:
    buffer = RecurrentRolloutBuffer(
        capacity=5,
        num_envs=1,
        sequence_length=2,
        burn_in=1,
        obs_spec={
            "global": (2,),
            "player": (3,),
            "entities": (2, 4),
            "entity_mask": (2,),
            "action": (2,),
            "action_mask": (6,),
        },
    )

    for step in range(5):
        buffer.add(
            obs={
                "global": np.array([step, step + 0.5], dtype=np.float32),
                "player": np.ones((3,), dtype=np.float32) * step,
                "entities": np.ones((2, 4), dtype=np.float32) * step,
                "entity_mask": np.array([True, step % 2 == 0]),
            },
            action=np.array([step, step + 1], dtype=np.int64),
            log_prob=np.array([-0.1 * step], dtype=np.float32),
            value=np.array([0.0], dtype=np.float32),
            reward=np.array([1.0], dtype=np.float32),
            done=np.array([step == 2]),
            truncated=np.array([False]),
            action_mask=np.array([True, True, False, True, False, True]),
            prev_reward=np.array([float(step)], dtype=np.float32),
            rnn_state=np.full((1, 1, 3), step, dtype=np.float32),
            episode_id=np.array([1 if step <= 2 else 2], dtype=np.uint64),
            task_id=np.array([7], dtype=np.int64),
        )

    assert buffer.is_full()
    buffer.compute_returns(last_value=np.array([0.0], dtype=np.float32), gamma=1.0, gae_lambda=1.0)
    batches = list(buffer.iter_sequences())

    assert len(batches) == 1
    batch = batches[0]
    assert batch.obs["global"].shape == (3, 3, 2)
    assert batch.actions.shape == (3, 3, 2)
    np.testing.assert_array_equal(
        batch.loss_mask,
        np.array(
            [
                [True, True, False],
                [False, True, False],
                [True, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(batch.obs["global"][:, :, 0], [[0, 1, 0], [1, 2, 0], [3, 4, 0]])
    np.testing.assert_allclose(
        batch.advantages, [[3.0, 2.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    )
    np.testing.assert_allclose(
        batch.prev_rewards, [[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
    )
    np.testing.assert_allclose(buffer.advantages[:, 0], [3.0, 2.0, 1.0, 2.0, 1.0])
    assert batch.rnn_state.shape == (1, 3, 3)
    np.testing.assert_array_equal(batch.rnn_state[0, :, 0], np.array([0.0, 1.0, 3.0]))
    np.testing.assert_array_equal(batch.episode_ids, [[1, 1, 0], [1, 1, 0], [2, 2, 0]])


def test_recurrent_buffer_minibatches_sequences() -> None:
    buffer = _tiny_buffer(capacity=3)
    for step in range(3):
        buffer.add(**_transition(step))

    batches = list(buffer.iter_sequences(minibatch_size=1))

    assert len(batches) == 3
    assert all(batch.obs["global"].shape == (1, 1, 1) for batch in batches)


def test_recurrent_buffer_splits_sequences_at_truncation() -> None:
    buffer = _tiny_buffer(capacity=3, sequence_length=3)
    for step in range(3):
        transition = _transition(step)
        transition["truncated"] = np.array([step == 1])
        transition["rnn_state"] = np.full((1, 1, 2), step, dtype=np.float32)
        buffer.add(**transition)
    buffer.compute_returns(last_value=np.array([0.0], dtype=np.float32), gamma=1.0, gae_lambda=1.0)

    batches = list(buffer.iter_sequences())

    assert len(batches) == 1
    batch = batches[0]
    assert batch.obs["global"].shape == (2, 3, 1)
    np.testing.assert_array_equal(
        batch.loss_mask,
        np.array(
            [
                [True, True, False],
                [True, False, False],
            ]
        ),
    )
    np.testing.assert_array_equal(batch.obs["global"][:, :, 0], [[0, 1, 0], [2, 0, 0]])
    assert batch.rnn_state.shape == (1, 2, 2)
    np.testing.assert_array_equal(batch.rnn_state[0, :, 0], [0.0, 2.0])


def test_recurrent_buffer_exports_flat_rollout_batch() -> None:
    buffer = _tiny_buffer(capacity=2)
    for step in range(2):
        buffer.add(**_transition(step))
    buffer.compute_returns(last_value=np.array([0.0], dtype=np.float32), gamma=1.0, gae_lambda=1.0)

    batch = buffer.to_batch(policy_version=9)

    assert batch.obs_global.shape == (2, 1, 1)
    assert batch.actions.shape == (2, 1)
    assert batch.rewards.shape == (2, 1)
    assert batch.rnn_states is not None
    assert batch.rnn_states.shape == (2, 1, 1, 2)
    np.testing.assert_array_equal(batch.rnn_states, np.zeros((2, 1, 1, 2), dtype=np.float32))
    assert batch.policy_version == 9


def test_recurrent_buffer_rejects_flat_lstm_state_export() -> None:
    buffer = _tiny_buffer(capacity=1)
    transition = _transition(0)
    transition["rnn_state"] = (
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
    )
    buffer.add(**transition)
    buffer.compute_returns(last_value=np.array([0.0], dtype=np.float32), gamma=1.0, gae_lambda=1.0)

    with pytest.raises(ValueError, match="LSTM"):
        buffer.to_batch(policy_version=1)


def test_recurrent_buffer_rejects_overfill() -> None:
    buffer = _tiny_buffer(capacity=1)
    buffer.add(**_transition(0))

    with pytest.raises(RuntimeError, match="full"):
        buffer.add(**_transition(1))


def test_recurrent_buffer_rejects_non_finite_rnn_state() -> None:
    buffer = _tiny_buffer(capacity=1)
    transition = _transition(0)
    transition["rnn_state"] = np.array([[[np.nan, 0.0]]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        buffer.add(**transition)


def _tiny_buffer(capacity: int, sequence_length: int = 1) -> RecurrentRolloutBuffer:
    return RecurrentRolloutBuffer(
        capacity=capacity,
        num_envs=1,
        sequence_length=sequence_length,
        obs_spec={
            "global": (1,),
            "player": (1,),
            "entities": (1, 1),
            "entity_mask": (1,),
            "action": (),
            "action_mask": (1,),
        },
    )


def _transition(step: int) -> dict[str, object]:
    return {
        "obs": {
            "global": np.array([step], dtype=np.float32),
            "player": np.array([0.0], dtype=np.float32),
            "entities": np.array([[0.0]], dtype=np.float32),
            "entity_mask": np.array([True]),
        },
        "action": np.array(step, dtype=np.int64),
        "log_prob": np.array([0.0], dtype=np.float32),
        "value": np.array([0.0], dtype=np.float32),
        "reward": np.array([0.0], dtype=np.float32),
        "prev_reward": np.array([float(step)], dtype=np.float32),
        "done": np.array([False]),
        "truncated": np.array([False]),
        "action_mask": np.array([True]),
        "rnn_state": np.zeros((1, 1, 2), dtype=np.float32),
    }
