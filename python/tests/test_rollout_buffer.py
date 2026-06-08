"""RolloutBatch field-contract tests (must match docs/distributed_training.md §3)."""

from __future__ import annotations

import dataclasses

import pytest
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


@pytest.mark.xfail(reason="GAE implementation lands in phase 3", strict=True)
def test_compute_gae() -> None:
    import numpy as np
    from hkrl.training.gae import compute_gae

    compute_gae(
        rewards=np.zeros((4, 1)),
        values=np.zeros((4, 1)),
        dones=np.zeros((4, 1)),
        truncateds=np.zeros((4, 1)),
        last_value=np.zeros((1,)),
    )
