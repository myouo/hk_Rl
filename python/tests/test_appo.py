"""APPO learner tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from hkrl.models.base import ActorCritic
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
from hkrl.training.appo import APPO
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer
from hkrl.utils.config import TrainConfig


def test_appo_ingest_filters_stale_and_future_batches() -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    appo = APPO(model, TrainConfig(algorithm="appo"), max_staleness=2)

    assert appo.ingest(_empty_batch(policy_version=3), current_version=3)
    assert appo.ingest(_empty_batch(policy_version=1), current_version=3)
    assert not appo.ingest(_empty_batch(policy_version=0), current_version=3)
    assert not appo.ingest(_empty_batch(policy_version=4), current_version=3)
    assert appo.queued_batches == 2


def test_appo_update_returns_metrics_changes_parameters_and_clears_queue() -> None:
    torch.manual_seed(123)
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    cfg = TrainConfig(
        algorithm="appo",
        epochs=2,
        minibatch_size=2,
        learning_rate=1.0e-2,
        entropy_coef=0.0,
    )
    appo = APPO(model, cfg, max_staleness=4)
    batch = _rollout_batch(model)

    assert appo.ingest(batch, current_version=2)
    before = [param.detach().clone() for param in model.parameters()]
    metrics = appo.update()

    assert appo.queued_batches == 0
    assert appo.current_version == 1
    for key in (
        "policy_loss",
        "value_loss",
        "action_entropy",
        "policy_kl",
        "explained_variance",
        "grad_norm",
        "policy_version",
        "samples",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["samples"] == 4.0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_appo_passes_rollout_rnn_states_to_model() -> None:
    model = _RnnAwareActorCritic()
    cfg = TrainConfig(
        algorithm="appo",
        epochs=1,
        minibatch_size=2,
        learning_rate=1.0e-2,
        entropy_coef=0.0,
    )
    appo = APPO(model, cfg, max_staleness=1)

    assert appo.ingest(_rnn_batch(policy_version=3), current_version=3)
    metrics = appo.update()

    assert metrics["samples"] == 4.0
    assert (1, 2, 3) in model.seen_rnn_shapes
    assert model.seen_rnn_shapes[-1] == (1, 4, 3)


class _RnnAwareActorCritic(ActorCritic):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.seen_rnn_shapes: list[tuple[int, ...] | None] = []

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.zeros((1, batch_size, 3), device=device)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        rnn_state: Any = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        batch_size = obs["global"].shape[0]
        return None, self.weight.expand(batch_size), self.initial_state(batch_size)

    def act(
        self,
        obs: dict[str, torch.Tensor],
        rnn_state: Any = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs["global"].shape[0]
        return (
            torch.zeros((batch_size, 12), dtype=torch.long),
            self.weight.expand(batch_size),
            self.weight.expand(batch_size),
            self.initial_state(batch_size),
        )

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rnn_state: Any = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.seen_rnn_shapes.append(None if rnn_state is None else tuple(rnn_state.shape))
        batch_size = actions.shape[0]
        value = self.weight.expand(batch_size)
        log_prob = self.weight.expand(batch_size)
        entropy = self.weight.expand(batch_size) * 0.0
        return log_prob, entropy, value


def _rollout_batch(model: MlpActorCritic) -> RolloutBatch:
    buffer = RolloutBuffer(
        capacity=4,
        num_envs=1,
        obs_spec={
            **_obs_spec(),
            "action": (12,),
            "action_mask": (_mask_dim(),),
        },
    )
    action_mask = np.ones((_mask_dim(),), dtype=bool)
    for step in range(4):
        obs = _numpy_obs(step)
        with torch.no_grad():
            action, log_prob, value, _ = model.act(
                _torch_obs(obs),
                action_mask=torch.as_tensor(action_mask[None, :]),
            )
        buffer.add(
            obs=obs,
            action=action.numpy(),
            log_prob=log_prob.numpy(),
            value=value.numpy(),
            reward=np.array([1.0 + step], dtype=np.float32),
            done=np.array([step == 3]),
            truncated=np.array([False]),
            action_mask=action_mask,
        )
    buffer.compute_returns(
        last_value=np.array([0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95
    )
    return buffer.to_batch(policy_version=1)


def _rnn_batch(policy_version: int) -> RolloutBatch:
    return RolloutBatch(
        obs_global=np.zeros((4, 1, 2), dtype=np.float32),
        obs_player=np.zeros((4, 1, 3), dtype=np.float32),
        obs_entities=np.zeros((4, 1, 4, 5), dtype=np.float32),
        entity_mask=np.ones((4, 1, 4), dtype=bool),
        actions=np.zeros((4, 1, 12), dtype=np.int64),
        log_probs=np.zeros((4, 1), dtype=np.float32),
        values=np.zeros((4, 1), dtype=np.float32),
        advantages=np.arange(1, 5, dtype=np.float32).reshape(4, 1),
        returns=np.arange(1, 5, dtype=np.float32).reshape(4, 1),
        rewards=np.ones((4, 1), dtype=np.float32),
        dones=np.array([[False], [False], [False], [True]]),
        truncateds=np.zeros((4, 1), dtype=bool),
        action_masks=np.ones((4, 1, _mask_dim()), dtype=bool),
        prev_actions=np.zeros((4, 1, 12), dtype=np.int64),
        rnn_states=np.arange(12, dtype=np.float32).reshape(4, 1, 1, 3),
        episode_ids=np.ones((4, 1), dtype=np.uint64),
        task_ids=np.zeros((4, 1), dtype=np.int64),
        policy_version=policy_version,
    )


def _empty_batch(policy_version: int) -> RolloutBatch:
    zeros = np.zeros((0, 1), dtype=np.float32)
    return RolloutBatch(
        obs_global=np.zeros((0, 1, 2), dtype=np.float32),
        obs_player=np.zeros((0, 1, 3), dtype=np.float32),
        obs_entities=np.zeros((0, 1, 4, 5), dtype=np.float32),
        entity_mask=np.zeros((0, 1, 4), dtype=bool),
        actions=np.zeros((0, 1, 12), dtype=np.int64),
        log_probs=zeros,
        values=zeros,
        advantages=zeros,
        returns=zeros,
        rewards=zeros,
        dones=np.zeros((0, 1), dtype=bool),
        truncateds=np.zeros((0, 1), dtype=bool),
        action_masks=np.zeros((0, 1, _mask_dim()), dtype=bool),
        prev_actions=np.zeros((0, 1, 12), dtype=np.int64),
        rnn_states=None,
        episode_ids=np.zeros((0, 1), dtype=np.uint64),
        task_ids=np.zeros((0, 1), dtype=np.int64),
        policy_version=policy_version,
    )


def _obs_spec() -> dict[str, tuple[int, ...]]:
    return {
        "global": (2,),
        "player": (3,),
        "entities": (4, 5),
        "entity_mask": (4,),
    }


def _numpy_obs(step: int) -> dict[str, np.ndarray]:
    return {
        "global": np.array([step, step + 0.5], dtype=np.float32),
        "player": np.ones((3,), dtype=np.float32) * step,
        "entities": np.ones((4, 5), dtype=np.float32) * (step + 1),
        "entity_mask": np.array([True, True, False, False]),
    }


def _torch_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "global": torch.as_tensor(obs["global"][None, :]),
        "player": torch.as_tensor(obs["player"][None, :]),
        "entities": torch.as_tensor(obs["entities"][None, :, :]),
        "entity_mask": torch.as_tensor(obs["entity_mask"][None, :]),
    }


def _mask_dim() -> int:
    return N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION
