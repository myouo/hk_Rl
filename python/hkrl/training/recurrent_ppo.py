"""Recurrent PPO over sequences (PRD Phase 5).

PPO trained with truncated BPTT on the RecurrentRolloutBuffer. Handles hidden
state propagation within a sequence, burn-in, and padded-timestep masking so the
loss only counts valid steps (docs/model_architecture.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic, RnnState
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer, RecurrentSequenceBatch
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import register_algo


@register_algo("recurrent_ppo")
class RecurrentPPO:
    """PPO learner for recurrent ActorCritic models."""

    def __init__(self, model: ActorCritic, config: TrainConfig) -> None:
        if config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if config.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")

        self.model = model
        self.cfg = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self._rng = np.random.default_rng(config.seed)

    def update(self, buffer: RecurrentRolloutBuffer) -> dict[str, float]:
        """Sequence-minibatch PPO update with masked loss; returns metrics."""
        device = _model_device(self.model)
        advantage_mean, advantage_std = _advantage_stats(buffer, device)
        sequence_minibatch_size = max(1, self.cfg.minibatch_size // buffer.sequence_length)

        self.model.train()
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "action_entropy": 0.0,
            "policy_kl": 0.0,
            "grad_norm": 0.0,
        }
        seen = 0

        for _ in range(self.cfg.epochs):
            for sequence in buffer.iter_sequences(
                minibatch_size=sequence_minibatch_size,
                shuffle=True,
                rng=self._rng,
            ):
                batch = _tensor_sequence(sequence, device)
                metrics, valid_count = self._update_sequence(
                    batch,
                    advantage_mean,
                    advantage_std,
                )
                for key, value in metrics.items():
                    totals[key] += value * valid_count
                seen += valid_count

        if seen == 0:
            raise ValueError("recurrent rollout buffer has no valid loss steps")

        metrics = {key: value / seen for key, value in totals.items()}
        metrics["explained_variance"] = self._explained_variance(
            buffer,
            sequence_minibatch_size,
            device,
        )
        return metrics

    def _update_sequence(
        self,
        batch: _TensorSequence,
        advantage_mean: Tensor,
        advantage_std: Tensor,
    ) -> tuple[dict[str, float], int]:
        valid = batch.loss_mask
        valid_count = int(valid.sum().detach().cpu())
        if valid_count == 0:
            return _zero_metrics(), 0

        advantages = _normalize_advantages(batch.advantages, advantage_mean, advantage_std)
        log_probs, entropy, values = self.model.evaluate_actions(
            batch.obs,
            batch.actions,
            rnn_state=batch.rnn_state,
            action_mask=batch.action_masks,
        )
        ratio = torch.exp(log_probs - batch.old_log_probs)
        unclipped_policy = ratio * advantages
        clipped_policy = (
            torch.clamp(
                ratio,
                1.0 - self.cfg.clip_range,
                1.0 + self.cfg.clip_range,
            )
            * advantages
        )
        policy_loss = -_masked_mean(torch.minimum(unclipped_policy, clipped_policy), valid)

        value_pred_clipped = batch.old_values + (values - batch.old_values).clamp(
            -self.cfg.clip_range,
            self.cfg.clip_range,
        )
        value_loss_unclipped = (values - batch.returns).square()
        value_loss_clipped = (value_pred_clipped - batch.returns).square()
        value_loss = 0.5 * _masked_mean(
            torch.maximum(value_loss_unclipped, value_loss_clipped),
            valid,
        )

        entropy_mean = _masked_mean(entropy, valid)
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_mean

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        approx_kl = _masked_mean(batch.old_log_probs - log_probs, valid)
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "action_entropy": float(entropy_mean.detach().cpu()),
            "policy_kl": float(approx_kl.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }, valid_count

    def _explained_variance(
        self,
        buffer: RecurrentRolloutBuffer,
        sequence_minibatch_size: int,
        device: torch.device,
    ) -> float:
        returns: list[Tensor] = []
        predictions: list[Tensor] = []
        with torch.no_grad():
            for sequence in buffer.iter_sequences(minibatch_size=sequence_minibatch_size):
                batch = _tensor_sequence(sequence, device)
                _, _, values = self.model.evaluate_actions(
                    batch.obs,
                    batch.actions,
                    rnn_state=batch.rnn_state,
                    action_mask=batch.action_masks,
                )
                returns.append(batch.returns[batch.loss_mask])
                predictions.append(values[batch.loss_mask])

        if not returns:
            return 0.0
        target = torch.cat(returns)
        predicted = torch.cat(predictions)
        target_var = torch.var(target, unbiased=False)
        if float(target_var.cpu()) < 1.0e-8:
            return 0.0
        residual_var = torch.var(target - predicted, unbiased=False)
        explained = 1.0 - residual_var / target_var
        return float(explained.cpu())


@dataclass(frozen=True)
class _TensorSequence:
    obs: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor
    action_masks: Tensor | None
    rnn_state: RnnState
    loss_mask: Tensor


def _tensor_sequence(sequence: RecurrentSequenceBatch, device: torch.device) -> _TensorSequence:
    obs = {
        "global": torch.as_tensor(sequence.obs["global"], dtype=torch.float32, device=device),
        "player": torch.as_tensor(sequence.obs["player"], dtype=torch.float32, device=device),
        "entities": torch.as_tensor(sequence.obs["entities"], dtype=torch.float32, device=device),
        "entity_mask": torch.as_tensor(
            sequence.obs["entity_mask"], dtype=torch.bool, device=device
        ),
        "prev_action": torch.as_tensor(sequence.prev_actions, dtype=torch.float32, device=device),
        "prev_reward": torch.as_tensor(sequence.prev_rewards, dtype=torch.float32, device=device),
    }
    action_masks = None
    if sequence.action_masks.ndim > 2:
        action_masks = torch.as_tensor(sequence.action_masks, dtype=torch.bool, device=device)

    return _TensorSequence(
        obs=obs,
        actions=torch.as_tensor(sequence.actions, dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(sequence.old_log_probs, dtype=torch.float32, device=device),
        old_values=torch.as_tensor(sequence.old_values, dtype=torch.float32, device=device),
        returns=torch.as_tensor(sequence.returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(sequence.advantages, dtype=torch.float32, device=device),
        action_masks=action_masks,
        rnn_state=_to_tensor_rnn_state(sequence.rnn_state, device),
        loss_mask=torch.as_tensor(sequence.loss_mask, dtype=torch.bool, device=device),
    )


def _advantage_stats(
    buffer: RecurrentRolloutBuffer,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    values: list[Tensor] = []
    for sequence in buffer.iter_sequences():
        advantages = torch.as_tensor(sequence.advantages, dtype=torch.float32, device=device)
        loss_mask = torch.as_tensor(sequence.loss_mask, dtype=torch.bool, device=device)
        values.append(advantages[loss_mask])

    if not values:
        raise ValueError("recurrent rollout buffer is empty")
    flat = torch.cat(values)
    if flat.numel() == 0:
        raise ValueError("recurrent rollout buffer has no valid loss steps")

    std = flat.std(unbiased=False)
    if float(std.cpu()) < 1.0e-8:
        std = torch.ones((), dtype=flat.dtype, device=flat.device)
    return flat.mean(), std


def _normalize_advantages(advantages: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (advantages - mean) / (std + 1.0e-8)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    masked = values[mask]
    if masked.numel() == 0:
        raise ValueError("loss mask selected no values")
    return masked.mean()


def _to_tensor_rnn_state(state: Any, device: torch.device) -> RnnState:
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_to_tensor_rnn_state(part, device) for part in state)
    return torch.as_tensor(state, dtype=torch.float32, device=device)


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _zero_metrics() -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "action_entropy": 0.0,
        "policy_kl": 0.0,
        "grad_norm": 0.0,
    }
