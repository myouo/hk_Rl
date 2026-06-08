"""Synchronous PPO (PRD Phase 3, ADR-0001).

Clipped-objective PPO over a flat RolloutBuffer. Model-agnostic via the
ActorCritic interface; logs the training metrics from docs/metrics.md
(policy_loss, value_loss, entropy, kl, explained_variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic
from hkrl.training.rollout_buffer import RolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import register_algo


@register_algo("ppo")
class PPO:
    """Vanilla clipped PPO learner.

    Reserve performance levers on the update path: ``torch.compile`` and AMP
    (mixed precision) — see docs/model_architecture.md §5.
    """

    def __init__(self, model: ActorCritic, config: TrainConfig) -> None:
        if config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if config.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")

        self.model = model
        self.cfg = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run ``epochs`` of minibatch updates; return training metrics."""
        device = _model_device(self.model)
        batch = _tensor_batch(buffer, device)
        advantages = _normalize_advantages(batch.advantages)
        num_samples = batch.old_log_probs.shape[0]
        if num_samples == 0:
            raise ValueError("rollout buffer is empty")

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
            indices = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, self.cfg.minibatch_size):
                mb = indices[start : start + self.cfg.minibatch_size]
                metrics = self._update_minibatch(batch, advantages, mb)
                mb_size = int(mb.numel())
                for key, value in metrics.items():
                    totals[key] += value * mb_size
                seen += mb_size

        metrics = {key: value / seen for key, value in totals.items()}
        metrics["explained_variance"] = self._explained_variance(batch)
        return metrics

    def _update_minibatch(
        self,
        batch: _TensorBatch,
        advantages: Tensor,
        indices: Tensor,
    ) -> dict[str, float]:
        obs = _index_obs(batch.obs, indices)
        actions = batch.actions.index_select(0, indices)
        old_log_probs = batch.old_log_probs.index_select(0, indices)
        returns = batch.returns.index_select(0, indices)
        old_values = batch.old_values.index_select(0, indices)
        mb_advantages = advantages.index_select(0, indices)
        action_masks = (
            None if batch.action_masks is None else batch.action_masks.index_select(0, indices)
        )

        log_probs, entropy, values = self.model.evaluate_actions(
            obs,
            actions,
            action_mask=action_masks,
        )
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped_policy = ratio * mb_advantages
        clipped_policy = (
            torch.clamp(
                ratio,
                1.0 - self.cfg.clip_range,
                1.0 + self.cfg.clip_range,
            )
            * mb_advantages
        )
        policy_loss = -torch.minimum(unclipped_policy, clipped_policy).mean()

        value_pred_clipped = old_values + (values - old_values).clamp(
            -self.cfg.clip_range,
            self.cfg.clip_range,
        )
        value_loss_unclipped = (values - returns).square()
        value_loss_clipped = (value_pred_clipped - returns).square()
        value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()

        entropy_mean = entropy.mean()
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_mean

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        approx_kl = (old_log_probs - log_probs).mean()
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "action_entropy": float(entropy_mean.detach().cpu()),
            "policy_kl": float(approx_kl.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

    def _explained_variance(self, batch: _TensorBatch) -> float:
        with torch.no_grad():
            _, _, values = self.model.evaluate_actions(
                batch.obs,
                batch.actions,
                action_mask=batch.action_masks,
            )
            target_var = torch.var(batch.returns, unbiased=False)
            if float(target_var.cpu()) < 1.0e-8:
                return 0.0
            residual_var = torch.var(batch.returns - values, unbiased=False)
            explained = 1.0 - residual_var / target_var
        return float(explained.cpu())


@dataclass(frozen=True)
class _TensorBatch:
    obs: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor
    action_masks: Tensor | None


def _tensor_batch(buffer: RolloutBuffer, device: torch.device) -> _TensorBatch:
    batch = buffer.to_batch(policy_version=0)
    obs = {
        "global": _flatten_time_env(batch.obs_global, device, dtype=torch.float32),
        "player": _flatten_time_env(batch.obs_player, device, dtype=torch.float32),
        "entities": _flatten_time_env(batch.obs_entities, device, dtype=torch.float32),
        "entity_mask": _flatten_time_env(batch.entity_mask, device, dtype=torch.bool),
    }
    action_masks = None
    if batch.action_masks.ndim > 2:
        action_masks = _flatten_time_env(batch.action_masks, device, dtype=torch.bool)

    return _TensorBatch(
        obs=obs,
        actions=_flatten_time_env(batch.actions, device, dtype=torch.long),
        old_log_probs=_flat_vector(batch.log_probs, device),
        old_values=_flat_vector(batch.values, device),
        returns=_flat_vector(batch.returns, device),
        advantages=_flat_vector(batch.advantages, device),
        action_masks=action_masks,
    )


def _flatten_time_env(array: object, device: torch.device, *, dtype: torch.dtype) -> Tensor:
    tensor = torch.as_tensor(array, dtype=dtype, device=device)
    if tensor.ndim < 2:
        raise ValueError("rollout arrays must have time and env dimensions")
    return tensor.reshape((-1, *tensor.shape[2:]))


def _flat_vector(array: object, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device).reshape(-1)


def _normalize_advantages(advantages: Tensor) -> Tensor:
    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.cpu()) < 1.0e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1.0e-8)


def _index_obs(obs: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {key: value.index_select(0, indices) for key, value in obs.items()}


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
