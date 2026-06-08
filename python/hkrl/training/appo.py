"""Asynchronous PPO-style learner (PRD Phase 6/8, §9.5).

For async multi-worker sampling where rollouts may be off-policy. Filters/
drops batches by ``policy_version`` and applies clipped PPO importance ratios to
tolerate bounded staleness (docs/distributed_training.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic
from hkrl.training.rollout_buffer import RolloutBatch
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import register_algo


@register_algo("appo")
class APPO:
    """Async PPO with staleness handling.

    Accepts RolloutBatches from many workers, drops those older than a version
    threshold, and corrects for off-policyness.
    """

    def __init__(self, model: ActorCritic, config: TrainConfig, max_staleness: int = 4) -> None:
        if config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if config.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")

        self.model = model
        self.cfg = config
        self.max_staleness = max_staleness
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.current_version = 0
        self._queue: list[RolloutBatch] = []

    def ingest(self, batch: RolloutBatch, current_version: int) -> bool:
        """Accept or reject a batch by staleness; returns True if used.

        ``current_version`` is the learner's policy version at intake time.
        """
        if batch.rewards.size == 0:
            return False
        if not _batch_has_finite_training_values(batch):
            return False
        if batch.policy_version > current_version:
            return False
        if current_version - batch.policy_version > self.max_staleness:
            return False
        self._queue.append(batch)
        return True

    def update(self) -> dict[str, float]:
        """Run an async update step over accepted batches; return metrics."""
        if not self._queue:
            raise ValueError("APPO update requires at least one accepted batch")

        device = _model_device(self.model)
        batch = _tensor_batch(self._queue, device)
        self._queue = []
        advantages = _normalize_advantages(batch.advantages)
        num_samples = batch.old_log_probs.shape[0]
        if num_samples == 0:
            raise ValueError("accepted APPO batches contain no samples")

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

        self.current_version += 1
        metrics = {key: value / seen for key, value in totals.items()}
        metrics["explained_variance"] = self._explained_variance(batch)
        metrics["policy_version"] = float(self.current_version)
        metrics["samples"] = float(num_samples)
        return metrics

    @property
    def queued_batches(self) -> int:
        return len(self._queue)

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
        rnn_state = None if batch.rnn_state is None else batch.rnn_state.index_select(1, indices)

        log_probs, entropy, values = self.model.evaluate_actions(
            obs,
            actions,
            rnn_state=rnn_state,
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
                rnn_state=batch.rnn_state,
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
    rnn_state: Tensor | None


def _tensor_batch(batches: list[RolloutBatch], device: torch.device) -> _TensorBatch:
    obs = {
        "global": _flatten_time_env(_concat(batches, "obs_global"), device, dtype=torch.float32),
        "player": _flatten_time_env(_concat(batches, "obs_player"), device, dtype=torch.float32),
        "entities": _flatten_time_env(
            _concat(batches, "obs_entities"), device, dtype=torch.float32
        ),
        "entity_mask": _flatten_time_env(_concat(batches, "entity_mask"), device, dtype=torch.bool),
        "prev_action": _flatten_time_env(
            _concat(batches, "prev_actions"), device, dtype=torch.float32
        ),
        "prev_reward": _flat_vector(_concat(batches, "prev_rewards"), device),
    }
    action_masks = None
    action_mask_array = _concat(batches, "action_masks")
    if action_mask_array.ndim > 2:
        action_masks = _flatten_time_env(action_mask_array, device, dtype=torch.bool)

    return _TensorBatch(
        obs=obs,
        actions=_flatten_time_env(_concat(batches, "actions"), device, dtype=torch.long),
        old_log_probs=_flat_vector(_concat(batches, "log_probs"), device),
        old_values=_flat_vector(_concat(batches, "values"), device),
        returns=_flat_vector(_concat(batches, "returns"), device),
        advantages=_flat_vector(_concat(batches, "advantages"), device),
        action_masks=action_masks,
        rnn_state=_flatten_rnn_states(batches, device),
    )


def _concat(batches: list[RolloutBatch], field: str) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(batch, field)) for batch in batches], axis=0)


def _batch_has_finite_training_values(batch: RolloutBatch) -> bool:
    for field in (
        "obs_global",
        "obs_player",
        "obs_entities",
        "log_probs",
        "values",
        "advantages",
        "returns",
        "rewards",
        "prev_rewards",
    ):
        if not np.isfinite(np.asarray(getattr(batch, field))).all():
            return False
    return batch.rnn_states is None or bool(np.isfinite(np.asarray(batch.rnn_states)).all())


def _flatten_time_env(array: object, device: torch.device, *, dtype: torch.dtype) -> Tensor:
    tensor = torch.as_tensor(array, dtype=dtype, device=device)
    if tensor.ndim < 2:
        raise ValueError("rollout arrays must have time and env dimensions")
    return tensor.reshape((-1, *tensor.shape[2:]))


def _flat_vector(array: object, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device).reshape(-1)


def _flatten_rnn_states(batches: list[RolloutBatch], device: torch.device) -> Tensor | None:
    states = [batch.rnn_states for batch in batches]
    if all(state is None for state in states):
        return None
    if any(state is None for state in states):
        raise ValueError("cannot mix RolloutBatches with and without rnn_states")

    array = np.concatenate([np.asarray(state) for state in states], axis=0)
    if array.ndim != 4:
        raise ValueError("rnn_states must have shape (time, layers, envs, hidden)")

    time, layers, envs, hidden = array.shape
    flat = np.transpose(array, (1, 0, 2, 3)).reshape(layers, time * envs, hidden)
    return torch.as_tensor(flat, dtype=torch.float32, device=device)


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
