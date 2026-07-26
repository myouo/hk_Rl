"""Asynchronous PPO-style learner (PRD Phase 6/8, §9.5).

For async multi-worker sampling where rollouts may be off-policy. Filters/
drops batches by ``policy_version`` and applies clipped PPO importance ratios to
tolerate bounded staleness (docs/distributed_training.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hkrl.models.base import ActorCritic
from hkrl.training.accelerator import TorchLearnerRuntime
from hkrl.training.numerics import require_finite_tensors
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
        self.runtime = TorchLearnerRuntime(model, config)
        self.optimizer = self.runtime.optimizer
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
        if not _batch_is_model_compatible(self.model, batch):
            return False
        self._queue.append(batch)
        return True

    def update(self) -> dict[str, float]:
        """Run an async update step over accepted batches; return metrics."""
        if not self._queue:
            raise ValueError("APPO update requires at least one accepted batch")

        device = self.runtime.device
        batch = _tensor_batch(self._queue, device)
        self._queue = []
        advantages = _normalize_advantages(
            batch.advantages,
            task_ids=batch.task_ids,
            by_task=self.cfg.learner.normalize_advantages_by_task,
        )
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
        optimizer_steps = 0
        epochs_completed = 0
        kl_early_stop = False

        for epoch in range(self.cfg.epochs):
            indices = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, self.cfg.minibatch_size):
                mb = indices[start : start + self.cfg.minibatch_size]
                metrics = self._update_minibatch(batch, advantages, mb)
                mb_size = int(mb.numel())
                for key, value in metrics.items():
                    totals[key] += value * mb_size
                seen += mb_size
                optimizer_steps += 1
                target_kl = self.cfg.learner.target_kl
                if target_kl is not None and metrics["policy_kl"] > target_kl:
                    kl_early_stop = True
                    break
            epochs_completed = epoch + 1
            if kl_early_stop:
                break

        self.current_version += 1
        metrics = {key: value / seen for key, value in totals.items()}
        metrics["explained_variance"] = self._explained_variance(batch)
        metrics["policy_version"] = float(self.current_version)
        metrics["samples"] = float(num_samples)
        metrics["task_count"] = float(torch.unique(batch.task_ids).numel())
        metrics["optimizer_steps"] = float(optimizer_steps)
        metrics["epochs_completed"] = float(epochs_completed)
        metrics["kl_early_stop"] = float(kl_early_stop)
        metrics.update(self.runtime.metric_flags())
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

        with self.runtime.autocast():
            log_probs, entropy, values = self.runtime.evaluate_actions(
                obs,
                actions,
                rnn_state=rnn_state,
                action_mask=action_masks,
            )
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
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
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * entropy_mean
            )
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
        require_finite_tensors(
            (
                ("model log_probs", log_probs),
                ("model entropy", entropy),
                ("model values", values),
                ("appo ratio", ratio),
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy_mean", entropy_mean),
                ("policy_kl", approx_kl),
                ("loss", loss),
            )
        )

        grad_norm = self.runtime.backward_step(
            loss,
            max_grad_norm=self.cfg.max_grad_norm,
        )
        metric_names = (
            "policy_loss",
            "value_loss",
            "action_entropy",
            "policy_kl",
            "grad_norm",
        )
        metric_values = (
            torch.stack(
                (
                    policy_loss.detach().float(),
                    value_loss.detach().float(),
                    entropy_mean.detach().float(),
                    approx_kl.detach().float(),
                    grad_norm.detach().float(),
                )
            )
            .cpu()
            .tolist()
        )
        return dict(zip(metric_names, metric_values, strict=True))

    def _explained_variance(self, batch: _TensorBatch) -> float:
        with torch.no_grad():
            with self.runtime.autocast():
                _, _, values = self.runtime.evaluate_actions(
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
    task_ids: Tensor
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
        task_ids=_flat_int_vector(_concat(batches, "task_ids"), device),
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


def _batch_is_model_compatible(model: ActorCritic, batch: RolloutBatch) -> bool:
    """Return False when a worker batch cannot be evaluated by this learner model."""
    try:
        device = _model_device(model)
        tensor_batch = _tensor_batch([_first_transition_batch(batch)], device)
        with torch.no_grad():
            log_probs, entropy, values = model.evaluate_actions(
                tensor_batch.obs,
                tensor_batch.actions,
                rnn_state=tensor_batch.rnn_state,
                action_mask=tensor_batch.action_masks,
            )
    except Exception:
        return False

    expected = tensor_batch.old_log_probs.shape
    return (
        tuple(log_probs.shape) == tuple(expected)
        and tuple(entropy.shape) == tuple(expected)
        and tuple(values.shape) == tuple(expected)
    )


def _first_transition_batch(batch: RolloutBatch) -> RolloutBatch:
    return RolloutBatch(
        obs_global=_first_time_env(batch.obs_global),
        obs_player=_first_time_env(batch.obs_player),
        obs_entities=_first_time_env(batch.obs_entities),
        entity_mask=_first_time_env(batch.entity_mask),
        actions=_first_time_env(batch.actions),
        log_probs=_first_time_env(batch.log_probs),
        values=_first_time_env(batch.values),
        advantages=_first_time_env(batch.advantages),
        returns=_first_time_env(batch.returns),
        rewards=_first_time_env(batch.rewards),
        dones=_first_time_env(batch.dones),
        truncateds=_first_time_env(batch.truncateds),
        action_masks=_first_time_env(batch.action_masks),
        prev_actions=_first_time_env(batch.prev_actions),
        prev_rewards=_first_time_env(batch.prev_rewards),
        rnn_states=None if batch.rnn_states is None else _first_rnn_state(batch.rnn_states),
        episode_ids=_first_time_env(batch.episode_ids),
        task_ids=_first_time_env(batch.task_ids),
        policy_version=batch.policy_version,
    )


def _first_time_env(array: object) -> np.ndarray:
    return np.asarray(array)[:1, :1].copy()


def _first_rnn_state(array: object) -> np.ndarray:
    return np.asarray(array)[:1, :, :1, :].copy()


def _flatten_time_env(array: object, device: torch.device, *, dtype: torch.dtype) -> Tensor:
    tensor = torch.as_tensor(array, dtype=dtype, device=device)
    if tensor.ndim < 2:
        raise ValueError("rollout arrays must have time and env dimensions")
    return tensor.reshape((-1, *tensor.shape[2:]))


def _flat_vector(array: object, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device).reshape(-1)


def _flat_int_vector(array: object, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.long, device=device).reshape(-1)


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


def _normalize_advantages(
    advantages: Tensor,
    *,
    task_ids: Tensor | None = None,
    by_task: bool = False,
) -> Tensor:
    if advantages.numel() <= 1:
        return advantages
    if not by_task:
        return _normalize_group(advantages)
    if task_ids is None or task_ids.shape != advantages.shape:
        raise ValueError("task_ids must match advantages for task-wise normalization")

    _, inverse, counts = torch.unique(
        task_ids,
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    group_counts = counts.to(dtype=advantages.dtype)
    group_sums = torch.zeros_like(group_counts).scatter_add_(
        0,
        inverse,
        advantages,
    )
    group_means = group_sums / group_counts
    centered = advantages - group_means.index_select(0, inverse)
    group_square_sums = torch.zeros_like(group_counts).scatter_add_(
        0,
        inverse,
        centered.square(),
    )
    group_std = (group_square_sums / group_counts).sqrt().clamp_min(1.0e-8)
    normalized = centered / group_std.index_select(0, inverse)
    enough_samples = counts.index_select(0, inverse) > 1
    return torch.where(enough_samples, normalized, advantages)


def _normalize_group(values: Tensor) -> Tensor:
    if values.numel() <= 1:
        return values
    centered = values - values.mean()
    std = values.std(unbiased=False)
    return centered / std.clamp_min(1.0e-8)


def _index_obs(obs: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {key: value.index_select(0, indices) for key, value in obs.items()}


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
