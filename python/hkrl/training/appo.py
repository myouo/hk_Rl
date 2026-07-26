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
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
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
        if config.burn_in != 0:
            raise ValueError(
                "APPO starts each chunk from its recorded hidden state and requires burn_in=0"
            )
        if max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")

        self.model = model
        self.cfg = config
        self.max_staleness = max_staleness
        self.runtime = TorchLearnerRuntime(model, config)
        self.optimizer = self.runtime.optimizer
        self.current_version = 0
        self._queue: list[RolloutBatch] = []
        self._accepted_signature: tuple[object, ...] | None = None

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
        if not _batch_has_valid_actions(self.model, batch):
            return False
        if not _batch_has_valid_action_masks(self.model, batch):
            return False
        signature = _batch_signature(batch)
        if self._accepted_signature is None:
            if not _batch_is_model_compatible(self.model, batch):
                return False
            self._accepted_signature = signature
        elif signature != self._accepted_signature:
            return False
        self._queue.append(batch)
        return True

    def update(self) -> dict[str, float]:
        """Run an async update step over accepted batches; return metrics."""
        if not self._queue:
            raise ValueError("APPO update requires at least one accepted batch")

        device = self.runtime.device
        batch = _tensor_batch(
            self._queue,
            device,
            sequence_length=self.cfg.sequence_length,
        )
        self._queue = []
        advantages = _normalize_advantages(
            batch.advantages,
            task_ids=batch.task_ids,
            by_task=self.cfg.learner.normalize_advantages_by_task,
            loss_mask=batch.loss_mask,
        )
        num_samples = int(batch.loss_mask.sum().detach().cpu())
        if num_samples == 0:
            raise ValueError("accepted APPO batches contain no samples")
        num_sequences = int(batch.loss_mask.shape[0])
        sequence_minibatch_size = max(
            1,
            self.cfg.minibatch_size // self.cfg.sequence_length,
        )

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
            indices = torch.randperm(num_sequences, device=device)
            for start in range(0, num_sequences, sequence_minibatch_size):
                raw_mb = indices[start : start + sequence_minibatch_size]
                mb, active_sequences = _pad_sequence_indices(
                    raw_mb,
                    sequence_minibatch_size
                    if self.runtime.compile_enabled
                    else int(raw_mb.numel()),
                )
                metrics, valid_count = self._update_minibatch(
                    batch,
                    advantages,
                    mb,
                    active_sequences,
                )
                for key, value in metrics.items():
                    totals[key] += value * valid_count
                seen += valid_count
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
        metrics["task_count"] = float(torch.unique(batch.task_ids[batch.loss_mask]).numel())
        metrics["sequence_count"] = float(num_sequences)
        metrics["sequence_length"] = float(self.cfg.sequence_length)
        metrics["bptt_enabled"] = float(
            batch.rnn_state is not None and self.cfg.sequence_length > 1
        )
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
        active_sequences: Tensor,
    ) -> tuple[dict[str, float], int]:
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
        loss_mask = batch.loss_mask.index_select(0, indices)
        loss_mask = loss_mask & active_sequences.unsqueeze(-1)
        valid_count = int(loss_mask.sum().detach().cpu())
        if valid_count == 0:
            raise ValueError("APPO sequence minibatch has no valid loss steps")

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
            policy_loss = -_masked_mean(
                torch.minimum(unclipped_policy, clipped_policy),
                loss_mask,
            )

            value_pred_clipped = old_values + (values - old_values).clamp(
                -self.cfg.clip_range,
                self.cfg.clip_range,
            )
            value_loss_unclipped = (values - returns).square()
            value_loss_clipped = (value_pred_clipped - returns).square()
            value_loss = 0.5 * _masked_mean(
                torch.maximum(value_loss_unclipped, value_loss_clipped),
                loss_mask,
            )

            entropy_mean = _masked_mean(entropy, loss_mask)
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * entropy_mean
            )
            approx_kl = _masked_mean((ratio - 1.0) - log_ratio, loss_mask)
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
        return dict(zip(metric_names, metric_values, strict=True)), valid_count

    def _explained_variance(self, batch: _TensorBatch) -> float:
        """Return pre-update explained variance without a second model forward."""
        with torch.no_grad():
            returns = batch.returns[batch.loss_mask]
            values = batch.old_values[batch.loss_mask]
            target_var = torch.var(returns, unbiased=False)
            if float(target_var.cpu()) < 1.0e-8:
                return 0.0
            residual_var = torch.var(returns - values, unbiased=False)
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
    loss_mask: Tensor


@dataclass(frozen=True)
class _SequenceDescriptor:
    batch: RolloutBatch
    env_index: int
    start: int
    length: int


def _tensor_batch(
    batches: list[RolloutBatch],
    device: torch.device,
    *,
    sequence_length: int,
) -> _TensorBatch:
    """Pack contiguous, episode-safe chunks for truncated BPTT.

    Every chunk has a fixed padded length, which keeps compiled learner graphs
    stable. ``loss_mask`` excludes padding. Recurrent chunks start from the
    behavior-policy hidden state recorded at that exact transition.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    descriptors = _sequence_descriptors(batches, sequence_length)
    if not descriptors:
        raise ValueError("RolloutBatch collection contains no transitions")

    first = descriptors[0].batch
    count = len(descriptors)
    obs_global = _zeros_like_sequences(first.obs_global, count, sequence_length, np.float32)
    obs_player = _zeros_like_sequences(first.obs_player, count, sequence_length, np.float32)
    obs_entities = _zeros_like_sequences(
        first.obs_entities,
        count,
        sequence_length,
        np.float32,
    )
    entity_mask = _zeros_like_sequences(first.entity_mask, count, sequence_length, bool)
    actions = _zeros_like_sequences(first.actions, count, sequence_length, np.int64)
    old_log_probs = np.zeros((count, sequence_length), dtype=np.float32)
    old_values = np.zeros((count, sequence_length), dtype=np.float32)
    returns = np.zeros((count, sequence_length), dtype=np.float32)
    advantages = np.zeros((count, sequence_length), dtype=np.float32)
    task_ids = np.zeros((count, sequence_length), dtype=np.int64)
    prev_actions = _zeros_like_sequences(first.prev_actions, count, sequence_length, np.int64)
    prev_rewards = np.zeros((count, sequence_length), dtype=np.float32)
    loss_mask = np.zeros((count, sequence_length), dtype=bool)

    first_action_masks = np.asarray(first.action_masks)
    action_masks: np.ndarray | None = None
    if first_action_masks.ndim > 2:
        action_masks = np.ones(
            (count, sequence_length, *first_action_masks.shape[2:]),
            dtype=bool,
        )

    initial_states: list[np.ndarray] = []
    has_rnn_state = first.rnn_states is not None
    for target_index, descriptor in enumerate(descriptors):
        batch = descriptor.batch
        source = slice(descriptor.start, descriptor.start + descriptor.length)
        target = slice(0, descriptor.length)
        env = descriptor.env_index
        obs_global[target_index, target] = np.asarray(batch.obs_global)[source, env]
        obs_player[target_index, target] = np.asarray(batch.obs_player)[source, env]
        obs_entities[target_index, target] = np.asarray(batch.obs_entities)[source, env]
        entity_mask[target_index, target] = np.asarray(batch.entity_mask)[source, env]
        actions[target_index, target] = np.asarray(batch.actions)[source, env]
        old_log_probs[target_index, target] = np.asarray(batch.log_probs)[source, env]
        old_values[target_index, target] = np.asarray(batch.values)[source, env]
        returns[target_index, target] = np.asarray(batch.returns)[source, env]
        advantages[target_index, target] = np.asarray(batch.advantages)[source, env]
        task_ids[target_index, target] = np.asarray(batch.task_ids)[source, env]
        prev_actions[target_index, target] = np.asarray(batch.prev_actions)[source, env]
        prev_rewards[target_index, target] = np.asarray(batch.prev_rewards)[source, env]
        loss_mask[target_index, target] = True
        if action_masks is not None:
            action_masks[target_index, target] = np.asarray(batch.action_masks)[source, env]

        if has_rnn_state:
            state = np.asarray(batch.rnn_states)
            if state.ndim != 4:
                raise ValueError("rnn_states must have shape (time, layers, envs, hidden)")
            initial_states.append(state[descriptor.start, :, env, :])

    rnn_state = None
    if has_rnn_state:
        # (sequence, layers, hidden) -> (layers, sequence, hidden)
        stacked = np.stack(initial_states, axis=0).transpose(1, 0, 2)
        rnn_state = torch.as_tensor(stacked, dtype=torch.float32, device=device)

    obs = {
        "global": torch.as_tensor(obs_global, dtype=torch.float32, device=device),
        "player": torch.as_tensor(obs_player, dtype=torch.float32, device=device),
        "entities": torch.as_tensor(obs_entities, dtype=torch.float32, device=device),
        "entity_mask": torch.as_tensor(entity_mask, dtype=torch.bool, device=device),
        "prev_action": torch.as_tensor(prev_actions, dtype=torch.float32, device=device),
        "prev_reward": torch.as_tensor(prev_rewards, dtype=torch.float32, device=device),
    }
    return _TensorBatch(
        obs=obs,
        actions=torch.as_tensor(actions, dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(old_log_probs, dtype=torch.float32, device=device),
        old_values=torch.as_tensor(old_values, dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        task_ids=torch.as_tensor(task_ids, dtype=torch.long, device=device),
        action_masks=(
            None
            if action_masks is None
            else torch.as_tensor(action_masks, dtype=torch.bool, device=device)
        ),
        rnn_state=rnn_state,
        loss_mask=torch.as_tensor(loss_mask, dtype=torch.bool, device=device),
    )


def _sequence_descriptors(
    batches: list[RolloutBatch],
    sequence_length: int,
) -> list[_SequenceDescriptor]:
    descriptors: list[_SequenceDescriptor] = []
    for batch in batches:
        rewards = np.asarray(batch.rewards)
        if rewards.ndim != 2:
            raise ValueError("RolloutBatch rewards must have shape (time, env)")
        time_steps, num_envs = rewards.shape
        dones = np.asarray(batch.dones, dtype=bool)
        truncateds = np.asarray(batch.truncateds, dtype=bool)
        episode_ids = np.asarray(batch.episode_ids)
        task_ids = np.asarray(batch.task_ids)
        if (
            dones.shape != rewards.shape
            or truncateds.shape != rewards.shape
            or episode_ids.shape != rewards.shape
            or task_ids.shape != rewards.shape
        ):
            raise ValueError("RolloutBatch lifecycle arrays must match rewards shape")

        for env_index in range(num_envs):
            segment_start = 0
            while segment_start < time_steps:
                segment_end = segment_start + 1
                while segment_end < time_steps:
                    previous = segment_end - 1
                    boundary = bool(dones[previous, env_index] or truncateds[previous, env_index])
                    boundary = boundary or bool(
                        episode_ids[segment_end, env_index] != episode_ids[previous, env_index]
                    )
                    boundary = boundary or bool(
                        task_ids[segment_end, env_index] != task_ids[previous, env_index]
                    )
                    if boundary:
                        break
                    segment_end += 1

                for start in range(segment_start, segment_end, sequence_length):
                    descriptors.append(
                        _SequenceDescriptor(
                            batch=batch,
                            env_index=env_index,
                            start=start,
                            length=min(sequence_length, segment_end - start),
                        )
                    )
                segment_start = segment_end
    return descriptors


def _zeros_like_sequences(
    source: object,
    count: int,
    sequence_length: int,
    dtype: type[np.generic] | type[bool],
) -> np.ndarray:
    array = np.asarray(source)
    if array.ndim < 2:
        raise ValueError("rollout arrays must have time and env dimensions")
    return np.zeros((count, sequence_length, *array.shape[2:]), dtype=dtype)


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
    if batch.discount_exponents is not None:
        exponents = np.asarray(batch.discount_exponents)
        if (
            exponents.shape != np.asarray(batch.rewards).shape
            or not np.isfinite(exponents).all()
            or not (exponents > 0.0).all()
        ):
            return False
    return batch.rnn_states is None or bool(np.isfinite(np.asarray(batch.rnn_states)).all())


def _batch_has_valid_actions(model: ActorCritic, batch: RolloutBatch) -> bool:
    actions = np.asarray(batch.actions)
    prev_actions = np.asarray(batch.prev_actions)
    if prev_actions.shape != actions.shape:
        return False
    if actions.ndim != 3 or not np.issubdtype(actions.dtype, np.integer):
        return False
    if not np.issubdtype(prev_actions.dtype, np.integer):
        return False
    policy = getattr(model, "policy", None)
    if policy is None:
        return True

    enable_macro = bool(getattr(policy, "enable_macro", False))
    expected_dim = 1 + 1 + N_BUTTONS + 1 + int(enable_macro)
    if actions.shape[-1] != expected_dim:
        return False
    n_macros = int(getattr(policy, "n_macros", 0))
    return _packed_actions_are_valid(
        actions,
        enable_macro=enable_macro,
        n_macros=n_macros,
    ) and _packed_actions_are_valid(
        prev_actions,
        enable_macro=enable_macro,
        n_macros=n_macros,
    )


def _packed_actions_are_valid(
    actions: np.ndarray,
    *,
    enable_macro: bool,
    n_macros: int,
) -> bool:
    if not np.isin(actions[..., 0], np.arange(N_MOVEMENT_X)).all():
        return False
    if not np.isin(actions[..., 1], np.arange(N_AIM_Y)).all():
        return False
    if not np.isin(actions[..., 2 : 2 + N_BUTTONS], (0, 1)).all():
        return False
    duration_index = 2 + N_BUTTONS
    if not np.isin(actions[..., duration_index], np.arange(N_DURATION)).all():
        return False
    if not enable_macro:
        return True

    macro_actions = actions[..., duration_index + 1]
    if not np.isin(macro_actions, np.arange(n_macros + 1)).all():
        return False
    macro_branch = macro_actions > 0
    canonical_primitives = (
        (actions[..., 0] == 1)
        & (actions[..., 1] == 1)
        & (actions[..., 2 : 2 + N_BUTTONS] == 0).all(axis=-1)
        & (actions[..., duration_index] == 0)
    )
    return bool((canonical_primitives | ~macro_branch).all())


def _batch_has_valid_action_masks(model: ActorCritic, batch: RolloutBatch) -> bool:
    masks = np.asarray(batch.action_masks)
    if masks.dtype != np.bool_:
        return False
    if masks.ndim <= 2:
        return True
    if masks.ndim != 3:
        return False
    policy = getattr(model, "policy", None)
    if policy is None:
        return True
    if masks.shape[-1] != int(getattr(policy, "mask_dim", -1)):
        return False

    actions = np.asarray(batch.actions)
    movement_group = masks[..., :N_MOVEMENT_X]
    aim_group = masks[..., N_MOVEMENT_X : N_MOVEMENT_X + N_AIM_Y]
    duration_start = N_MOVEMENT_X + N_AIM_Y + N_BUTTONS
    duration_group = masks[..., duration_start : duration_start + N_DURATION]
    groups = [movement_group, aim_group, duration_group]
    primitive_path = np.ones(actions.shape[:-1], dtype=bool)

    if bool(getattr(policy, "enable_macro", False)):
        macro_group = masks[..., duration_start + N_DURATION :]
        groups.append(macro_group)
        macro_actions = actions[..., 3 + N_BUTTONS]
        if not _selected_categorical_mask(macro_group, macro_actions).all():
            return False
        primitive_path = macro_actions == 0

    if not all(bool(group.any(axis=-1).all()) for group in groups):
        return False
    if not (_selected_categorical_mask(movement_group, actions[..., 0]) | ~primitive_path).all():
        return False
    if not (_selected_categorical_mask(aim_group, actions[..., 1]) | ~primitive_path).all():
        return False
    if not (
        _selected_categorical_mask(
            duration_group,
            actions[..., 2 + N_BUTTONS],
        )
        | ~primitive_path
    ).all():
        return False

    button_actions = actions[..., 2 : 2 + N_BUTTONS]
    button_mask = masks[
        ...,
        N_MOVEMENT_X + N_AIM_Y : N_MOVEMENT_X + N_AIM_Y + N_BUTTONS,
    ]
    return bool(((button_actions == 0) | button_mask | ~primitive_path[..., np.newaxis]).all())


def _selected_categorical_mask(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.take_along_axis(mask, values[..., np.newaxis], axis=-1)[..., 0]


def _batch_signature(batch: RolloutBatch) -> tuple[object, ...]:
    """Return the shape contract that must be identical across queued workers."""

    def tail(field: str) -> tuple[int, ...]:
        array = np.asarray(getattr(batch, field))
        return tuple(int(dim) for dim in array.shape[2:])

    action_masks = np.asarray(batch.action_masks)
    state = None if batch.rnn_states is None else np.asarray(batch.rnn_states)
    state_shape: tuple[int, ...] | None = None
    if state is not None:
        if state.ndim != 4:
            return ("invalid-rnn-state",)
        state_shape = (int(state.shape[1]), int(state.shape[3]))
    return (
        tail("obs_global"),
        tail("obs_player"),
        tail("obs_entities"),
        tail("entity_mask"),
        tail("actions"),
        None if action_masks.ndim <= 2 else tail("action_masks"),
        tail("prev_actions"),
        state_shape,
    )


def _batch_is_model_compatible(model: ActorCritic, batch: RolloutBatch) -> bool:
    """Return False when a worker batch cannot be evaluated by this learner model."""
    try:
        device = _model_device(model)
        tensor_batch = _tensor_batch(
            [_first_transition_batch(batch)],
            device,
            sequence_length=1,
        )
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
        discount_exponents=(
            None if batch.discount_exponents is None else _first_time_env(batch.discount_exponents)
        ),
    )


def _first_time_env(array: object) -> np.ndarray:
    return np.asarray(array)[:1, :1].copy()


def _first_rnn_state(array: object) -> np.ndarray:
    return np.asarray(array)[:1, :, :1, :].copy()


def _normalize_advantages(
    advantages: Tensor,
    *,
    task_ids: Tensor | None = None,
    by_task: bool = False,
    loss_mask: Tensor | None = None,
) -> Tensor:
    if loss_mask is None:
        loss_mask = torch.ones_like(advantages, dtype=torch.bool)
    if loss_mask.shape != advantages.shape:
        raise ValueError("loss_mask must match advantages")
    valid_advantages = advantages[loss_mask]
    if valid_advantages.numel() <= 1:
        result = torch.zeros_like(advantages)
        result[loss_mask] = valid_advantages
        return result
    if not by_task:
        normalized = _normalize_group(valid_advantages)
        result = torch.zeros_like(advantages)
        result[loss_mask] = normalized
        return result
    if task_ids is None or task_ids.shape != advantages.shape:
        raise ValueError("task_ids must match advantages for task-wise normalization")
    valid_task_ids = task_ids[loss_mask]

    _, inverse, counts = torch.unique(
        valid_task_ids,
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    group_counts = counts.to(dtype=valid_advantages.dtype)
    group_sums = torch.zeros_like(group_counts).scatter_add_(
        0,
        inverse,
        valid_advantages,
    )
    group_means = group_sums / group_counts
    centered = valid_advantages - group_means.index_select(0, inverse)
    group_square_sums = torch.zeros_like(group_counts).scatter_add_(
        0,
        inverse,
        centered.square(),
    )
    group_std = (group_square_sums / group_counts).sqrt().clamp_min(1.0e-8)
    normalized = centered / group_std.index_select(0, inverse)
    enough_samples = counts.index_select(0, inverse) > 1
    normalized = torch.where(enough_samples, normalized, valid_advantages)
    result = torch.zeros_like(advantages)
    result[loss_mask] = normalized
    return result


def _normalize_group(values: Tensor) -> Tensor:
    if values.numel() <= 1:
        return values
    centered = values - values.mean()
    std = values.std(unbiased=False)
    return centered / std.clamp_min(1.0e-8)


def _index_obs(obs: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {key: value.index_select(0, indices) for key, value in obs.items()}


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("loss mask selected no values")
    return selected.mean()


def _pad_sequence_indices(indices: Tensor, target_size: int) -> tuple[Tensor, Tensor]:
    """Pad only the last sequence minibatch so compiled shapes remain stable."""
    size = int(indices.numel())
    if size <= 0 or target_size <= 0 or size > target_size:
        raise ValueError("invalid APPO sequence minibatch size")
    active = torch.arange(target_size, device=indices.device) < size
    if size == target_size:
        return indices, active
    padding = indices[:1].expand(target_size - size)
    return torch.cat((indices, padding)), active


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
