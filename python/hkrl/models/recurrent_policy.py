"""Entity-attention + recurrent ActorCritic (docs/model_architecture.md §2).

The project's high-ceiling model: encoders -> masked attention -> GRU/LSTM memory
-> hybrid heads + value. Threads ``entity_mask`` through attention and ``rnn_state``
through the recurrence. Trained with truncated BPTT (+ optional burn-in) by
training/recurrent_ppo.py.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic, RnnState
from hkrl.models.encoders import EntityEncoder, GlobalEncoder, PlayerEncoder
from hkrl.models.entity_attention import EntityTransformerEncoder
from hkrl.models.heads import (
    ACTION_TENSOR_DIM_NO_MACRO,
    CompositeActionDistribution,
    HybridPolicyHead,
    ValueHead,
)
from hkrl.spaces import DEFAULT_N_MACROS
from hkrl.utils.registry import register_model


@register_model("entity_attention_gru")
class EntityAttentionRecurrentAC(ActorCritic):
    """Encoders -> attention(entity_context) -> [global|player|context|prev_action|
    prev_reward] -> GRU/LSTM -> heads.

    ``rnn_type`` in {"gru", "lstm"}. ``initial_state`` returns the zeroed hidden
    (tuple for LSTM). Reset hidden at episode boundaries (and optionally between
    bosses in a linear sequence — config-controlled).
    """

    def __init__(
        self,
        obs_dims: dict[str, Any],
        entity_hidden: int = 128,
        attention_layers: int = 2,
        attention_heads: int = 4,
        rnn_type: str = "gru",
        rnn_hidden: int = 256,
        enable_macro: bool = True,
        n_macros: int = DEFAULT_N_MACROS,
        max_entities: int = 64,
    ) -> None:
        super().__init__()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        self.rnn_type = rnn_type
        self.rnn_hidden = rnn_hidden
        self.action_dim = ACTION_TENSOR_DIM_NO_MACRO + (1 if enable_macro else 0)
        global_dim = _flat_dim(obs_dims, "global")
        player_dim = _flat_dim(obs_dims, "player")
        entity_shape = _shape(obs_dims, "entities")
        entity_dim = entity_shape[-1]

        self.global_encoder = GlobalEncoder(global_dim, entity_hidden)
        self.player_encoder = PlayerEncoder(player_dim, entity_hidden)
        self.entity_encoder = EntityEncoder(entity_dim, entity_hidden, n_types=256, n_ids=4096)
        self.entity_attention = EntityTransformerEncoder(
            dim=entity_hidden,
            layers=attention_layers,
            heads=attention_heads,
        )
        self.prev_action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, entity_hidden),
            nn.Tanh(),
        )
        scale = torch.ones((self.action_dim,), dtype=torch.float32)
        scale[0] = 2.0
        scale[1] = 2.0
        scale[ACTION_TENSOR_DIM_NO_MACRO - 1] = 3.0
        if enable_macro:
            scale[-1] = float(max(n_macros, 1))
        self.register_buffer("_prev_action_scale", scale)
        memory_input_dim = entity_hidden * 4 + 1
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(memory_input_dim, rnn_hidden, batch_first=True)
        self.policy = HybridPolicyHead(
            rnn_hidden,
            enable_macro=enable_macro,
            n_macros=n_macros,
        )
        self.value = ValueHead(rnn_hidden)
        self.max_entities = max_entities

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        hidden = torch.zeros((1, batch_size, self.rnn_hidden), device=device)
        if self.rnn_type == "lstm":
            return hidden, torch.zeros_like(hidden)
        return hidden

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        memory_input, is_sequence = self._memory_input(obs)
        batch_size = memory_input.shape[0]
        if rnn_state is None:
            rnn_state = self.initial_state(batch_size, device=memory_input.device)

        memory_out, next_state = self.rnn(memory_input, rnn_state)
        if not is_sequence:
            memory_out = memory_out[:, 0]
        dist = self.policy(memory_out, action_mask=action_mask)
        value = self.value(memory_out)
        return dist, value, next_state

    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        with torch.no_grad():
            dist, value, next_state = self.forward(
                obs,
                rnn_state=rnn_state,
                action_mask=action_mask,
            )
            action = dist.mode() if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value, next_state

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dist, value, _ = self.forward(obs, rnn_state=rnn_state, action_mask=action_mask)
        return dist.log_prob(actions), dist.entropy(), value

    def _memory_input(self, obs: dict[str, Tensor]) -> tuple[Tensor, bool]:
        global_state = obs["global"]
        is_sequence = global_state.ndim == 3
        if global_state.ndim not in (2, 3):
            raise ValueError("obs['global'] must have shape (B, G) or (B, T, G)")

        if not is_sequence:
            prepared = {
                "global": obs["global"].unsqueeze(1),
                "player": obs["player"].unsqueeze(1),
                "entities": obs["entities"].unsqueeze(1),
                "entity_mask": obs["entity_mask"].unsqueeze(1),
            }
        else:
            prepared = obs

        global_seq = prepared["global"]
        player_seq = prepared["player"]
        entities_seq = prepared["entities"]
        mask_seq = prepared["entity_mask"]
        batch_size, seq_len = global_seq.shape[:2]
        entity_count = entities_seq.shape[-2]
        entity_dim = entities_seq.shape[-1]

        global_emb = self.global_encoder(global_seq.reshape(batch_size * seq_len, -1))
        player_emb = self.player_encoder(player_seq.reshape(batch_size * seq_len, -1))
        entities = entities_seq.reshape(batch_size * seq_len, entity_count, entity_dim)
        entity_mask = mask_seq.reshape(batch_size * seq_len, entity_count)
        entity_type = entities[..., 1] if entity_dim > 1 else torch.zeros_like(entity_mask)
        entity_id = entities[..., 0] if entity_dim > 0 else None
        entity_embs = self.entity_encoder(entities, entity_type=entity_type, entity_id=entity_id)
        entity_context = self.entity_attention(entity_embs, entity_mask)

        prev_action = _optional_sequence_feature(
            obs,
            "prev_action",
            batch_size=batch_size,
            seq_len=seq_len,
            feature_dim=self.action_dim,
            device=global_seq.device,
            dtype=global_seq.dtype,
        )
        prev_reward = _optional_sequence_feature(
            obs,
            "prev_reward",
            batch_size=batch_size,
            seq_len=seq_len,
            feature_dim=1,
            device=global_seq.device,
            dtype=global_seq.dtype,
        )
        scale = self._prev_action_scale.to(device=prev_action.device, dtype=prev_action.dtype)
        prev_action = prev_action / scale.clamp_min(1.0)
        prev_action_emb = self.prev_action_encoder(prev_action.reshape(batch_size * seq_len, -1))
        prev_reward = prev_reward.reshape(batch_size * seq_len, 1)

        memory_input = torch.cat(
            [global_emb, player_emb, entity_context, prev_action_emb, prev_reward],
            dim=-1,
        )
        return memory_input.reshape(batch_size, seq_len, -1), is_sequence


def _shape(obs_dims: dict[str, Any], key: str) -> tuple[int, ...]:
    if key not in obs_dims:
        raise KeyError(f"obs_dims missing {key!r}")
    value = obs_dims[key]
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, int):
        return (value,)
    return tuple(int(dim) for dim in value)


def _flat_dim(obs_dims: dict[str, Any], key: str) -> int:
    shape = _shape(obs_dims, key)
    if not shape:
        raise ValueError(f"obs_dims[{key!r}] must have at least one dimension")
    return math.prod(shape)


def _optional_sequence_feature(
    obs: dict[str, Tensor],
    key: str,
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    shape = (batch_size, seq_len, feature_dim)
    value = obs.get(key)
    if value is None:
        return torch.zeros(shape, dtype=dtype, device=device)

    tensor = value.to(device=device, dtype=dtype)
    if feature_dim == 1:
        if tensor.shape == (batch_size,):
            tensor = tensor.reshape(batch_size, 1, 1)
        elif tensor.shape == (batch_size, seq_len):
            tensor = tensor.unsqueeze(-1)
        elif tensor.shape == (batch_size, 1) and seq_len == 1:
            tensor = tensor.reshape(batch_size, 1, 1)
    elif tensor.shape == (batch_size, feature_dim) and seq_len == 1:
        tensor = tensor.unsqueeze(1)

    if tensor.shape != shape:
        raise ValueError(f"obs[{key!r}] must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor
