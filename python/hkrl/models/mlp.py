"""MLP Actor-Critic baseline (PRD Phase 3).

Non-recurrent, fixed-vector model for the single-boss baseline. Concatenates
global + player + (flattened, masked) entity features and runs an MLP trunk into
the hybrid heads. Useful as the ablation floor vs attention / attention+GRU.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic, RnnState
from hkrl.models.heads import CompositeActionDistribution, HybridPolicyHead, ValueHead
from hkrl.utils.registry import register_model


@register_model("mlp")
class MlpActorCritic(ActorCritic):
    """Feedforward baseline. ``initial_state`` returns None (no recurrence)."""

    def __init__(
        self,
        obs_dims: dict[str, Any],
        hidden: int = 256,
        enable_macro: bool = True,
        n_macros: int = 11,
    ) -> None:
        super().__init__()
        global_dim = _flat_dim(obs_dims, "global")
        player_dim = _flat_dim(obs_dims, "player")
        entity_dim = _flat_dim(obs_dims, "entities")
        input_dim = global_dim + player_dim + entity_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = HybridPolicyHead(hidden, enable_macro=enable_macro, n_macros=n_macros)
        self.value = ValueHead(hidden)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        del batch_size, device
        return None

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        del rnn_state
        hidden = self.trunk(_flatten_observation(obs))
        return self.policy(hidden, action_mask=action_mask), self.value(hidden), None

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


def _flatten_observation(obs: dict[str, Tensor]) -> Tensor:
    global_state = obs["global"].flatten(start_dim=-1)
    player_state = obs["player"].flatten(start_dim=-1)
    entities = obs["entities"]
    entity_mask = obs.get("entity_mask")
    if entity_mask is not None:
        mask = entity_mask.to(device=entities.device, dtype=entities.dtype).unsqueeze(-1)
        entities = entities * mask
    return torch.cat(
        [
            global_state,
            player_state,
            entities.flatten(start_dim=-2),
        ],
        dim=-1,
    )


def _flat_dim(obs_dims: dict[str, Any], key: str) -> int:
    if key not in obs_dims:
        raise KeyError(f"obs_dims missing {key!r}")

    value = obs_dims[key]
    if hasattr(value, "shape"):
        shape = tuple(int(dim) for dim in value.shape)
    elif isinstance(value, int):
        shape = (value,)
    else:
        shape = tuple(int(dim) for dim in value)

    if not shape:
        raise ValueError(f"obs_dims[{key!r}] must have at least one dimension")
    return math.prod(shape)
