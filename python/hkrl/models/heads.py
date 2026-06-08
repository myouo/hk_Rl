"""Policy heads (per action component) + value head (docs/action_space.md §2).

One head per hybrid-action component. Each head applies the relevant slice of the
action mask (masked logits -> -inf) before forming a Categorical/Bernoulli. The
composite log-prob/entropy is the sum across components.
"""

from __future__ import annotations

from torch import Tensor, nn

from hkrl.spaces import DURATION_TICKS, N_AIM_Y, N_BUTTONS, N_MOVEMENT_X


class HybridPolicyHead(nn.Module):
    """Produces per-component distributions for the hybrid action space.

    Components: movement_x Categorical(3), aim_y Categorical(3),
    buttons Bernoulli(9), duration Categorical(4), macro Categorical(M+1) optional.
    """

    def __init__(self, in_dim: int, enable_macro: bool = True, n_macros: int = 11) -> None:
        super().__init__()
        self.movement_x = nn.Linear(in_dim, N_MOVEMENT_X)
        self.aim_y = nn.Linear(in_dim, N_AIM_Y)
        self.buttons = nn.Linear(in_dim, N_BUTTONS)
        self.duration = nn.Linear(in_dim, len(DURATION_TICKS))
        self.macro = nn.Linear(in_dim, n_macros + 1) if enable_macro else None

    def forward(self, x: Tensor, action_mask: Tensor | None = None) -> object:
        """Return a composite distribution object (log_prob/entropy/sample sum over
        components, masked per the canonical action_mask layout).

        TODO(phase-3): build masked Categorical/Bernoulli per component; return a
        small composite-distribution helper.
        """
        raise NotImplementedError


class ValueHead(nn.Module):
    """Scalar state-value V(memory_out)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError  # TODO(phase-3)
