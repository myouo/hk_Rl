"""Policy heads (per action component) + value head (docs/action_space.md §2).

One head per hybrid-action component. Each head applies the relevant slice of the
action mask (masked logits -> -inf) before forming a Categorical/Bernoulli. The
composite log-prob/entropy is the sum across components.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X

ACTION_TENSOR_DIM_NO_MACRO = 1 + 1 + N_BUTTONS + 1


@dataclass(frozen=True)
class CompositeActionDistribution:
    """Distribution over the packed training action tensor.

    Tensor order: movement_x, aim_y, buttons[0:9], duration, optional macro.
    """

    movement_x: Categorical
    aim_y: Categorical
    buttons: Bernoulli
    duration: Categorical
    macro: Categorical | None = None

    @property
    def action_dim(self) -> int:
        return ACTION_TENSOR_DIM_NO_MACRO + (1 if self.macro is not None else 0)

    def sample(self) -> Tensor:
        parts = [
            self.movement_x.sample().unsqueeze(-1),
            self.aim_y.sample().unsqueeze(-1),
            self.buttons.sample().to(dtype=torch.long),
            self.duration.sample().unsqueeze(-1),
        ]
        if self.macro is not None:
            parts.append(self.macro.sample().unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def mode(self) -> Tensor:
        parts = [
            self.movement_x.logits.argmax(dim=-1, keepdim=True),
            self.aim_y.logits.argmax(dim=-1, keepdim=True),
            (self.buttons.probs > 0.5).to(dtype=torch.long),
            self.duration.logits.argmax(dim=-1, keepdim=True),
        ]
        if self.macro is not None:
            parts.append(self.macro.logits.argmax(dim=-1, keepdim=True))
        return torch.cat(parts, dim=-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        movement_x, aim_y, buttons, duration, macro = self._unpack(actions)
        log_prob = (
            self.movement_x.log_prob(movement_x)
            + self.aim_y.log_prob(aim_y)
            + self.buttons.log_prob(buttons).sum(dim=-1)
            + self.duration.log_prob(duration)
        )
        if self.macro is not None and macro is not None:
            log_prob = log_prob + self.macro.log_prob(macro)
        return log_prob

    def entropy(self) -> Tensor:
        entropy = (
            self.movement_x.entropy()
            + self.aim_y.entropy()
            + self.buttons.entropy().sum(dim=-1)
            + self.duration.entropy()
        )
        if self.macro is not None:
            entropy = entropy + self.macro.entropy()
        return entropy

    def _unpack(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions last dimension must be {self.action_dim}, got {actions.shape[-1]}"
            )

        actions = actions.to(dtype=torch.long)
        offset = 0
        movement_x = actions[..., offset]
        offset += 1
        aim_y = actions[..., offset]
        offset += 1
        buttons = actions[..., offset : offset + N_BUTTONS].to(dtype=self.buttons.logits.dtype)
        offset += N_BUTTONS
        duration = actions[..., offset]
        offset += 1
        macro = actions[..., offset] if self.macro is not None else None
        return movement_x, aim_y, buttons, duration, macro


class HybridPolicyHead(nn.Module):
    """Produces per-component distributions for the hybrid action space.

    Components: movement_x Categorical(3), aim_y Categorical(3),
    buttons Bernoulli(9), duration Categorical(4), macro Categorical(M+1) optional.
    """

    def __init__(self, in_dim: int, enable_macro: bool = True, n_macros: int = 11) -> None:
        super().__init__()
        if n_macros < 0:
            raise ValueError("n_macros must be non-negative")

        self.enable_macro = enable_macro
        self.n_macros = n_macros
        self.mask_dim = N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION
        if enable_macro:
            self.mask_dim += n_macros + 1

        self.movement_x = nn.Linear(in_dim, N_MOVEMENT_X)
        self.aim_y = nn.Linear(in_dim, N_AIM_Y)
        self.buttons = nn.Linear(in_dim, N_BUTTONS)
        self.duration = nn.Linear(in_dim, N_DURATION)
        self.macro = nn.Linear(in_dim, n_macros + 1) if enable_macro else None

    def forward(self, x: Tensor, action_mask: Tensor | None = None) -> CompositeActionDistribution:
        """Return a composite distribution object (log_prob/entropy/sample sum over
        components, masked per the canonical action_mask layout).
        """
        movement_logits = self.movement_x(x)
        aim_logits = self.aim_y(x)
        button_logits = self.buttons(x)
        duration_logits = self.duration(x)
        macro_logits = self.macro(x) if self.macro is not None else None

        if action_mask is not None:
            mask = action_mask.to(device=x.device, dtype=torch.bool)
            if mask.shape[-1] != self.mask_dim:
                raise ValueError(
                    f"action_mask last dimension must be {self.mask_dim}, got {mask.shape[-1]}"
                )

            offset = 0
            movement_logits = _mask_categorical_logits(
                movement_logits,
                mask[..., offset : offset + N_MOVEMENT_X],
                "movement_x",
            )
            offset += N_MOVEMENT_X
            aim_logits = _mask_categorical_logits(
                aim_logits,
                mask[..., offset : offset + N_AIM_Y],
                "aim_y",
            )
            offset += N_AIM_Y
            button_logits = _mask_button_logits(
                button_logits,
                mask[..., offset : offset + N_BUTTONS],
            )
            offset += N_BUTTONS
            duration_logits = _mask_categorical_logits(
                duration_logits,
                mask[..., offset : offset + N_DURATION],
                "duration",
            )
            offset += N_DURATION
            if macro_logits is not None:
                macro_logits = _mask_categorical_logits(
                    macro_logits,
                    mask[..., offset:],
                    "macro",
                )

        return CompositeActionDistribution(
            movement_x=Categorical(logits=movement_logits),
            aim_y=Categorical(logits=aim_logits),
            buttons=Bernoulli(logits=button_logits),
            duration=Categorical(logits=duration_logits),
            macro=Categorical(logits=macro_logits) if macro_logits is not None else None,
        )


class ValueHead(nn.Module):
    """Scalar state-value V(memory_out)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.v(x).squeeze(-1)


def _mask_categorical_logits(logits: Tensor, mask: Tensor, name: str) -> Tensor:
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError(f"action_mask has no valid entries for {name}")
    return torch.where(mask, logits, torch.full_like(logits, _masked_logit_value(logits)))


def _mask_button_logits(logits: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, logits, torch.full_like(logits, _masked_logit_value(logits)))


def _masked_logit_value(logits: Tensor) -> float:
    if logits.dtype in (torch.float16, torch.bfloat16):
        return -1.0e4
    return -1.0e9
