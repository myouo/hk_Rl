"""Config-driven PyTorch learner acceleration with safe CPU fallbacks.

The game-facing policy loop stays local and uncompiled.  This module is only
for large-batch learner updates, where AMP, ``torch.compile`` and fused Adam can
improve GPU utilization without changing rollout or checkpoint semantics.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic
from hkrl.training.numerics import require_finite_tensor
from hkrl.utils.config import TrainConfig


class TorchLearnerRuntime:
    """Own optimizer/AMP/compile state for one learner-side model."""

    def __init__(self, model: ActorCritic, config: TrainConfig) -> None:
        self.model = model
        self.device = _model_device(model)
        self.amp_dtype = _resolve_amp_dtype(
            config.learner.amp_dtype,
            device=self.device,
        )
        self.optimizer, self.fused_optimizer = _build_adam(
            model,
            learning_rate=config.learning_rate,
            mode=config.learner.fused_optimizer,
            device=self.device,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16,
        )
        compile_mode = _resolve_compile_mode(
            config.learner.compile_mode,
            device=self.device,
        )
        self.compile_mode = compile_mode
        evaluator: Any = model.evaluate_actions
        if compile_mode is not None:
            evaluator = torch.compile(
                evaluator,
                mode=compile_mode,
                dynamic=False,
            )
        self.evaluate_actions = evaluator

    @property
    def amp_enabled(self) -> bool:
        return self.amp_dtype is not None

    @property
    def compile_enabled(self) -> bool:
        return self.compile_mode is not None

    def autocast(self) -> AbstractContextManager[Any]:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def backward_step(self, loss: Tensor, *, max_grad_norm: float) -> Tensor:
        """Backpropagate, unscale before clipping, and take one optimizer step."""

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            require_finite_tensor("grad_norm", grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return grad_norm

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        require_finite_tensor("grad_norm", grad_norm)
        self.optimizer.step()
        return grad_norm

    def metric_flags(self) -> dict[str, float]:
        return {
            "amp_enabled": float(self.amp_enabled),
            "compile_enabled": float(self.compile_enabled),
            "fused_optimizer": float(self.fused_optimizer),
        }


def _resolve_amp_dtype(mode: str, *, device: torch.device) -> torch.dtype | None:
    if mode == "off":
        return None
    if mode == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if _cuda_bf16_supported(device) else torch.float16
    if mode == "float16":
        if device.type != "cuda":
            raise ValueError("learner.amp_dtype='float16' requires a CUDA learner")
        return torch.float16
    if mode == "bfloat16":
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("learner.amp_dtype='bfloat16' requires CPU or CUDA")
        if device.type == "cuda" and not _cuda_bf16_supported(device):
            raise ValueError("learner.amp_dtype='bfloat16' is unsupported by this CUDA device")
        return torch.bfloat16
    raise ValueError(f"unsupported learner.amp_dtype {mode!r}")


def _resolve_compile_mode(mode: str, *, device: torch.device) -> str | None:
    if mode == "off":
        return None
    if mode == "auto":
        return "default" if device.type == "cuda" else None
    if mode in {"default", "reduce-overhead", "max-autotune"}:
        return mode
    raise ValueError(f"unsupported learner.compile_mode {mode!r}")


def _build_adam(
    model: ActorCritic,
    *,
    learning_rate: float,
    mode: str,
    device: torch.device,
) -> tuple[torch.optim.Adam, bool]:
    if mode == "off":
        return torch.optim.Adam(model.parameters(), lr=learning_rate), False
    if mode == "on" and device.type != "cuda":
        raise ValueError("learner.fused_optimizer='on' requires a CUDA learner")

    use_fused = device.type == "cuda"
    if use_fused:
        try:
            return (
                torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    fused=True,
                ),
                True,
            )
        except (RuntimeError, TypeError, ValueError):
            if mode == "on":
                raise
    return torch.optim.Adam(model.parameters(), lr=learning_rate), False


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _cuda_bf16_supported(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return bool(torch.cuda.is_bf16_supported())


__all__ = ["TorchLearnerRuntime"]
