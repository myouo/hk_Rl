"""Learner acceleration hardware-selection tests."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from hkrl.models.mlp import MlpActorCritic
from hkrl.training import accelerator
from hkrl.training.accelerator import (
    TorchLearnerRuntime,
    _cuda_bf16_supported,
    _resolve_amp_dtype,
)
from hkrl.utils.config import TrainConfig


def test_auto_amp_uses_float16_on_pre_ampere_cuda(
    monkeypatch,
) -> None:
    """A software BF16 fallback must not make V100 the auto-selected path."""

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    dtype = _resolve_amp_dtype("auto", device=torch.device("cuda:0"))

    assert dtype == torch.float16


def test_auto_amp_uses_bfloat16_on_supported_ampere_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert _cuda_bf16_supported(torch.device("cuda:0"))
    dtype = _resolve_amp_dtype("auto", device=torch.device("cuda:0"))

    assert dtype == torch.bfloat16


def test_auto_compile_falls_back_to_eager_when_dynamo_is_unsupported(
    monkeypatch,
) -> None:
    monkeypatch.setattr(accelerator, "_torch_compile_supported", lambda: False)

    assert (
        accelerator._resolve_compile_mode(
            "auto",
            device=torch.device("cuda:0"),
        )
        is None
    )
    with pytest.raises(ValueError, match="TorchDynamo"):
        accelerator._resolve_compile_mode(
            "reduce-overhead",
            device=torch.device("cuda:0"),
        )


def test_amp_overflow_is_skipped_and_reported_without_crashing() -> None:
    model = MlpActorCritic(
        {
            "global": (1,),
            "player": (1,),
            "entities": (1, 1),
            "entity_mask": (1,),
        },
        hidden=4,
        enable_macro=False,
    )
    runtime = TorchLearnerRuntime(model, TrainConfig(algorithm="appo"))
    scaler = _BackoffGradScaler()
    runtime.scaler = scaler  # type: ignore[assignment]
    parameter = next(model.parameters())
    before = parameter.detach().clone()

    grad_norm, step_skipped = runtime.backward_step(
        parameter.sum(),
        max_grad_norm=0.5,
    )

    assert float(grad_norm) == 0.0
    assert float(step_skipped) == 1.0
    assert scaler.step_called
    assert scaler.get_scale() == 512.0
    assert torch.equal(parameter.detach(), before)


class _BackoffGradScaler:
    """Minimal scaler double that simulates scaled-gradient overflow."""

    def __init__(self) -> None:
        self._scale = 1024.0
        self.step_called = False

    def is_enabled(self) -> bool:
        return True

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * torch.tensor(float("inf"))

    def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, _optimizer: torch.optim.Optimizer) -> None:
        self.step_called = True

    def update(self) -> None:
        self._scale /= 2.0

    def get_scale(self) -> float:
        return self._scale
