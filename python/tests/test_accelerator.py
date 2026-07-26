"""Learner acceleration hardware-selection tests."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from hkrl.training.accelerator import _cuda_bf16_supported, _resolve_amp_dtype


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
