"""Policy/value head tests."""

from __future__ import annotations

import pytest
import torch
from hkrl.models.heads import HybridPolicyHead, ValueHead
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
from torch import nn


def test_hybrid_policy_head_samples_and_evaluates_actions() -> None:
    head = HybridPolicyHead(in_dim=4, enable_macro=True, n_macros=2)
    x = torch.zeros((2, 4), dtype=torch.float32)

    dist = head(x)
    actions = dist.sample()

    assert actions.shape == (2, 13)
    assert dist.log_prob(actions).shape == (2,)
    assert dist.entropy().shape == (2,)
    assert torch.isfinite(dist.log_prob(actions)).all()


def test_hybrid_policy_head_respects_action_mask() -> None:
    head = HybridPolicyHead(in_dim=4, enable_macro=True, n_macros=2)
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    with torch.no_grad():
        head.buttons.bias.fill_(8.0)

    mask = torch.tensor(
        [
            [
                False,
                False,
                True,  # movement_x = 2 only
                True,
                False,
                False,  # aim_y = 0 only
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,  # buttons: 0 and 2 may be pressed
                False,
                False,
                False,
                True,  # duration = 3 only
                False,
                True,
                False,  # macro = 1 only
            ]
        ],
        dtype=torch.bool,
    )

    dist = head(torch.zeros((1, 4), dtype=torch.float32), action_mask=mask)
    mode = dist.mode()

    assert mode.tolist() == [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    assert torch.isfinite(dist.log_prob(mode)).all()


def test_hybrid_policy_head_uses_safe_noop_for_unvalidated_empty_mask_group() -> None:
    head = HybridPolicyHead(in_dim=4, enable_macro=False)
    mask = torch.ones(
        (1, N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION),
        dtype=torch.bool,
    )
    mask[:, :N_MOVEMENT_X] = False

    dist = head(torch.zeros((1, 4), dtype=torch.float32), action_mask=mask)

    assert dist.mode()[0, 0].item() == 1
    assert torch.isfinite(dist.log_prob(dist.mode())).all()


def test_hybrid_policy_head_rejects_wrong_mask_length() -> None:
    head = HybridPolicyHead(in_dim=4, enable_macro=False)

    with pytest.raises(ValueError, match="action_mask"):
        head(torch.zeros((1, 4), dtype=torch.float32), action_mask=torch.ones((1, 3)))


def test_macro_policy_probability_excludes_ignored_primitive_branches() -> None:
    head = HybridPolicyHead(in_dim=4, enable_macro=True, n_macros=2)
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    dist = head(torch.zeros((1, 4), dtype=torch.float32))
    primitive = torch.zeros((1, 13), dtype=torch.long)
    macro = primitive.clone()
    macro[:, -1] = 1

    primitive_log_prob = (
        -torch.log(torch.tensor(3.0))
        - torch.log(torch.tensor(3.0))
        - 9.0 * torch.log(torch.tensor(2.0))
        - torch.log(torch.tensor(4.0))
    )
    macro_log_prob = -torch.log(torch.tensor(3.0))
    expected_entropy = -macro_log_prob + (1.0 / 3.0) * -primitive_log_prob

    torch.testing.assert_close(
        dist.log_prob(primitive),
        (macro_log_prob + primitive_log_prob).reshape(1),
    )
    torch.testing.assert_close(dist.log_prob(macro), macro_log_prob.reshape(1))
    torch.testing.assert_close(dist.entropy(), expected_entropy.reshape(1))


def test_value_head_returns_flat_values() -> None:
    value = ValueHead(in_dim=4)
    out = value(torch.zeros((3, 4), dtype=torch.float32))

    assert out.shape == (3,)
