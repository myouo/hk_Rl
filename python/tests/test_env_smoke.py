"""Env import/wiring smoke tests (no live game required).

Behavioral env tests need a live HKRLEnvMod connection and are marked
``integration`` (skipped by default). Here we only assert the class wires up.
"""

from __future__ import annotations

import pytest


def test_env_module_imports() -> None:
    from hkrl.env import HKRLEnv

    assert HKRLEnv is not None


def test_registry_has_builtin_components() -> None:
    # Importing the packages registers their components by name.
    import hkrl.models.mlp
    import hkrl.models.recurrent_policy
    import hkrl.training.ppo
    import hkrl.transport.tcp  # noqa: F401
    from hkrl.utils import registry

    assert "mlp" in registry.available("model")
    assert "entity_attention_gru" in registry.available("model")
    assert "tcp" in registry.available("transport")
    assert "ppo" in registry.available("algo")


@pytest.mark.integration
def test_random_policy_episode() -> None:
    pytest.skip("requires live Hollow Knight + HKRLEnvMod (phase 1+)")
