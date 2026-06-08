"""Pytest fixtures and import-safety checks.

Unit tests exercise protocol layout, env validation, rollout/training helpers,
and distributed plumbing without requiring a live Hollow Knight process.
Integration tests requiring a live game connection are marked ``integration`` and
skipped by default.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def schema_version() -> int:
    from hkrl.protocol import SCHEMA_VERSION

    return SCHEMA_VERSION
