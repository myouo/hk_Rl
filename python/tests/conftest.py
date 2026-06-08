"""Pytest fixtures and import-safety for the skeleton stage.

At the interface-placeholder stage, tests assert that the package *imports* and
that constants/layouts are self-consistent. Behavioral tests are marked xfail/skip
until their phase lands (see per-file markers). Integration tests requiring a live
game connection are marked ``integration`` and skipped by default.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def schema_version() -> int:
    from hkrl.protocol import SCHEMA_VERSION

    return SCHEMA_VERSION
