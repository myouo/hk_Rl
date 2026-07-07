"""Pluggable transports between GameWorker and HKRLEnvMod.

All transports implement :class:`hkrl.transport.base.Transport`. Select by name
via the registry (see hkrl.utils.registry). TCP is the supported live
HKRLEnvMod path; shared-memory is an explicit opt-in in-process prototype until
the mod ships an OS shared-memory server. See docs/protocol.md.
"""

from __future__ import annotations

from hkrl.transport.base import Transport

__all__ = ["Transport"]
