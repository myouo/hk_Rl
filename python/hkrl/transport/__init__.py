"""Pluggable transports between GameWorker and HKRLEnvMod.

All transports implement :class:`hkrl.transport.base.Transport`. Select by name
via the registry (see hkrl.utils.registry); TCP for portability/cross-machine,
shared-memory for lowest-latency single-machine. See docs/protocol.md.
"""

from __future__ import annotations

from hkrl.transport.base import Transport

__all__ = ["Transport"]
