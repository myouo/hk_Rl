"""Generated FlatBuffers Python bindings live here.

Run ``make gen-schema`` (or ``scripts/gen_schema.sh``) to populate this package
from ``schema/hkrl.fbs``. The generated subpackage (``hkrl.schema.HKRL.*``) is a
build artifact and is gitignored — do NOT hand-edit it.

This ``__init__`` stays in source so the package imports cleanly before codegen;
high-level decode helpers that wrap the bindings live in ``hkrl.protocol``.
"""

from __future__ import annotations

import sys
from contextlib import suppress
from importlib import import_module

with suppress(ModuleNotFoundError):
    # flatc emits intra-package imports like ``from HKRL.Action import Action``.
    # Register a top-level alias so generated modules work from inside hkrl.schema.
    sys.modules.setdefault("HKRL", import_module(f"{__name__}.HKRL"))
