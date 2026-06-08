# `schema/` — Single Source of Truth

`hkrl.fbs` is the **only** definition of the observation, action, and protocol
wire format. Both the C# mod and the Python package consume generated bindings
from it — never the other way around.

```text
                ┌───────────────┐
                │  hkrl.fbs     │  ← edit here, only here
                └───────┬───────┘
            flatc       │       flatc
        --csharp        │        --python
        ┌───────────────┴───────────────┐
        ▼                               ▼
 mod/HKRLEnvMod/Schema/        python/hkrl/schema/hkrl/
   (C# bindings)                 (Python bindings)
```

## Regenerate

```bash
make gen-schema        # both languages
make gen-schema-py     # Python only
make gen-schema-cs     # C# only
```

Requires [`flatc`](https://github.com/google/flatbuffers/releases) on `PATH`.
Check with `flatc --version` (recommend ≥ 24.3.25 to match the `flatbuffers`
Python runtime pinned in `python/pyproject.toml`).

## Evolution rules

1. **Append-only.** Add new fields at the *end* of a table with an explicit
   default. Never reorder or reuse a removed field slot — that breaks
   forward/backward compatibility.
2. **Bump the version.** Increment `SCHEMA_VERSION` in `python/hkrl/protocol.py`
   and `mod/HKRLEnvMod/Transport/Protocol.cs`, and the `schema_version` carried
   in each message. Record the change in `CHANGELOG.md`.
3. **Document semantics.** Field units/ranges/normalization live in
   `docs/observation_schema.md` and `docs/action_space.md`. Keep them in sync.
4. **Generated code is gitignored.** See `.gitignore`; bindings are build
   artifacts, not source.

## Why FlatBuffers

Zero-copy decode on the hot path (every tick), strict schema versioning, and
first-class C#/Python codegen. Rationale: [`docs/adr/0002`](../docs/adr/0002-serialization-flatbuffers.md).
