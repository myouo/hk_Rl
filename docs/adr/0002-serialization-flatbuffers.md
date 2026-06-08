# ADR-0002: Serialization — FlatBuffers as single source of truth

- Status: **Accepted**
- Date: 2026-06-08

## Context

Observation/action/protocol cross a C# (mod) ↔ Python (worker) boundary every
tick. We need: high decode performance on the hot path, strict schema versioning
to prevent two-sided drift, and code generation for both languages. Candidates:
FlatBuffers, Protobuf, MessagePack.

## Decision

Define everything once in [`schema/hkrl.fbs`](../../schema/hkrl.fbs). Generate C#
and Python bindings via `flatc` (`make gen-schema`). Generated code is a build
artifact (gitignored), never hand-edited. A `Transport` abstraction lets an MVP
start with a simpler codec and switch without touching env/model code.

## Rationale

- **Zero-copy decode** every tick — read fields directly from the buffer, no full
  deserialization. Matters for SPS.
- **Single source of truth** — one `.fbs` generates both ends; impossible to drift
  silently. `schema_version` gates compatibility at runtime.
- **Append-only evolution** — add fields at table end with defaults; old/new peers
  interoperate.
- First-class **C#/Python codegen**.

## Consequences

- Requires `flatc` in the dev/build environment (documented in `schema/README.md`).
- Generated dirs are gitignored; CI/build must run codegen.
- Slightly more ceremony than MessagePack, repaid by perf + safety.

## Alternatives rejected

- **MessagePack** — simplest, no codegen, but weak schema constraints; two ends
  drift easily; poor long-term extensibility.
- **Protobuf** — good codegen + evolution, but decode requires full
  deserialization; hot-path perf trails FlatBuffers' zero-copy.
