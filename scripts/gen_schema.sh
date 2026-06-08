#!/usr/bin/env bash
# Generate FlatBuffers bindings for both Python and C# from the single source of
# truth (schema/hkrl.fbs). Idempotent; safe to re-run. See schema/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FBS="$ROOT/schema/hkrl.fbs"
PY_OUT="$ROOT/python/hkrl/schema"
CS_OUT="$ROOT/mod/HKRLEnvMod/Schema"

if ! command -v flatc >/dev/null 2>&1; then
  echo "error: 'flatc' not found on PATH." >&2
  echo "Install FlatBuffers compiler: https://github.com/google/flatbuffers/releases" >&2
  exit 1
fi

echo "flatc: $(flatc --version)"
echo "schema: $FBS"

mkdir -p "$PY_OUT" "$CS_OUT"
flatc --python -o "$PY_OUT" "$FBS"
flatc --csharp -o "$CS_OUT" "$FBS"

echo "Generated Python bindings -> $PY_OUT"
echo "Generated C# bindings     -> $CS_OUT"
