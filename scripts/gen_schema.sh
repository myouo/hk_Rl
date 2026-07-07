#!/usr/bin/env bash
# Generate FlatBuffers bindings for both Python and C# from the single source of
# truth (schema/hkrl.fbs). Idempotent; safe to re-run. See schema/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FBS="$ROOT/schema/hkrl.fbs"
PY_OUT="$ROOT/python/hkrl/schema"
CS_OUT="$ROOT/mod/HKRLEnvMod/Schema"
PY_FLATC="${HKRL_PY_FLATC:-${HKRL_FLATC:-flatc}}"
CS_FLATC="${HKRL_CSHARP_FLATC:-${HKRL_FLATC:-flatc}}"
REQUIRED_CSHARP_FLATC_VERSION="${HKRL_CSHARP_FLATC_VERSION:-23.5.26}"

if ! command -v "$PY_FLATC" >/dev/null 2>&1; then
  echo "error: '$PY_FLATC' not found on PATH." >&2
  echo "Install FlatBuffers compiler: https://github.com/google/flatbuffers/releases" >&2
  exit 1
fi
if ! command -v "$CS_FLATC" >/dev/null 2>&1; then
  echo "error: '$CS_FLATC' not found on PATH." >&2
  echo "Install FlatBuffers compiler: https://github.com/google/flatbuffers/releases" >&2
  exit 1
fi

PY_FLATC_VERSION="$("$PY_FLATC" --version)"
CS_FLATC_VERSION="$("$CS_FLATC" --version)"
if [[ "$CS_FLATC_VERSION" != *"$REQUIRED_CSHARP_FLATC_VERSION"* ]]; then
  echo "error: C# schema generation requires flatc ${REQUIRED_CSHARP_FLATC_VERSION}; got: ${CS_FLATC_VERSION}" >&2
  echo "Use environment-mod-build.yml, or set HKRL_CSHARP_FLATC to a matching flatc." >&2
  exit 2
fi

echo "python flatc: $PY_FLATC_VERSION"
echo "csharp flatc: $CS_FLATC_VERSION"
echo "schema: $FBS"

mkdir -p "$PY_OUT" "$CS_OUT"
rm -rf "$PY_OUT/HKRL" "$CS_OUT/HKRL"
"$PY_FLATC" --python -o "$PY_OUT" "$FBS"
"$CS_FLATC" --csharp -o "$CS_OUT" "$FBS"

echo "Generated Python bindings -> $PY_OUT"
echo "Generated C# bindings     -> $CS_OUT"
