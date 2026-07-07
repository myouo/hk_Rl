#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFS_DIR="${ROOT}/.refs/Managed"
FLATC="${HKRL_FLATC:-flatc}"
REQUIRED_FLATC_VERSION="${HKRL_CSHARP_FLATC_VERSION:-23.5.26}"

cd "${ROOT}"

FLATC_VERSION="$("${FLATC}" --version)"
if [[ "${FLATC_VERSION}" != *"${REQUIRED_FLATC_VERSION}"* ]]; then
  echo "C# mod build requires flatc ${REQUIRED_FLATC_VERSION}; got: ${FLATC_VERSION}" >&2
  echo "Use the C# build CI environment, not the Python hkrl env flatc." >&2
  exit 2
fi

"${FLATC}" --csharp -o mod/HKRLEnvMod/Schema schema/hkrl.fbs
mkdir -p "${REFS_DIR}"

dotnet build mod/ci-stubs/UnityEngine/UnityEngine.csproj -c Release
dotnet build mod/ci-stubs/UnityEngine.CoreModule/UnityEngine.CoreModule.csproj -c Release
dotnet build mod/ci-stubs/Assembly-CSharp/Assembly-CSharp.csproj -c Release
dotnet build mod/ci-stubs/MMHOOK_Assembly-CSharp/MMHOOK_Assembly-CSharp.csproj -c Release
dotnet build mod/ci-stubs/PlayMaker/PlayMaker.csproj -c Release

dotnet build mod/HKRLEnvMod/HKRLEnvMod.csproj \
  -c Release \
  -p:HollowKnightManaged="${REFS_DIR}" \
  -p:TreatWarningsAsErrors=true
