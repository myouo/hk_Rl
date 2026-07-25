#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/remote/bootstrap_learner_env.sh [options]

Options:
  --env-name NAME          Conda environment name (default: hkrl-learner)
  --torch-index-url URL    Optional PyTorch wheel index selected at pytorch.org
  --reinstall-torch        Force reinstall torch (useful after a CPU-only install)
  --allow-cpu              Permit a CPU-only environment for local validation
  -h, --help               Show this help

Run this script from a Linux training device with NVIDIA drivers installed.
The repository's environment.yml is intentionally CPU-only and is not used here.
EOF
}

ENV_NAME="hkrl-learner"
TORCH_INDEX_URL=""
REINSTALL_TORCH=0
ALLOW_CPU=0

while (($# > 0)); do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || {
        echo "--env-name requires a value" >&2
        exit 2
      }
      ENV_NAME="$2"
      shift 2
      ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || {
        echo "--torch-index-url requires a value" >&2
        exit 2
      }
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --reinstall-torch)
      REINSTALL_TORCH=1
      shift
      ;;
    --allow-cpu)
      ALLOW_CPU=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$ENV_NAME" ]] || {
  echo "--env-name must not be empty" >&2
  exit 2
}

command -v conda >/dev/null 2>&1 || {
  echo "conda was not found. Install Miniconda/Miniforge first." >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! conda run --name "$ENV_NAME" python --version >/dev/null 2>&1; then
  conda create --yes --name "$ENV_NAME" python=3.10 pip
fi

conda run --name "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel

TORCH_INSTALL_ARGS=(install --upgrade torch)
if ((REINSTALL_TORCH)); then
  TORCH_INSTALL_ARGS+=(--force-reinstall)
fi
if [[ -n "$TORCH_INDEX_URL" ]]; then
  TORCH_INSTALL_ARGS+=(--index-url "$TORCH_INDEX_URL")
fi
conda run --name "$ENV_NAME" python -m pip "${TORCH_INSTALL_ARGS[@]}"

conda run --name "$ENV_NAME" python -m pip install \
  -e "${REPO_ROOT}/python[dev,logging,distributed]" \
  "jupyterlab>=4" \
  "nbformat>=5.9" \
  "nbclient>=0.10" \
  "ipykernel>=6.29"

conda run --name "$ENV_NAME" python -m ipykernel install \
  --user \
  --name "$ENV_NAME" \
  --display-name "Python ($ENV_NAME)"

conda run --name "$ENV_NAME" python -c \
  "import torch; print({'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})"

if ((ALLOW_CPU == 0)); then
  conda run --name "$ENV_NAME" python -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable. Re-run with the wheel index selected at https://pytorch.org/get-started/locally/ and --reinstall-torch.'"
fi

echo
echo "Environment ready: $ENV_NAME"
echo "Start JupyterLab with:"
echo "  conda run --name $ENV_NAME jupyter lab --no-browser"
