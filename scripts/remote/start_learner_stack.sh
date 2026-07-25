#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_CONFIG="${HKRL_TRAIN_CONFIG:-configs/train/ssh_remote_learner.yaml}"
HKRL_PYTHON_BIN="${HKRL_PYTHON_BIN:-python}"

if [[ -z "${HKRL_AUTH_TOKEN:-}" ]]; then
  echo "error: HKRL_AUTH_TOKEN must be set and match the Windows GameWorker" >&2
  exit 2
fi

if [[ "$#" -gt 0 ]]; then
  TASK_PATHS=("$@")
else
  TASK_PATHS=(
    "configs/tasks/gruz_mother.yaml"
    "configs/tasks/hornet_protector.yaml"
    "configs/tasks/mantis_lords.yaml"
  )
fi

cd "${REPO_ROOT}"

for path in "${TRAIN_CONFIG}" "${TASK_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "error: required config does not exist: ${path}" >&2
    exit 2
  fi
done

learner_pid=""
registry_pid=""

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${learner_pid}" ]]; then
    kill "${learner_pid}" 2>/dev/null || true
  fi
  if [[ -n "${registry_pid}" ]]; then
    kill "${registry_pid}" 2>/dev/null || true
  fi
  if [[ -n "${learner_pid}" ]]; then
    wait "${learner_pid}" 2>/dev/null || true
  fi
  if [[ -n "${registry_pid}" ]]; then
    wait "${registry_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONUNBUFFERED=1 "${HKRL_PYTHON_BIN}" scripts/run_checkpoint_server.py \
  --config "${TRAIN_CONFIG}" \
  --bind 127.0.0.1:5601 &
registry_pid="$!"

PYTHONUNBUFFERED=1 "${HKRL_PYTHON_BIN}" scripts/run_learner.py \
  --config "${TRAIN_CONFIG}" \
  --tasks "${TASK_PATHS[@]}" \
  --bind 127.0.0.1:5600 \
  --serve-forever &
learner_pid="$!"

echo "HKRL remote stack started: learner=127.0.0.1:5600 registry=127.0.0.1:5601"
echo "Both ports are loopback-only; connect from Windows with the SSH tunnel script."

set +e
wait -n "${learner_pid}" "${registry_pid}"
stack_status="$?"
set -e
exit "${stack_status}"
