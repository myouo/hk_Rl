#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/linux/start_ssh_tunnel.sh --remote USER@HOST [options]

Options:
  --ssh-port PORT              SSH server port (default: 22)
  --identity-file PATH         SSH private key
  --local-learner-port PORT    Local rollout forward (default: 5600)
  --local-registry-port PORT   Local registry forward (default: 5601)
  --remote-learner-port PORT   Remote learner port (default: 5600)
  --remote-registry-port PORT  Remote registry port (default: 5601)
  --dry-run                    Print resolved topology without connecting
  -h, --help                   Show this help
EOF
}

REMOTE=""
SSH_PORT=22
IDENTITY_FILE=""
LOCAL_LEARNER_PORT=5600
LOCAL_REGISTRY_PORT=5601
REMOTE_LEARNER_PORT=5600
REMOTE_REGISTRY_PORT=5601
DRY_RUN=0

while (($# > 0)); do
  case "$1" in
    --remote)
      [[ $# -ge 2 ]] || hkrl_die "--remote requires a value"
      REMOTE="$2"
      shift 2
      ;;
    --ssh-port)
      [[ $# -ge 2 ]] || hkrl_die "--ssh-port requires a value"
      SSH_PORT="$2"
      shift 2
      ;;
    --identity-file)
      [[ $# -ge 2 ]] || hkrl_die "--identity-file requires a value"
      IDENTITY_FILE="$2"
      shift 2
      ;;
    --local-learner-port)
      LOCAL_LEARNER_PORT="$2"
      shift 2
      ;;
    --local-registry-port)
      LOCAL_REGISTRY_PORT="$2"
      shift 2
      ;;
    --remote-learner-port)
      REMOTE_LEARNER_PORT="$2"
      shift 2
      ;;
    --remote-registry-port)
      REMOTE_REGISTRY_PORT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      hkrl_die "unknown option: $1"
      ;;
  esac
done

[[ -n "${REMOTE}" ]] || hkrl_die "--remote is required"
[[ "${REMOTE}" != -* && ! "${REMOTE}" =~ [[:space:]] ]] \
  || hkrl_die "--remote must be an SSH alias or user@host without whitespace"
for item in \
  "${SSH_PORT}:ssh port" \
  "${LOCAL_LEARNER_PORT}:local learner port" \
  "${LOCAL_REGISTRY_PORT}:local registry port" \
  "${REMOTE_LEARNER_PORT}:remote learner port" \
  "${REMOTE_REGISTRY_PORT}:remote registry port"; do
  hkrl_validate_port "${item%%:*}" "${item#*:}"
done
((LOCAL_LEARNER_PORT != LOCAL_REGISTRY_PORT)) \
  || hkrl_die "local learner and registry ports must be different"
command -v ssh >/dev/null 2>&1 || hkrl_die "ssh was not found"

SSH_ARGS=(
  -N
  -T
  -p "${SSH_PORT}"
  -o BatchMode=yes
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
  -L "127.0.0.1:${LOCAL_LEARNER_PORT}:127.0.0.1:${REMOTE_LEARNER_PORT}"
  -L "127.0.0.1:${LOCAL_REGISTRY_PORT}:127.0.0.1:${REMOTE_REGISTRY_PORT}"
)
if [[ -n "${IDENTITY_FILE}" ]]; then
  [[ -f "${IDENTITY_FILE}" ]] || hkrl_die \
    "SSH identity file does not exist: ${IDENTITY_FILE}"
  SSH_ARGS+=(-o IdentitiesOnly=yes -i "${IDENTITY_FILE}")
fi
SSH_ARGS+=("${REMOTE}")

if ((DRY_RUN)); then
  python3 - \
    "${REMOTE}" \
    "${LOCAL_LEARNER_PORT}" \
    "${LOCAL_REGISTRY_PORT}" \
    "${REMOTE_LEARNER_PORT}" \
    "${REMOTE_REGISTRY_PORT}" <<'PY'
import json
import sys

remote, local_learner, local_registry, remote_learner, remote_registry = sys.argv[1:]
print(
    json.dumps(
        {
            "learner_forward": (
                f"127.0.0.1:{local_learner} -> 127.0.0.1:{remote_learner}"
            ),
            "note": "The real-time Hollow Knight action loop remains local.",
            "registry_forward": (
                f"127.0.0.1:{local_registry} -> 127.0.0.1:{remote_registry}"
            ),
            "remote": remote,
        },
        sort_keys=True,
    )
)
PY
  exit 0
fi

printf '%s\n' \
  "SSH tunnel active in this foreground terminal; press Ctrl+C to stop it."
exec ssh "${SSH_ARGS[@]}"
