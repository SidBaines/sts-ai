#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sync_back.sh <ssh-target> <remote-dir> <local-dir> [interval-seconds]

Continuously rsync rollout artifacts from a RunPod pod to this machine.

Examples:
  scripts/runpod/sync_back.sh runpod-podA /workspace/SlayTheSpireAI/data/rollouts/a40_sweep data/rollouts/a40_sweep
  scripts/runpod/sync_back.sh "root@203.0.113.10 -p 22022" /workspace/SlayTheSpireAI/data/rollouts/a40_sweep data/rollouts/a40_sweep

The ssh-target may be an SSH config alias, or a quoted "root@<ip> -p <port>" target.
Default interval: 600 seconds.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

SSH_TARGET="$1"
REMOTE_DIR="${2%/}"
LOCAL_DIR="${3%/}"
INTERVAL_SECONDS="${4:-600}"

timestamp() {
  date -Is
}

ssh_host="$SSH_TARGET"
ssh_options=(-o StrictHostKeyChecking=no)

if [[ "$SSH_TARGET" == *" "* ]]; then
  read -r -a target_parts <<< "$SSH_TARGET"
  ssh_host="${target_parts[0]}"
  if [[ ${#target_parts[@]} -gt 1 ]]; then
    ssh_options+=("${target_parts[@]:1}")
  fi
fi

ssh_command="ssh"
for option in "${ssh_options[@]}"; do
  ssh_command+=" $(printf '%q' "$option")"
done

mkdir -p "$LOCAL_DIR"

printf '[%s] Sync loop started: %s:%s/ -> %s/ every %ss\n' \
  "$(timestamp)" "$ssh_host" "$REMOTE_DIR" "$LOCAL_DIR" "$INTERVAL_SECONDS"
printf '[%s] Pulling JSONL, .meta.json, .error.json, and logs/ from the remote output directory.\n' "$(timestamp)"

while true; do
  printf '[%s] rsync cycle starting\n' "$(timestamp)"
  if rsync -avz --partial -e "$ssh_command" "${ssh_host}:${REMOTE_DIR}/" "${LOCAL_DIR}/"; then
    printf '[%s] rsync cycle complete\n' "$(timestamp)"
  else
    rc=$?
    printf '[%s] rsync failed with rc=%s; will retry after %ss\n' "$(timestamp)" "$rc" "$INTERVAL_SECONDS" >&2
  fi
  sleep "$INTERVAL_SECONDS"
done
