#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: setup_pod.sh [repo-dir]

Prepare a RunPod Ubuntu pod for SlayTheSpireAI vLLM rollouts.
Default repo-dir: /workspace/SlayTheSpireAI
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="${1:-/workspace/SlayTheSpireAI}"

log() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*"
}

if [[ ! -d "$REPO_DIR" ]]; then
  printf 'ERROR: repo directory does not exist: %s\n' "$REPO_DIR" >&2
  printf 'Clone the repo first, or pass its path as the first argument.\n' >&2
  exit 2
fi

REPO_DIR="$(cd "$REPO_DIR" && pwd)"
cd "$REPO_DIR"

apt_prefix=()
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    apt_prefix=(sudo)
  else
    printf 'ERROR: missing root privileges and sudo is unavailable; cannot install apt packages.\n' >&2
    exit 2
  fi
fi

python_headers_available() {
  python3 - <<'PY'
import os
import sysconfig

include_dir = sysconfig.get_config_var("INCLUDEPY")
raise SystemExit(0 if include_dir and os.path.exists(os.path.join(include_dir, "Python.h")) else 1)
PY
}

need_apt=0
for cmd in git cmake g++ make python3 rsync; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    need_apt=1
  fi
done

if ! python3 -m venv --help >/dev/null 2>&1; then
  need_apt=1
fi

if ! python_headers_available; then
  need_apt=1
fi

if (( need_apt )); then
  if ! command -v apt-get >/dev/null 2>&1; then
    printf 'ERROR: build dependencies are missing and apt-get is unavailable.\n' >&2
    exit 2
  fi
  log "Installing Ubuntu build dependencies with apt-get"
  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    python3-dev \
    python3-venv \
    rsync
else
  log "Build dependencies already available; skipping apt-get"
fi

if [[ ! -x .venv/bin/python ]]; then
  log "Creating project virtualenv at .venv"
  python3 -m venv .venv
else
  log "Project virtualenv already exists"
fi

log "Upgrading pip"
.venv/bin/pip install -U pip

log "Installing project with vLLM optional dependencies"
.venv/bin/pip install -e '.[vllm]'

log "Building C++ lightspeed simulator"
bash scripts/build_lightspeed.sh

log "setup complete: RunPod pod is ready for vLLM rollouts"
printf 'Reminder: run "huggingface-cli login" or export HF_TOKEN before gated Gemma/Llama weights.\n'
