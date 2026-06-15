#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_sweep_on_pod.sh <models-file> [out-dir] [seeds]

Runs one model per scripts/run_sweep.py process, sequentially, on the current GPU.

Defaults:
  out-dir: data/rollouts/a40_sweep
  seeds:   3,4,5,6,7,8
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

MODELS_ARG="$1"
OUT_DIR="${2:-data/rollouts/a40_sweep}"
SEEDS="${3:-3,4,5,6,7,8}"
# Concurrent rollouts per arm (the throughput lever). Capped in practice by the
# number of seeds, so raising it only helps when SEEDS is large. Override via env.
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_DECISIONS="${MAX_DECISIONS:-200}"

ORIGINAL_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ "$MODELS_ARG" = /* ]]; then
  MODELS_FILE="$MODELS_ARG"
elif [[ -f "$MODELS_ARG" ]]; then
  MODELS_FILE="$REPO_DIR/$MODELS_ARG"
elif [[ -f "$ORIGINAL_CWD/$MODELS_ARG" ]]; then
  MODELS_FILE="$ORIGINAL_CWD/$MODELS_ARG"
else
  printf 'ERROR: models file not found: %s\n' "$MODELS_ARG" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  printf 'ERROR: .venv/bin/python not found. Run scripts/runpod/setup_pod.sh first.\n' >&2
  exit 2
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

timestamp() {
  date -Is
}

banner() {
  printf '\n[%s] ===== %s =====\n' "$(timestamp)" "$*"
}

models=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(trim "$line")"
  if [[ -n "$line" ]]; then
    models+=("$line")
  fi
done < "$MODELS_FILE"

if [[ ${#models[@]} -eq 0 ]]; then
  printf 'ERROR: no models found in %s\n' "$MODELS_FILE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

banner "Starting sweep"
printf 'Repo:        %s\n' "$REPO_DIR"
printf 'Models file: %s\n' "$MODELS_FILE"
printf 'Output dir:  %s\n' "$OUT_DIR"
printf 'Seeds:       %s\n' "$SEEDS"
printf 'Batch size:  %s   Max decisions: %s\n' "$BATCH_SIZE" "$MAX_DECISIONS"
printf 'Models:      %s\n' "${models[*]}"

successes=()
failures=()

for model in "${models[@]}"; do
  safe_model="${model//\//_}"
  safe_model="${safe_model//:/_}"
  log_file="$LOG_DIR/${safe_model}.log"

  banner "START model=${model}"
  {
    printf '\n[%s] START model=%s\n' "$(timestamp)" "$model"
    printf '[%s] log_file=%s\n' "$(timestamp)" "$log_file"
  } | tee -a "$log_file"

  # vLLM does not reliably free GPU memory between in-process model loads.
  # A fresh Python process per model is slower but avoids fragmented/stale GPU state.
  cmd=(
    .venv/bin/python scripts/run_sweep.py
    --backend vllm
    --models "$model"
    --thinking both
    --combat-control llm
    --max-tokens 4096
    --temperature 0
    --battle-simulations 50
    --max-decisions "$MAX_DECISIONS"
    --batch-size "$BATCH_SIZE"
    --seeds "$SEEDS"
    --output-dir "$OUT_DIR"
  )

  set +e
  PYTHONPATH=src "${cmd[@]}" 2>&1 | tee -a "$log_file"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    successes+=("$model")
    printf '[%s] FINISH model=%s rc=0\n' "$(timestamp)" "$model" | tee -a "$log_file"
    banner "FINISH model=${model} rc=0"
  else
    failures+=("$model (rc=$rc)")
    printf '[%s] FAILED model=%s rc=%s; continuing to next model\n' "$(timestamp)" "$model" "$rc" | tee -a "$log_file"
    banner "FAILED model=${model} rc=${rc}; continuing"
  fi
done

banner "Sweep summary"
printf 'Succeeded (%s):\n' "${#successes[@]}"
if [[ ${#successes[@]} -gt 0 ]]; then
  printf '  %s\n' "${successes[@]}"
else
  printf '  none\n'
fi

printf 'Failed (%s):\n' "${#failures[@]}"
if [[ ${#failures[@]} -gt 0 ]]; then
  printf '  %s\n' "${failures[@]}"
  exit 1
else
  printf '  none\n'
fi
