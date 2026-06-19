#!/usr/bin/env bash
# Lower-variance paired eval: K rollouts/seed on the frozen EVAL split for the
# base model and a trained ADAPTER, then a paired per-seed floor delta + sign test
# (compare_paired.py). Reused by run_offline_pg.sh and run_grpo.sh.
#
# Env in: MODEL, ADAPTER (dir or ""), OUT_EVAL (output root), K (rollouts/seed),
#         CONCURRENCY, REPO_DIR. Writes OUT_EVAL/{base,trained}/ + paired.json.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/SlayTheSpireAI}"
cd "$REPO_DIR"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
ADAPTER="${ADAPTER:-}"
OUT_EVAL="${OUT_EVAL:-data/eval_paired}"
K="${K:-4}"
CONCURRENCY="${CONCURRENCY:-96}"
PY=".venv/bin/python"

SAMPLING=(--thinking --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192)
GAME=(--combat-control llm --max-act 3 --battle-simulations 50)
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONPATH=src

LOGS="$OUT_EVAL/logs"; mkdir -p "$LOGS"
log() { printf '\n========== [%s] %s ==========\n' "$(date -Is)" "$*"; }

# One eval arm draws the frozen eval split's exact seeds, K rollouts each, skipping
# its exclusions (exclusions live in the same config as the splits).
log "eval BASE: eval split x K=$K (no adapter)"
$PY scripts/run_until.py \
  --model "$MODEL" --backend vllm \
  --seeds-config configs/frozen_seeds.json --split eval --rollouts-per-seed "$K" \
  --concurrency "$CONCURRENCY" \
  "${SAMPLING[@]}" "${GAME[@]}" \
  --output-dir "$OUT_EVAL/base" 2>&1 | tee "$LOGS/eval_base.log"

if [ -n "$ADAPTER" ]; then
  log "eval TRAINED: same eval split x K=$K, adapter=$ADAPTER"
  $PY scripts/run_until.py \
    --model "$MODEL" --backend vllm \
    --adapter-path "$ADAPTER" --max-lora-rank 16 \
    --seeds-config configs/frozen_seeds.json --split eval --rollouts-per-seed "$K" \
    --concurrency "$CONCURRENCY" \
    "${SAMPLING[@]}" "${GAME[@]}" \
    --output-dir "$OUT_EVAL/trained" 2>&1 | tee "$LOGS/eval_trained.log"

  log "paired compare (per-seed floor delta + sign test, beside agent_invalid)"
  $PY scripts/compare_paired.py \
    --base "$OUT_EVAL/base" --trained "$OUT_EVAL/trained" \
    --metric final_floor --out "$OUT_EVAL/paired.json" 2>&1 | tee "$LOGS/paired.log"
else
  log "no ADAPTER given; base-only eval (skipping paired compare)"
fi
log "DONE. Eval artifacts under $OUT_EVAL (base/, trained/, paired.json, logs/)."
