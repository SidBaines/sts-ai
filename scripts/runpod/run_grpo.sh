#!/usr/bin/env bash
# On-policy GRPO stage (deliverable 4): the in-process loop regenerates K rollouts
# per train-split seed at temp>0 from the CURRENT policy (streaming orchestrator),
# computes group-relative advantages, runs a clipped+KL update, and hot-swaps the
# new adapter into the resident vLLM engine — repeating for --num-iterations. Then
# a paired eval of base vs the final adapter on the frozen eval split.
#
# DRY-RUN FIRST: docs/grpo_dryrun_checklist.md §5 (1 iter, 2 seeds x G=2, tiny model)
# — the in-process vLLM-sleep <-> trainer GPU hand-off is the #1 thing to validate.
#
# Env in: MODEL, OUT (artifact root), CONCURRENCY, K (eval rollouts/seed),
#         ITERS, GROUP_SIZE, SEEDS_PER_ITER, KL_BETA, CLIP_EPS, LR, REPO_DIR.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/SlayTheSpireAI}"
cd "$REPO_DIR"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
OUT="${OUT:-data/grpo}"
CONCURRENCY="${CONCURRENCY:-96}"
K="${K:-4}"
ITERS="${ITERS:-20}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SEEDS_PER_ITER="${SEEDS_PER_ITER:-8}"
KL_BETA="${KL_BETA:-0.02}"
CLIP_EPS="${CLIP_EPS:-0.2}"
LR="${LR:-1e-5}"
PY=".venv/bin/python"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONPATH=src
LOGS="$OUT/logs"; mkdir -p "$LOGS"
log() { printf '\n========== [%s] %s ==========\n' "$(date -Is)" "$*"; }

log "PHASE 1/2  GRPO loop: $ITERS iters x ($SEEDS_PER_ITER seeds x G=$GROUP_SIZE) at temp>0, in-process LoRA hot-swap"
$PY scripts/run_grpo.py \
  --base-model "$MODEL" --tokenizer "$MODEL" \
  --train-seeds-config configs/frozen_seeds.json --train-split train \
  --out-dir "$OUT/grpo" \
  --num-iterations "$ITERS" --group-size "$GROUP_SIZE" --seeds-per-iter "$SEEDS_PER_ITER" \
  --concurrency "$CONCURRENCY" \
  --temperature 1.0 --top-p 0.95 --top-k 64 \
  --combat-control llm --max-act 3 --max-decisions 1500 --battle-simulations 50 \
  --clip-eps "$CLIP_EPS" --kl-beta "$KL_BETA" --learning-rate "$LR" \
  2>&1 | tee "$LOGS/1_grpo.log"

# The loop prints {"final_adapter": ...}; the canonical location is the last iteration's adapter.
FINAL_ADAPTER=$(find "$OUT/grpo" -maxdepth 2 -type d -name adapter | sort -V | tail -1)
[ -n "$FINAL_ADAPTER" ] || { echo "ERROR: no adapter produced under $OUT/grpo"; exit 3; }
log "final adapter: $FINAL_ADAPTER"

log "PHASE 2/2  paired eval base vs GRPO-final on the frozen eval split (K=$K/seed)"
MODEL="$MODEL" ADAPTER="$FINAL_ADAPTER" OUT_EVAL="$OUT/eval" K="$K" \
  CONCURRENCY="$CONCURRENCY" REPO_DIR="$REPO_DIR" \
  bash scripts/runpod/eval_paired.sh 2>&1 | tee "$LOGS/2_eval.log"

log "DONE. $OUT/eval/paired.json shows GRPO-final vs base (read beside the agent_invalid rate)."
