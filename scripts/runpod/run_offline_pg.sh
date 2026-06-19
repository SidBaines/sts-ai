#!/usr/bin/env bash
# Offline signed-advantage PG stepping stone (deliverable 3): on the SAME raw
# rollouts the RWR run used, replace exp(floor/beta) resampling with a signed
# advantage = floor - baseline (push below-baseline trajectories DOWN) via the
# clipped+KL PG trainer, then eval base-vs-trained on the frozen eval split.
#
# Reuses already-generated train rollouts (does NOT regenerate). Start from the
# raw rollouts + per-trajectory final_floor, NOT the built SFT dataset (RWR already
# filtered/replicated that).
#
# Env in: MODEL, OUT (artifact root, expects OUT/train_rollouts/<label>/ to exist),
#         CONCURRENCY, K (eval rollouts/seed), REPO_DIR.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/SlayTheSpireAI}"
cd "$REPO_DIR"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
OUT="${OUT:-data/offline_pg}"
CONCURRENCY="${CONCURRENCY:-96}"
K="${K:-4}"
PY=".venv/bin/python"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONPATH=src
LOGS="$OUT/logs"; mkdir -p "$LOGS"
log() { printf '\n========== [%s] %s ==========\n' "$(date -Is)" "$*"; }

# discover_rollouts globs seed_*_r*.jsonl non-recursively → point at the agent-label LEAF dir.
ROLLOUT_DIR=$(find "$OUT/train_rollouts" -mindepth 1 -maxdepth 1 -type d | head -1)
[ -n "$ROLLOUT_DIR" ] || { echo "ERROR: no rollout leaf dir under $OUT/train_rollouts (generate train rollouts first, e.g. via run_until --split train --hints on)"; exit 3; }

log "PHASE 1/3  build OFFLINE signed-advantage PG dataset from $ROLLOUT_DIR"
$PY scripts/build_pg_dataset.py \
  --rollout-dir "$ROLLOUT_DIR" --tokenizer "$MODEL" \
  --mode offline --baseline median --allow-thinking \
  --out "$OUT/pg/offline.jsonl" 2>&1 | tee "$LOGS/1_build.log"
n_ex=$(wc -l < "$OUT/pg/offline.jsonl" | tr -d ' ')
log "built $n_ex PG examples"
[ "$n_ex" -ge 1 ] || { echo "ERROR: empty PG dataset; aborting"; exit 3; }

log "PHASE 2/3  train LoRA with the clipped+KL PG loss (mu=1, signed advantage)"
$PY scripts/train_pg.py \
  --dataset "$OUT/pg/offline.jsonl" --base-model "$MODEL" \
  --manifest "$OUT/pg/offline.jsonl.manifest.json" \
  --out "$OUT/pg/adapter" \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 1 --per-device-batch-size 1 --grad-accum 16 --max-seq-len 4096 \
  --learning-rate 1e-5 --clip-eps 0.2 --kl-beta 0.02 \
  --wandb-project sts-e4b-offline-pg --run-name offline-signed-adv \
  2>&1 | tee "$LOGS/2_train.log"

log "PHASE 3/3  paired eval base vs PG-trained on the frozen eval split (K=$K/seed)"
MODEL="$MODEL" ADAPTER="$OUT/pg/adapter" OUT_EVAL="$OUT/eval" K="$K" \
  CONCURRENCY="$CONCURRENCY" REPO_DIR="$REPO_DIR" \
  bash scripts/runpod/eval_paired.sh 2>&1 | tee "$LOGS/3_eval.log"

log "DONE. Compare $OUT/eval/paired.json against the RWR run's paired result to decide if advantage-shaping beats RWR."
