#!/usr/bin/env bash
# iter2 offline-BC pipeline: RWR weighting + hinted rollouts + native-thinking
# format, on Gemma-4 E4B, end-to-end on one CUDA H100.
#
#   Phase 1  generate   native-thinking + hinted full-game rollouts (TRAIN split, vLLM)
#   Phase 2  build       RWR-weighted SFT dataset (broadens pool past the act-boss filter)
#   Phase 3  train       TRL LoRA adapter (assistant-only loss)
#   Phase 4  eval-base   base model on the frozen EVAL split (no hints, no adapter)
#   Phase 5  eval-train   trained adapter on the SAME eval seeds
#   Phase 6  compare      paired report
#
# `set -e` is deliberate: a phase failure stops the chain and leaves logs so we
# can see exactly where it broke (per the "skip the smoke test" decision).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/SlayTheSpireAI}"
cd "$REPO_DIR"

MODEL="${MODEL:-google/gemma-4-E4B-it}"
OUT="${OUT:-data/iter2_rwr_hinted}"
CONCURRENCY="${CONCURRENCY:-96}"
PY=".venv/bin/python"

# Sampling (matches the data-gen + prior-eval config so base re-establishes the
# baseline on the same seeds): native thinking, temp 1.0, Gemma nucleus params.
SAMPLING=(--thinking --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192)
GAME=(--combat-control llm --max-act 3 --battle-simulations 50)

# flashinfer JIT needs nvcc; fall back to the native torch sampler if the pod's
# toolkit doesn't match the cu13 torch wheels (negligible cost). Harmless if unused.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONPATH=src

LOGS="$OUT/logs"
mkdir -p "$LOGS"
log() { printf '\n========== [%s] %s ==========\n' "$(date -Is)" "$*"; }

# ---------------------------------------------------------------------------
log "PHASE 1/6  generate train-split rollouts (seeds 200-499, hints on, native thinking)"
$PY scripts/run_until.py \
  --model "$MODEL" --backend vllm \
  --target 300 --seed-start 200 --concurrency "$CONCURRENCY" \
  --exclude-seeds-config configs/frozen_seeds.json \
  --preserve-special-tokens on \
  --hints on \
  "${SAMPLING[@]}" "${GAME[@]}" \
  --output-dir "$OUT/train_rollouts" 2>&1 | tee "$LOGS/1_generate.log"

n_meta=$(find "$OUT/train_rollouts" -name '*.meta.json' | wc -l | tr -d ' ')
log "generated $n_meta rollout metas"
[ "$n_meta" -ge 1 ] || { echo "ERROR: no rollouts produced; aborting"; exit 3; }

# discover_rollouts globs seed_*_r*.jsonl NON-recursively, so point it at the
# agent-label leaf dir, not the parent train_rollouts.
ROLLOUT_DIR=$(find "$OUT/train_rollouts" -mindepth 1 -maxdepth 1 -type d | head -1)

# ---------------------------------------------------------------------------
log "PHASE 2/6  build RWR-weighted SFT dataset"
$PY scripts/build_sft_dataset.py \
  --rollout-dir "$ROLLOUT_DIR" \
  --tokenizer "$MODEL" \
  --weighting-mode rwr --rwr-beta 5.0 --rwr-baseline median --rwr-max-multiplier 8 \
  --allow-thinking \
  --out "$OUT/sft/train.jsonl" 2>&1 | tee "$LOGS/2_build.log"

n_ex=$(wc -l < "$OUT/sft/train.jsonl" | tr -d ' ')
log "built $n_ex SFT examples"
[ "$n_ex" -ge 1 ] || { echo "ERROR: empty dataset; aborting"; exit 3; }

# ---------------------------------------------------------------------------
log "PHASE 3/6  train LoRA (TRL, assistant-only loss)"
$PY scripts/train_policy.py \
  --backend trl --base-model "$MODEL" \
  --dataset "$OUT/sft/train.jsonl" \
  --manifest "$OUT/sft/train.manifest.json" \
  --out "$OUT/adapter" \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 1 --per-device-batch-size 1 --grad-accum 16 --max-seq-len 4096 --learning-rate 1e-4 \
  --wandb-project sts-e4b-offline-bc --run-name iter2-rwr-hinted-thinking \
  2>&1 | tee "$LOGS/3_train.log"

# ---------------------------------------------------------------------------
# Eval on the FROZEN eval split (seeds 47-151 minus exclusions = 100 seeds),
# disjoint from the 200-499 train split. NO hints (measure the unaided policy).
log "PHASE 4/6  eval BASE on frozen eval split"
$PY scripts/run_until.py \
  --model "$MODEL" --backend vllm \
  --target 100 --seed-start 47 --concurrency "$CONCURRENCY" \
  --exclude-seeds-config configs/frozen_seeds.json \
  "${SAMPLING[@]}" "${GAME[@]}" \
  --output-dir "$OUT/eval/base" 2>&1 | tee "$LOGS/4_eval_base.log"

log "PHASE 5/6  eval TRAINED adapter on the same eval seeds"
$PY scripts/run_until.py \
  --model "$MODEL" --backend vllm \
  --adapter-path "$OUT/adapter" --max-lora-rank 16 \
  --target 100 --seed-start 47 --concurrency "$CONCURRENCY" \
  --exclude-seeds-config configs/frozen_seeds.json \
  "${SAMPLING[@]}" "${GAME[@]}" \
  --output-dir "$OUT/eval/trained" 2>&1 | tee "$LOGS/5_eval_trained.log"

# ---------------------------------------------------------------------------
log "PHASE 6/6  compare"
$PY scripts/compare_models.py "$OUT/eval" 2>&1 | tee "$LOGS/6_compare.log" || true
$PY - "$OUT/eval" <<'PY' 2>&1 | tee "$LOGS/6_headline.log"
import glob, json, sys
root = sys.argv[1]
for arm in ("base", "trained"):
    metas = [json.load(open(p)) for p in glob.glob(f"{root}/{arm}/**/*.meta.json", recursive=True)]
    n = len(metas)
    clears = sum(1 for m in metas if m.get("final_act", 1) >= 2 or "VICTORY" in str(m.get("outcome")))
    wins = sum(1 for m in metas if "VICTORY" in str(m.get("outcome")))
    floors = [m.get("final_floor", 0) for m in metas]
    inval = sum(m.get("n_invalid", 0) for m in metas)
    dec = sum(m.get("n_decisions", 0) for m in metas)
    mf = sum(floors) / max(n, 1)
    print(f"{arm:8s} n={n} boss_clear={clears}/{n} ({100*clears/max(n,1):.0f}%) "
          f"wins={wins} mean_floor={mf:.2f} invalid_rate={inval/max(dec,1):.4f}")
PY

log "DONE. Artifacts under $OUT (adapter/, eval/, sft/, logs/)."
