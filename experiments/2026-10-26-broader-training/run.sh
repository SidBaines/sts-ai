#!/usr/bin/env bash
# Fixed continuation settings; stop through the run's STOP file, not script edits.
set -u
cd "$(dirname "$0")/../.."
SCREEN=${1:?run name}
ROOT=data/llm_runs/$SCREEN
if [[ -e "$ROOT/STOP" ]]; then exit 0; fi
echo "start broader training $(date '+%F %T')"
PYTHONPATH=src PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-mlx/bin/python -m sts_llm.mlx_training \
  --mlx-model data/models/k2-horizon-0.9b-mlx \
  --adapter data/llm_runs/stage1_screen_ps_20261026/lr_3e-5/adapters/iter_0048.safetensors \
  --train data/llm_runs/broader_states_selected_20261026/train.jsonl \
  --train-eval data/llm_runs/broader_states_selected_20261026/train_eval.jsonl \
  --probe data/llm_runs/broader_states_selected_20261026/probe.jsonl \
  --output "$ROOT/lr_3e-5" --iterations 256 --max-run-seconds 28800 \
  --states-per-iteration 4 --group-size 8 --effort medium --menu-order per_sample \
  --temperature 0.6 --max-new-tokens 4096 --learning-rate 3e-5 \
  --loss-normalization fixed_token_budget --loss-token-budget 2048 --kl-beta 0 \
  --invalid-reward -1 --eval-every 32 --eval-orders 1 --eval-concurrency 16 \
  --save-every 8 --seed 1729 --eval-seed 1729 > "$ROOT/lr_3e-5.log" 2>&1 &
TRAINER=$!
echo "trainer_pid $TRAINER" >> "$ROOT/pids.txt"
python3 -u experiments/2026-10-26-broader-training/watch.py "$ROOT" "$TRAINER" \
  > "$ROOT/watch.log" 2>&1 &
WATCHER=$!
echo "watcher_pid $WATCHER" >> "$ROOT/pids.txt"
wait "$TRAINER"
RESULT=$?
echo "end learning rate 3e-5 exit $RESULT $(date '+%F %T')"
wait "$WATCHER"
if [[ $RESULT -ne 0 ]]; then
  echo "SCREEN_FAILED exit $RESULT"
  exit "$RESULT"
fi
echo "SCREEN_DONE $(date '+%F %T')"
