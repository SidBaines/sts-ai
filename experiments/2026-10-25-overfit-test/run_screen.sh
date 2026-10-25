#!/usr/bin/env bash
# Stage-1 learning-rate screen: one run per learning rate, one after another.
# Settings are fixed on research/experiments/2026-10-25-overfit-test.md; do not
# change them mid-screen. Launch detached with launch_screen.py. Create
# data/llm_runs/<screen>/STOP to stop after the current run (and the current
# run's own OUTPUT/STOP to stop it between iterations).
set -u
cd "$(dirname "$0")/../.."
SCREEN=${1:?screen directory name under data/llm_runs}
shift
ROOT=data/llm_runs/$SCREEN
mkdir -p "$ROOT"
for LR in "$@"; do
  if [[ -e "$ROOT/STOP" ]]; then
    echo "screen stopped before learning rate $LR"
    break
  fi
  echo "start learning rate $LR $(date '+%F %T')"
  PYTHONPATH=src PYTHONUNBUFFERED=1 .venv-mlx/bin/python -m sts_llm.mlx_training \
    --mlx-model data/models/k2-horizon-0.9b-mlx \
    --train data/llm_runs/stage1_states_20261025/train.jsonl \
    --probe data/llm_runs/stage1_states_20261025/probe.jsonl \
    --output "$ROOT/lr_$LR" \
    --iterations 48 --states-per-iteration 4 --group-size 8 --effort medium \
    --temperature 0.6 --max-new-tokens 4096 --learning-rate "$LR" \
    --loss-normalization fixed_token_budget --loss-token-budget 2048 --kl-beta 0 \
    --invalid-reward -1 --eval-every 24 --eval-orders 1 --save-every 8 --seed 1729 \
    > "$ROOT/lr_$LR.log" 2>&1
  echo "end learning rate $LR exit $? $(date '+%F %T')"
done
echo "SCREEN_DONE $(date '+%F %T')"
