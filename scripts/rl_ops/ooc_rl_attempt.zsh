#!/bin/zsh
# One supervised attempt of the ooc_grpo_v1 OOC-GRPO run. Launched detached
# (own session) by rl_supervisor.zsh so this zsh is the process-group leader:
# the babysitter's STOP/CONT and the supervisor's stall kill -9 target the
# whole group (this wrapper + the python) without touching the supervisor.
#
# The run_grpo argument list below IS the v1 run definition — treat it as
# provenance. Copy this file (and point RL_ATTEMPT_SCRIPT at the copy) for a
# different run rather than editing it in place.
#
# argv: <attempt-number> <policy-seed-salt> <start-iteration> <resume-adapter-or-NONE>
ATTEMPT=${1:?attempt number}
SALT=${2:?policy seed salt}
START_ITER=${3:?start iteration}
RESUME=${4:?resume adapter path or NONE}

HERE=${0:A:h}
cd ${HERE:h:h}

OUT=${RL_OUT_DIR:-data/rl/ooc_grpo_v1}
OPS=${RL_OPS_DIR:-data/rl_ops/${OUT:t}}
mkdir -p "$OPS"
STATUS=$OPS/run.status
note() { print -r -- "[$(date '+%m-%d %H:%M:%S')] attempt-$ATTEMPT: $*" >> "$STATUS" }

resume_args=()
if [[ "$RESUME" != "NONE" ]]; then
  resume_args=(--start-iteration $START_ITER --resume-adapter "$RESUME")
fi

note "starting (pid $$, salt $SALT, start-iter $START_ITER, resume $RESUME)"
PYTHONPATH=src .venv/bin/python scripts/run_grpo.py \
  --backend mlx \
  --base-model mlx-community/gemma-4-e4b-it-bf16 \
  --tokenizer mlx-community/gemma-4-e4b-it-bf16 \
  --train-seeds-config configs/competence/ooc_rl_train16_v1.json --train-split ooc_rl_train16 \
  --combat-control search --battle-simulations 50 \
  --num-iterations 12 --seeds-per-iter 16 --group-size 6 \
  --concurrency 12 --temperature 0.7 \
  --max-seq-len 1024 --max-decisions 250 \
  --output-contract action_text --max-retries 2 \
  --kl-beta 0.02 --learning-rate 1e-5 \
  --per-device-batch-size 1 --grad-accum 8 \
  --train-example-cap 1000 \
  --wandb-project sts-ooc-rl --run-name ${OUT:t} \
  --out-dir $OUT \
  --policy-seed-salt $SALT \
  $resume_args >> $OPS/run.log 2>&1
rc=$?
note "run_grpo exited rc=$rc"
print -r -- "$rc" > "$OPS/attempt_${ATTEMPT}.rc"
