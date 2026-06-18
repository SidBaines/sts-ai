#!/usr/bin/env bash
# SCRATCH (temporary): full-game (--max-act 3) concurrency benchmark for E4B-it
# thinking. Unlike the first (act-1, instant-sample) bench, this lets each arm
# WARM UP so rollouts progress into the game (realistic longer prompts), then
# samples for a longer window, measuring BOTH aggregate dec/s and completed
# rollouts/min (the metric that actually matters for full-game collection), plus
# preemption (longer contexts may hit the KV cache where act-1 did not) and the
# avg in-flight decision depth (maturity indicator).
set +e
cd /workspace/SlayTheSpireAI
export HF_TOKEN=$(cat /workspace/.hf_token)
export VLLM_USE_FLASHINFER_SAMPLER=0
R=/workspace/fullgame_bench.txt
: > "$R"
SEED=300000
WARMUP=300   # let rollouts progress into the game so prompts are realistic
SAMPLE=180   # longer sample window than the first bench

free_gpu() {
  pkill -9 -f run_until.py 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  for i in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u:-0}" -lt 2000 ] && break
    sleep 4
  done
}

for C in 64 128 256 384; do
  OUT=data/rollouts/fgbench_c$C
  rm -rf "$OUT"
  D=$OUT/vllm_gemma_4_E4B_it_thinking_8192
  echo "[fg] launching full-game concurrency=$C ..." | tee -a "$R"
  nohup env PYTHONPATH=src .venv/bin/python scripts/run_until.py \
    --model google/gemma-4-E4B-it --backend vllm --thinking \
    --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192 \
    --max-act 3 --combat-control llm --battle-simulations 50 \
    --concurrency "$C" --target $((C * 3)) --max-decisions 1500 \
    --seed-start "$SEED" --output-dir "$OUT" > /workspace/fg_c$C.log 2>&1 &
  PID=$!
  SEED=$((SEED + 5000))
  started=0
  for i in $(seq 1 72); do   # up to 6 min for load + ramp to C in flight
    n=$(cat $D/*.jsonl 2>/dev/null | wc -l)
    if [ "${n:-0}" -gt "$C" ]; then started=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "[fg] c=$C died early (see fg_c$C.log)" | tee -a "$R"; break; fi
    sleep 5
  done
  if [ "$started" -eq 1 ]; then
    sleep "$WARMUP"
    m0=$(ls $D/*.meta.json 2>/dev/null | wc -l)
    a=$(cat $D/*.jsonl 2>/dev/null | wc -l)
    sleep "$SAMPLE"
    b=$(cat $D/*.jsonl 2>/dev/null | wc -l)
    m1=$(ls $D/*.meta.json 2>/dev/null | wc -l)
    started_n=$(ls $D/*.jsonl 2>/dev/null | wc -l)
    inflight=$(( started_n - m1 ))
    preempt=$(grep -ic preempt /workspace/fg_c$C.log 2>/dev/null)
    gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    python3 -c "
ds=($b-$a)/$SAMPLE
comp=($m1-$m0)*60.0/$SAMPLE
depth=$b/max(1,$started_n)
print(f'[fg] RESULT C=$C -> {ds:.1f} dec/s | {comp:.1f} completed/min sampled | ~$inflight in flight | depth~{depth:.0f} dec/started-roll | preempt=$preempt | mem=${gmem}MiB')
" | tee -a "$R"
  fi
  kill "$PID" 2>/dev/null
  free_gpu
  echo "[fg] c=$C teardown done" | tee -a "$R"
done
echo "[fg] ALL DONE" | tee -a "$R"
