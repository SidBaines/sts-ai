#!/usr/bin/env bash
# SCRATCH (temporary): measure steady-state aggregate dec/s vs --concurrency for
# the E4B-it thinking config, to see if the live run's ~7 dec/s is concurrency-
# limited. Runs each arm to steady state, samples 60s, then frees the GPU.
set +e
cd /workspace/SlayTheSpireAI
export HF_TOKEN=$(cat /workspace/.hf_token)
export VLLM_USE_FLASHINFER_SAMPLER=0
RESULTS=/workspace/bench_results2.txt
: > "$RESULTS"
SEED=200000

free_gpu() {
  pkill -9 -f run_until.py 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  for i in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -lt 2000 ] && break
    sleep 4
  done
}

for C in 256 384 512; do
  OUT=data/rollouts/bench_c$C
  rm -rf "$OUT"
  D=$OUT/vllm_gemma_4_E4B_it_thinking_8192
  echo "[bench] launching concurrency=$C ..." | tee -a "$RESULTS"
  nohup env PYTHONPATH=src .venv/bin/python scripts/run_until.py \
    --model google/gemma-4-E4B-it --backend vllm --thinking \
    --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 8192 \
    --max-act 1 --combat-control llm --battle-simulations 50 \
    --concurrency "$C" --target $((C * 2)) --max-decisions 200 \
    --seed-start "$SEED" --output-dir "$OUT" > /workspace/bench_c$C.log 2>&1 &
  PID=$!
  SEED=$((SEED + 2000))
  started=0
  for i in $(seq 1 72); do   # up to 6 min for load + CUDA-graph capture + ramp
    n=$(cat $D/*.jsonl 2>/dev/null | wc -l)
    if [ "${n:-0}" -gt "$C" ]; then started=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "[bench] c=$C process died early (see bench_c$C.log)" | tee -a "$RESULTS"; break
    fi
    sleep 5
  done
  if [ "$started" -eq 1 ]; then
    sleep 25   # settle to steady state
    a=$(cat $D/*.jsonl 2>/dev/null | wc -l)
    sleep 60
    b=$(cat $D/*.jsonl 2>/dev/null | wc -l)
    inflight=$(( $(ls $D/*.jsonl 2>/dev/null | wc -l) - $(ls $D/*.meta.json 2>/dev/null | wc -l) ))
    preempt=$(grep -ic "preempt" /workspace/bench_c$C.log 2>/dev/null)
    gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    python3 -c "print(f'[bench] RESULT concurrency=$C -> {($b-$a)/60:.1f} dec/s aggregate (delta {$b-$a} over 60s, ~$inflight in flight, preempt_events=$preempt, gpu_mem=${gmem}MiB)')" | tee -a "$RESULTS"
  fi
  kill "$PID" 2>/dev/null
  free_gpu
  echo "[bench] c=$C teardown done (GPU used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB)" | tee -a "$RESULTS"
done
echo "[bench] ALL DONE" | tee -a "$RESULTS"
