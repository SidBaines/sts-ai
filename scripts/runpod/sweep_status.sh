#!/usr/bin/env bash
# Read-only snapshot of a running (or finished) sweep on a pod. Safe to run anytime;
# it only reads. Shows: runner state, recent per-model banners, completed arms
# (rollouts / decisions / invalid), in-flight arm decision progress, GPU, last vLLM line.
#
# Usage: scripts/runpod/sweep_status.sh [ssh-target] [out-dir] [repo-dir] [sweep-log]
set -uo pipefail

HOST="${1:-runpod-sts-podA}"
OUT="${2:-/workspace/SlayTheSpireAI/data/rollouts/a40_sweep}"
REPO="${3:-/workspace/SlayTheSpireAI}"
SWEEP_LOG="${4:-/workspace/sweep.out}"

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$HOST" bash -s "$OUT" "$REPO" "$SWEEP_LOG" <<'REMOTE'
set -uo pipefail
OUT="$1"; REPO="$2"; SWEEP_LOG="$3"
cd "$REPO" 2>/dev/null || true

echo "== sweep status @ $(date -u +%H:%M:%SZ) UTC =="
if pgrep -f run_sweep_on_pod >/dev/null 2>&1; then echo "runner: RUNNING"; else echo "runner: STOPPED"; fi

echo "-- recent model banners --"
grep -hE 'START model=|done:|FAILED to load|SUMMARY|succeeded|failed' "$SWEEP_LOG" 2>/dev/null | tail -14

OUT="$OUT" .venv/bin/python - <<'PY'
import json, glob, os, collections
base = os.environ["OUT"]
done, dec, inval = collections.Counter(), collections.Counter(), collections.Counter()
for f in glob.glob(base + "/**/*.meta.json", recursive=True):
    a = os.path.basename(os.path.dirname(f))
    try:
        m = json.load(open(f))
    except Exception:
        continue
    done[a] += 1; dec[a] += m.get("n_decisions", 0); inval[a] += m.get("n_invalid", 0)
print("-- completed arms (finalized rollouts) --")
for a in sorted(done):
    print(f"  {a:44s} rollouts={done[a]:2d} dec={dec[a]:6d} invalid={inval[a]}")
print(f"  TOTAL finalized rollouts: {sum(done.values())}")

jl = collections.defaultdict(list)
for f in glob.glob(base + "/**/*.jsonl", recursive=True):
    a = os.path.basename(os.path.dirname(f))
    jl[a].append(sum(1 for _ in open(f)))
inflight = {a: xs for a, xs in jl.items() if len(xs) > done.get(a, 0)}
if inflight:
    print("-- in-flight arm (decisions written per rollout) --")
    for a, xs in sorted(inflight.items()):
        xs = sorted(xs)
        print(f"  {a:44s} rollouts_started={len(xs)} finalized={done.get(a,0)} "
              f"dec[min/med/max]={xs[0]}/{xs[len(xs)//2]}/{xs[-1]}")
PY

echo "-- GPU --"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
echo "-- last vLLM line --"
tail -1 "$SWEEP_LOG" 2>/dev/null | tr '\r' '\n' | tail -1
REMOTE
