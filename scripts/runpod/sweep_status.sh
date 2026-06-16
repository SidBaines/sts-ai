#!/usr/bin/env bash
# Read-only snapshot of a sweep on a pod. Safe to run anytime; it only reads.
#
# Works across pods/GPUs and runs with NO hardcoded paths: with just an ssh
# target it auto-detects the active output dir (from the running run_sweep.py
# process, else the most-recently-written rollout), pulls logs from inside that
# run tree, and reports orchestrator/concurrency + GPU. Pass an explicit out-dir
# to inspect a specific (e.g. finished) run.
#
# Usage: scripts/runpod/sweep_status.sh [ssh-target] [out-dir] [repo-dir]
set -uo pipefail

HOST="${1:-runpod-sts-h100}"
OUT="${2:-}"                              # empty => auto-detect on the pod
REPO="${3:-/workspace/SlayTheSpireAI}"

# Pass OUT/REPO as env vars (not positionals): an empty positional arg is dropped
# when it travels through ssh's command-string join, which breaks `set -u`.
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$HOST" "OUT='$OUT' REPO='$REPO' bash -s" <<'REMOTE'
set -uo pipefail
OUT="${OUT:-}"; REPO="${REPO:-/workspace/SlayTheSpireAI}"
cd "$REPO" 2>/dev/null || true

echo "== sweep status @ $(date -u +%H:%M:%SZ) UTC =="

# --- runner state ---
if pgrep -f run_sweep_on_pod >/dev/null 2>&1; then runner=RUNNING; else runner=STOPPED; fi
echo "host: $(hostname)  runner: $runner"

# --- auto-detect the output dir if not given ---
if [ -z "$OUT" ]; then
  # 1) the --output-dir of a live run_sweep.py process (the authoritative active run)
  OUT=$(pgrep -af 'run_sweep.py' 2>/dev/null \
        | grep -oE -- '--output-dir[ =][^ ]+' | head -1 | sed -E 's/--output-dir[ =]//')
fi
if [ -z "$OUT" ]; then
  # 2) fall back to the most-recently-written rollout's sweep dir (two levels up from a *.jsonl)
  latest=$(ls -t data/rollouts/*/*/*.jsonl 2>/dev/null | head -1)
  [ -n "$latest" ] && OUT=$(dirname "$(dirname "$latest")")
fi
if [ -z "$OUT" ]; then
  echo "(no run found under data/rollouts/*/*/. Available sweep dirs:)"
  ls -d data/rollouts/*/ 2>/dev/null | sed 's/^/  /'
  echo "Pass an out-dir explicitly: sweep_status.sh <host> <out-dir>"
fi
echo "out-dir: ${OUT:-<none>}"
# show sibling runs so multi-run pods are discoverable
sibs=$(ls -d data/rollouts/*/ 2>/dev/null | sed 's#data/rollouts/##;s#/##' | tr '\n' ' ')
[ -n "$sibs" ] && echo "runs on pod: $sibs"

if [ -n "${OUT:-}" ]; then
OUT="$OUT" .venv/bin/python - <<'PY'
import json, glob, os, collections
base = os.environ["OUT"].rstrip("/")
metas = glob.glob(base + "/**/*.meta.json", recursive=True)
done, dec, inval = collections.Counter(), collections.Counter(), collections.Counter()
orch = conc = None
for f in metas:
    a = os.path.basename(os.path.dirname(f))
    try:
        m = json.load(open(f))
    except Exception:
        continue
    done[a] += 1; dec[a] += m.get("n_decisions", 0); inval[a] += m.get("n_invalid", 0)
    ex = m.get("extra", {}) or {}
    orch = orch or ex.get("orchestrator"); conc = conc if conc is not None else ex.get("concurrency")
if orch or conc is not None:
    print(f"orchestrator: {orch}  concurrency: {conc}")
print("-- completed arms (finalized rollouts) --")
for a in sorted(done):
    inv = inval[a]; rate = (inv / dec[a]) if dec[a] else 0
    print(f"  {a:44s} rollouts={done[a]:2d} dec={dec[a]:6d} invalid={inv} ({rate:.1%})")
print(f"  TOTAL finalized rollouts: {sum(done.values())}")

jl = collections.defaultdict(list)
for f in glob.glob(base + "/**/*.jsonl", recursive=True):
    a = os.path.basename(os.path.dirname(f))
    jl[a].append(sum(1 for _ in open(f)))
inflight = {a: xs for a, xs in jl.items() if len(xs) > done.get(a, 0)}
if inflight:
    print("-- in-flight arm (continuous batching: decisions written per rollout) --")
    for a, xs in sorted(inflight.items()):
        xs = sorted(xs)
        print(f"  {a:44s} started={len(xs)} finalized={done.get(a,0)} "
              f"dec[min/med/max]={xs[0]}/{xs[len(xs)//2]}/{xs[-1]}")
PY
fi

# --- GPU (name makes the pod's accelerator obvious across A40/H100/etc.) ---
echo "-- GPU --"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
  || echo "  (nvidia-smi unavailable)"

# --- live tail from the per-model logs INSIDE the run tree (no nohup path needed) ---
if [ -n "${OUT:-}" ] && ls "$OUT"/logs/*.log >/dev/null 2>&1; then
  echo "-- recent arm events (from $OUT/logs) --"
  grep -hE 'specs=|done:|FAILED to load' "$OUT"/logs/*.log 2>/dev/null | tail -8
  latest_log=$(ls -t "$OUT"/logs/*.log 2>/dev/null | head -1)
  echo "-- last line ($(basename "$latest_log")) --"
  tail -1 "$latest_log" 2>/dev/null | tr '\r' '\n' | tail -1
fi
REMOTE
