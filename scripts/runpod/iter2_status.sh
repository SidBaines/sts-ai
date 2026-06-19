#!/usr/bin/env bash
# One-shot progress snapshot for the iter2 RWR+hinted pipeline run.
# Usage: bash scripts/runpod/iter2_status.sh
ALIAS="${ALIAS:-runpod-20260619_SidsH100_iter2b}"
POD="${POD:-8qllohk2hiado9}"

echo "=== pod ($POD) ==="
runpodctl pod list 2>/dev/null | grep -E "NAME|$POD" || echo "(not listed — likely STOPPED, i.e. run finished)"

ssh -o ConnectTimeout=15 "$ALIAS" '
cd /workspace/SlayTheSpireAI
echo "=== phase ==="
grep -E "PHASE [0-9]/6" /workspace/pipeline_resume.log 2>/dev/null | tail -1 || echo "(no phase marker yet)"
pgrep -f run_iter2_resume.sh >/dev/null && echo "pipeline: RUNNING" || echo "pipeline: ENDED"
echo "=== rollouts (train target=300, eval target=100/arm) ==="
for d in train_rollouts eval/base eval/trained; do
  n=$(find "data/iter2_rwr_hinted/$d" -name "*.meta.json" 2>/dev/null | wc -l | tr -d " ")
  [ "$n" != "0" ] && echo "  $d: $n done"
done
echo "=== outcomes + mean floor so far ==="
python3 - <<PY
import glob,json,collections
c=collections.Counter(); floors=[]
for p in glob.glob("data/iter2_rwr_hinted/**/*.meta.json",recursive=True):
    try: m=json.load(open(p))
    except Exception: continue
    c[str(m.get("outcome"))[:28]]+=1; floors.append(m.get("final_floor",0) or 0)
for k,v in c.most_common(): print(f"  {v:4d}  {k}")
if floors: print(f"  mean_floor={sum(floors)/len(floors):.1f}  n={len(floors)}")
PY
echo "=== latest active log (last 6 lines) ==="
ls -t data/iter2_rwr_hinted/logs/*.log 2>/dev/null | head -1 | xargs -I{} sh -c "echo {}; tail -6 {}" 2>/dev/null
'
