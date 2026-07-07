#!/usr/bin/env bash
# Resume an interrupted GRPO run from the latest adapter on HuggingFace.
#
# The GRPO loop pushes each completed iteration's adapter to HF
# (grpo/iter_<N>/adapter). After a pod loss, on a FRESH pod (repo cloned +
# setup_pod.sh + .[train-cuda] + sim built + HF_TOKEN exported), this script:
#   1. finds the highest iter_<N> with a uploaded adapter on $HF_REPO,
#   2. downloads it locally,
#   3. relaunches run_grpo.sh with START_ITERATION=N+1, RESUME_ADAPTER=<local>.
# No cross-iteration optimizer state is needed (each iter trains fresh from the
# adapter), so the adapter weights are a sufficient checkpoint.
#
# Env in: MODEL, HF_REPO (required), OUT, ITERS, and the rest of run_grpo.sh's env.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/SlayTheSpireAI}"
cd "$REPO_DIR"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
HF_REPO="${HF_REPO:?set HF_REPO (e.g. user/sts-e4b-grpo) — the run's HF model repo}"
ITERS="${ITERS:-20}"
PY=".venv/bin/python"
export PYTHONPATH=src
export HF_TOKEN="${HF_TOKEN:-$(grep -E '^HF_TOKEN=' .env 2>/dev/null | head -1 | cut -d= -f2-)}"

echo "Locating latest completed adapter on $HF_REPO ..."
LATEST=$($PY - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
files = HfApi().list_repo_files(sys.argv[1], repo_type="model")
done = sorted(
    int(f.split("/")[1].removeprefix("iter_"))
    for f in files
    if f.startswith("grpo/iter_") and f.endswith("/adapter/adapter_model.safetensors")
)
print(done[-1] if done else -1)
PY
)
[ "$LATEST" -ge 0 ] 2>/dev/null || { echo "ERROR: no completed iter_<N>/adapter on $HF_REPO"; exit 3; }
NEXT=$((LATEST + 1))
if [ "$NEXT" -ge "$ITERS" ]; then echo "All $ITERS iterations already complete (latest=$LATEST). Nothing to resume."; exit 0; fi

LOCAL_ADAPTER="$REPO_DIR/data/grpo_resume/iter_${LATEST}_adapter"
echo "Latest completed = iter_$LATEST. Downloading its adapter -> $LOCAL_ADAPTER"
$PY - "$HF_REPO" "$LATEST" "$LOCAL_ADAPTER" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, latest, dest = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(
    repo_id=repo, repo_type="model",
    allow_patterns=[f"grpo/iter_{latest}/adapter/*"],
    local_dir=dest + "_dl",
)
# flatten to the adapter dir
import shutil, os
src = os.path.join(dest + "_dl", "grpo", f"iter_{latest}", "adapter")
if os.path.exists(dest): shutil.rmtree(dest)
shutil.copytree(src, dest)
print("downloaded adapter to", dest)
PY

echo "Resuming GRPO from iteration $NEXT (of $ITERS) ..."
START_ITERATION="$NEXT" RESUME_ADAPTER="$LOCAL_ADAPTER" MODEL="$MODEL" HF_REPO="$HF_REPO" ITERS="$ITERS" \
  bash scripts/runpod/run_grpo.sh
