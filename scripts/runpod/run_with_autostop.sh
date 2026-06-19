#!/usr/bin/env bash
# Wrapper: run the iter2 pipeline, then STOP this pod on exit — success OR
# failure — to halt GPU billing. Stop (not delete) keeps the disk + artifacts.
#
# An EXIT trap (not an appended last line) is used deliberately: the pipeline
# runs `set -e`, so a phase failure exits early; a last-line stop would then
# never run and the pod would bill all night. The trap fires on any exit.
#
# Pod id + API key come from RunPod's injected pod env. On these pods those vars
# live in PID 1's environment (/proc/1/environ), NOT the ssh/nohup shell, so the
# stop call reads them from there. curl is blocked by RunPod's WAF, so the call
# goes through python `requests`. Override with $RP_POD_ID if needed.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/SlayTheSpireAI}"

stop_self() {
  code=$?
  echo "[autostop] pipeline exited code=$code; attempting to stop this pod to halt GPU billing"
  RP_POD_ID="${RP_POD_ID:-}" python3 <<'PY' || echo "[autostop] WARN: stop call failed — stop manually: runpodctl pod stop <id>"
import os, requests
def proc1_env():
    env = {}
    try:
        with open("/proc/1/environ", "rb") as f:
            for kv in f.read().split(b"\0"):
                if b"=" in kv:
                    k, v = kv.split(b"=", 1)
                    env[k.decode(errors="replace")] = v.decode(errors="replace")
    except Exception:
        pass
    return env
penv = proc1_env()
pod_id = os.environ.get("RP_POD_ID") or os.environ.get("RUNPOD_POD_ID") or penv.get("RUNPOD_POD_ID")
key = os.environ.get("RUNPOD_API_KEY") or penv.get("RUNPOD_API_KEY")
if not pod_id or not key:
    raise SystemExit("[autostop] missing pod id or API key; cannot self-stop")
q = 'mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod_id
r = requests.post("https://api.runpod.io/graphql?api_key=" + key, json={"query": q}, timeout=30)
print("[autostop] stop pod", pod_id, "-> http", r.status_code, r.text[:200])
PY
}
trap stop_self EXIT

bash "$REPO_DIR/scripts/runpod/run_iter2_pipeline.sh"
