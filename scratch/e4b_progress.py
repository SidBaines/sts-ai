#!/usr/bin/env python3
"""SCRATCH (temporary): live progress + rate table for the E4B-thinking H100 run.

Pulls just the small *.meta.json sidecars from the pod (fast; the big .jsonl are
left to sync_back), then prints a snapshot: completion counts, the in-flight
pipeline (so a "filling" pipeline doesn't look stalled), outcome mix, floor
reach, and throughput rates. Re-run anytime.

    PYTHONPATH=src .venv/bin/python scratch/e4b_progress.py

Notes:
- Reads run2.pid/run2_start.txt if present (the resumed run), else run.pid/run_start.txt.
- "in flight" = started - completed: at high --concurrency with long full-game
  rollouts, completions are back-loaded, so watch `started` / `in flight` climb
  even while `completed` is flat. That is the pipeline filling, not a stall.
- Completed-rollout stats are survivorship-biased early (fast deaths finish first).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone

POD = "runpod-e4b-h100"
REMOTE = "/workspace/SlayTheSpireAI/data/rollouts/e4b_think_perf"
LOCAL = "data/rollouts/e4b_think_perf"
ARM = "vllm_gemma_4_E4B_it_thinking_8192"
TARGET = 5000


def pull() -> tuple[datetime | None, bool, int, int]:
    os.makedirs(LOCAL, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-rtz",
            "--include=*/", "--include=*.meta.json", "--exclude=*",
            "-e", "ssh -o StrictHostKeyChecking=no",
            f"{POD}:{REMOTE}/", f"{LOCAL}/",
        ],
        check=False,
    )
    # one round-trip: start time, alive, and live pod completed/started counts
    remote = (
        f'D={REMOTE}/{ARM}; '
        '(cat /workspace/run2_start.txt 2>/dev/null || cat /workspace/run_start.txt 2>/dev/null); echo "|||"; '
        '( kill -0 $(cat /workspace/run2.pid 2>/dev/null) 2>/dev/null '
        '|| kill -0 $(cat /workspace/run.pid 2>/dev/null) 2>/dev/null ) && echo ALIVE || echo DEAD; echo "|||"; '
        'ls $D/*.meta.json 2>/dev/null | wc -l; echo "|||"; '
        'ls $D/*.jsonl 2>/dev/null | wc -l'
    )
    out = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", POD, remote],
        capture_output=True, text=True,
    ).stdout
    parts = [p.strip() for p in out.split("|||")]
    start = None
    if parts and parts[0]:
        try:
            start = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    alive = len(parts) > 1 and "ALIVE" in parts[1]
    pod_completed = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    pod_started = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    return start, alive, pod_completed, pod_started


def outcome_of(m: dict) -> str:
    o = str(m.get("outcome", ""))
    if "VICTORY" in o:
        return "victory"
    if "LOSS" in o or "DEATH" in o:
        return "death"
    if m.get("stopped_reason") == "max_decisions":
        return "budget_cut"
    if m.get("stopped_reason") == "agent_invalid":
        return "agent_invalid"
    if m.get("stopped_reason") == "simulator_error":
        return "sim_error"
    return o or m.get("stopped_reason") or "other"


def fmt_dur(secs: float) -> str:
    secs = int(secs)
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m{secs % 60:02d}s"


def main() -> None:
    start, alive, pod_completed, pod_started = pull()
    now = datetime.now(timezone.utc)
    paths = glob.glob(os.path.join(LOCAL, "**", "*.meta.json"), recursive=True)
    metas, mtimes = [], []
    for p in paths:
        try:
            metas.append(json.load(open(p)))
            mtimes.append(os.path.getmtime(p))
        except Exception:
            continue

    n = len(metas)
    elapsed = (now - start).total_seconds() if start else 0.0
    print("=" * 64)
    print(f" E4B-it · thinking · full-game   run {'ALIVE' if alive else 'STOPPED'}")
    print(f" started {start.strftime('%H:%M:%SZ') if start else '?'}  ·  elapsed {fmt_dur(elapsed)}")
    print("=" * 64)
    in_flight = max(0, pod_started - pod_completed)
    print(f" pipeline  : {pod_started} started · {pod_completed} completed · {in_flight} in flight (pod, live)")
    print(f"             (high concurrency back-loads completions for long rollouts;")
    print(f"              watch 'started'/'in flight' climb even while 'completed' lags)")
    if not n:
        print(" no completed rollouts synced yet")
        return

    start_ts = start.timestamp() if start else 0.0
    run_completed = sum(1 for t in mtimes if t >= start_ts)  # this-run only (excl. prior run's metas)
    overall_rpm = run_completed / (elapsed / 60) if elapsed else 0.0
    cutoff = time.time() - 600
    recent = sum(1 for t in mtimes if t >= cutoff)
    print("-" * 64)
    print(f" completed (synced, all runs): {n}  ·  this run since resume: {run_completed}")
    print(f" rate      : {overall_rpm:5.1f} completed/min since resume   |   {recent/10.0:5.1f}/min synced last 10m")

    oc = Counter(outcome_of(m) for m in metas)
    print("-" * 64)
    print(" outcome           count    %")
    for k in ["victory", "death", "budget_cut", "agent_invalid", "sim_error"]:
        if oc.get(k):
            print(f"   {k:<15} {oc[k]:>5}  {100*oc[k]/n:4.0f}%")

    floors = [int(m.get("final_floor", 0)) for m in metas]
    acts = Counter(int(m.get("final_act", 1)) for m in metas)
    total_dec = sum(int(m.get("n_decisions", 0)) for m in metas)
    n_inval = sum(int(m.get("n_invalid", 0)) for m in metas)
    print("-" * 64)
    print(f" final floor : mean {statistics.mean(floors):.1f} · median {int(statistics.median(floors))} · max {max(floors)}")
    print(" reached act : " + "  ".join(f"act{a}:{acts[a]}" for a in sorted(acts)))
    print(f" invalid dec : {n_inval} / {total_dec} ({100*n_inval/max(1,total_dec):.1f}% of decisions)")
    print("=" * 64)


if __name__ == "__main__":
    main()
