#!/usr/bin/env python
"""Live monitor for a running GRPO dir: every ~2 min, push partial-iteration
rollout stats and mid-pass training curves to a dedicated wandb run
(``<out-dir name>-live``) and refresh the rl_progress plots.

Read-only with respect to the run; exits when the main run's DONE file appears
or the out-dir goes quiet for an hour. Relaunch it (or let the supervisor's
``ensure_watcher`` do so) after a pause longer than that quiet timeout.

Usage:
    .venv/bin/python scripts/rl_ops/rl_live_watcher.py <out-dir> [<done-file>]

Env: RL_WANDB_PROJECT (default ``sts-ooc-rl``).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

OUT_DIR = Path(sys.argv[1])
DONE_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else None

import wandb  # noqa: E402
from rl_progress import collect_iteration, plot, _iter_index  # noqa: E402

run = wandb.init(
    project=os.environ.get("RL_WANDB_PROJECT", "sts-ooc-rl"),
    name=OUT_DIR.name + "-live",
    resume="allow",
)
tick = 0
last_change = time.time()
last_signature = None

while True:
    tick += 1
    payload: dict[str, float] = {}
    iter_dirs = sorted((d for d in OUT_DIR.glob("iter_*") if d.is_dir()), key=_iter_index)
    if iter_dirs:
        current = collect_iteration(iter_dirs[-1])
        payload["live/iteration"] = float(current["iteration"])
        payload["live/rollouts_done_this_iter"] = float(current["n_rollouts"])
        if current["floors"]:
            payload["live/partial_mean_floor"] = statistics.mean(current["floors"])
            payload["live/partial_max_floor"] = float(max(current["floors"]))
        payload["live/decision_invalid_rate"] = current["decision_invalid_rate"]
        payload["live/decision_retry_rate"] = current["decision_retry_rate"]
        trainer_log = iter_dirs[-1] / "adapter" / "trainer_log.json"
        if trainer_log.exists():
            try:
                history = json.loads(trainer_log.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                history = []
            if history:
                last = history[-1]
                payload["live/train_step"] = float(last.get("step", 0))
                for key in ("loss", "mean_kl", "mean_advantage"):
                    if isinstance(last.get(key), (int, float)):
                        payload[f"live/train_{key}"] = float(last[key])
        try:
            done_iters = [collect_iteration(d) for d in iter_dirs]
            plot([it for it in done_iters if it["n_rollouts"] > 0], OUT_DIR / "progress_plots")
        except Exception:
            pass
    if payload:
        signature = json.dumps(payload, sort_keys=True)
        if signature != last_signature:
            last_change = time.time()
            last_signature = signature
        wandb.log(payload, step=tick)
    if DONE_FILE is not None and DONE_FILE.exists():
        break
    if time.time() - last_change > 3600:
        break
    time.sleep(120)

run.finish()
