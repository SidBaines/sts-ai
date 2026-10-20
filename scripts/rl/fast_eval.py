#!/usr/bin/env python3
"""Evaluate greedy policies in disposable processes with batched native games."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.batched_runner import configure_cpu, evaluation_summary, run_multiprocess
from sts_rl.fresh_study import atomic_json, validate_world_range
from sts_rl.recording import file_identity


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Frozen champion composite")
    p.add_argument("--combat")
    p.add_argument("--ooc")
    p.add_argument("--world-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--games-per-process", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120., help="Seconds without a completed decision per process")
    p.add_argument("--no-skip-rares", action="store_true", default=False,
                   help="overlay: at a card reward whose cards are all rare, take one rather than skipping")
    p.add_argument("--take-card", action="append", default=[], metavar="NAME",
                   help="overlay: always take this card (upgraded copies too) when a reward offers it; repeatable")
    p.add_argument("--out", required=True)
    return p


def overlays_from(config):
    """Menu overlay keywords for sts_rl.common.apply_overlays, or None when no rule is set."""
    overlays = {}
    if config.get("no_skip_rares"):
        overlays["no_skip_rares"] = True
    if config.get("take_card"):
        overlays["take_cards"] = tuple(config["take_card"])
    return overlays or None


def prepare_output(config):
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")
    from sts_env._backend.lightspeed_import import native_metadata
    run = {"schema": "sts_batched_eval_v1", "config": config, "created_unix": time.time(),
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "checkpoints": {key: file_identity(config[key]) if config.get(key) else None
                           for key in ("checkpoint", "combat", "ooc")},
           "native": native_metadata(), "python": sys.version}
    if "games_per_process" in config:
        start, n, processes = config["world_start"], config["n"], config["processes"]
        run["batching"] = {"games_per_process": config["games_per_process"],
                           "processes": min(processes, n),
                           "world_seeds_by_process": [list(range(start + i, start + n, processes))
                                                      for i in range(min(processes, n))]}
    atomic_json(out / "run.json", run)
    return out


def main(argv=None):
    p = parser()
    config = vars(p.parse_args(argv))
    if (config["processes"] < 1 or config["games_per_process"] < 1 or
            not math.isfinite(config["timeout"]) or config["timeout"] <= 0):
        p.error("Require positive processes, games per process and finite positive timeout")
    validate_world_range(config["world_start"], config["n"])
    configure_cpu()
    out = prepare_output(config)
    start = time.perf_counter()
    try:
        records = run_multiprocess(range(config["world_start"], config["world_start"] + config["n"]),
            config["checkpoint"], config["combat"], config["ooc"], processes=config["processes"],
            games_per_process=config["games_per_process"], timeout=config["timeout"],
            overlays=overlays_from(config))
    except Exception as exc:
        summary = evaluation_summary(getattr(exc, "records", []), config["n"])
        summary.update(error=f"{type(exc).__name__}: {exc}", confidence_claims_allowed=False,
                       requeues=getattr(exc, "requeues", summary["requeues"]))
        atomic_json(out / "summary.json", summary)
        raise
    summary = evaluation_summary(records, config["n"])
    elapsed = time.perf_counter() - start
    summary.update(seconds=elapsed, games_per_second=summary["completed"] / elapsed,
                   decisions_per_second=sum(sum(r["decisions"].values()) for r in records) / elapsed)
    atomic_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("completed", "wins", "faults", "cuts", "requeues", "seconds",
                                              "games_per_second", "decisions_per_second")}))


if __name__ == "__main__":
    main()
