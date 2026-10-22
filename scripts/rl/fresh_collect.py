#!/usr/bin/env python3
"""Fresh-only collection/evaluation; see sts_rl.fresh_collection for protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_collection import DEFAULT_PROBABILITIES, run_collection, validate_config
from sts_rl.fresh_study import StudyPolicy
from sts_rl.teacher_votes import RULES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("eval", "collect"), required=True)
    p.add_argument("--checkpoint", required=True, help="Frozen legacy champion composite")
    p.add_argument("--combat")
    p.add_argument("--ooc")
    p.add_argument("--out", required=True)
    p.add_argument("--world-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0, help="Finite per-RPC seconds")
    p.add_argument("--deadline-utc", help="Absolute ISO UTC time, e.g. 2026-10-16T08:00:00Z")
    p.add_argument("--bank", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--teacher-labels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacher-probabilities", type=json.loads, default=DEFAULT_PROBABILITIES,
                   help='Frozen group probabilities as JSON, e.g. {"BOSS2":1,"MONSTER1":0.04}')
    p.add_argument("--max-labels", type=int, default=0, help="Combat labels; stop between worlds, 0=unbounded")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--simulations", type=int, default=1000)
    p.add_argument("--vote-rule", choices=RULES, default=argparse.SUPPRESS, help="Default best_line")
    p.add_argument("--teacher-plays", action="store_true", default=argparse.SUPPRESS, help="Benchmark only; exclude corpus from training")
    p.add_argument("--record-search", action="store_true", default=argparse.SUPPRESS, help="Save per-search diagnostics")
    p.add_argument("--self-consistency-every", type=int, default=100,
                   help="TEST worlds only, hash-selected fraction 1/N, disjoint draw/RNG/search seeds; 0=disabled")
    return p


def main(argv=None):
    config = vars(parser().parse_args(argv))
    if config.get("vote_rule") == "best_line":
        config.pop("vote_rule")
    validate_config(config)
    policy = StudyPolicy(config["checkpoint"], config["combat"], config["ooc"])
    summary = run_collection(config, policy)
    print(json.dumps({k: summary[k] for k in ("event", "status", "cohort_complete", "completed",
                                             "faults", "cuts", "censored", "unstarted", "wins")}))


if __name__ == "__main__":
    main()
