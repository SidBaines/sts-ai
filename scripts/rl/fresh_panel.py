#!/usr/bin/env python3
"""Freeze a whole-world bank grid and compare champion/replacement combat heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_panel import run_panel


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True, nargs="+", help="Completed receipted collection directories")
    p.add_argument("--bank-world-start", type=int, help="Optional exact expected bank seedgrid start")
    p.add_argument("--bank-n", type=int, help="Optional exact expected bank seedgrid size")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--combat", required=True, help="entity_fresh_v1 replacement combat head")
    p.add_argument("--out", required=True)
    p.add_argument("--starts-per-group", type=json.loads,
                   default={"BOSS2": 300, "BOSS1": 100, "BOSS3": 100,
                            "ELITE2": 150, "ELITE3": 100, "MONSTER2": 100, "MONSTER3": 100,
                            "EVENT1": 100, "EVENT2": 100, "EVENT3": 100})
    p.add_argument("--resamples", type=int, default=8, help="Salted repeats, plus exact unsalted control -1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--deadline-utc")
    return p


def main(argv=None):
    summary = run_panel(vars(parser().parse_args(argv)))
    print(json.dumps({k: summary[k] for k in ("event", "status", "cohort_complete", "grid_complete",
                                             "valid", "faults", "cuts", "censored", "unstarted")}))


if __name__ == "__main__":
    main()
