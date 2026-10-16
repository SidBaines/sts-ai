#!/usr/bin/env python3
"""Validate C6 panel receipts and report the predeclared G1 new-minus-old contrast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_panel_comparison import compare_panel_runs, write_comparison
from sts_rl.fresh_statistics import BOOTSTRAPS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True, help="Completed NEW combat-student panel directory")
    parser.add_argument("--old", required=True, help="Completed OLD combat-student panel directory")
    parser.add_argument("--grid", required=True, help="Common frozen grid JSON supplied to both runs")
    parser.add_argument("--expectations", required=True, help="Root-frozen expectations JSON")
    parser.add_argument("--out", required=True, help="Fresh output JSON file; overwrites rejected")
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS, help="Default 10000; seed fixed at 1919")
    args = parser.parse_args(argv)
    try:
        if Path(args.out).exists():
            raise FileExistsError(f"Output already exists: {args.out}")
        if args.bootstraps < 2:
            raise ValueError("Require at least two bootstrap resamples")
        report = compare_panel_runs(args.new, args.old, args.grid, args.expectations, resamples=args.bootstraps)
        write_comparison(args.out, report)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"fresh_panel_compare: {exc}\n")
    primary = report["groups"]["BOSS2:resampled"]
    print(json.dumps({"output": str(Path(args.out).resolve()), "grid_hash": report["grid_hash"],
                     "control_exact_match": True, "BOSS2_resampled_role": primary["role"],
                     **{m: {k: primary[m][k] for k in ("paired_difference", "ci95", "rows", "worlds")}
                        for m in ("win", "hp_lost")}}, sort_keys=True))


if __name__ == "__main__":
    main()
