#!/usr/bin/env python3
"""Freeze one start per source world/group, then compare paired combat heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_panel import DEFAULT_TARGETS, freeze_panel_grid, run_panel


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True, nargs="+", help="Completed receipted collection directories")
    p.add_argument("--bank-world-start", type=int, help="Optional exact expected bank seedgrid start")
    p.add_argument("--bank-n", type=int, help="Optional exact expected bank seedgrid size")
    p.add_argument("--checkpoint")
    p.add_argument("--combat", help="entity_fresh_v1 or entity_controlled_v1 replacement combat head")
    p.add_argument("--out", required=True, help="New run directory, or new grid JSON file with --freeze-grid-only")
    p.add_argument("--grid", help="Pre-frozen immutable grid JSON; exact bank, targets and resamples must match")
    p.add_argument("--freeze-grid-only", action="store_true", help="Create --out grid JSON without loading any model")
    p.add_argument("--starts-per-group", type=json.loads,
                   default=dict(DEFAULT_TARGETS), help="Positive source-WORLD counts per target group (JSON)")
    p.add_argument("--resamples", type=int, default=8, help="Salted repeats, plus exact unsalted control -1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--deadline-utc")
    return p


def main(argv=None):
    p = parser()
    config = vars(p.parse_args(argv))
    if config["freeze_grid_only"]:
        if config["grid"] or config["checkpoint"] or config["combat"]:
            p.error("--freeze-grid-only takes bank/targets/resamples/out, without --grid or model arguments")
        frozen = freeze_panel_grid(config)
        print(json.dumps({k: frozen[k] for k in ("grid_hash", "selection_protocol", "resamples", "targets", "coverage",
                                                "coverage_sufficient", "coverage_shortfall", "absent_groups")}))
        return
    if not config["checkpoint"] or not config["combat"]:
        p.error("Panel comparison requires --checkpoint and --combat; freeze first with --freeze-grid-only")
    summary = run_panel(config)
    print(json.dumps({k: summary[k] for k in ("event", "status", "cohort_complete", "grid_complete",
                     "valid", "faults", "cuts", "censored", "unstarted", "grid_hash", "selection_protocol",
                     "targets", "coverage", "coverage_sufficient", "results_complete", "missing_results",
                     "invalid_results") if k in summary}))


if __name__ == "__main__":
    main()
