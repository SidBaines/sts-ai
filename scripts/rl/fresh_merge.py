#!/usr/bin/env python3
"""Validate/merge completed fresh collection or eval process-shards, exactly."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sts_rl.fresh_cohort import merge_cohorts, validate_cohorts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--world-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", help="Omit for validation only; merged output must be fresh")
    a = p.parse_args(argv)
    if a.out:
        report = merge_cohorts(a.inputs, a.out, world_start=a.world_start, n=a.n)
    else:
        cohort = validate_cohorts(a.inputs, world_start=a.world_start, n=a.n)
        report = {"status": "completed", "completed": len(cohort["records"]),
                  "science_identity": cohort["science_identity"]}
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, allow_nan=False))


if __name__ == "__main__":
    main()
