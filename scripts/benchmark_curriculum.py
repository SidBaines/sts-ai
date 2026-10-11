#!/usr/bin/env python3
"""Bounded persistent-worker scaling panel; reports native faults separately."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from sts_policy.data import atomic_json, collect_world
from sts_policy.workers import CurriculumWorker


def panel(worlds: int, seed_start: int, counts: list[int], fast: bool) -> list[dict]:
    rows = []
    def lane(seeds):
        out = []
        with CurriculumWorker({"fast_observations": fast}, timeout=10) as worker:
            for seed in seeds:
                result = collect_world(worker, seed, epsilon_ooc=0.05, epsilon_combat=0.02)
                out.append({"seed": seed, "records": len(result["records"]),
                            "status": result["status"], "error": result.get("error")})
        return out
    for count in counts:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=count) as pool:
            result = list(pool.map(lane, [list(range(seed_start+i, seed_start+worlds, count)) for i in range(count)]))
        flat = [world for lane_result in result for world in lane_result]
        seconds = time.monotonic() - started
        records = sum(world["records"] for world in flat)
        row = {"workers": count, "fast_observations": fast, "seconds": seconds,
               "worlds": worlds, "records": records, "records_per_second": records / seconds,
               "faults": [world for world in flat if world["status"] != "ok"]}
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--workers", default="1,4,8,12")
    parser.add_argument("--fast-observations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = [int(n) for n in args.workers.split(",")]
    if args.worlds < 1 or min(counts) < 1:
        parser.error("worlds/workers must be positive")
    atomic_json(args.output, panel(args.worlds, args.seed_start, counts, args.fast_observations))

if __name__ == "__main__":
    main()
