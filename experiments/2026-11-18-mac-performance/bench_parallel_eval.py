"""Time the deployed multiprocessing evaluator and retain trajectory signatures for every world."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from sts_rl.batched_runner import run_multiprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--processes", required=True, type=int)
    p.add_argument("--games", default=224, type=int)
    p.add_argument("--slots", default=8, type=int)
    args = p.parse_args()
    start = time.perf_counter()
    rows = run_multiprocess(
        range(91_001_000_000, 91_001_000_000 + args.games),
        args.checkpoint,
        processes=args.processes,
        games_per_process=args.slots,
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    records = {}
    for row in rows:
        record = {
            k: row[k]
            for k in (
                "status",
                "win",
                "action_hash",
                "terminal_hash",
                "bosses",
                "behavior",
            )
        }
        records[row["world_seed"]] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
    args.out.write_text(
        json.dumps(
            dict(
                seconds=elapsed,
                worlds=len(records),
                wins=sum(r["win"] is True for r in rows),
                faults=sum(r["status"] != "completed" for r in rows),
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(args.out, round(elapsed, 2), flush=True)


if __name__ == "__main__":
    main()
