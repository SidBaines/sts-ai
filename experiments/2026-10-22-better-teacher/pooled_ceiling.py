#!/usr/bin/env python3
"""How often two independent pooled teachers choose the same move: the ceiling for a student's agreement.

    pooled_ceiling.py BENCH_DIR --out ceiling.jsonl [--per-act 60] [--samples 24] [--threads 3]

Replays boss decisions recorded by a teacher benchmark and runs the pooled-searches teacher twice, with
disjoint sampled futures and search seeds (24 each), at each decision.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts_policy.curriculum import public_hash  # noqa: E402
from sts_rl.common import WORKER_CONFIG  # noqa: E402
from sts_rl.fresh_collection import task  # noqa: E402
from sts_rl.fresh_panel import FreshPanelWorker  # noqa: E402
from sts_rl.teacher_votes import votes_from_details  # noqa: E402
from teacher_noise import positions  # noqa: E402  (same experiment folder)

OFFSETS = (4096, 8192)  # disjoint from label seeds (1-24), repeats (1025-1048) and noise runs (2049-2056)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench")
    p.add_argument("--out", required=True)
    p.add_argument("--per-act", type=int, default=60)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--simulations", type=int, default=1000)
    args = p.parse_args()
    rows = positions(args.bench, args.per_act)
    out, lock = open(args.out, "w"), threading.Lock()

    def run(slot):
        worker = FreshPanelWorker(config={**WORKER_CONFIG, "fast_observations": False}, timeout=120.)
        try:
            for r in rows[slot::args.threads]:
                observation, _ = worker.reset(task(r["world_seed"], r["expected_provenance"], r["prefix"], r["public_hash"]))
                if public_hash(observation) != r["public_hash"]:
                    raise RuntimeError(f"Replay mismatch at world {r['world_seed']}")
                base = int(r["public_hash"][:10], 16) * 4096
                choices = []
                for offset in OFFSETS:
                    futures = [offset + i for i in range(1, args.samples + 1)]
                    details = worker.search_details_public(args.simulations, futures, [base + f for f in futures], base)
                    votes = votes_from_details(details, "pooled_mean") if details else {}
                    choices.append(max(votes, key=votes.get) if votes else None)
                with lock:
                    out.write(json.dumps({"group": r["group"], "public_hash": r["public_hash"],
                                          "legal": len(observation["legal_actions"]), "choices": choices}) + "\n")
                    out.flush()
        finally:
            worker.close()

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(run, range(args.threads)))
    out.close()
    results = [json.loads(line) for line in open(args.out)]
    for act in "123":
        sub = [r for r in results if r["group"] == f"BOSS{act}" and None not in r["choices"]]
        if sub:
            agree = sum(r["choices"][0] == r["choices"][1] for r in sub) / len(sub)
            print(f"Act {act} boss: two independent pooled teachers agree on {100*agree:.0f}% of {len(sub)} decisions")


if __name__ == "__main__":
    main()
