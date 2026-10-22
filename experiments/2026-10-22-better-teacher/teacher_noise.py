#!/usr/bin/env python3
"""Split the teacher's vote disagreement into search noise and disagreement between sampled futures.

    teacher_noise.py BENCH_DIR --out noise.jsonl [--per-act 100] [--k 8] [--threads 4] [--simulations 1000]

Replays boss-fight decisions recorded by a teacher benchmark (fresh_collect --teacher-plays) and runs K
searches twice: (A) one sampled future, K search seeds; (B) K sampled futures, one search seed. If A's
searches already disagree as much as B's, the split votes are search noise, not hidden information.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import public_hash  # noqa: E402
from sts_rl.common import WORKER_CONFIG  # noqa: E402
from sts_rl.fresh_collection import task  # noqa: E402
from sts_rl.fresh_panel import FreshPanelWorker  # noqa: E402
from sts_rl.teacher_votes import votes_from_details  # noqa: E402

FUTURE_OFFSET = 2048  # disjoint from the label seeds (1-24) and self-consistency repeats (1025-1048)


def positions(directory, per_act):
    rows = []
    for file in sorted(Path(directory).glob("shard_*/worlds/*.json.gz")):
        record = json.load(gzip.open(file))
        for s in record["labels"]:
            if s.get("search_details") and s["group"].startswith("BOSS"):
                rows.append({k: s[k] for k in ("world_seed", "prefix", "public_hash", "group", "expected_provenance")})
    chosen = []
    for act in "123":
        group = sorted((r for r in rows if r["group"] == f"BOSS{act}"),
                       key=lambda r: hashlib.sha256(r["public_hash"].encode()).hexdigest())
        chosen += group[:per_act]
    return chosen


def agreement(votes, k):
    """Probability that two different searches of the K chose the same action."""
    return sum(c * (c - 1) for c in votes.values()) / (k * (k - 1))


def compact(details):
    return [{"chosen": d["chosen"], "winning": d["winning_sequence_found"],
             "edges": [[e["index"], e["visits"], round(e["mean_evaluation"], 3)] for e in d["edges"]]} for d in details]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench")
    p.add_argument("--out", required=True)
    p.add_argument("--per-act", type=int, default=100)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--simulations", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=120.)
    p.add_argument("--any-build", action="store_true",
                   help="Replay positions recorded on another native build (the public state hash is still checked)")
    args = p.parse_args()
    rows = positions(args.bench, args.per_act)
    out, lock = open(args.out, "a"), threading.Lock()
    done = {json.loads(line)["public_hash"] for line in open(args.out)} if Path(args.out).stat().st_size else set()
    todo = [r for r in rows if r["public_hash"] not in done]
    print(f"{len(rows)} positions, {len(todo)} to do", flush=True)

    def run(slot):
        worker = FreshPanelWorker(config={**WORKER_CONFIG, "fast_observations": False}, timeout=args.timeout)
        try:
            for r in todo[slot::args.threads]:
                observation, _ = worker.reset(task(r["world_seed"], None if args.any_build else r["expected_provenance"],
                                                     r["prefix"], r["public_hash"]))
                if public_hash(observation) != r["public_hash"]:
                    raise RuntimeError(f"Replay mismatch at world {r['world_seed']}")
                base = int(r["public_hash"][:10], 16) * 4096
                k = args.k
                same_future = worker.search_details_public(args.simulations, [FUTURE_OFFSET] * k,
                                                           [base + FUTURE_OFFSET + i for i in range(1, k + 1)], base)
                many_futures = worker.search_details_public(args.simulations, [FUTURE_OFFSET + i for i in range(1, k + 1)],
                                                            [base + FUTURE_OFFSET] * k, base)
                result = {"public_hash": r["public_hash"], "group": r["group"], "world_seed": r["world_seed"],
                          "legal": len(observation["legal_actions"])}
                for name, details in (("same_future", same_future), ("many_futures", many_futures)):
                    result[name] = {"abstain": not details, "details": compact(details)}
                    for rule in ("best_line", "mean_value"):
                        votes = votes_from_details(details, rule) if details else {}
                        result[name][rule] = {"votes": votes, "agreement": agreement(votes, k) if votes else None}
                with lock:
                    out.write(json.dumps(result) + "\n")
                    out.flush()
        finally:
            worker.close()

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(run, range(args.threads)))
    summarise(args.out)


def summarise(path):
    rows = [json.loads(line) for line in open(path)]
    for act in "123":
        sub = [r for r in rows if r["group"] == f"BOSS{act}" and not r["same_future"]["abstain"]
               and not r["many_futures"]["abstain"]]
        if not sub:
            continue
        line = [f"Act {act} boss, {len(sub)} positions:"]
        for rule in ("best_line", "mean_value"):
            a = sum(r["same_future"][rule]["agreement"] for r in sub) / len(sub)
            b = sum(r["many_futures"][rule]["agreement"] for r in sub) / len(sub)
            line.append(f"{rule}: two searches agree {100*a:.0f}% in one future, {100*b:.0f}% across futures")
        print(" ".join(line))


if __name__ == "__main__":
    main()
