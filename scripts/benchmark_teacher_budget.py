#!/usr/bin/env python3
"""CPU-only paired privileged teacher queries on identical public frontiers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import math
from pathlib import Path
import statistics
import time

from sts_env.env import SimulatorError
from sts_policy.curriculum import public_hash
from sts_policy.workers import CurriculumWorker
try:
    from scripts.evaluate_nonllm_baselines import matched_tasks, task_identity
except ModuleNotFoundError:
    from evaluate_nonllm_baselines import matched_tasks, task_identity


def summarize(rows):
    pairs = {}
    for a, b in ((50, 200), (200, 1000), (50, 1000)):
        differing = sum(r["labels"][str(a)] != r["labels"][str(b)] for r in rows)
        pairs[f"{a}_vs_{b}"] = {"different": differing, "frontiers": len(rows),
                                 "disagreement_rate": differing / len(rows) if rows else None}
    latency = {}
    for simulations in (50, 200, 1000):
        values = sorted(r["latency_seconds"][str(simulations)] for r in rows)
        latency[str(simulations)] = {"mean_ms": 1000 * statistics.mean(values) if values else None,
                                    "median_ms": 1000 * statistics.median(values) if values else None,
                                    "p95_ms": 1000 * values[min(len(values) - 1, int(math.ceil(.95 * len(values))) - 1)] if values else None}
    return {"paired_disagreement": pairs, "query_rpc_latency": latency,
            "worlds": len({r["world_seed"] for r in rows})}


def scenario_identity(task):
    """Match the same replay/start/task bounds while varying teacher search budget."""
    fields = task.to_dict()
    fields.pop("battle_simulations")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "final"), default="dev")
    parser.add_argument("--frontiers", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--search-seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if min(args.frontiers, args.workers, args.seconds, args.timeout) <= 0:
        parser.error("positive counts and time bounds required")
    tasks = matched_tasks(bank=args.tasks, kind="battle", split=args.split,
                          episodes=args.frontiers + 16, seed=args.seed, horizon=256, simulations=200)
    started = time.monotonic()
    deadline = started + args.seconds
    chunks = [list(enumerate(tasks))[i::args.workers] for i in range(args.workers)]
    rows, faults, skipped = [], [], []
    def lane(chunk):
        result, errors, ended = [], [], []
        with CurriculumWorker({"fast_observations": True, "fast_replay": True}, timeout=args.timeout) as worker:
            for position, task in chunk:
                if time.monotonic() >= deadline:
                    break
                try:
                    observation, _ = worker.reset(task)
                    offset = 2 * (position % 4)
                    terminated = truncated = False
                    for _ in range(offset):
                        action = worker.teacher_action(200, search_seed=args.search_seed)
                        observation, _, terminated, truncated, _ = worker.step(action)
                        if terminated or truncated:
                            break
                    if terminated or truncated:
                        ended.append({"task_id": task_identity(task), "reason": "fight_ended_before_requested_frontier", "offset": offset})
                        continue
                    before = public_hash(observation)
                    labels, latencies = {}, {}
                    # Rotate query order to avoid a fixed cold-cache advantage.
                    budgets = (50, 200, 1000)
                    shift = position % 3
                    for simulations in budgets[shift:] + budgets[:shift]:
                        tick = time.perf_counter()
                        labels[str(simulations)] = worker.teacher_action(simulations, search_seed=args.search_seed)
                        latencies[str(simulations)] = time.perf_counter() - tick
                    if public_hash(worker.observation) != before:
                        raise RuntimeError("teacher query changed parent public observation")
                    result.append({"position": position, "task_id": task_identity(task),
                                   "world_seed": task.seed, "split": args.split, "offset": offset,
                                   "public_hash": before, "labels": labels, "latency_seconds": latencies,
                                   "legal_candidates": len(observation["legal_actions"]),
                                   "label_descriptions": {k: next(a["description"] for a in observation["legal_actions"] if a["index"] == v) for k, v in labels.items()}})
                except SimulatorError as error:
                    errors.append({"world_seed": task.seed, "task_id": task_identity(task),
                                   "error": type(error).__name__, "message": str(error)})
        return result, errors, ended
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(lane, chunk) for chunk in chunks]):
            collected, errors, ended = future.result()
            rows.extend(collected); faults.extend(errors); skipped.extend(ended)
    queried_frontiers = len(rows)
    rows = sorted(rows, key=lambda row: row["position"])[:args.frontiers]
    report = {"search_seed": args.search_seed, "simulation_budgets": [50, 200, 1000],
              "frontiers": rows, "faults": faults, "skipped": skipped,
              "queried_frontiers_including_reserve": queried_frontiers,
              "elapsed_seconds": time.monotonic() - started, "summary": summarize(rows),
              "privilege_note": "Search sees hidden simulator state; these labels/identities/latencies are diagnostics, never model input. Native seeds 0 and 1 alias; only seed 1 used."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"frontiers": len(rows), "faults": len(faults), "skipped": len(skipped),
                      "elapsed_seconds": report["elapsed_seconds"], "summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
