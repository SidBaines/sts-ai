#!/usr/bin/env python3
"""Compare the per-decision RPC evaluator and batched runners on identical worlds.

Use the queue runner for a full benchmark. Partial results are saved after every
configuration. OS failures and exhausted budgets are reported as unavailable;
programming errors and incorrect K=1 trajectories fail the benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for _name in ("PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fast_eval import prepare_output
from sts_rl.batched_runner import (compare_records, configure_cpu, evaluation_summary,
                                   run_in_process, run_multiprocess)
from sts_rl.controlled_policy import policy_factory
from sts_rl.fresh_collection import play_world
from sts_rl.fresh_study import atomic_json, validate_world_range
from sts_rl.recording import file_identity


class _Control:
    def check(self):
        pass


class _Progress:
    def add(self, **kwargs):
        pass


def existing_records(seeds, policy, timeout):
    from sts_rl.common import WORKER_CONFIG
    from sts_rl.fresh_panel import FreshPanelWorker
    worker = FreshPanelWorker(config={**WORKER_CONFIG, "fast_observations": False}, timeout=timeout)
    try:
        records = []
        for seed in seeds:
            records.append(play_world(worker, policy, seed, {"mode": "eval", "bank": False},
                                      _Control(), _Progress(), {}, lambda _: None))
            if len(records) % 32 == 0:
                print(json.dumps({"progress": "existing_p1", "worlds_recorded": len(records)}), flush=True)
        return records
    finally:
        worker.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--combat")
    p.add_argument("--ooc")
    p.add_argument("--world-start", type=int, default=91_001_000_000)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--processes", type=int, default=7, help="Maximum seven processes")
    p.add_argument("--timeout", type=float, default=120.)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", help="Existing evaluator summary.json for an independent K=1 gate")
    p.add_argument("--skip-existing", action="store_true", help="Skip an already measured RPC baseline")
    p.add_argument("--single-records", help="Reuse a saved K=1 summary for this exact policy and world cohort")
    p.add_argument("--max-seconds", type=float, default=540., help="Budget between configurations; queue can stop a job")
    a = p.parse_args(argv)
    if not 1 <= a.processes <= 7 or not 1 <= a.n <= 256:
        p.error("Require processes <=7 and n <=256")
    validate_world_range(a.world_start, a.n)
    configure_cpu()
    out = prepare_output(vars(a))
    seeds = list(range(a.world_start, a.world_start + a.n))
    policy = policy_factory(a.checkpoint, a.combat, a.ooc)
    report = {"config": vars(a), "results": {}, "comparisons": {}, "minimum_aggregate_games_per_second": 4.5}
    started = time.perf_counter()

    def measure(name, function):
        if time.perf_counter() - started >= a.max_seconds:
            report["results"][name] = {"unavailable": "Benchmark time budget exhausted"}
            atomic_json(out / "benchmark.json", report)
            return None
        before = time.perf_counter()
        try:
            records = function()
        except (PermissionError, OSError) as exc:
            report["results"][name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
            atomic_json(out / "benchmark.json", report)
            print(json.dumps({name: report["results"][name]}), flush=True)
            return None
        seconds = time.perf_counter() - before
        summary = evaluation_summary(records)
        decisions = sum(len(r["action_prefix"]) for r in records)
        row = {"seconds": seconds, "games_per_second": summary["completed"] / seconds,
               "decisions_per_second": decisions / seconds, "decisions": decisions,
               **{k: summary[k] for k in ("completed", "faults", "cuts", "wins")}}
        report["results"][name] = row
        atomic_json(out / (name + ".json"), summary)
        atomic_json(out / "benchmark.json", report)
        print(json.dumps({name: row}), flush=True)
        return records

    baseline = None if a.skip_existing else measure("existing_p1", lambda: existing_records(seeds, policy, a.timeout))
    if a.single_records:
        single = json.loads(Path(a.single_records).read_text())["records"]
        if sorted(r["world_seed"] for r in single) != seeds:
            raise ValueError("Saved K=1 records do not match the benchmark world cohort")
        saved_run = json.loads(Path(a.single_records).with_name("run.json").read_text())
        current_run = json.loads((out / "run.json").read_text())
        if saved_run["checkpoints"] != current_run["checkpoints"]:
            raise ValueError("Saved K=1 records belong to different checkpoints")
        report["reused_single_records"] = file_identity(a.single_records)
        previous_path = Path(a.single_records).with_name("benchmark.json")
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text())
            timing = previous["results"].get("batched_p1_k1", {})
            if "games_per_second" in timing:
                report["results"]["batched_p1_k1"] = {**timing, "reused_timing": True}
                report["reused_timing_source"] = file_identity(previous_path)
    else:
        single = measure("batched_p1_k1", lambda: run_in_process(seeds, policy, 1))
    if single is None:
        raise RuntimeError("K=1 gate did not run")
    references = baseline
    if a.reference:
        all_rows = json.loads(Path(a.reference).read_text())["records"]
        references = [r for r in all_rows if r["world_seed"] in set(seeds)]
    if references is not None:
        comparison = compare_records(references, single)
        report["comparisons"]["existing_vs_k1"] = comparison
        atomic_json(out / "benchmark.json", report)
        if comparison["differences"]:
            raise RuntimeError("K=1 failed existing evaluator action-hash reproduction")
    for k in (8, 32):
        rows = measure(f"batched_p1_k{k}", lambda: run_in_process(seeds, policy, k, reference_records=single))
        if rows is not None:
            comparison = compare_records(single, rows)
            report["comparisons"][f"k1_vs_k{k}"] = comparison
            atomic_json(out / "benchmark.json", report)
            print(json.dumps({f"k1_vs_k{k}": comparison}), flush=True)
    choices = [(row["games_per_second"], k) for k in (1, 8, 32)
               if "games_per_second" in (row := report["results"].get(f"batched_p1_k{k}", {}))]
    if not choices:
        atomic_json(out / "benchmark.json", report)
        raise RuntimeError("No batched timing completed within the benchmark budget")
    _, best = max(choices)
    report["best_k"] = best
    rows = measure(f"batched_p{a.processes}_k{best}", lambda: run_multiprocess(seeds, a.checkpoint, a.combat, a.ooc,
            processes=a.processes, games_per_process=best, timeout=a.timeout, reference_records=single))
    if rows is not None:
        report["comparisons"][f"k1_vs_p{a.processes}_k{best}"] = compare_records(single, rows)
    aggregate = report["results"][f"batched_p{a.processes}_k{best}"]
    if a.processes == 7 and "games_per_second" in aggregate:
        report["speedup_vs_existing_7_lanes_approx_3_games_s"] = aggregate["games_per_second"] / 3.
        report["stop_for_insufficient_speedup"] = aggregate["games_per_second"] < 4.5
    atomic_json(out / "benchmark.json", report)
    print(json.dumps({"best_k": best, "report": str(out / "benchmark.json")}), flush=True)


if __name__ == "__main__":
    main()
