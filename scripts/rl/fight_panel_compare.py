#!/usr/bin/env python3
"""Paired greedy fight-panel comparison of two worker settings on bank fight starts.

Replays fixed bank fights (replay is exact: 300/300 outcomes reproduced) under a
baseline and a variant condition and reports per-group win rates and final-HP,
paired per fight. Conditions are FilteringWorker options, e.g. forced_play.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import TaskSpec  # noqa: E402
from sts_policy.losses import PPOConfig  # noqa: E402
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_policy.rollout_training import collect_rollouts, task_identity  # noqa: E402
from sts_policy.workers import CurriculumWorker  # noqa: E402
from sts_rl.common import WORKER_CONFIG, FilteringWorker, close_workers, load_policy  # noqa: E402


def run(model, rows, workers: int, options: dict) -> dict[str, tuple[bool, float]]:
    pool = [FilteringWorker(CurriculumWorker(config=WORKER_CONFIG, timeout=60.0), dead_actions=False, slim=True,
                            **options) for _ in range(workers)]
    tasks = [TaskSpec.from_dict(r["task"]) for r in rows]
    try:
        res = collect_rollouts(model, pool, tasks, collate=collate_public, forward=forward_policy, config=PPOConfig(),
                               max_seconds=7200, deterministic=True, deferred_bootstrap=True, reuse_encodings=False)
    finally:
        close_workers(pool)
    out = {}
    for e in res["episodes"]:
        out[e["task_id"]] = (e.get("task_outcome") == "battle_victory", e.get("hp_cost"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", default="data/rl_runs/banks/fights_main_s1_i8.jsonl")
    p.add_argument("--groups", default="BOSS2:300,ELITE2:300,BOSS1:200,BOSS3:90,ELITE3:81")
    p.add_argument("--variant", default="forced_play", help="FilteringWorker option for the variant ('' for none)")
    p.add_argument("--variant-checkpoint", help="compare a second checkpoint instead of (or as well as) a worker option")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    rows = [json.loads(l) for l in Path(args.bank).read_text().splitlines() if l.strip()]
    random.Random(11).shuffle(rows)
    by = defaultdict(list)
    for r in rows:
        by[f"{r['room']}{r['act']}"].append(r)
    picked = [r for g, n in (x.split(":") for x in args.groups.split(",")) for r in by[g][: int(n)]]
    base = run(model, picked, args.workers, {})
    variant_model = model
    if args.variant_checkpoint:
        variant_model, _ = load_policy(args.variant_checkpoint)
        variant_model.eval()
    var = run(variant_model, picked, args.workers, {args.variant: True} if args.variant else {})
    report = {}
    for g in dict.fromkeys(f"{r['room']}{r['act']}" for r in picked):
        ids = [task_identity(TaskSpec.from_dict(r["task"])) for r in picked if f"{r['room']}{r['act']}" == g]
        ids = [i for i in ids if i in base and i in var]
        a = np.array([base[i][0] for i in ids], float)
        b = np.array([var[i][0] for i in ids], float)
        d = b - a
        report[g] = {"n": len(ids), "base_win": a.mean(), "variant_win": b.mean(), "diff": d.mean(),
                     "se": d.std() / max(1, np.sqrt(len(d))), "changed": int((d != 0).sum())}
    print(json.dumps(report, indent=1, default=float))
    Path(args.output).write_text(json.dumps(report, default=float))


if __name__ == "__main__":
    main()
