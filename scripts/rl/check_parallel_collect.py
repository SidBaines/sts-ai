#!/usr/bin/env python3
"""Check CollectorPool against single-process collection.

1. Greedy evaluation: per-world records must be identical.
2. OOC collection: transitions must survive the round trip into update_ppo
   (encoding identity re-bound), and report throughput for both paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.losses import PPOConfig  # noqa: E402
from sts_policy.model_agent import collate_public  # noqa: E402
from sts_policy.rollout_training import update_ppo  # noqa: E402
from sts_rl.common import close_workers, load_policy, make_workers, phase_forward, train_tasks  # noqa: E402
from sts_rl.evaluate import evaluate  # noqa: E402
from sts_rl.ooc_trainer import collect_ooc_batch  # noqa: E402
from sts_rl.parallel_collect import CollectorPool  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-n", type=int, default=48)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--collectors", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    ppo = PPOConfig(gamma=1.0, gae_lambda=.95, epochs=1, minibatch_size=4096, kl_guard_before_update=True,
                    target_kl=.05)
    report = {}
    pool = CollectorPool(args.collectors, workers=args.workers, checkpoint=args.checkpoint)
    workers = make_workers(args.workers, dead_action_filter=True)
    try:
        t = time.time(); local = evaluate(model, workers, n=args.eval_n); report["eval_local_s"] = time.time() - t
        t = time.time(); par = evaluate(model, pool, n=args.eval_n); report["eval_pool_s"] = time.time() - t
        report["eval_records_identical"] = local["records"] == par["records"]
        report["eval_n"] = len(local["records"])
        tasks = train_tasks(7_950_000_000, args.games)
        t = time.time()
        a = collect_ooc_batch(model, workers, tasks, forward=phase_forward(combat="greedy"), ppo=ppo,
                              credit="macro", gae_lambda=.95, chunk=128)
        report["collect_local_s"] = time.time() - t
        t = time.time()
        b = pool.collect_ooc(model, tasks, combat="greedy", temperature=1.0, ppo=ppo, credit="macro",
                             gae_lambda=.95, chunk=128, seed=1)
        report["collect_pool_s"] = time.time() - t
        report["kept_local"] = sum(len(e) for e in a["kept"])
        report["kept_pool"] = sum(len(e) for e in b["kept"])
        report["games_local"], report["games_pool"] = len(a["game_episodes"]), len(b["game_episodes"])
        flat = [t for ep in b["kept"] for t in ep]
        out = update_ppo(model, flat, collate=collate_public, forward=phase_forward(combat="greedy"), config=ppo,
                         micro_batch=1024)
        report["pool_update_steps"] = out["steps"]
        report["pool_reused_encodings"] = out["reused_encoding_count"]
    finally:
        close_workers(workers)
        pool.close()
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
