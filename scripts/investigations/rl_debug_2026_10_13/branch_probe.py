#!/usr/bin/env python3
"""Paired counterfactual branches at OOC decisions under greedy continuation.

For sampled non-forced OOC decisions in greedy original-policy episodes, force each
alternative legal action (same world, same prefix) and continue with the greedy
policy to the end. Compares final floor/outcome with the recorded greedy episode.
Measures (a) how noisy a paired action-difference is versus independent games and
(b) which decision types have alternatives that beat the policy's choice.
Diagnostic only; TRAIN worlds.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import re
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rollout_probe import WORKER_CONFIG  # noqa: E402

from sts_policy.curriculum import TaskSpec
from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public, forward_policy, policy_from_payload
from sts_policy.rollout_training import collect_rollouts
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker


def episodes(rows):
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if r["terminated"] or r["truncated"]:
            out.append(cur)
            cur = []
    return out


def label(desc: str) -> str:
    d = re.sub(r"\d+", "#", desc)
    for pre in ("take card", "buy card remove", "buy card", "buy potion", "buy relic", "take relic", "take boss relic",
                "select card index"):
        if d.startswith(pre):
            return pre
    m = re.search(r"room=(\w+)", d)
    if m:
        return "map:" + m.group(1)
    return d.split(" (")[0][:28]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decisions", type=int, default=600)
    p.add_argument("--max-alts", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    data = pickle.load(open(args.probe, "rb"))
    rng = random.Random(0)
    eps = episodes(data["rows"])
    final = {e["world_seed"]: (e["floor"], e["task_outcome"]) for e in data["episodes"]}
    cands = []
    for ep in eps:
        seed = ep[0]["seed"]
        if seed not in final:
            continue
        idx = [i for i, r in enumerate(ep) if r["phase"] != "combat" and r["n_legal"] > 1 and r["legal"]]
        for i in rng.sample(idx, min(2, len(idx))):
            cands.append((seed, ep, i))
    rng.shuffle(cands)
    cands = cands[: args.decisions]
    tasks, meta = [], []
    for seed, ep, i in cands:
        r = ep[i]
        prefix = [x["action_index"] for x in ep[:i]]
        alts = [a for a in range(r["n_legal"]) if a != r["action_index"]]
        rng.shuffle(alts)
        for a in alts[: args.max_alts]:
            tasks.append(TaskSpec(seed=seed, kind="run", prefix=tuple(prefix + [a]), seek_combat=False,
                                  max_act=3, horizon=4096, battle_simulations=200))
            meta.append({"seed": seed, "i": i, "floor_at": r["floor"], "act": r["act"], "kind": r["kind"],
                         "chosen": r["action"], "alt": r["legal"][a], "base_floor": final[seed][0],
                         "base_outcome": final[seed][1]})
    print(f"{len(cands)} decisions, {len(tasks)} branch rollouts", flush=True)
    model = policy_from_payload(load_checkpoint(args.checkpoint)).eval()
    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=20.0) for _ in range(args.workers)]
    try:
        res = collect_rollouts(model, workers, tasks, collate=collate_public, forward=forward_policy,
                               config=PPOConfig(), max_seconds=3600, deterministic=True, deferred_bootstrap=True,
                               reuse_encodings=False)
    finally:
        for w in workers:
            w.close()
    by_id = {}
    from sts_policy.rollout_training import task_identity
    for e in res["episodes"]:
        by_id[e["task_id"]] = e
    rows = []
    for t, m in zip(tasks, meta):
        e = by_id.get(task_identity(t))
        if e is None:
            continue
        rows.append({**m, "alt_floor": e["floor"], "alt_outcome": e["task_outcome"],
                     "diff": e["floor"] - m["base_floor"]})
    json.dump(rows, open(args.output, "w"))
    d = np.array([r["diff"] for r in rows])
    print(json.dumps({"branches": len(rows), "faults": len(res["faults"]), "tie_frac": float((d == 0).mean()),
                      "diff_sd": float(d.std()), "diff_mean_alt_minus_policy": float(d.mean()),
                      "frac_alt_better": float((d > 0).mean()), "frac_alt_worse": float((d < 0).mean())}))
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["kind"], label(r["chosen"]), label(r["alt"]))].append(r["diff"])
    print("alt minus policy-choice, by (screen, policy choice, alternative), n>=8:")
    for k, v in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 8:
            v = np.array(v)
            print(f"  {str(k):75s} n={len(v):4d} mean={v.mean():+6.2f} se={v.std()/np.sqrt(len(v)):.2f} ties={np.mean(v==0):.2f}")


if __name__ == "__main__":
    main()
