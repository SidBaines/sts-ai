#!/usr/bin/env python3
"""Replay the learner's own boss fights under different combat controllers.

From probe episodes (full action traces, fresh TRAIN starts with empty prefix),
reconstruct the state at the first combat decision on a boss floor and play the
fight with: learned combat (greedy), learned combat (sampled, K times), and the
privileged native search at several simulation budgets. Answers: are boss deaths
mainly combat-skill or deck/HP (strategic) failures? Diagnostic only, TRAIN worlds.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rollout_probe import WORKER_CONFIG  # noqa: E402

from sts_policy.curriculum import TaskSpec
from sts_policy.losses import masked_distribution
from sts_policy.model_agent import collate_public, forward_policy, policy_from_payload
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


def fatal_starts(rows, floors):
    """Start of the fight in which the run died, for deaths on the given floors."""
    starts = []
    for ep in episodes(rows):
        last = ep[-1]
        if not (last["terminated"] and last["reward"] < -0.5 and last["phase"] == "combat" and last["floor"] in floors):
            continue
        i = len(ep) - 1
        while i > 0 and ep[i - 1]["phase"] == "combat" and ep[i - 1]["floor"] == last["floor"]:
            i -= 1
        starts.append({"seed": last["seed"], "prefix": [x["action_index"] for x in ep[:i]], "floor": last["floor"],
                       "hp": ep[i]["hp"], "original_died": True})
    return starts


def boss_starts(rows, floors):
    starts = []
    for ep in episodes(rows):
        for i, r in enumerate(ep):
            if r["phase"] == "combat" and r["floor"] in floors:
                died_here = ep[-1]["terminated"] and ep[-1]["floor"] == r["floor"] and ep[-1]["reward"] < -0.5
                starts.append({"seed": r["seed"], "prefix": [x["action_index"] for x in ep[:i]],
                               "floor": r["floor"], "hp": r["hp"], "original_died": died_here})
                break
    return starts


def play(worker, model, start, controller, sims, rng_seed):
    task = TaskSpec(seed=start["seed"], kind="battle", prefix=tuple(start["prefix"]), seek_combat=False,
                    horizon=4096, max_act=3, battle_simulations=200)
    obs, info = worker.reset(task)
    gen = torch.Generator().manual_seed(rng_seed)
    for _ in range(4096):
        if controller == "search":
            action = worker.teacher_action(sims, rng_seed)
        else:
            with torch.no_grad():
                logits, _, mask = forward_policy(model, collate_public([obs], "cpu"))
            if controller == "greedy":
                pos = int(logits.argmax(-1))
            else:
                probs = masked_distribution(logits, mask).probs[0]
                pos = int(torch.multinomial(probs, 1, generator=gen))
            action = int(obs["legal_actions"][pos]["index"])
        obs, reward, term, trunc, info = worker.step(action)
        if term or trunc:
            return {"win": info.get("task_outcome") == "battle_victory",
                    "end_hp": (obs.get("state", {}).get("combat", {}) or {}).get("player_cur_hp", obs.get("state", {}).get("cur_hp"))}
    return {"win": False, "end_hp": None, "cut": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--floors", default="16")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--sims", default="200,2000")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--fatal", action="store_true", help="replay the fight each run died in")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(1)
    rows = pickle.load(open(args.probe, "rb"))["rows"]
    floors = {int(x) for x in args.floors.split(",")} if "-" not in args.floors else set()
    if args.fatal:
        floors = {int(a) for a in args.floors.split("-")} if "-" in args.floors else floors
        if "-" in args.floors:
            lo, hi = (int(a) for a in args.floors.split("-")); floors = set(range(lo, hi + 1))
        starts = fatal_starts(rows, floors)
    else:
        starts = boss_starts(rows, floors)
    random.Random(0).shuffle(starts)
    # balance: half original deaths, half survivals where possible
    died = [s for s in starts if s["original_died"]][: args.n // 2]
    lived = [s for s in starts if not s["original_died"]][: args.n - len(died)]
    starts = died + lived
    model = policy_from_payload(load_checkpoint(args.checkpoint)).eval()
    controllers = [("greedy", None)] + [("sample", k) for k in range(args.samples)] + \
                  [("search", int(s)) for s in args.sims.split(",")]
    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=120.0) for _ in range(args.workers)]

    def run(i):
        w = workers[i % len(workers)]
        s = starts[i]
        res = {"seed": s["seed"], "floor": s["floor"], "hp": s["hp"], "original_died": s["original_died"]}
        for name, arg in controllers:
            key = name if name == "greedy" else f"{name}{arg}"
            try:
                res[key] = play(w, model, s, name, arg if name == "search" else None, 1000 + (arg or 0))
            except Exception as e:  # simulator faults are reported, not counted as losses
                res[key] = {"fault": type(e).__name__}
        return res

    out = []
    try:
        with ThreadPoolExecutor(len(workers)) as ex:
            # each worker is used by exactly one thread at a time
            for b in range(0, len(starts), len(workers)):
                out.extend(ex.map(run, range(b, min(b + len(workers), len(starts)))))
                print(f"{len(out)}/{len(starts)}", flush=True)
    finally:
        for w in workers:
            w.close()
    json.dump(out, open(args.output, "w"), indent=1)
    keys = [k for k in out[0] if isinstance(out[0][k], dict)]
    for group, sel in (("orig-died", lambda r: r["original_died"]), ("orig-survived", lambda r: not r["original_died"]), ("all", lambda r: True)):
        g = [r for r in out if sel(r)]
        summary = {}
        for k in keys:
            vals = [r[k] for r in g if "fault" not in r[k]]
            summary[k] = f"{sum(v['win'] for v in vals)}/{len(vals)}"
        print(group, len(g), "start HP %.1f" % (sum(r['hp'] or 0 for r in g) / max(1, len(g))), json.dumps(summary))


if __name__ == "__main__":
    main()
