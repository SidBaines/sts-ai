#!/usr/bin/env python3
"""Matched-state comparison of two combat heads along one player's recorded games.

    combat_diff.py SUMMARY.json --old OLD.pt --new NEW.pt --out rows.jsonl [--n 1024] [--processes 8]

Replays each completed game in SUMMARY (a fast_eval summary.json played by OLD) and, at every combat
decision with more than one legal action, records both checkpoints' greedy actions. OLD's choice must
equal the recorded action (checked), so every row is a state OLD actually reached.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def worker(args):
    records, old_path, new_path, out_path = args
    from sts_rl.batched_runner import BatchedPolicy, configure_cpu
    from sts_rl.common import WORKER_CONFIG
    from sts_rl.controlled_policy import policy_factory
    from sts_rl.fresh_collection import task
    from sts_rl.fresh_panel import FreshPanelEpisode
    from sts_rl.fresh_study import policy_observation
    import torch
    configure_cpu()
    old, new = BatchedPolicy(policy_factory(old_path)), BatchedPolicy(policy_factory(new_path))
    worlds = []
    for r in records:
        episode = FreshPanelEpisode({**WORKER_CONFIG, "fast_observations": False})
        obs, _ = episode.reset(task(r["world_seed"]))
        worlds.append({"seed": r["world_seed"], "episode": episode, "obs": obs, "actions": r["action_prefix"], "step": 0})
    mismatches = 0
    with open(out_path, "w") as out:
        while worlds:
            asks = [w for w in worlds if w["obs"]["phase"] == "combat"
                    and len(policy_observation(w["obs"])["legal_actions"]) > 1]
            if asks:
                old_scores, new_scores = [], []
                a_old = old.choose_actions([w["obs"] for w in asks], details=old_scores)
                a_new = new.choose_actions([w["obs"] for w in asks], details=new_scores)
                for w, ao, an, so, sn in zip(asks, a_old, a_new, old_scores, new_scores):
                    if ao != w["actions"][w["step"]]:
                        mismatches += 1
                    state, legal = w["obs"]["state"], policy_observation(w["obs"])["legal_actions"]
                    pos = {int(a["index"]): i for i, a in enumerate(legal)}
                    po, pn = torch.softmax(so, 0), torch.softmax(sn, 0)
                    combat = state.get("combat", {})
                    out.write(json.dumps({
                        "world": w["seed"], "step": w["step"], "act": state.get("act"), "floor": state.get("floor"),
                        "room": str(state.get("room", "")).rsplit(".", 1)[-1], "turn": combat.get("turn"),
                        "hp": state.get("cur_hp"), "max_hp": state.get("max_hp"),
                        "enemies": sorted(e["name"] for e in combat.get("enemies", [])),
                        "legal": [a["description"] for a in legal], "old": pos[ao], "new": pos[an],
                        "recorded": pos.get(w["actions"][w["step"]]),
                        "p_old_of_old": round(float(po[pos[ao]]), 4), "p_new_of_old": round(float(pn[pos[ao]]), 4),
                        "p_new_of_new": round(float(pn[pos[an]]), 4), "p_old_of_new": round(float(po[pos[an]]), 4),
                    }) + "\n")
            for w in list(worlds):
                obs, _, terminated, truncated, _ = w["episode"].step(w["actions"][w["step"]])
                w["obs"], w["step"] = obs, w["step"] + 1
                if terminated or truncated or w["step"] >= len(w["actions"]):
                    w["episode"].close() if hasattr(w["episode"], "close") else None
                    worlds.remove(w)
    return mismatches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--processes", type=int, default=8)
    a = p.parse_args()
    if Path(a.out).exists():
        p.error("Output exists")
    records = sorted((r for r in json.load(open(a.summary))["records"] if r["status"] == "completed"),
                     key=lambda r: r["world_seed"])[:a.n]
    parts = [f"{a.out}.part{i}" for i in range(a.processes)]
    jobs = [(records[i::a.processes], a.old, a.new, parts[i]) for i in range(a.processes)]
    with multiprocessing.get_context("spawn").Pool(a.processes) as pool:
        mismatches = pool.map(worker, jobs)
    with open(a.out, "w") as out:
        for part in parts:
            out.write(open(part).read())
            os.remove(part)
    print(json.dumps({"worlds": len(records), "old_vs_recorded_mismatches": sum(mismatches)}))


if __name__ == "__main__":
    main()
