#!/usr/bin/env python3
"""Hand-rule OOC overlays on a learned policy: estimates strategic headroom.

Each rule only overrides the learned choice on its own screen when it fires;
everything else (all combat, other OOC screens) is the unchanged learned policy.
Same TRAIN worlds as rollout_probe (paired comparison). Diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rollout_probe import WORKER_CONFIG, tasks_for  # noqa: E402

from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public, forward_policy, policy_from_payload
from sts_policy.rollout_training import collect_rollouts
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker

BOSS_FLOORS = {16, 33, 50}


def rule_choice(obs: dict, rules: set[str], smith_frac: float) -> int | None:
    if obs.get("phase") == "combat":
        return None
    legal = [a.get("description", "") for a in obs.get("legal_actions", [])]
    if len(legal) < 2:
        return None
    st = obs.get("state", {})
    hp, mx = float(st.get("cur_hp") or 0), float(st.get("max_hp") or 80)
    text = str(obs.get("text", ""))
    find = lambda pred: next((i for i, a in enumerate(legal) if pred(a)), None)
    if "nokey" in rules:
        if any(a.startswith("recall") for a in legal) or any(a == "take key" for a in legal):
            # fall through to other rules first; if none fire, pick best non-key option later
            pass
    if "smith" in rules and any(a.startswith("smith") for a in legal):
        next_is_boss = int(st.get("floor") or 0) + 1 in BOSS_FLOORS
        threshold = max(smith_frac, 0.75 if next_is_boss else smith_frac)
        if hp >= threshold * mx:
            return find(lambda a: a.startswith("smith"))
        r = find(lambda a: a.startswith("rest"))
        if r is not None:
            return r
    if "remove" in rules:
        r = find(lambda a: a.startswith("buy card remove"))
        if r is not None:
            return r
        if re.search(r"Choose a card to (SACRIFICE|remove|REMOVE|purge|PURGE)", text) or "remove" in text.split("Choose a card", 1)[-1][:60].lower():
            for name in ("Strike", "Defend"):
                r = find(lambda a, n=name: re.match(rf"select card index \d+: {n} \[", a) is not None)
                if r is not None:
                    return r
    if "elite" in rules and int(st.get("act") or 0) <= 2 and hp >= 0.7 * mx:
        r = find(lambda a: "room=ELITE" in a)
        if r is not None:
            return r
    if "nokey" in rules:
        bad = [i for i, a in enumerate(legal) if a.startswith("recall") or a == "take key"]
        if bad:
            return "AVOID", bad  # handled by caller: mask these out
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rules", default="")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--world-start", type=int, default=5_000_000_000)
    p.add_argument("--smith-frac", type=float, default=0.55)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    rules = {r for r in args.rules.split(",") if r}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = policy_from_payload(load_checkpoint(args.checkpoint)).eval()
    stash: dict[str, list] = {"obs": []}
    fired: dict[str, int] = {}

    def collate(observations, device="cpu"):
        stash["obs"] = list(observations)
        return collate_public(observations, device)

    def forward(m, batch):
        logits, values, mask = forward_policy(m, batch)
        logits = logits.clone()
        for row, obs in enumerate(stash["obs"]):
            choice = rule_choice(obs, rules, args.smith_frac)
            if choice is None:
                continue
            if isinstance(choice, tuple):
                for i in choice[1]:
                    if mask[row].sum() > 1:
                        logits[row, i] = -1e9
                fired["nokey"] = fired.get("nokey", 0) + 1
                continue
            fired[obs["legal_actions"][choice]["description"].split(" ")[0]] = fired.get(
                obs["legal_actions"][choice]["description"].split(" ")[0], 0) + 1
            logits[row] = -1e9
            logits[row, choice] = 0.0
        return logits, values, mask

    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=10.0) for _ in range(8)]
    started = time.time()
    try:
        res = collect_rollouts(model, workers, tasks_for(args.world_start, args.episodes), collate=collate,
                               forward=forward, config=PPOConfig(), max_seconds=1200,
                               deterministic=not args.stochastic, deferred_bootstrap=False, reuse_encodings=False)
    finally:
        for w in workers:
            w.close()
    eps = res["episodes"]
    floors = np.array([e["floor"] for e in eps])
    summary = {"rules": sorted(rules), "episodes": len(eps), "faults": len(res["faults"]),
               "mean_floor": float(floors.mean()), "act1_clear": float((floors > 17).mean()),
               "act2_clear": float((floors > 34).mean()),
               "wins": sum(e["task_outcome"] in ("act_clear", "victory") for e in eps), "fired": fired,
               "elapsed": time.time() - started}
    print(json.dumps(summary))
    if args.output:
        pickle.dump({"summary": summary, "episodes": eps}, open(args.output, "wb"))


if __name__ == "__main__":
    main()
