#!/usr/bin/env python3
"""Instrumented TRAIN-only rollouts: per-transition phase/value/reward/GAE/MC return.

Diagnostic only. Uses fresh TRAIN-split worlds from an unused seed namespace.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import torch

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public, forward_policy, policy_from_payload
from sts_policy.rollout_training import collect_rollouts
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker

WORKER_CONFIG = {"gamma": 1.0, "reward_mode": "progress", "terminal_hp_weight": 0.0,
                 "progress_floor_weight": 0.05, "goal_act_clear": True,
                 "fast_observations": True, "fast_replay": True}


def decision_kind(obs: dict) -> str:
    if obs.get("phase") == "combat":
        return "combat"
    state = obs.get("state", {})
    screen = str(state.get("screen_state", "")).rsplit(".", 1)[-1]
    descs = " | ".join(a.get("description", "") for a in obs.get("legal_actions", []))
    if "map node" in descs:
        return "map"
    if "take card" in descs or "skip card" in descs:
        return "card_reward"
    return screen.lower() or "ooc_other"


def tasks_for(start: int, count: int, max_act: int = 3) -> list[TaskSpec]:
    tasks, seed = [], start
    while len(tasks) < count:
        if world_split(seed) == "train":
            tasks.append(TaskSpec(seed=seed, kind="run", seek_combat=False, max_act=max_act,
                                  horizon=4096, battle_simulations=200))
        seed += 1
    return tasks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--world-start", type=int, default=5_000_000_000)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=900)
    p.add_argument("--keep-obs", action="store_true")
    p.add_argument("--greedy-phase", choices=("none", "combat", "ooc"), default="none")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = policy_from_payload(load_checkpoint(args.checkpoint)).eval()
    tasks = tasks_for(args.world_start, args.episodes)
    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=10.0) for _ in range(args.workers)]
    started = time.time()
    def forward(m, batch):
        logits, values, mask = forward_policy(m, batch)
        combat = batch[0][:, 7] > .5
        scale = torch.full_like(values, 1.0 / args.temperature)
        if args.greedy_phase == "combat":
            scale = torch.where(combat, torch.full_like(values, 1e4), scale)
        elif args.greedy_phase == "ooc":
            scale = torch.where(~combat, torch.full_like(values, 1e4), scale)
        return (logits * scale[:, None]).masked_fill(~mask, -1e9), values, mask

    try:
        result = collect_rollouts(model, workers, tasks, collate=collate_public, forward=forward,
                                  config=PPOConfig(), device="cpu", max_seconds=args.max_seconds,
                                  deterministic=args.deterministic, deferred_bootstrap=True,
                                  reuse_encodings=False)
    finally:
        for w in workers:
            w.close()
    rows = []
    for t in result["transitions"]:
        obs = t.observation
        st = obs.get("state", {})
        legal = obs["legal_actions"]
        rows.append({
            "seed": t.world_seed, "phase": obs.get("phase"), "kind": decision_kind(obs),
            "act": st.get("act"), "floor": st.get("floor"), "hp": st.get("cur_hp"),
            "n_legal": len(legal), "action": legal[t.action_position].get("description", ""),
            "action_index": t.action_index, "legal": [a.get("description", "") for a in legal] if obs.get("phase") != "combat" else None,
            "max_hp": st.get("max_hp"), "room": st.get("room"),
            "logp": t.behavior_log_prob, "value": t.behavior_value, "reward": t.reward,
            "terminated": t.terminated, "truncated": t.truncated, "next_value": t.next_value,
            "adv": t.advantage, "ret": t.returns,
            **({"obs": obs} if args.keep_obs else {}),
        })
    out = {"checkpoint": args.checkpoint, "rows": rows, "episodes": result["episodes"],
           "faults": result["faults"], "elapsed": time.time() - started,
           "deterministic": args.deterministic}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    eps = result["episodes"]
    floors = [e["floor"] for e in eps]
    print(json.dumps({"episodes": len(eps), "faults": len(result["faults"]),
                      "mean_floor": sum(floors) / max(1, len(floors)),
                      "wins": sum(e["task_outcome"] in ("act_clear", "victory") for e in eps),
                      "transitions": len(rows), "elapsed": time.time() - started}))


if __name__ == "__main__":
    main()
