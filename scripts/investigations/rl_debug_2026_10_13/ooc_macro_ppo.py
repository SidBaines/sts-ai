#!/usr/bin/env python3
"""Bounded diagnostic: large-batch OOC-only PPO with frozen greedy combat and macro credit.

Tests whether a low-noise, cross-critic-free OOC advantage lets PPO *improve* the
original composite, which the v2 campaign never achieved. TRAIN-split worlds only,
from namespaces disjoint from the campaigns and from the evaluation set.
Writes only to --out. Not a campaign condition.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rollout_probe import WORKER_CONFIG, tasks_for  # noqa: E402

from sts_policy.losses import PPOConfig, generalized_advantages
from sts_policy.model_agent import collate_public, forward_policy, policy_from_payload
from sts_policy.rollout_training import collect_rollouts, update_ppo
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker


def greedy_combat_forward(model, batch):
    logits, values, mask = forward_policy(model, batch)
    combat = batch[0][:, 7] > .5
    scale = torch.where(combat, torch.full_like(values, 1e4), torch.ones_like(values))
    return (logits * scale[:, None]).masked_fill(~mask, -1e9), values, mask


def macro_ooc_transitions(transitions, gae_lambda):
    """Episode-contiguous transitions -> OOC-only transitions with macro GAE.

    Rewards between consecutive OOC decisions (incl. all combat) are summed onto the
    earlier OOC decision; bootstrap is the OOC critic's value at the next OOC decision.
    """
    episodes, cur = [], []
    for t in transitions:
        cur.append(t)
        if t.terminated or t.truncated:
            episodes.append(cur)
            cur = []
    out = []
    for ep in episodes:
        idx = [i for i, t in enumerate(ep) if t.observation.get("phase") != "combat"]
        if not idx:
            continue
        rewards, values, term, trunc, nxt = [], [], [], [], []
        for k, i in enumerate(idx):
            j = idx[k + 1] if k + 1 < len(idx) else None
            seg = ep[i:j] if j is not None else ep[i:]
            rewards.append(sum(t.reward for t in seg))
            values.append(ep[i].behavior_value)
            if j is not None:
                term.append(False); trunc.append(False); nxt.append(ep[j].behavior_value)
            else:
                last = ep[-1]
                term.append(bool(last.terminated)); trunc.append(not last.terminated)
                nxt.append(0.0 if last.terminated else float(last.next_value))
        adv, ret = generalized_advantages(rewards, values, term, trunc, nxt, gamma=1.0, gae_lambda=gae_lambda)
        for k, i in enumerate(idx):
            t = ep[i]
            t.advantage, t.returns = adv[k], ret[k]
            if len(t.observation["legal_actions"]) > 1:
                out.append(t)
    return out


def evaluate(model, workers, world_start, n):
    res = collect_rollouts(model, workers, tasks_for(world_start, n), collate=collate_public,
                           forward=forward_policy, config=PPOConfig(), max_seconds=1800,
                           deterministic=True, deferred_bootstrap=True)
    floors = np.array([e["floor"] for e in res["episodes"]])
    return {"n": len(floors), "faults": len(res["faults"]), "mean_floor": float(floors.mean()),
            "se": float(floors.std() / np.sqrt(len(floors))),
            "act1_clear": float((floors > 17).mean()), "act2_clear": float((floors > 34).mean()),
            "wins": int(sum(e["task_outcome"] in ("act_clear", "victory") for e in res["episodes"]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--episodes", type=int, default=768)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-n", type=int, default=1024)
    p.add_argument("--eval-world-start", type=int, default=5_000_000_000)
    p.add_argument("--train-world-start", type=int, default=7_000_000_000)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    def emit(rec):
        rec["t"] = time.time()
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = policy_from_payload(load_checkpoint(args.checkpoint))
    for name, prm in model.named_parameters():
        prm.requires_grad_(name.startswith("ooc."))
    combat_before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("combat.")}
    optimizer = torch.optim.AdamW([p_ for n_, p_ in model.named_parameters() if n_.startswith("ooc.")], lr=args.lr)
    cfg = PPOConfig(gamma=1.0, gae_lambda=args.gae_lambda, clip_ratio=args.clip, entropy_weight=args.entropy,
                    epochs=args.epochs, minibatch_size=args.minibatch, learning_rate=args.lr,
                    kl_guard_before_update=True, target_kl=0.05)
    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=10.0) for _ in range(args.workers)]
    emit({"event": "start", "args": vars(args)})
    try:
        emit({"event": "eval", "iteration": 0, **evaluate(model.eval(), workers, args.eval_world_start, args.eval_n)})
        for it in range(1, args.iterations + 1):
            t0 = time.time()
            model.eval()
            res = collect_rollouts(model, workers, tasks_for(args.train_world_start + it * 10_000_000, args.episodes),
                                   collate=collate_public, forward=greedy_combat_forward, config=cfg,
                                   max_seconds=1800, deterministic=False, deferred_bootstrap=True)
            floors = np.array([e["floor"] for e in res["episodes"]])
            ooc = macro_ooc_transitions(res["transitions"], args.gae_lambda)
            adv = np.array([t.advantage for t in ooc])
            upd = update_ppo(model, ooc, collate=collate_public, forward=forward_policy, config=cfg,
                             device="cpu", seed=it, optimizer=optimizer)
            optimizer = upd.pop("optimizer")
            emit({"event": "train", "iteration": it, "collect_floor": float(floors.mean()),
                  "ooc_choice_transitions": len(ooc), "adv_raw_std": float(adv.std()), "adv_raw_mean": float(adv.mean()),
                  **{k: upd.get(k) for k in ("steps", "approx_kl", "clip_fraction", "entropy", "value_loss", "policy_loss", "early_stop")},
                  "seconds": time.time() - t0})
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in combat_before.items()), "combat changed"
            torch.save({"model_state": model.state_dict(), "iteration": it}, out / f"iter_{it}.pt")
            if it % args.eval_every == 0 or it == args.iterations:
                emit({"event": "eval", "iteration": it, **evaluate(model.eval(), workers, args.eval_world_start, args.eval_n)})
    finally:
        for w in workers:
            w.close()
    emit({"event": "done"})


if __name__ == "__main__":
    main()
