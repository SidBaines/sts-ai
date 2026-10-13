#!/usr/bin/env python3
"""Policy-gradient signal-to-noise: cosine between gradients from disjoint episode halves.

Mirrors update_ppo's first minibatch-free step at ratio==1: loss = -mean(A_norm * logpi)
with A_norm normalised over the whole buffer (as update_ppo does). Reports per-phase
(combat vs OOC actor parameters) cosines for several group sizes, plus the relative
size of entropy and value-loss gradients on the shared encoders.
Diagnostic only; TRAIN-split worlds from an unused namespace.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rollout_probe import WORKER_CONFIG, tasks_for  # noqa: E402

from sts_policy.losses import PPOConfig, masked_distribution
from sts_policy.model_agent import cached_public_collator, collate_public, forward_policy, policy_from_payload
from sts_policy.rollout_training import collect_rollouts
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker


def split_episodes(transitions):
    eps, cur = [], []
    for t in transitions:
        cur.append(t)
        if t.terminated or t.truncated:
            eps.append(cur)
            cur = []
    if cur:
        eps.append(cur)
    return eps


def flat_grad(model, prefix_filter):
    parts = []
    for name, p in model.named_parameters():
        if prefix_filter(name):
            parts.append((p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(parts).clone()


def group_grads(model, collate, trans, adv_norm, which):
    """Return dict of flattened grads for 'pg', 'ent', 'val' losses over transitions."""
    model.zero_grad(set_to_none=True)
    batch = collate([t.observation for t in trans], "cpu")
    logits, values, mask = forward_policy(model, batch)
    dist = masked_distribution(logits, mask)
    actions = torch.tensor([t.action_position for t in trans])
    a = torch.tensor([adv_norm[id(t)] for t in trans])
    if which == "pg":
        loss = -(a * dist.log_prob(actions)).mean()
    elif which == "ent":
        loss = -0.01 * dist.entropy().mean()
    else:
        ret = torch.tensor([t.returns for t in trans])
        loss = 0.5 * 0.5 * (values - ret).square().mean()
    loss.backward()
    out = {}
    for phase in ("combat", "ooc"):
        out[phase + "_actor"] = flat_grad(model, lambda n, ph=phase: n.startswith(ph + ".") and ".critic." not in n)
        out[phase + "_encoder"] = flat_grad(model, lambda n, ph=phase: n.startswith(ph + ".state_encoder"))
    return out


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--world-start", type=int, default=6_000_000_000)
    p.add_argument("--chunk", type=int, default=4096, help="transitions per forward chunk")
    p.add_argument("--macro", action="store_true", help="greedy frozen combat + macro OOC advantages")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sizes", default="16,32,64,128")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = policy_from_payload(load_checkpoint(args.checkpoint)).eval()
    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=10.0) for _ in range(args.workers)]
    from ooc_macro_ppo import greedy_combat_forward, macro_ooc_transitions
    try:
        res = collect_rollouts(model, workers, tasks_for(args.world_start, args.episodes), collate=collate_public,
                               forward=greedy_combat_forward if args.macro else forward_policy, config=PPOConfig(), max_seconds=1200,
                               deferred_bootstrap=True, reuse_encodings=True)
    finally:
        for w in workers:
            w.close()
    trans = res["transitions"]
    eps = split_episodes(trans)
    if args.macro:
        keep = {id(t) for t in macro_ooc_transitions(trans, 0.95)}
        eps = [[t for t in ep if id(t) in keep] for ep in eps]
        eps = [ep for ep in eps if ep]
        trans = [t for ep in eps for t in ep]
    print(json.dumps({"episodes": len(eps), "transitions": len(trans),
                      "mean_floor": float(np.mean([e["floor"] for e in res["episodes"]]))}), flush=True)
    raw = np.array([t.advantage for t in trans])
    norm = (raw - raw.mean()) / max(raw.std(), 1e-8)
    adv_norm = {id(t): float(v) for t, v in zip(trans, norm)}
    collate = cached_public_collator([t.observation for t in trans], preencoded=[t.encoded_observation for t in trans])
    model.train()

    def grads_for(episode_group, which):
        flat = [t for ep in episode_group for t in ep]
        total = None
        for i in range(0, len(flat), args.chunk):
            part = flat[i:i + args.chunk]
            g = group_grads(model, collate, part, adv_norm, which)
            w = len(part) / len(flat)
            total = {k: v * w for k, v in g.items()} if total is None else {k: total[k] + g[k] * w for k in g}
        return total

    rng = random.Random(0)
    report = {}
    for size in [int(x) for x in args.sizes.split(",")]:
        if 2 * size > len(eps):
            continue
        cs = {}
        for rep in range(4):
            order = list(range(len(eps)))
            rng.shuffle(order)
            A = [eps[i] for i in order[:size]]
            B = [eps[i] for i in order[size:2 * size]]
            ga, gb = grads_for(A, "pg"), grads_for(B, "pg")
            for k in ga:
                cs.setdefault(k, []).append(cos(ga[k], gb[k]))
        report[f"pg_cos_{size}eps"] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in cs.items()}
        print(f"halves of {size} episodes:", {k: round(float(np.mean(v)), 3) for k, v in cs.items()}, flush=True)
    # relative gradient norms on the full buffer
    full = {w: grads_for(eps, w) for w in ("pg", "ent", "val")}
    norms = {w: {k: float(v.norm()) for k, v in g.items()} for w, g in full.items()}
    print("full-buffer grad norms:", json.dumps(norms, indent=1))
    print("cos(value-grad, pg-grad) on encoders:",
          {k: round(cos(full['val'][k], full['pg'][k]), 3) for k in ("combat_encoder", "ooc_encoder")})
    print("cos(entropy-grad, pg-grad) on actors:",
          {k: round(cos(full['ent'][k], full['pg'][k]), 3) for k in ("combat_actor", "ooc_actor")})


if __name__ == "__main__":
    main()
