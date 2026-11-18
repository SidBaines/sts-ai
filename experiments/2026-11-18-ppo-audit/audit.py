"""Frozen-champion diagnostics and held-out PPO probes. Candidate weights stay in memory."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sts_policy.encoding import collate_observations
from sts_policy.losses import PPOConfig, masked_distribution
from sts_policy.model_agent import collate_public
from sts_policy.rollout_training import update_ppo
from sts_rl.common import load_policy, phase_forward, train_tasks
from sts_rl.ooc_trainer import ReturnNormalizer, card_reward_has_items, ooc_forward
from sts_rl.parallel_collect import CollectorPool

ARMS = {
    "control": (0.0003, 3, 0.01, False),
    "lower_lr": (0.0001, 3, 0.01, False),
    "one_epoch": (0.0003, 1, 0.01, False),
    "more_entropy": (0.0003, 3, 0.03, False),
    "keep_adam": (0.0003, 3, 0.01, True),
}


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def group(seed_array, values):
    _, ids = np.unique(seed_array, return_inverse=True)
    return np.bincount(ids, weights=values) / np.bincount(ids)


def interval(values):
    rng = np.random.default_rng(9303)
    means = rng.choice(values, (3000, len(values)), replace=True).mean(axis=1)
    return [float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()]


def subtype(t):
    if t.menu_kind != "card_reward":
        return t.menu_kind
    return (
        "card_with_items"
        if card_reward_has_items(t.observation)
        else "card_without_items"
    )


@torch.no_grad()
def inspect_policy(model, rows, ref=None):
    output = defaultdict(list)
    for i in range(0, len(rows), 512):
        part = rows[i : i + 512]
        batch = collate_observations([t.encoded_observation for t in part])
        logits, values = model(*batch)
        d = masked_distribution(logits, batch[2])
        prob = d.probs
        best = prob.argmax(1)
        pos = torch.tensor([t.action_position for t in part])
        output["logp"].extend(d.log_prob(pos).tolist())
        output["value"].extend(values.tolist())
        output["argmax"].extend(best.tolist())
        output["maxprob"].extend(prob.max(1).values.tolist())
        output["entropy"].extend(d.entropy().tolist())
        if ref is not None:
            old, _ = ref(*batch)
            old_d = masked_distribution(old, batch[2])
            output["kl"].extend(
                (old_d.probs * (old_d.logits - d.logits)).sum(1).tolist()
            )
            output["flip"].extend((old_d.probs.argmax(1) != best).tolist())
    return {k: np.asarray(v) for k, v in output.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--previous", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "results.json").exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    model, payload = load_policy(args.checkpoint)
    model.eval()
    previous, _ = load_policy(args.previous)
    previous.eval()
    normalizer = ReturnNormalizer.restore(payload, enabled=True)
    ppo = PPOConfig(
        gamma=1.0,
        gae_lambda=1.0,
        clip_ratio=0.2,
        value_weight=0.5,
        entropy_weight=0.0,
        epochs=3,
        minibatch_size=4096,
        learning_rate=0.0003,
        target_kl=0.05,
        kl_guard_before_update=True,
    )
    tasks = train_tasks(124_000_000_000, 512)
    pool = CollectorPool(
        6,
        workers=192,
        checkpoint=str(args.checkpoint),
        threads=1,
        in_process=True,
        dead_action_filter=True,
        hide_ooc_potions=True,
        progress_floor_weight=0.05,
    )
    try:
        data = pool.collect_ooc(
            model,
            tasks,
            combat="greedy",
            temperature=1.0,
            ppo=ppo,
            credit="macro",
            gae_lambda=1.0,
            chunk=128,
            seed=18420,
            value_scale=normalizer.scale,
        )
    finally:
        pool.close()
    records = data["game_episodes"]
    terminal = {
        r["world_seed"] for r in records if r["terminated"] and not r["truncated"]
    }
    assert len(records) + data["faults"] == len(tasks)
    rows = [t for ep in data["kept"] for t in ep if t.world_seed in terminal]
    world = np.array([t.world_seed for t in rows])
    assert set(world) == terminal
    out = {
        "config": vars(args)
        | {
            "checkpoint": str(args.checkpoint),
            "previous": str(args.previous),
            "out": str(args.out),
        },
        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "worlds": [t.seed for t in tasks],
        "completed": len(terminal),
        "faults": data["faults"],
        "censored": len(records) - len(terminal),
        "wins": sum(r["task_outcome"] in ("victory", "act_clear") for r in records),
        "kept": len(rows),
        "scale": normalizer.scale,
        "folds": [],
    }
    adv = np.array([t.advantage for t in rows])
    ret = np.array([t.returns for t in rows])
    val = np.array([t.behavior_value for t in rows])
    act = np.rint([3 * t.encoded_observation.state_features[4] for t in rows]).astype(
        int
    )
    kinds = np.array([subtype(t) for t in rows])
    # Compare the old policy to the champion, evaluated on the champion's own states.
    baseline = inspect_policy(model, rows, ref=previous)
    behavior = np.array([t.behavior_log_prob for t in rows])
    out["logp_max_abs_recompute"] = float(np.max(np.abs(baseline["logp"] - behavior)))
    out["max_gae1_identity_error"] = float(np.max(np.abs(ret - val - adv)))
    out["critic"] = {}
    for label, mask in [
        ("all", np.ones(len(rows), bool)),
        *((str(a), act == a) for a in (1, 2, 3)),
    ]:
        err = ret[mask] - val[mask]
        out["critic"][label] = dict(
            n=int(mask.sum()),
            bias=float(err.mean()),
            ev=float(1 - err.var() / ret[mask].var()),
            mse=float(np.mean(err**2)),
            constant_mse=float(ret[mask].var()),
        )
    out["menus"] = {}
    for label in sorted(set(kinds)):
        mask = kinds == label
        out["menus"][label] = dict(
            n=int(mask.sum()),
            worlds=len(set(world[mask])),
            saturated=float((baseline["maxprob"][mask] > 0.99).mean()),
            old_to_new_kl=float(baseline["kl"][mask].mean()),
            flip=float(baseline["flip"][mask].mean()),
            entropy=float(baseline["entropy"][mask].mean()),
        )
    write(args.out / "diagnostics.json", out)
    print("Frozen collection complete", len(rows), out["critic"], flush=True)
    worlds = sorted(terminal)
    random.Random(838).shuffle(worlds)
    halves = np.array_split(worlds, 2)
    changes = defaultdict(dict)
    for fold, train_worlds in enumerate(halves):
        mask = np.isin(world, train_worlds)
        train = [r for r, t in zip(rows, mask) if t]
        test = [r for r, t in zip(rows, mask) if not t]
        sc = deepcopy(normalizer)
        sc.update(ret[mask])
        foldrec = {"fold": fold, "ntrain": len(train), "ntest": len(test), "arms": {}}
        for arm, (lr, epochs, entropy, carry) in ARMS.items():
            clone = deepcopy(model)
            for name, param in clone.named_parameters():
                param.requires_grad_(name.startswith("ooc."))
            clone.to("mps")
            opt = torch.optim.AdamW(
                [p for p in clone.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=0.0,
            )
            if carry:
                opt.load_state_dict(deepcopy(payload["optimizer_state"]))
                for params in opt.param_groups:
                    params.update(lr=lr, weight_decay=0.0)
            result = update_ppo(
                clone,
                train,
                collate=collate_public,
                forward=ooc_forward(
                    phase_forward(combat="greedy"), critic_gradient="detached"
                ),
                config=replace(ppo, epochs=epochs, learning_rate=lr),
                device="mps",
                seed=834 + fold,
                optimizer=opt,
                micro_batch=1024,
                return_scale=sc.scale,
                entropy_weights=[entropy * bool(t.strategic_menu) for t in train],
            )
            clone.cpu().eval()
            z = inspect_policy(clone, test, ref=model)
            ratio = np.exp(z["logp"] - baseline["logp"][~mask])
            advtest = adv[~mask]
            diff = (
                np.minimum(ratio * advtest, np.clip(ratio, 0.8, 1.2) * advtest)
                - advtest
            )
            byworld = group(world[~mask], diff)
            for seed, x in zip(sorted(set(world[~mask])), byworld):
                changes[arm][int(seed)] = float(x)
            foldrec["arms"][arm] = dict(
                steps=result["steps"],
                stopped=result["early_stop"],
                surrogate=float(byworld.mean()),
                kl=float(group(world[~mask], z["kl"]).mean()),
                flip=float(group(world[~mask], z["flip"]).mean()),
                critic_mse=float(
                    group(
                        world[~mask], (z["value"] * sc.scale - ret[~mask]) ** 2
                    ).mean()
                ),
            )
            print(fold, arm, foldrec["arms"][arm], flush=True)
        out["folds"].append(foldrec)
        write(args.out / "partial.json", out)
    out["paired_surrogate"] = {}
    for arm in ARMS:
        v = np.array([changes[arm][w] for w in worlds])
        c = np.array([changes["control"][w] for w in worlds])
        out["paired_surrogate"][arm] = {
            "vs_start": interval(v),
            "vs_control": interval(v - c),
        }
    write(args.out / "results.json", out)


if __name__ == "__main__":
    main()
