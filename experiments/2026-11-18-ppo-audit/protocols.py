"""Evaluate frozen policies on matched worlds and count top actions on mixed reward screens."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import torch
from audit import inspect_policy, subtype, write
from history import pairs

from sts_policy.losses import PPOConfig
from sts_rl.common import (
    close_workers,
    load_policy,
    make_workers,
    phase_forward,
    train_tasks,
)
from sts_rl.ooc_trainer import ReturnNormalizer, collect_ooc_batch


def lane(args):
    checkpoint, tasks, seed = args
    torch.set_num_threads(1)
    model, payload = load_policy(checkpoint)
    model.eval()
    scale = ReturnNormalizer.restore(payload, enabled=True).scale
    result = {}
    counts = Counter()
    for name, combat, ooc in [
        ("sampling", "greedy", "sample"),
        ("exact_combat", "argmax", "sample"),
        ("exact_ooc", "greedy", "argmax"),
        ("deployment", "argmax", "argmax"),
    ]:
        torch.manual_seed(seed)
        workers = make_workers(
            16,
            in_process=True,
            dead_action_filter=True,
            hide_ooc_potions=True,
            progress_floor_weight=0.05,
        )
        try:
            data = collect_ooc_batch(
                model,
                workers,
                tasks,
                forward=phase_forward(combat=combat, ooc=ooc),
                ppo=PPOConfig(gae_lambda=1.0),
                credit="macro",
                gae_lambda=1.0,
                chunk=32,
                value_scale=scale,
            )
        finally:
            close_workers(workers)
        assert not data["faults"] and len(data["game_episodes"]) == len(tasks)
        assert all(
            r["terminated"] and not r["truncated"] for r in data["game_episodes"]
        )
        result[name] = {
            r["world_seed"]: r["task_outcome"] in ("victory", "act_clear")
            for r in data["game_episodes"]
        }
        if name == "sampling":
            mixed = [
                t for ep in data["kept"] for t in ep if subtype(t) == "card_with_items"
            ]
            if mixed:
                probs = inspect_policy(model, mixed)
                for t, i, prob in zip(mixed, probs["argmax"], probs["maxprob"]):
                    description = t.observation["legal_actions"][i]["description"]
                    action = next(
                        (
                            s
                            for s in (
                                "take gold",
                                "take relic",
                                "take potion",
                                "take key",
                                "take card",
                                "skip card",
                                "skip rewards",
                            )
                            if description.startswith(s)
                        ),
                        "other",
                    )
                    counts["all"] += 1
                    if prob > 0.99:
                        counts[action] += 1
    return result, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    tasks = train_tasks(124_000_000_000, 256)
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as pool:
        chunks = list(
            pool.map(
                lane, [(args.checkpoint, tasks[i::4], 64510 + i) for i in range(4)]
            )
        )
    results = {
        name: {k: v for arms, _ in chunks for k, v in arms[name].items()}
        for name in chunks[0][0]
    }
    counts = sum((counts for _, counts in chunks), Counter())
    base = results["sampling"]
    comparisons = {k: pairs(base, v) for k, v in results.items()}
    args.out.mkdir(exist_ok=True, parents=True)
    write(
        args.out / "protocols.json",
        {
            "n": len(tasks),
            "comparisons": comparisons,
            "outcomes": results,
            "mixed_menus": counts,
        },
    )
    print(comparisons, counts, flush=True)


if __name__ == "__main__":
    main()
