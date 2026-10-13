"""Deployment-style evaluation on fixed TRAIN-eval worlds, with per-world outcomes.

Greedy play by default (what is deployed). Per-world records make every
comparison pairable: same world seeds -> identical starting runs.
"""
from __future__ import annotations

import gc
import math
from typing import Any, Sequence

import numpy as np
import torch

from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public, forward_policy
from sts_policy.rollout_training import collect_rollouts

from .common import BOSS_FLOORS, EVAL_WORLD_START, WIN_OUTCOMES, phase_forward, train_tasks


def _episodes(transitions: Sequence[Any]) -> list[list[Any]]:
    out, cur = [], []
    for t in transitions:
        cur.append(t)
        if t.terminated or t.truncated:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def episode_record(trajectory: Sequence[Any], episode: dict[str, Any]) -> dict[str, Any]:
    rec = {"seed": episode["world_seed"], "floor": episode["floor"],
           "win": episode.get("task_outcome") in WIN_OUTCOMES, "steps": episode["steps"]}
    for boss in BOSS_FLOORS:
        entry = next((t for t in trajectory if t.observation.get("phase") == "combat"
                      and t.observation.get("state", {}).get("floor") == boss), None)
        rec[f"boss{boss}_hp"] = (entry.observation["state"].get("cur_hp") if entry is not None else None)
    return rec


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    floors = np.array([r["floor"] for r in records], dtype=float)
    out = {"n": len(records), "mean_floor": float(floors.mean()),
           "se": float(floors.std() / math.sqrt(max(1, len(floors)))),
           "act1_clear": float((floors > 17).mean()), "act2_clear": float((floors > 34).mean()),
           "wins": int(sum(r["win"] for r in records))}
    for boss, name in zip(BOSS_FLOORS, ("act1", "act2", "act3")):
        reached = [r for r in records if r[f"boss{boss}_hp"] is not None]
        out[f"{name}_boss_reach"] = len(reached) / max(1, len(records))
        out[f"{name}_boss_pass"] = (sum((r["floor"] > boss) or r["win"] for r in reached) / len(reached)
                                    if reached else None)
        out[f"{name}_boss_entry_hp"] = (float(np.mean([r[f"boss{boss}_hp"] for r in reached])) if reached else None)
    return out


def eval_forward(mode: str):
    # "greedy" is an exact argmax (first index wins exact ties), so evaluation is
    # reproducible at any worker count. Logit-scaling "greedy" is NOT: exact
    # ties (duplicate card copies on select screens) then break randomly and the
    # run diverges chaotically.
    return {"greedy": forward_policy,
            "sample": phase_forward(),
            "greedy_combat": phase_forward(combat="greedy")}[mode]


@torch.no_grad()
def evaluate_tasks(model: torch.nn.Module, workers: Sequence[Any], tasks: Sequence[Any], *,
                   mode: str = "greedy", max_seconds: float = 3600.,
                   chunk: int = 256) -> tuple[list[dict[str, Any]], int, float]:
    """Per-world records for ``tasks``: (records, faults, elapsed collector seconds)."""
    model.eval()
    forward = eval_forward(mode)
    # Chunked: collect_rollouts keeps every transition (full observations) until it
    # returns, so evaluating 1,024 games at once held several GB. Each chunk is
    # reduced to per-world records immediately and then released.
    records, faults, elapsed = [], 0, 0.0
    for start in range(0, len(tasks), chunk):
        result = collect_rollouts(model, workers, tasks[start:start + chunk], collate=collate_public,
                                  forward=forward, config=PPOConfig(), max_seconds=max_seconds,
                                  deterministic=mode == "greedy", deferred_bootstrap=True, reuse_encodings=False)
        by_seed = {e["world_seed"]: e for e in result["episodes"]}
        for trajectory in _episodes(result["transitions"]):
            episode = by_seed.get(trajectory[0].world_seed)
            if episode is not None:
                records.append(episode_record(trajectory, episode))
        faults += len(result["faults"])
        elapsed += result["elapsed_seconds"]
        del result, by_seed
        gc.collect()
    return records, faults, elapsed


def finish_evaluation(records: list[dict[str, Any]], faults: int, elapsed: float, *, mode: str,
                      world_start: int) -> dict[str, Any]:
    records = sorted(records, key=lambda r: r["seed"])
    summary = summarize(records)
    summary.update(faults=faults, mode=mode, world_start=world_start, elapsed=elapsed)
    return {"summary": summary, "records": records}


def evaluate(model: torch.nn.Module, workers: Sequence[Any], *, n: int = 1024,
             world_start: int = EVAL_WORLD_START, mode: str = "greedy",
             max_seconds: float = 3600., chunk: int = 256) -> dict[str, Any]:
    """mode: greedy (deployment), sample (training-time), greedy_combat (OOC sampled).

    ``workers`` may also be a CollectorPool (sts_rl.parallel_collect), which
    splits the worlds across collector processes; records are identical."""
    tasks = train_tasks(world_start, n)
    if hasattr(workers, "evaluate_tasks"):
        records, faults, elapsed = workers.evaluate_tasks(model, tasks, mode=mode, max_seconds=max_seconds,
                                                          chunk=chunk)
    else:
        records, faults, elapsed = evaluate_tasks(model, workers, tasks, mode=mode, max_seconds=max_seconds,
                                                  chunk=chunk)
    return finish_evaluation(records, faults, elapsed, mode=mode, world_start=world_start)


def paired(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]], *, bootstrap: int = 2000) -> dict[str, Any]:
    """b minus a on common world seeds (faulted worlds drop out of both)."""
    fa, fb = {r["seed"]: r for r in a}, {r["seed"]: r for r in b}
    common = sorted(set(fa) & set(fb))
    d = np.array([fb[s]["floor"] - fa[s]["floor"] for s in common], dtype=float)
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(bootstrap)] if len(d) else [0.0]
    return {"n": len(common), "diff": float(d.mean()) if len(d) else 0.0,
            "se": float(d.std() / math.sqrt(max(1, len(d)))),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": [int(sum(fa[s]["win"] for s in common)), int(sum(fb[s]["win"] for s in common))]}


def paired_wins(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Act 3 boss wins on common worlds, with an exact McNemar (sign) test on discordant worlds.

    Wins are rare (~2%), so floor-based paired SEs say little about them: at
    1,024 worlds a 1-point win-rate change is ~1 SE. Use >= 4,096 worlds."""
    fa, fb = {r["seed"]: r for r in a}, {r["seed"]: r for r in b}
    common = sorted(set(fa) & set(fb))
    only_a = sum(fa[s]["win"] and not fb[s]["win"] for s in common)
    only_b = sum(fb[s]["win"] and not fa[s]["win"] for s in common)
    n = only_a + only_b
    tail = sum(math.comb(n, k) for k in range(max(only_a, only_b), n + 1)) / 2 ** n if n else 1.0
    wins_a = sum(fa[s]["win"] for s in common)
    wins_b = sum(fb[s]["win"] for s in common)
    return {"n": len(common), "wins": [wins_a, wins_b],
            "rate": [wins_a / max(1, len(common)), wins_b / max(1, len(common))],
            "only_a": only_a, "only_b": only_b, "p_two_sided": min(1.0, 2 * tail)}

