#!/usr/bin/env python3
"""Summarize policy metrics with world-cluster uncertainty and explicit censoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

SUCCESS = {"battle_victory", "victory", "act_clear", "PLAYER_VICTORY"}
FAILURE = {"battle_defeat", "defeat", "PLAYER_LOSS"}


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def world_bootstrap(rows: Iterable[tuple[Any, float]], *, replicates: int = 2000,
                    seed: int = 0) -> dict[str, Any]:
    """Equal-world mean; resample worlds, preserving dependent episodes within each.

    The estimand gives each world equal weight, rather than overweighting worlds
    that supplied many fights. No interval is reported for fewer than two worlds.
    """
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    groups: dict[str, list[float]] = defaultdict(list)
    for world, value in rows:
        if world is None:
            raise ValueError("world identity is required for clustered uncertainty")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metric values must be finite")
        groups[str(world)].append(number)
    means = [sum(values) / len(values) for _, values in sorted(groups.items())]
    result = {"worlds": len(means), "observations": sum(map(len, groups.values())),
              "world_mean": sum(means) / len(means) if means else None,
              "ci95": None, "bootstrap_replicates": replicates,
              "estimand": "equal-world mean; worlds resampled as clusters"}
    if len(means) >= 2:
        rng = random.Random(seed)
        samples = [sum(rng.choice(means) for _ in means) / len(means) for _ in range(replicates)]
        result["ci95"] = [_quantile(samples, .025), _quantile(samples, .975)]
    return result


def summarize_live(episodes: list[dict[str, Any]], faults: list[dict[str, Any]], *,
                   replicates: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Faults never enter losses; truncated/unresolved episodes do not become losses."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fault_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        kind = str(episode.get("task_kind", "unknown"))
        if episode.get("fault", False):
            fault_groups[kind].append(episode)
        else:
            groups[kind].append(episode)
    for fault in faults:
        fault_groups[str(fault.get("task_kind", "unknown"))].append(fault)
    output = {}
    for kind in sorted(set(groups) | set(fault_groups)):
        rows = groups[kind]
        decisive = []
        counts = Counter()
        reasons = Counter()
        outcomes = Counter()
        for episode in rows:
            outcome = str(episode.get("task_outcome") or episode.get("outcome", "")).rsplit(".", 1)[-1]
            outcomes[outcome] += 1
            terminated, truncated = bool(episode.get("terminated")), bool(episode.get("truncated"))
            counts["terminated"] += terminated
            counts["truncated"] += truncated
            reasons[str(episode.get("termination_reason", ""))] += 1
            if terminated and not truncated and outcome in SUCCESS | FAILURE:
                success = outcome in SUCCESS
                counts["successes" if success else "losses"] += 1
                decisive.append((episode.get("world_seed"), float(success)))
            else:
                counts["unresolved"] += 1
        def metric(name: str):
            values = [(e.get("world_seed"), float(e[name])) for e in rows
                      if e.get(name) is not None and math.isfinite(float(e[name]))]
            return world_bootstrap(values, replicates=replicates, seed=seed)
        output[kind] = {"valid_episodes": len(rows), "faults": len(fault_groups[kind]),
                        "fault_worlds": len({f.get("world_seed") for f in fault_groups[kind]}),
                        **{key: counts[key] for key in ("terminated", "truncated", "successes", "losses", "unresolved")},
                        "termination_reasons": dict(reasons),
                        "outcomes": dict(outcomes),
                        "success_among_decisive_episodes": counts["successes"] / len(decisive) if decisive else None,
                        "decisive_success": world_bootstrap(decisive, replicates=replicates, seed=seed),
                        "reward_all_valid": metric("reward"), "hp_cost_all_valid": metric("hp_cost"),
                        "floor_all_valid": metric("floor"),
                        "success_denominator": "only non-fault, non-truncated, recognized terminal outcomes",
                        "censoring_note": "conditional success can be biased when truncation/fault rates differ; report these counts alongside it"}
    return output


def summarize_bc(rows: list[dict[str, Any]], *, replicates: int = 2000,
                 seed: int = 0) -> dict[str, Any]:
    """Optional per-observation correct/loss records permit world-cluster intervals."""
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phases[str(row.get("phase", "unknown"))].append(row)
    return {phase: {"accuracy": world_bootstrap([(r["world_seed"], float(r["correct"])) for r in records],
                                               replicates=replicates, seed=seed),
                    "loss": world_bootstrap([(r["world_seed"], float(r["loss"])) for r in records],
                                           replicates=replicates, seed=seed)}
            for phase, records in phases.items()}


def summarize_metrics(metrics: dict[str, Any], *, replicates: int = 2000,
                      seed: int = 0) -> dict[str, Any]:
    rounds = metrics.get("rounds", [])
    episodes = list(metrics.get("episodes", [])) + [e for r in rounds for e in r.get("episodes", [])]
    faults = list(metrics.get("faults", [])) + [f for r in rounds for f in r.get("faults", [])]
    result = {"command": metrics.get("command"), "encoder_version": metrics.get("encoder_version"),
              "config": metrics.get("config", {}), "bc_training": metrics.get("training"),
              "bc_development": metrics.get("development"),
              "live": summarize_live(episodes, faults, replicates=replicates, seed=seed)}
    if len(rounds) > 1:
        result["live_by_round"] = [{"round": i, "live": summarize_live(r.get("episodes", []), r.get("faults", []),
                                  replicates=replicates, seed=seed)} for i, r in enumerate(rounds)]
        result["pooling_note"] = "multiple learner rounds may use different policy checkpoints; pooled rollout summaries are not a fixed-checkpoint evaluation"
    if metrics.get("bc_records"):
        result["bc_by_world"] = summarize_bc(metrics["bc_records"], replicates=replicates, seed=seed)
    else:
        result["bc_ci_note"] = "aggregate BC metrics do not support world-cluster confidence intervals; per-world correct/loss records are required"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    reports = [{"file": str(path), **summarize_metrics(json.loads(path.read_text()),
                replicates=args.bootstrap, seed=args.seed)} for path in args.metrics]
    text = json.dumps({"reports": reports}, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
