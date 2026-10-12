"""Fixed full-game TRAIN mixture and prespecified DEV-only v2 cohorts."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from .curriculum import TaskSpec, validate_provenance, world_split
from .long_run_tasks import (ContinuationPool, INTERFACE_SHA, NATIVE_SHA, ROUND_STRIDE,
    _train_worlds, _write_immutable, add_learner_continuations, load_continuations,
    task_identity, write_task_batch, write_learner_continuations)

VERSION = "sts_long_curriculum_tasks_v2"
PANEL_VERSION = "sts_long_curriculum_dev_v2"
TRAIN_WORLD_START = 100000000
TRAIN_WORLD_STOP = 100000000000  # Below the frozen v1 Act 2 full-start stream.
FRACTIONS = {"scratch": .70, "act2": .20, "act3": .10}


def _integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")


def mixture_counts(episodes: int) -> dict[str, int]:
    """Exact largest remainders; fresh scratch wins ties, avoiding float ties."""
    _integer(episodes, "episode count", 1, 10000)
    weights = {"scratch": 7, "act2": 2, "act3": 1}
    counts = {key: episodes * weight // 10 for key, weight in weights.items()}
    priority = {"scratch": 0, "act2": 1, "act3": 2}
    order = sorted(weights, key=lambda key: (-(episodes * weights[key] % 10), priority[key]))
    for key in order[:episodes - sum(counts.values())]:
        counts[key] += 1
    return counts


def _saved_start(row: dict[str, Any], act: int, pool: ContinuationPool) -> tuple[TaskSpec, dict[str, Any]]:
    task = TaskSpec.from_dict(row)
    long = row.get("long_metadata", {})
    source = long.get("start_source", "native_teacher")
    floor = long.get("start_floor") if source == "learner" else row.get("curriculum_metadata", {}).get("floor")
    if (world_split(task.seed) != "train" or row.get("split") != "train" or
            source not in ("native_teacher", "learner") or not task.prefix or
            len(task.ooc_previews) != len(task.prefix) or task.max_act != 3 or task.ascension != 0 or
            task.kind not in ("run", "battle") or
            not isinstance(task.expected_public_hash, str) or
            re.fullmatch(r"[a-f0-9]{64}", task.expected_public_hash) is None or
            isinstance(floor, bool) or not isinstance(floor, int) or
            not 17 * (act - 1) <= floor <= 17 * act or
            (source == "learner" and long.get("start_act") != act) or
            (source == "native_teacher" and row.get("curriculum_metadata", {}).get("act") != act)):
        raise ValueError("Invalid healthy TRAIN saved frontier")
    validate_provenance(pool.provenance, task.expected_provenance or {})
    prefix = {"prefix": list(task.prefix), "ooc_previews": list(task.ooc_previews)}
    metadata = {"component": f"act{act}", "start_source": source, "start_act": act,
                "start_floor": floor, "source_frontier_id": task_identity(task),
                "source_public_hash": task.expected_public_hash,
                "source_prefix_sha256": hashlib.sha256(json.dumps(prefix, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()}
    return task, metadata


def sample_v2(pool: ContinuationPool, *, round_index: int, episodes: int = 128,
              seed: int = 11, horizon: int = 4096, simulations: int = 200,
              world_start: int = TRAIN_WORLD_START) -> dict[str, Any]:
    """All starts target full Act 3 victory; source labels and overlap stay explicit."""
    _integer(round_index, "round", 0, 999999)
    _integer(world_start, "TRAIN namespace", TRAIN_WORLD_START, TRAIN_WORLD_STOP - ROUND_STRIDE)
    _integer(horizon, "horizon", 1, 100000)
    _integer(simulations, "simulations", 1, 1000000)
    if (pool.provenance.get("environment_version") != "0.1.1" or
            pool.provenance.get("python_interface_sha256") != INTERFACE_SHA or
            pool.provenance.get("native_provenance", {}).get("module_sha256") != NATIVE_SHA):
        raise ValueError("Saved source is not corrected native 0.1.1")
    counts = mixture_counts(episodes)
    start = world_start + round_index * ROUND_STRIDE
    if start + ROUND_STRIDE > TRAIN_WORLD_STOP:
        raise ValueError("TRAIN seed block overlaps the frozen v1 phase namespace")
    fresh = _train_worlds(start, counts["scratch"], start + ROUND_STRIDE)
    pairs = [(TaskSpec(seed=world, kind="run", max_act=3, horizon=horizon,
                      battle_simulations=simulations, seek_combat=False),
              {"component": "scratch", "start_source": "fresh_scratch", "start_act": 1,
               "start_floor": 0}) for world in fresh]
    rng = random.Random(f"{VERSION}:{seed}:{round_index}")
    for act in (2, 3):
        rows = list(pool.by_act.get(act, ()))
        if counts[f"act{act}"] > len(rows):
            raise ValueError("Saved-start quota exceeds available distinct TRAIN worlds")
        rng.shuffle(rows)
        chosen = rows[:counts[f"act{act}"]]
        if len({TaskSpec.from_dict(row).seed for row in chosen}) != len(chosen):
            raise ValueError("Duplicate TRAIN saved world within act quota")
        for row in chosen:
            source, metadata = _saved_start(row, act, pool)
            pairs.append((replace(source, kind="run", max_act=3, horizon=horizon,
                                  battle_simulations=simulations, seek_combat=False), metadata))
    rng.shuffle(pairs)
    identities = [task_identity(task) for task, _ in pairs]
    if len(identities) != episodes or len(set(identities)) != episodes:
        raise ValueError("Duplicate or incomplete immutable v2 task batch")
    saved = {task.seed for task, meta in pairs if meta["start_source"] != "fresh_scratch"}
    source_counts = {label: sum(meta["start_source"] == label for _, meta in pairs)
                     for label in ("fresh_scratch", "native_teacher", "learner")}
    metadata = {"version": VERSION, "stage": "full3_continuous", "stream": "full3_continuous",
        "round_index": round_index, "optimizer_seed": seed, "fractions": dict(FRACTIONS),
        "counts": counts, "scratch_seed_block": [start, start + ROUND_STRIDE],
        "source": pool.source_identity, "source_provenance": pool.provenance,
        "tasks": [{"task_id": task_identity(task), "world_seed": task.seed,
                   **meta, "goal_act": 3, "horizon": horizon} for task, meta in pairs],
        "source_episode_counts": source_counts,
        "world_inventory": {"requested_episodes": episodes, "distinct_worlds": len(set(fresh) | saved),
            "fresh_worlds": len(fresh), "saved_worlds": len(saved),
            "fresh_saved_overlap": len(set(fresh) & saved)},
        "continuation_reuse": "TRAIN saved frontiers reused across rounds; distinct worlds are a union, not source sums"}
    return {"tasks": [task for task, _ in pairs], "metadata": metadata}


def fixed_v2_dev_panels(output: str | Path, *, world_start: int = 400000,
                        screen_episodes: int = 48, confirm_episodes: int = 192,
                        retention_episodes: int = 192, horizon: int = 4096,
                        retention_horizon: int = 2048, simulations: int = 200,
                        seed: int = 777) -> dict[str, Any]:
    """Only new DEV definitions; confirmation/retention share paired worlds."""
    _integer(world_start, "DEV namespace", 400000, TRAIN_WORLD_START - 100000)
    for name, count in (("screen", screen_episodes), ("confirmation", confirm_episodes)):
        _integer(count, name + " count", 1, 10000)
    if retention_episodes != confirm_episodes:
        raise ValueError("Retention must pair the complete confirmation world cohort")
    for name, value in (("horizon", horizon), ("retention horizon", retention_horizon), ("simulations", simulations)):
        _integer(value, name, 1, 1000000)
    worlds = []
    for world in range(world_start, world_start + 100000):
        if world_split(world) == "dev":
            worlds.append(world)
            if len(worlds) == screen_episodes + confirm_episodes:
                break
    if len(worlds) != screen_episodes + confirm_episodes:
        raise ValueError("DEV seed block exhausted")
    screen, confirm = worlds[:screen_episodes], worlds[screen_episodes:]
    random.Random(f"{PANEL_VERSION}:{seed}:screen").shuffle(screen)
    random.Random(f"{PANEL_VERSION}:{seed}:confirmation").shuffle(confirm)
    output = Path(output).resolve()
    panels = {}
    for name, selected, max_act, limit in (("screen_full3", screen, 3, horizon),
            ("confirm_full3", confirm, 3, horizon), ("retention_act1", confirm, 1, retention_horizon)):
        tasks = [TaskSpec(seed=world, kind="run", max_act=max_act, horizon=limit,
                         battle_simulations=simulations, seek_combat=False) for world in selected]
        batch = {"tasks": tasks, "metadata": {"version": PANEL_VERSION, "split": "dev", "panel": name,
            "policy_seed": seed, "tasks": [{"task_id": task_identity(task), "world_seed": task.seed,
                "component": name, "start_source": "fresh_scratch", "start_act": 1,
                "start_floor": 0, "goal_act": max_act, "horizon": limit} for task in tasks]}}
        record = write_task_batch(batch, output / f"{name}.jsonl")
        panels[name] = {**record, "max_act": max_act, "horizon": limit, "worlds": list(selected),
                       "episodes": len(selected), "simulations": simulations, "split": "dev"}
    definition = {"version": PANEL_VERSION, "world_start": world_start, "policy_seed": seed,
                  "panels": panels, "confirmation_disjoint_from_screen": True,
                  "retention_paired_with_confirmation": True,
                  "final_embargo": "No FINAL tasks or states are generated or opened"}
    manifest = output / "v2_panels.json"
    _write_immutable(manifest, json.dumps(definition, sort_keys=True, indent=2) + "\n")
    return {**definition, "manifest": str(manifest)}
