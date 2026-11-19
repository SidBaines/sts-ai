"""Measure combat exploration on real games and collect bounded, on-policy samples."""

from __future__ import annotations
import argparse
import gzip
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from sts_policy.losses import PPOConfig
from sts_policy.model_agent import collate_public
from sts_policy.rollout_training import collect_rollouts
from sts_rl.common import (
    JsonlLog,
    close_workers,
    load_policy,
    make_workers,
    phase_forward,
    train_tasks,
)
from sts_rl.fresh_study import atomic_json
from sts_rl.online_combat import ROLES, action_diagnostics, samples_from_game
from sts_rl.ooc_trainer import split_episodes
from sts_rl.recording import begin_run

RUNS = ROOT / "data" / "rl_runs"
BASE = RUNS / "online_combat_from_9700"
CHAMP = RUNS / "champions" / "champion_9700741dca15.pt"
TARGETS = (0.05, 0.12, 0.22, 0.025)
GRID = np.geomspace(0.0001, 100.0, 81).tolist()
STARTS = {
    "calibration": 240_000_000_000,
    "exploration": 241_000_000_000,
    "fit": 242_000_000_000,
}
COUNTS = {"calibration": 192, "exploration": 768, "fit": 3072}


def temperatures(target):
    if target == 0:
        return dict.fromkeys(ROLES, 0.0001)
    calibration = json.loads((BASE / "calibration" / "summary.json").read_text())
    return calibration["temperatures"][str(target)]


def shard(work):
    index, tasks, stage, temps, out = work
    torch.set_num_threads(1)
    torch.manual_seed(83013 + index)
    model, _ = load_policy(str(CHAMP))
    workers = make_workers(
        16,
        in_process=True,
        dead_action_filter=True,
        hide_ooc_potions=True,
        progress_floor_weight=0.05,
    )
    forward = phase_forward(
        combat_temperature=temps["combat"], boss_temperature=temps["boss_specialist"]
    )
    rows = {role: [] for role in ROLES}
    worlds, failures = {}, []
    try:
        for begin in range(0, len(tasks), 32):
            subset = tasks[begin : begin + 32]
            result = collect_rollouts(
                model,
                workers,
                subset,
                collate=collate_public,
                forward=forward,
                config=PPOConfig(gamma=1.0, gae_lambda=1.0),
                max_seconds=3600,
                deferred_bootstrap=True,
            )
            for ep in split_episodes(result["transitions"]):
                samples = samples_from_game(ep)
                for role in ROLES:
                    rows[role].extend(samples[role])
            for ep in result["episodes"]:
                complete = ep["terminated"] and not ep["truncated"] and not ep["fault"]
                worlds[ep["world_seed"]] = {
                    "win": ep["task_outcome"] in ("victory", "act_clear")
                    if complete
                    else None,
                    "act3": ep["max_act_reached"] >= 3 if complete else None,
                }
            failures.extend(result["faults"])
            for fault in result["faults"]:
                worlds[fault["world_seed"]] = {"win": None, "act3": None}
            print(
                f"{stage} worker {index}: {min(len(tasks), begin + 32)}/{len(tasks)}",
                flush=True,
            )
            del result
    finally:
        close_workers(workers)
    settings = {
        role: GRID if stage == "calibration" else [temps[role]] for role in ROLES
    }
    stats = {
        role: action_diagnostics(getattr(model, role), rows[role], settings[role])
        for role in ROLES
    }
    if stage in ("calibration", "fit"):
        part = out / f"rows-{index}.pt.gz"
        with gzip.open(part.with_suffix(".tmp"), "wb", compresslevel=1) as f:
            torch.save(rows, f)
        part.with_suffix(".tmp").replace(part)
    atomic_json(
        out / f"shard-{index}.json", dict(worlds=worlds, failures=failures, stats=stats)
    )


def collect(stage, out, target, processes=4):
    temps = temperatures(target)
    n = COUNTS[stage]
    tasks = train_tasks(STARTS[stage], n)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(
        stage=stage,
        target=target,
        temperatures=temps,
        world_start=STARTS[stage],
        games=n,
        processes=processes,
    )
    if (out / "config.json").exists() and json.loads(
        (out / "config.json").read_text()
    ) != config:
        raise ValueError("existing pilot folder has different settings")
    atomic_json(out / "config.json", config)
    begin_run(out, config, CHAMP)
    log = JsonlLog(out / "progress.jsonl")
    with ProcessPoolExecutor(processes, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(shard, (i, tasks[i::processes], stage, temps, out))
            for i in range(processes)
            if not (out / f"shard-{i}.json").is_file()
        ]
        for f in as_completed(futures):
            f.result()
            log.emit({"event": "shard_done"})
    shards = [
        json.loads((out / f"shard-{i}.json").read_text()) for i in range(processes)
    ]
    worlds = {int(k): v for s in shards for k, v in s["worlds"].items()}
    if (
        set(worlds) != {t.seed for t in tasks}
        or sum(len(s["worlds"]) for s in shards) != n
    ):
        raise ValueError("pilot world roster mismatch")
    counts = {k: sum(w[k] is True for w in worlds.values()) for k in ("win", "act3")}
    counts["unknown"] = sum(w["win"] is None for w in worlds.values())
    summary = dict(config=config, counts=counts, exploration={}, temperatures={})
    for role in ROLES:
        stats = [s["stats"][role] for s in shards]
        total = sum(s["choices"] for s in stats)
        if total == 0:
            raise ValueError(f"no meaningful {role} decisions")
        means = np.average(
            [s["expected_rates"] for s in stats],
            axis=0,
            weights=[s["choices"] for s in stats],
        )
        actual = sum(s["actual"] for s in stats) / total
        summary["exploration"][role] = dict(
            choices=total,
            actual_rate=actual,
            expected_rates=means.tolist(),
            max_log_prob_error=max(
                (
                    s["max_log_prob_error"]
                    for s in stats
                    if s["max_log_prob_error"] is not None
                ),
                default=None,
            ),
        )
        if stage == "calibration":
            for tar in TARGETS:
                if not means[0] <= tar <= means[-1]:
                    raise ValueError("temperature grid does not bracket target")
                summary["temperatures"].setdefault(str(tar), {})[role] = float(
                    np.exp(np.interp(tar, means, np.log(GRID)))
                )
    atomic_json(out / "worlds.json", worlds)
    atomic_json(out / "summary.json", summary)
    log.emit({"event": "done", "worlds": n, **counts})
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=STARTS)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--target", type=float, default=0.0)
    p.add_argument("--processes", type=int, default=4)
    args = p.parse_args()
    collect(args.stage, args.out, args.target, args.processes)
