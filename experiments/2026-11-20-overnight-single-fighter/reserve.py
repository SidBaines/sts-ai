"""Check the planned world ranges against previous runs before opening the new panels."""

from bisect import bisect_left
from collections import Counter
from itertools import count, islice
import json
import os
from pathlib import Path
import subprocess

from sts_policy.curriculum import world_split
from sts_rl.fresh_study import atomic_json
from study import CFG, ROOT, RUNS, OUT, check_inputs


def training(start, n):
    return list(islice((i for i in count(start) if world_split(i) == "train"), n))


def reserve():
    check_inputs()
    names = (
        "screen",
        "feedback_screen",
        "bank",
        "retention",
        "development",
        "rl_panel",
        "rl_confirmation",
    )
    rosters = {name: range(CFG[name][0], sum(CFG[name])) for name in names}
    rosters["feedback"] = training(*CFG["feedback"])
    for i, start in enumerate(CFG["backup_bank_starts"], 1):
        rosters[f"bank-backup-{i}"] = range(start, start + CFG["bank"][1])
    rosters["smoke-train"] = training(CFG["smoke_train_base"] + 10_000_000, 48)
    rosters["smoke-eval"] = range(CFG["smoke_eval"], CFG["smoke_eval"] + 16)
    for i in range(1, CFG["rl_iterations"] + 1):
        start = CFG["rl_train_base"] + 500_000_000 * CFG["rl_seed"] + 10_000_000 * i
        rosters[f"rl-{i:02}"] = training(start, CFG["rl_games"])
    seeds = sorted(s for worlds in rosters.values() for s in worlds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Planned assignments overlap")

    def check(left, right, path):
        if type(left) is int and type(right) is int:
            i = bisect_left(seeds, left)
            if i < len(seeds) and seeds[i] < right:
                raise ValueError(f"Previously assigned world {seeds[i]} in {path}")

    def scan(obj, path):
        if isinstance(obj, list):
            for v in obj:
                scan(v, path)
        if not isinstance(obj, dict) or str(obj.get("name", "")).startswith("overnight-fighter-"):
            return
        n = next(
            (obj[k] for k in ("n", "games", "worlds", "episodes", "count") if type(obj.get(k)) is int),
            1,
        )
        for k, v in obj.items():
            if k in ("windows", "reserved_windows") and isinstance(v, dict):
                for span in v.values():
                    if isinstance(span, list) and len(span) == 2:
                        check(*span, path)
            elif k in ("seeds", "world_seeds", "world_seeds_by_process") and isinstance(v, list):
                stack = list(v)
                while stack:
                    item = stack.pop()
                    if isinstance(item, list):
                        stack.extend(item)
                    elif type(item) is int:
                        check(item, item + 1, path)
            elif type(v) is int and (
                k in ("world_seed", "start", "first_seed") or k.endswith(("world_start", "seed_start"))
            ):
                check(v, v + max(1, n), path)
            elif k == "cmd" and isinstance(v, list) and "--world-start" in v:
                left = int(v[v.index("--world-start") + 1])
                size = int(v[v.index("--n") + 1]) if "--n" in v else 1
                check(left, left + size, path)
            else:
                scan(v, path)
        if all(type(obj.get(k)) is int for k in ("train_world_start", "seed", "iterations", "games")):
            for i in range(1, obj["iterations"] + 1):
                left = obj["train_world_start"] + 500_000_000 * obj["seed"] + 10_000_000 * i
                check(left, left + 10 * obj["games"], path)

    gitdir = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT, text=True
    ).strip()
    archive = Path(gitdir).parent / "data/worktree-archive/2026-11-18"
    roots = [RUNS, *sorted(archive.glob("*/data/rl_runs")), archive / "nonllm-curriculum/data"]
    counts = Counter()
    for root in roots:
        for parent, dirs, files in os.walk(root):
            dirs[:] = [
                d
                for d in dirs
                if d not in ("record_objects", "checkpoints", "__pycache__")
                and Path(parent) / d != OUT
                and not d.startswith("overnight-fighter-")
            ]
            for name in set(files) & {
                "config.json",
                "run.json",
                "queue.json",
                "training-plan.json",
                "production-plan.json",
                "world-plan.json",
            }:
                path = Path(parent) / name
                scan(json.loads(path.read_bytes()), path)
                counts[name] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUT / "world-plan.json",
        dict(
            windows={k: [min(v), max(v) + 1] for k, v in rosters.items()},
            counts={k: len(v) for k, v in rosters.items()},
            historical_files=dict(counts),
            overlaps=[],
        ),
    )
    atomic_json(OUT / "rl-worlds.json", {k: v for k, v in rosters.items() if k.startswith("rl-")})
    print(f"Reserved {len(seeds):,} distinct worlds against {sum(counts.values())} historical records.")


if __name__ == "__main__":
    reserve()
