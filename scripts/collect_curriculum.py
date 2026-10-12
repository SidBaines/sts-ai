#!/usr/bin/env python3
"""Collect new base-native labels/tasks; old research corpora are incompatible."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from sts_policy.data import collect_dataset
from sts_policy.curriculum import world_split


def selected_worlds(seed_start: int, count: int, split: str) -> list[int]:
    """Request exactly count worlds without executing another split's seeds."""
    if count < 1 or split not in {"train", "dev", "final", "all"}:
        raise ValueError("positive count and a known split are required")
    worlds = []
    for world in range(seed_start, seed_start + max(1024, 100 * count)):
        if split == "all" or world_split(world) == split:
            worlds.append(world)
            if len(worlds) == count:
                return worlds
    raise RuntimeError("Requested split exhausted the bounded seed range")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--split", choices=("train", "dev", "final", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-act", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--epsilon-ooc", type=float, default=0.0)
    parser.add_argument("--epsilon-combat", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--fast-observations", action="store_true")
    args = parser.parse_args()
    if args.worlds < 1 or args.workers < 1:
        parser.error("worlds/workers must be positive")
    result = collect_dataset(args.output, selected_worlds(args.seed_start, args.worlds, args.split),
                             workers=args.workers, timeout=args.timeout, fast_observations=args.fast_observations,
                             deadline_seconds=args.deadline_seconds, max_act=args.max_act,
                             horizon=args.horizon, simulations=args.simulations,
                             epsilon_ooc=args.epsilon_ooc, epsilon_combat=args.epsilon_combat)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
