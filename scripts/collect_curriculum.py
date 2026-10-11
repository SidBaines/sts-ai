#!/usr/bin/env python3
"""Collect new base-native labels/tasks; old research corpora are incompatible."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from sts_policy.data import collect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--worlds", type=int, default=32)
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
    result = collect_dataset(args.output, list(range(args.seed_start, args.seed_start + args.worlds)),
                             workers=args.workers, timeout=args.timeout, fast_observations=args.fast_observations,
                             deadline_seconds=args.deadline_seconds, max_act=args.max_act,
                             horizon=args.horizon, simulations=args.simulations,
                             epsilon_ooc=args.epsilon_ooc, epsilon_combat=args.epsilon_combat)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
