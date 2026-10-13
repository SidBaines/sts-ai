#!/usr/bin/env python3
"""Evaluate a checkpoint on fixed TRAIN-eval worlds; optional paired comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from sts_rl.common import EVAL_WORLD_START, ORIGINAL_CHECKPOINT, close_workers, load_policy, make_workers  # noqa: E402
from sts_rl.evaluate import evaluate, paired  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--world-start", type=int, default=EVAL_WORLD_START)
    p.add_argument("--mode", choices=("greedy", "sample", "greedy_combat"), default="greedy")
    p.add_argument("--dead-action-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--baseline", help="eval JSON to pair against")
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    workers = make_workers(args.workers, dead_action_filter=args.dead_action_filter)
    try:
        result = evaluate(model, workers, n=args.n, world_start=args.world_start, mode=args.mode)
    finally:
        close_workers(workers)
    result["summary"].update(checkpoint=args.checkpoint, dead_action_filter=args.dead_action_filter)
    if args.baseline:
        result["paired"] = paired(json.loads(Path(args.baseline).read_text())["records"], result["records"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result))
    print(json.dumps({**result["summary"], **({"paired": result["paired"]} if "paired" in result else {})}))


if __name__ == "__main__":
    main()
