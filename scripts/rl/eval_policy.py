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
    p.add_argument("--dominated-filter", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--forced-play", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--smith-above", type=float, default=None, help="overlay: drop 'rest' at HP fraction >= this")
    p.add_argument("--prefer-remove", action=argparse.BooleanOptionalAction, default=False,
                   help="overlay: drop potion buys while a card removal is affordable")
    p.add_argument("--elite-above", type=float, default=None,
                   help="overlay: path into an elite whenever one is offered at HP fraction >= this")
    p.add_argument("--skip-cards-after", type=int, default=None, help="overlay: skip every card reward after this floor")
    p.add_argument("--save-potions-above", type=float, default=None,
                   help="overlay: no potion drinking in hallway fights while HP fraction >= this")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--collectors", type=int, default=1, help=">1: parallel collector processes")
    p.add_argument("--baseline", help="eval JSON to pair against")
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    options = dict(dead_action_filter=args.dead_action_filter, dominated=args.dominated_filter,
                   forced_play=args.forced_play, smith_above=args.smith_above, prefer_remove=args.prefer_remove,
                   elite_above=args.elite_above, skip_cards_after=args.skip_cards_after,
                   save_potions_above=args.save_potions_above)
    if args.collectors > 1:
        from sts_rl.parallel_collect import CollectorPool
        workers = CollectorPool(args.collectors, workers=args.workers, checkpoint=args.checkpoint, **options)
    else:
        workers = make_workers(args.workers, **options)
    try:
        result = evaluate(model, workers, n=args.n, world_start=args.world_start, mode=args.mode)
    finally:
        workers.close() if hasattr(workers, "evaluate_tasks") else close_workers(workers)
    result["summary"].update(checkpoint=args.checkpoint, **options)
    if args.baseline:
        result["paired"] = paired(json.loads(Path(args.baseline).read_text())["records"], result["records"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result))
    print(json.dumps({**result["summary"], **({"paired": result["paired"]} if "paired" in result else {})}))


if __name__ == "__main__":
    main()
