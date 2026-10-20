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

from sts_rl.recording import begin_run
from sts_env.env import SimulatorError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--world-start", type=int, default=EVAL_WORLD_START)
    p.add_argument("--mode", choices=("greedy", "sample", "greedy_combat"), default="greedy")
    p.add_argument("--dead-action-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hide-ooc-potions", action=argparse.BooleanOptionalAction, default=False,
                   help="deployment protocol: hide potion drinking and discarding out of combat")
    p.add_argument("--dominated-filter", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--forced-play", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--smith-above", type=float, default=None, help="overlay: drop 'rest' at HP fraction >= this")
    p.add_argument("--prefer-remove", action=argparse.BooleanOptionalAction, default=False,
                   help="overlay: drop potion buys while a card removal is affordable")
    p.add_argument("--elite-above", default=None,
                   help="overlay: path into an offered elite at HP fraction >= this; or per act, e.g. '1:0.7,2:0.9,3:0.9'")
    p.add_argument("--skip-cards-after", type=int, default=None, help="overlay: skip every card reward after this floor")
    p.add_argument("--save-potions-above", type=float, default=None,
                   help="overlay: no potion drinking in hallway fights while HP fraction >= this")
    p.add_argument("--avoid-boss-relics", default="", help="overlay: comma-separated boss relics never to take")
    # Not BooleanOptionalAction: a flag whose name starts with "no-" would be read as the negative form.
    p.add_argument("--no-skip-rares", action="store_true", default=False,
                   help="overlay: at a card reward whose cards are all rare, take one rather than skipping")
    p.add_argument("--take-card", action="append", default=[], metavar="NAME",
                   help="overlay: always take this card (upgraded copies too) when a reward offers it; repeatable")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--collectors", type=int, default=1, help=">1: parallel collector processes")
    p.add_argument("--in-process", action=argparse.BooleanOptionalAction, default=False,
                   help="run games inside each collecting process (see src/sts_rl/README.md)")
    p.add_argument("--chunk", type=int, default=256, help="worlds per collection chunk (memory versus tail time)")
    p.add_argument("--baseline", help="eval JSON to pair against")
    args = p.parse_args()
    if Path(args.output).exists():
        p.error('Output already contains results; choose a fresh output')
    begin_run(str(args.output)+'.record', vars(args), args.checkpoint, inputs=[args.baseline] if args.baseline else [])
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    elite_above = args.elite_above
    if elite_above is not None:
        elite_above = ({int(a): float(t) for a, t in (part.split(":") for part in elite_above.split(","))}
                       if ":" in elite_above else float(elite_above))
    options = dict(dead_action_filter=args.dead_action_filter, hide_ooc_potions=args.hide_ooc_potions,
                   dominated=args.dominated_filter,
                   forced_play=args.forced_play, smith_above=args.smith_above, prefer_remove=args.prefer_remove,
                   elite_above=elite_above, skip_cards_after=args.skip_cards_after,
                   save_potions_above=args.save_potions_above,
                   avoid_boss_relics=tuple(x for x in args.avoid_boss_relics.split(",") if x),
                   no_skip_rares=args.no_skip_rares, take_cards=tuple(args.take_card), in_process=args.in_process)
    if args.collectors > 1:
        from sts_rl.parallel_collect import CollectorPool
        workers = CollectorPool(args.collectors, workers=args.workers, checkpoint=args.checkpoint, **options)
    else:
        workers = make_workers(args.workers, **options)
    try:
        result = evaluate(model, workers, n=args.n, world_start=args.world_start, mode=args.mode, chunk=args.chunk)
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
