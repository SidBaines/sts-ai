#!/usr/bin/env python3
"""Full-game evaluation of a search-augmented agent: greedy policy everywhere,
public-information search votes for combat decisions in BOSS rooms.

Public search (PanelWorker.search_votes_public) re-seeds every battle RNG stream
and shuffles the draw order in each sampled copy, so it does not foresee hidden
randomness. Needs the RL build with BattleContext.copy/reseed_rngs
(STS_LIGHTSPEED_BUILD=external/sts_lightspeed_rl/build_pub). Records match
sts_rl.evaluate (seed, floor, win, steps, boss*_hp), so compare_evals.py pairs them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import (BOSS_FLOORS, EVAL_WORLD_START, WIN_OUTCOMES, WORKER_CONFIG, FilteringWorker,  # noqa: E402
                           load_policy, train_tasks)
from sts_rl.evaluate import finish_evaluation, paired  # noqa: E402
from sts_rl.panel import PanelWorker  # noqa: E402

LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--world-start", type=int, default=EVAL_WORLD_START)
    p.add_argument("--simulations", type=int, default=1000)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--rooms", default="BOSS", help="rooms whose fights use search (e.g. BOSS or BOSS,ELITE)")
    p.add_argument("--min-act", type=int, default=1, help="search only in fights from this act on")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--baseline")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    rooms = tuple(args.rooms.split(","))
    tasks = train_tasks(args.world_start, args.n)
    started = time.time()

    def run(slot: int) -> tuple[list[dict], int]:
        inner = PanelWorker(config=WORKER_CONFIG, timeout=1800.0)
        game = FilteringWorker(inner, dead_actions=True, slim=False)
        records, faults = [], 0
        try:
            for task in tasks[slot::args.threads]:
                try:
                    obs, info = game.submit("reset", task).result()
                    term = trunc = False
                    steps, boss_hp = 0, {}
                    while not (term or trunc):
                        legal = obs["legal_actions"]
                        state = obs.get("state", {})
                        room = str(state.get("room", "")).rsplit(".", 1)[-1]
                        if obs.get("phase") == "combat":
                            floor = state.get("floor")
                            if floor in BOSS_FLOORS and floor not in boss_hp:
                                boss_hp[floor] = state.get("cur_hp")
                        votes = {}
                        if (obs.get("phase") == "combat" and room in rooms and len(legal) > 1
                                and int(state.get("act") or 1) >= args.min_act):
                            votes = inner.search_votes_public(args.simulations, list(range(1, args.samples + 1)),
                                                              task.seed % 100_000 + steps)
                        if votes:
                            action = max(votes.items(), key=lambda kv: kv[1])[0]
                        else:  # policy (also the fallback when search abstains)
                            with LOCK, torch.no_grad():
                                logits, _, _ = forward_policy(model, collate_public([obs]))
                            action = int(legal[int(logits[0][: len(legal)].argmax())]["index"])
                        obs, _, term, trunc, info = game.submit("step", int(action)).result()
                        steps += 1
                    rec = {"seed": task.seed, "floor": obs.get("state", {}).get("floor"),
                           "win": info.get("task_outcome") in WIN_OUTCOMES, "steps": steps}
                    for boss in BOSS_FLOORS:
                        rec[f"boss{boss}_hp"] = boss_hp.get(boss)
                    records.append(rec)
                except Exception:
                    faults += 1
                    game.close()
                    inner = PanelWorker(config=WORKER_CONFIG, timeout=1800.0)
                    game = FilteringWorker(inner, dead_actions=True, slim=False)
        finally:
            game.close()
        return records, faults

    with ThreadPoolExecutor(args.threads) as ex:
        parts = list(ex.map(run, range(args.threads)))
    records = [r for rs, _ in parts for r in rs]
    result = finish_evaluation(records, sum(f for _, f in parts), time.time() - started,
                               mode=f"greedy+public_search[{args.rooms}]", world_start=args.world_start)
    result["summary"].update(checkpoint=args.checkpoint, simulations=args.simulations, samples=args.samples)
    if args.baseline:
        result["paired"] = paired(json.loads(Path(args.baseline).read_text())["records"], records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result))
    print(json.dumps({**result["summary"], **({"paired": result["paired"]} if "paired" in result else {})}))


if __name__ == "__main__":
    main()
