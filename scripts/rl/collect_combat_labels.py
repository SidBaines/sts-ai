#!/usr/bin/env python3
"""P2 combat expert iteration, data step: DAgger labels on run-ending fight types.

Plays the argmax policy on TRAIN worlds. In boss and elite fights, and in Act 2+
hallway fights entered below --low-hp of max HP, every decision is labelled with
native search votes over shuffled hidden draw orders (see PanelEpisode.search_votes).
A fraction of other combat decisions is stored as rehearsal. The policy's own
action is always played (learner state distribution).

--public uses PanelEpisode.search_votes_public (battle RNG streams re-seeded per
sample as well as the draw order shuffled), so labels do not encode foreknowledge
of hidden randomness; it needs the RL build with BattleContext.reseed_rngs
(external/sts_lightspeed_rl/build_pub). --boss-only labels boss fights only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import ORIGINAL_CHECKPOINT, WORKER_CONFIG, FilteringWorker, load_policy, train_tasks  # noqa: E402
from sts_rl.panel import PanelWorker  # noqa: E402

LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--world-start", type=int, default=9_000_000_000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT)
    p.add_argument("--simulations", type=int, default=2000)
    p.add_argument("--draw-orders", type=int, default=3)
    p.add_argument("--low-hp", type=float, default=0.5)
    p.add_argument("--rehearsal-rate", type=float, default=0.05)
    p.add_argument("--public", action="store_true", help="public-information search labels (re-seeded RNG)")
    p.add_argument("--boss-only", action="store_true", help="label only boss fights")
    p.add_argument("--record-dir", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    record_dir = Path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    tasks = iter(train_tasks(args.world_start, args.n))
    task_lock = threading.Lock()
    seeds = list(range(1, args.draw_orders + 1))

    def play(slot: int) -> list[dict]:
        rng = random.Random(slot)
        inner = PanelWorker(config=WORKER_CONFIG, timeout=600.0)
        game = FilteringWorker(inner)
        shard = (record_dir / f"shard_{slot:02d}.jsonl").open("a")
        out = []
        try:
            while True:
                with task_lock:
                    task = next(tasks, None)
                if task is None:
                    return out
                started, labelled = time.time(), 0
                try:
                    obs, info = game.submit("reset", task).result()
                    terminated = truncated = False
                    fight_label = False
                    previous_phase = "out_of_combat"
                    while not (terminated or truncated):
                        state = obs.get("state", {})
                        phase = obs.get("phase")
                        if phase == "combat" and previous_phase != "combat":
                            room = str(state.get("room", ""))
                            hp, max_hp = state.get("cur_hp") or 0, state.get("max_hp") or 80
                            fight_label = ("BOSS" in room if args.boss_only else
                                           ("BOSS" in room or "ELITE" in room or
                                            (int(state.get("act") or 1) >= 2 and hp < args.low_hp * max_hp)))
                        previous_phase = phase
                        with LOCK, torch.no_grad():
                            logits, _, _ = forward_policy(model, collate_public([obs], "cpu"))
                        chosen = obs["legal_actions"][int(logits[0].argmax())]
                        if phase == "combat" and len(obs["legal_actions"]) > 1:
                            if fight_label:
                                search = inner.search_votes_public if args.public else inner.search_votes
                                votes = search(args.simulations, seeds, task.seed % 100000 + labelled)
                                if votes:  # public search abstains on draw-index card selection
                                    shard.write(json.dumps({"type": "combat", "seed": task.seed, "obs": obs,
                                                            "votes": votes, "policy_index": chosen["index"]}) + "\n")
                                    labelled += 1
                            elif rng.random() < args.rehearsal_rate:
                                shard.write(json.dumps({"type": "rehearsal", "seed": task.seed, "obs": obs}) + "\n")
                        obs, _, terminated, truncated, info = game.submit("step", int(chosen["index"])).result()
                    rec = {"seed": task.seed, "floor": obs.get("state", {}).get("floor"), "labelled": labelled,
                           "seconds": time.time() - started}
                except Exception as exc:
                    rec = {"seed": task.seed, "fault": f"{type(exc).__name__}: {exc}"[:200]}
                shard.flush()
                with LOCK:
                    with (record_dir / "progress.jsonl").open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                out.append(rec)
        finally:
            game.close()
            shard.close()

    with ThreadPoolExecutor(args.threads) as ex:
        results = [r for part in ex.map(play, range(args.threads)) for r in part]
    print(json.dumps({"games": len(results), "faults": sum("fault" in r for r in results),
                      "labelled": sum(r.get("labelled", 0) for r in results)}))


if __name__ == "__main__":
    main()
