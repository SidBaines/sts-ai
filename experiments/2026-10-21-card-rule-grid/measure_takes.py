#!/usr/bin/env python3
"""Play the champion greedily under the study protocol and log every card-reward offer.

    measure_takes.py CHECKPOINT WORLD_START N THREADS OUT.jsonl

One line per card-reward menu: world seed, floor, act, the offered cards (name,
type, rarity), the policy's probability for each legal option, and the chosen
description. Out-of-combat potion actions are hidden, as in fast_eval.py.
TRAIN worlds only. Run with the frozen study runtime and STS_LIGHTSPEED_BUILD set.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import threading

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import load_policy, make_workers, train_tasks  # noqa: E402
from sts_env.env import SimulatorError  # noqa: E402

CARD = re.compile(r"^take card (.+?) \[(\w+), (\w+)\]")
LOCK = threading.Lock()


def main() -> None:
    checkpoint, world_start, n, threads, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    torch.set_num_threads(1)
    model, _ = load_policy(checkpoint)
    model.eval()
    tasks = train_tasks(world_start, n)
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = out.open("w")

    def play(slot: int) -> int:
        worker = make_workers(1, dead_action_filter=True, hide_ooc_potions=True)[0]
        games = 0
        for task in tasks[slot::threads]:
            rows = []
            try:
                obs, _ = worker.submit("reset", task).result()
                term = trunc = False
                while not (term or trunc):
                    legal = [a["description"] for a in obs["legal_actions"]]
                    with LOCK, torch.no_grad():
                        logits, _, _ = forward_policy(model, collate_public([obs]))
                    row = logits[0][: len(legal)]
                    pos = int(row.argmax())
                    offers = [(i, m) for i, d in enumerate(legal) if (m := CARD.match(d))]
                    if obs.get("phase") != "combat" and offers:
                        probs = torch.softmax(row, -1).tolist()
                        state = obs.get("state", {})
                        rows.append({"seed": task.seed, "floor": state.get("floor"), "act": state.get("act"),
                                     "offers": [{"name": m[1], "type": m[2], "rarity": m[3], "p": probs[i]} for i, m in offers],
                                     "legal": legal, "p": probs, "chosen": legal[pos]})
                    obs, _, term, trunc, _ = worker.submit("step", int(obs["legal_actions"][pos]["index"])).result()
            except SimulatorError:
                worker.close()
                worker = make_workers(1, dead_action_filter=True, hide_ooc_potions=True)[0]
                continue
            with LOCK:
                for r in rows:
                    sink.write(json.dumps(r) + "\n")
            games += 1
        worker.close()
        return games

    with ThreadPoolExecutor(threads) as ex:
        games = sum(ex.map(play, range(threads)))
    sink.close()
    print(json.dumps({"games": games, "checkpoint": checkpoint, "world_start": world_start, "n": n}))


if __name__ == "__main__":
    main()
