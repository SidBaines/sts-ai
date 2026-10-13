#!/usr/bin/env python3
"""Bank of learner-reached fight starts (boss / elite) as replayable battle tasks.

Plays a checkpoint (exact argmax) on TRAIN worlds and records, for every fight in
the selected rooms, the native action prefix at the fight's first decision, plus
the run's actual outcome of that fight. Output: JSONL of TaskSpec dicts + metadata.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import TaskSpec  # noqa: E402
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_policy.workers import CurriculumWorker  # noqa: E402
from sts_rl.common import WORKER_CONFIG, FilteringWorker, load_policy, train_tasks  # noqa: E402

from sts_rl.recording import begin_run
from sts_env.env import SimulatorError

LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--world-start", type=int, default=11_000_000_000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--rooms", default="BOSS,ELITE")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error('Output already contains results; choose a fresh output')
    begin_run(str(args.output)+'.record', vars(args), args.checkpoint)
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    rooms = args.rooms.split(",")
    tasks = iter(train_tasks(args.world_start, args.n))
    lock = threading.Lock()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = out.open("w")

    def play(slot: int) -> int:
        game = FilteringWorker(CurriculumWorker(config=WORKER_CONFIG, timeout=30.0))
        count = 0
        try:
            while True:
                with lock:
                    task = next(tasks, None)
                if task is None:
                    return count
                try:
                    obs, info = game.submit("reset", task).result()
                    actions, previous_phase, open_fight = [], "out_of_combat", None
                    world_rows = []
                    terminated = truncated = False
                    while not (terminated or truncated):
                        state = obs.get("state", {})
                        if obs.get("phase") == "combat" and previous_phase != "combat":
                            room = str(state.get("room", "")).rsplit(".", 1)[-1]
                            if room in rooms:
                                open_fight = {"task": TaskSpec(seed=task.seed, kind="battle", prefix=tuple(actions),
                                                               seek_combat=False, max_act=3, horizon=4096).to_dict(),
                                              "room": room, "floor": state.get("floor"), "act": state.get("act"),
                                              "hp": state.get("cur_hp"), "max_hp": state.get("max_hp")}
                        if obs.get("phase") != "combat" and previous_phase == "combat" and open_fight:
                            open_fight["won"] = True
                            world_rows.append(open_fight)
                            open_fight = None
                        previous_phase = obs.get("phase")
                        with LOCK, torch.no_grad():
                            logits, _, _ = forward_policy(model, collate_public([obs], "cpu"))
                        action = int(obs["legal_actions"][int(logits[0].argmax())]["index"])
                        obs, _, terminated, truncated, info = game.submit("step", action).result()
                        actions.append(action)
                    if open_fight:  # the run ended inside this fight
                        open_fight["won"] = info.get("task_outcome") in ("victory", "act_clear")
                        world_rows.append(open_fight)
                    with LOCK:
                        for row in world_rows:
                            sink.write(json.dumps(row)+'\n')
                        sink.flush()
                    count += len(world_rows)
                except Exception as exc:
                    if not isinstance(exc, SimulatorError):
                        raise
                    print("fault", task.seed, type(exc).__name__, str(exc)[:100], flush=True)
        finally:
            game.close()

    try:
        with ThreadPoolExecutor(args.threads) as ex:
            total = sum(ex.map(play, range(args.threads)))
    finally:
        sink.close()
    print(json.dumps({"fights": total}))


if __name__ == "__main__":
    main()
