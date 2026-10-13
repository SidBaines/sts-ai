#!/usr/bin/env python3
"""X2: play the original policy, but let the fight-panel teacher make card-reward choices.

Each thread owns a game worker (dead-action-filtered, as in the baseline) and a
PanelWorker. When the policy's argmax at an OOC screen is a card choice (take a
card / skip the card reward), the teacher scores every card option's afterstate
on identical salted fights and plays the best. Everything else is the unchanged
argmax policy. Per-world records are written for paired comparison with the
argmax baseline on the same seeds. --no-teacher reproduces the baseline loop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import TaskSpec  # noqa: E402
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import (EVAL_WORLD_START, ORIGINAL_CHECKPOINT, WORKER_CONFIG, FilteringWorker,  # noqa: E402
                           load_policy, train_tasks)
from sts_rl.evaluate import paired, summarize  # noqa: E402
from sts_rl.panel import PanelWorker  # noqa: E402
from sts_policy.workers import CurriculumWorker  # noqa: E402

LOCK = threading.Lock()


def is_card_choice(description: str) -> bool:
    return description.startswith("take card ") or description.startswith("skip card")


def teacher_fights(encounters: dict, boss_samples: int, elite_samples: int, hallway: int) -> list:
    fights = [(encounters["boss"], "BOSS", k) for k in range(boss_samples)]
    for e in encounters["elites"][:2]:
        fights += [(e, "ELITE", k) for k in range(elite_samples)]
    fights += [(e, "MONSTER", 0) for e in encounters["hallway"][:hallway]]
    return fights


def score(fights: list[dict]) -> float | None:
    valid = [f for f in fights if "fault" not in f and not f.get("ub")]
    if not valid:
        return None
    def mean(room, key):
        rows = [f for f in valid if f["room"] == room]
        if not rows:
            return 0.0
        if key == "win":
            return sum(f["win"] for f in rows) / len(rows)
        return sum(f["hp_after"] / max(1, f["max_hp"]) for f in rows) / len(rows)
    return (mean("BOSS", "win") + 0.5 * mean("ELITE", "win") + 0.25 * mean("BOSS", "hp")
            + 0.25 * mean("ELITE", "hp") + 0.1 * mean("MONSTER", "hp"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--world-start", type=int, default=EVAL_WORLD_START)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT)
    p.add_argument("--panel-checkpoint", default=ORIGINAL_CHECKPOINT, help="combat policy used inside panel fights")
    p.add_argument("--teacher", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-act-teacher", type=int, default=3)
    p.add_argument("--boss-samples", type=int, default=4)
    p.add_argument("--elite-samples", type=int, default=2)
    p.add_argument("--hallway", type=int, default=2)
    p.add_argument("--baseline", default="data/rl_runs/baselines/orig_argmax_1024.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    tasks = train_tasks(args.world_start, args.n)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = out_path.with_suffix(".jsonl").open("a")

    def play(task_iter, slot):
        game = FilteringWorker(CurriculumWorker(config=WORKER_CONFIG, timeout=30.0))
        panel = PanelWorker(config=WORKER_CONFIG, timeout=900.0) if args.teacher else None
        records = []
        try:
            for task in task_iter:
                started = time.time()
                try:
                    obs, info = game.submit("reset", task).result()
                    actions, steps, teacher_calls, changed = [], 0, 0, 0
                    trajectory_hp = {}
                    terminated = truncated = False
                    while not (terminated or truncated):
                        legal = obs["legal_actions"]
                        with LOCK, torch.no_grad():
                            logits, _, _ = forward_policy(model, collate_public([obs], "cpu"))
                        position = int(logits[0].argmax())
                        chosen = legal[position]
                        state = obs.get("state", {})
                        if obs.get("phase") == "combat" and state.get("floor") in (16, 33, 50):
                            trajectory_hp.setdefault(state["floor"], state.get("cur_hp"))
                        if (panel is not None and obs.get("phase") != "combat" and len(legal) > 1
                                and is_card_choice(chosen["description"])
                                and int(state.get("act") or 0) <= args.max_act_teacher):
                            options = [a for a in legal if is_card_choice(a["description"])]
                            if len(options) > 1:
                                spec = TaskSpec(seed=task.seed, kind="run", prefix=tuple(actions), max_act=3,
                                                horizon=4096, seek_combat=False)
                                enc = panel.encounters(spec)
                                fights = teacher_fights(enc, args.boss_samples, args.elite_samples, args.hallway)
                                scored = panel.afterstates(spec, [[a["index"]] for a in options], fights,
                                                           args.panel_checkpoint, False, ["BOSS"])
                                best, best_score = None, None
                                for a, r in zip(options, scored):
                                    s = score(r.get("fights", []))
                                    if s is not None and (best_score is None or s > best_score + 1e-9):
                                        best, best_score = a, s
                                teacher_calls += 1
                                if best is not None and best["index"] != chosen["index"]:
                                    chosen, changed = best, changed + 1
                        obs, reward, terminated, truncated, info = game.submit("step", int(chosen["index"])).result()
                        actions.append(int(chosen["index"]))
                        steps += 1
                    state = obs.get("state", {})
                    rec = {"seed": task.seed, "floor": state.get("floor"),
                           "win": info.get("task_outcome") in ("victory", "act_clear"), "steps": steps,
                           "boss16_hp": trajectory_hp.get(16), "boss33_hp": trajectory_hp.get(33),
                           "boss50_hp": trajectory_hp.get(50), "teacher_calls": teacher_calls,
                           "teacher_changed": changed, "seconds": time.time() - started}
                except Exception as exc:
                    rec = {"seed": task.seed, "fault": f"{type(exc).__name__}: {exc}"[:200]}
                with LOCK:
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                records.append(rec)
        finally:
            game.close()
            if panel is not None:
                panel.close()
        return records

    shared = iter(tasks)
    lock = threading.Lock()

    def task_iter():
        while True:
            with lock:
                task = next(shared, None)
            if task is None:
                return
            yield task

    started = time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        results = [r for part in ex.map(lambda s: play(task_iter(), s), range(args.threads)) for r in part]
    valid = [r for r in results if "fault" not in r]
    summary = summarize(valid)
    summary.update(faults=len(results) - len(valid), seconds=time.time() - started, teacher=args.teacher,
                   teacher_changed=int(sum(r["teacher_changed"] for r in valid)),
                   teacher_calls=int(sum(r["teacher_calls"] for r in valid)))
    result = {"summary": summary, "records": valid, "config": vars(args)}
    if args.baseline and Path(args.baseline).exists():
        result["paired"] = paired(json.loads(Path(args.baseline).read_text())["records"], valid)
    out_path.write_text(json.dumps(result))
    print(json.dumps({**summary, **({"paired": result.get("paired")} if "paired" in result else {})}))


if __name__ == "__main__":
    main()
