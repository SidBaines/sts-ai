#!/usr/bin/env python3
"""Play the policy with the fight-panel teacher making selected OOC decisions.

Two uses of one loop:
  * evaluation  (eval worlds, --output): paired with the argmax baseline
  * label collection (TRAIN worlds, --record-dir): every teacher decision is
    written with its observation and per-option scores; with --rehearsal-rate,
    a fraction of other OOC observations is stored for behaviour rehearsal.

Decision types: card (take card / skip card reward) and rest (rest vs smith+card).
The policy is an exact argmax; the teacher only acts when the policy's argmax is
of a covered type, so reward pickup order etc. stay the policy's.
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

from sts_policy.curriculum import TaskSpec  # noqa: E402
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_policy.workers import CurriculumWorker  # noqa: E402
from sts_rl.common import (EVAL_WORLD_START, ORIGINAL_CHECKPOINT, WORKER_CONFIG, FilteringWorker,  # noqa: E402
                           load_policy, train_tasks)
from sts_rl.evaluate import paired, summarize  # noqa: E402
from sts_rl.panel import PanelWorker  # noqa: E402
from sts_rl.teacher import DECISION_TYPES, expand_rest_options, teach  # noqa: E402

LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--world-start", type=int, default=EVAL_WORLD_START)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT, help="policy playing the game")
    p.add_argument("--panel-checkpoint", default=ORIGINAL_CHECKPOINT, help="combat policy inside panel fights")
    p.add_argument("--types", default="card", help="comma list of: card, rest")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("--boss-samples", type=int, default=4)
    p.add_argument("--elite-samples", type=int, default=2)
    p.add_argument("--hallway", type=int, default=2)
    p.add_argument("--baseline", default="data/rl_runs/baselines/orig_argmax_1024.json")
    p.add_argument("--output", help="eval result JSON (records + summary)")
    p.add_argument("--record-dir", help="write teacher labels / rehearsal observations here")
    p.add_argument("--rehearsal-rate", type=float, default=0.0)
    args = p.parse_args()
    types = [t for t in args.types.split(",") if t]
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    tasks = train_tasks(args.world_start, args.n)
    record_dir = Path(args.record_dir) if args.record_dir else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
    progress = Path(args.output or (str(record_dir) + "/progress")).with_suffix(".jsonl")
    progress.parent.mkdir(parents=True, exist_ok=True)

    def kind_of(description: str) -> str | None:
        return next((t for t in types if DECISION_TYPES[t](description)), None)

    def play(task_iter, slot):
        rng = random.Random(slot)
        game = FilteringWorker(CurriculumWorker(config=WORKER_CONFIG, timeout=30.0))
        panel = None if args.no_teacher else PanelWorker(config=WORKER_CONFIG, timeout=1800.0)
        shard = (record_dir / f"shard_{slot:02d}.jsonl").open("a") if record_dir else None
        records = []
        try:
            for task in task_iter:
                started = time.time()
                try:
                    obs, info = game.submit("reset", task).result()
                    actions, steps, calls, changed = [], 0, 0, 0
                    boss_hp: dict[int, int] = {}
                    pending_follow: int | None = None
                    terminated = truncated = False
                    while not (terminated or truncated):
                        legal = obs["legal_actions"]
                        state = obs.get("state", {})
                        if obs.get("phase") == "combat" and state.get("floor") in (16, 33, 50):
                            boss_hp.setdefault(state["floor"], state.get("cur_hp"))
                        with LOCK, torch.no_grad():
                            logits, _, _ = forward_policy(model, collate_public([obs], "cpu"))
                        chosen = legal[int(logits[0].argmax())]
                        if pending_follow is not None:
                            # second step of a teacher-chosen [smith, card] option
                            follow = next((a for a in legal if a["index"] == pending_follow), None)
                            if follow is not None:
                                if shard is not None:
                                    shard.write(json.dumps({"type": "rest_select", "seed": task.seed, "obs": obs,
                                                            "label_index": pending_follow,
                                                            "policy_index": chosen["index"]}) + "\n")
                                changed += follow["index"] != chosen["index"]
                                chosen = follow
                            pending_follow = None
                        elif panel is not None and obs.get("phase") != "combat" and len(legal) > 1:
                            kind = kind_of(chosen["description"])
                            if kind is not None:
                                spec = TaskSpec(seed=task.seed, kind="run", prefix=tuple(actions), max_act=3,
                                                horizon=4096, seek_combat=False)
                                if kind == "card":
                                    options = [[a["index"]] for a in legal if DECISION_TYPES["card"](a["description"])]
                                else:
                                    options = expand_rest_options(panel, spec, legal)
                                if len(options) > 1:
                                    scored = teach(panel, spec, options, args.panel_checkpoint,
                                                   boss_samples=args.boss_samples, elite_samples=args.elite_samples,
                                                   hallway=args.hallway)
                                    valid = [s for s in scored if s["score"] is not None]
                                    calls += 1
                                    if valid:
                                        best = max(valid, key=lambda s: s["score"])
                                        first = best["option"][0]
                                        if shard is not None:
                                            shard.write(json.dumps({"type": kind, "seed": task.seed, "obs": obs,
                                                                    "scores": scored, "label_index": first,
                                                                    "policy_index": chosen["index"]}) + "\n")
                                        changed += first != chosen["index"]
                                        chosen = next(a for a in legal if a["index"] == first)
                                        if len(best["option"]) > 1:
                                            pending_follow = best["option"][1]
                            elif shard is not None and rng.random() < args.rehearsal_rate:
                                shard.write(json.dumps({"type": "rehearsal", "seed": task.seed, "obs": obs}) + "\n")
                        obs, _, terminated, truncated, info = game.submit("step", int(chosen["index"])).result()
                        actions.append(int(chosen["index"]))
                        steps += 1
                    state = obs.get("state", {})
                    rec = {"seed": task.seed, "floor": state.get("floor"),
                           "win": info.get("task_outcome") in ("victory", "act_clear"), "steps": steps,
                           "boss16_hp": boss_hp.get(16), "boss33_hp": boss_hp.get(33), "boss50_hp": boss_hp.get(50),
                           "teacher_calls": calls, "teacher_changed": changed, "seconds": time.time() - started}
                except Exception as exc:
                    rec = {"seed": task.seed, "fault": f"{type(exc).__name__}: {exc}"[:200]}
                with LOCK:
                    with progress.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    if shard is not None:
                        shard.flush()
                records.append(rec)
        finally:
            game.close()
            if panel is not None:
                panel.close()
            if shard is not None:
                shard.close()
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
    summary.update(faults=len(results) - len(valid), seconds=time.time() - started, types=types,
                   teacher=not args.no_teacher, teacher_changed=int(sum(r["teacher_changed"] for r in valid)),
                   teacher_calls=int(sum(r["teacher_calls"] for r in valid)))
    result = {"summary": summary, "records": valid, "config": vars(args)}
    if args.baseline and Path(args.baseline).exists() and args.world_start == EVAL_WORLD_START:
        result["paired"] = paired(json.loads(Path(args.baseline).read_text())["records"], valid)
    if args.output:
        Path(args.output).write_text(json.dumps(result))
    print(json.dumps({**summary, **({"paired": result["paired"]} if "paired" in result else {})}))


if __name__ == "__main__":
    main()
