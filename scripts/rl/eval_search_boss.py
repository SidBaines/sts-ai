#!/usr/bin/env python3
"""Full-game canonical RNG-resampled native-search evaluation.

Boss/elite search reseeds battle RNGs, canonicalizes unordered pile groups and
uses public-observation-derived sampling seeds. Other latent native fields are
not certified public-state-only; report this as its own simulator condition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
import math

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import public_hash
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import (BOSS_FLOORS, EVAL_WORLD_START, WIN_OUTCOMES, WORKER_CONFIG, FilteringWorker,  # noqa: E402
                           JsonlLog, load_policy, train_tasks)
from sts_rl.evaluate import finish_evaluation, paired  # noqa: E402
from sts_rl.panel import PanelWorker  # noqa: E402

from sts_rl.recording import begin_run
from sts_env.env import SimulatorError

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
    p.add_argument('--progress-output', required=True, help='JSONL progress log; use the queue job log path')
    p.add_argument('--seconds',type=float,default=7200.,help='Finite total collection budget; unstarted worlds remain explicit')
    p.add_argument('--query-timeout',type=float,default=60.)
    args = p.parse_args()
    if args.n<1 or args.threads<1 or args.samples<1 or args.simulations<1 or not math.isfinite(args.seconds) or args.seconds<=0 or not math.isfinite(args.query_timeout) or args.query_timeout<=0:
        p.error('Search evaluation requires positive finite bounds')
    if Path(args.output).exists():
        p.error('Output already contains results; choose a fresh output')
    begin_run(str(args.output)+'.record', vars(args), args.checkpoint, inputs=[args.baseline] if args.baseline else [])
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    rooms = tuple(args.rooms.split(","))
    tasks = train_tasks(args.world_start, args.n)
    started = time.time()
    progress=JsonlLog(args.progress_output)
    completed=[0]
    progress.emit({'event':'start','config':vars(args)})

    def run(slot: int) -> tuple[list[dict], int]:
        inner = PanelWorker(config=WORKER_CONFIG, timeout=args.query_timeout)
        game = FilteringWorker(inner, dead_actions=True, slim=False)
        records, faults = [], 0
        try:
            for task in tasks[slot::args.threads]:
                if time.time()-started>=args.seconds:
                    break
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
                                                              int(public_hash(obs)[:8], 16))
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
                    progress_record=rec
                except SimulatorError as exc:
                    faults += 1
                    progress_record={'seed':task.seed,'fault':str(exc),'fault_record_path':inner.last_fault_path}
                    game.close()
                    inner = PanelWorker(config=WORKER_CONFIG, timeout=args.query_timeout)
                    game = FilteringWorker(inner, dead_actions=True, slim=False)
                with LOCK:
                    completed[0]+=1
                    progress.emit({'event':'world','completed':completed[0],'requested':args.n,
                                   'elapsed_seconds':time.time()-started,**progress_record})
        finally:
            game.close()
        return records, faults

    with ThreadPoolExecutor(args.threads) as ex:
        parts = list(ex.map(run, range(args.threads)))
    records = [r for rs, _ in parts for r in rs]
    if not records:
        raise RuntimeError('No valid search evaluation worlds; inspect progress/fault records')
    result = finish_evaluation(records, sum(f for _, f in parts), time.time() - started,
                               mode=f"greedy+public_search[{args.rooms}]", world_start=args.world_start)
    result["summary"].update(checkpoint=args.checkpoint, simulations=args.simulations, samples=args.samples)
    result['summary'].update(requested=args.n,unstarted=args.n-completed[0],cohort_completed=completed[0]==args.n)
    if args.baseline:
        result["paired"] = paired(json.loads(Path(args.baseline).read_text())["records"], records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result))
    progress.emit({'event':'done','output':args.output,**result['summary']})
    print(json.dumps({**result["summary"], **({"paired": result["paired"]} if "paired" in result else {})}))


if __name__ == "__main__":
    main()
