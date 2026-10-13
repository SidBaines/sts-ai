#!/usr/bin/env python3
"""X1: does a fight-panel score at a mid-act state predict the run's real boss outcome?

Samples one out-of-combat map decision per episode from a probe pickle (argmax
original-policy runs with action traces), computes the panel at that state with
K seeds, and reports split-half reliability and AUC for predicting whether the
run then beat that act's boss, versus an HP-only baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import TaskSpec  # noqa: E402
from sts_rl.common import ORIGINAL_CHECKPOINT, WORKER_CONFIG  # noqa: E402
from sts_rl.panel import PanelWorker, default_fights, summarize_panel  # noqa: E402


def episodes(rows):
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if r["terminated"] or r["truncated"]:
            out.append(cur)
            cur = []
    return out


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True)
    p.add_argument("--act", type=int, default=1)
    p.add_argument("--floor-min", type=int, default=6)
    p.add_argument("--floor-max", type=int, default=12)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT)
    p.add_argument("--output", required=True)
    p.add_argument("--sample", action="store_true", help="sampled combat (default greedy, as deployed)")
    p.add_argument("--full-hp-boss", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    boss_floor = {1: 16, 2: 33, 3: 50}[args.act]
    rows = pickle.load(open(args.probe, "rb"))["rows"]
    rng = random.Random(0)
    cands = []
    for ep in episodes(rows):
        idx = [i for i, r in enumerate(ep) if r["phase"] != "combat" and r["kind"] == "map"
               and args.floor_min <= (r["floor"] or 0) <= args.floor_max and r["act"] == args.act]
        if idx:
            i = rng.choice(idx)
            final = ep[-1]
            beat = (final["floor"] or 0) > boss_floor or final["reward"] > 0.5
            cands.append({"seed": ep[0]["seed"], "prefix": [r["action_index"] for r in ep[:i]], "floor": ep[i]["floor"],
                          "hp": ep[i]["hp"], "beat_boss": bool(beat), "final_floor": final["floor"]})
    rng.shuffle(cands)
    cands = cands[: args.states]
    workers = [PanelWorker(config=WORKER_CONFIG, timeout=600) for _ in range(args.workers)]

    def run(i):
        c = cands[i]
        w = workers[i % len(workers)]
        spec = TaskSpec(seed=c["seed"], kind="run", prefix=tuple(c["prefix"]), max_act=3, horizon=4096)
        enc = w.encounters(spec)
        res = w.panel(spec, default_fights(enc, args.samples), args.checkpoint, args.sample,
                      ("BOSS",) if args.full_hp_boss else ())
        half = [summarize_panel([r for r in res if r.get("salt", 0) % 2 == k]) for k in (0, 1)]
        return {**c, "prefix": None, "encounters": enc, "panel": summarize_panel(res), "half": half, "raw": res}

    started = time.time()
    out = []
    try:
        with ThreadPoolExecutor(len(workers)) as ex:
            for b in range(0, len(cands), len(workers)):
                out.extend(ex.map(run, range(b, min(b + len(workers), len(cands)))))
                print(f"{len(out)}/{len(cands)} {time.time() - started:.0f}s", flush=True)
    finally:
        for w in workers:
            w.close()
    Path(args.output).write_text(json.dumps(out))
    labels = [o["beat_boss"] for o in out]
    report = {"states": len(out), "base_rate": float(np.mean(labels)), "seconds": time.time() - started}
    for key in ("boss_win", "elite_win", "hallway_win"):
        s = [o["panel"][key] if o["panel"][key] is not None else 0.0 for o in out]
        a = [o["half"][0][key] or 0.0 for o in out]
        b = [o["half"][1][key] or 0.0 for o in out]
        report[key] = {"auc": auc(s, labels), "split_half_r": float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None,
                       "mean": float(np.mean(s))}
    hp_neg_loss = [-(o["panel"]["boss_hp_loss"] or 0) for o in out]
    report["boss_hp_left"] = {"auc": auc(hp_neg_loss, labels)}
    report["entry_hp_only"] = {"auc": auc([o["hp"] or 0 for o in out], labels)}
    combo = [(o["panel"]["boss_win"] or 0) + 0.5 * (o["panel"]["elite_win"] or 0) for o in out]
    report["boss+elite"] = {"auc": auc(combo, labels)}
    report["faults"] = sum(o["panel"]["faults"] for o in out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
