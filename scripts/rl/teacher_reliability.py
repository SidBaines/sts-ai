#!/usr/bin/env python3
"""Is the teacher's per-decision ranking signal or noise?

For sampled card-reward decisions from a probe pickle, score every card option's
afterstate with 2K salts, split salts into two halves, and correlate the option
score *differences* (vs the policy's own choice) across halves. Near-zero means a
single teacher call cannot rank options; positive means usable per-decision signal.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.curriculum import TaskSpec  # noqa: E402
from sts_rl.common import ORIGINAL_CHECKPOINT, WORKER_CONFIG  # noqa: E402
from sts_rl.panel import PanelWorker  # noqa: E402
from sts_rl.teacher import is_card_choice, score_fights  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True)
    p.add_argument("--decisions", type=int, default=60)
    p.add_argument("--k", type=int, default=8, help="salts per half")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = pickle.load(open(args.probe, "rb"))["rows"]
    eps, cur = [], []
    for r in rows:
        cur.append(r)
        if r["terminated"] or r["truncated"]:
            eps.append(cur)
            cur = []
    rng = random.Random(0)
    cands = []
    for ep in eps:
        idx = [i for i, r in enumerate(ep) if r["phase"] != "combat" and r["legal"] and r["n_legal"] > 1
               and is_card_choice(r["action"]) and sum(is_card_choice(a) for a in r["legal"]) > 1]
        if idx:
            i = rng.choice(idx)
            r = ep[i]
            cands.append({"seed": r["seed"], "prefix": [x["action_index"] for x in ep[:i]], "floor": r["floor"],
                          "options": [k for k, a in enumerate(r["legal"]) if is_card_choice(a)],
                          "labels": [a for a in r["legal"] if is_card_choice(a)], "policy": r["action_index"]})
    rng.shuffle(cands)
    cands = cands[: args.decisions]
    workers = [PanelWorker(config=WORKER_CONFIG, timeout=1800) for _ in range(args.workers)]

    def run(i):
        c, w = cands[i], workers[i % len(workers)]
        spec = TaskSpec(seed=c["seed"], kind="run", prefix=tuple(c["prefix"]), max_act=3, horizon=4096, seek_combat=False)
        menu = w.menu(spec)
        options = [[int(menu[k]["index"])] for k in c["options"]]
        enc = w.encounters(spec)
        fights = [(enc["boss"], "BOSS", s) for s in range(2 * args.k)]
        fights += [(e, "ELITE", s) for e in enc["elites"][:2] for s in range(args.k)]
        fights += [(e, "MONSTER", s) for e in enc["hallway"][:2] for s in range(2)]
        res = w.afterstates(spec, options, fights, ORIGINAL_CHECKPOINT, False, ["BOSS"])
        halves = [[score_fights([f for f in r.get("fights", []) if f["salt"] % 2 == h]) for r in res] for h in (0, 1)]
        return {**c, "prefix": None, "halves": halves, "full": [score_fights(r.get("fights", [])) for r in res]}

    with ThreadPoolExecutor(args.workers) as ex:
        out = list(ex.map(run, range(len(cands))))
    for w in workers:
        w.close()
    Path(args.output).write_text(json.dumps(out))
    a, b = [], []
    agree = []
    for o in out:
        h0, h1 = o["halves"]
        if any(x is None for x in h0 + h1):
            continue
        base0, base1 = np.mean(h0), np.mean(h1)
        a += [x - base0 for x in h0]
        b += [x - base1 for x in h1]
        agree.append(int(np.argmax(h0) == np.argmax(h1)))
    print(json.dumps({"decisions": len(agree), "option_diff_split_half_r": float(np.corrcoef(a, b)[0, 1]),
                      "argmax_agreement_between_halves": float(np.mean(agree)),
                      "chance_agreement": float(np.mean([1 / len(o["options"]) for o in out]))}))


if __name__ == "__main__":
    main()
