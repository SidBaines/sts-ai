"""Paired base-vs-trained analysis of data/eval_iter1 (same world seeds 600-659).

The report compared marginal means (+1.15 floor ~= 1 SE). Because both arms ran
the identical world-seed block, the within-seed delta removes game-difficulty
variance and gives a much tighter CI. No GPU needed.
"""
from __future__ import annotations

import glob
import json
import math
import os

EVAL = os.path.join(os.path.dirname(__file__), "..", "data", "eval_iter1")


def load(arm: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in glob.glob(os.path.join(EVAL, arm, "*", "*.meta.json")):
        with open(path) as f:
            m = json.load(f)
        out[m["world_seed"]] = m
    return out


def boss_clear(m: dict) -> int:
    # Same rule as reward.py: cleared act 1 boss => reached act 2+ or VICTORY.
    return int(m["final_act"] > 1 or "VICTORY" in m["outcome"])


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> None:
    base = load("base")
    trained = load("trained")
    seeds = sorted(set(base) & set(trained))
    print(f"paired seeds: {len(seeds)} (base {len(base)}, trained {len(trained)})\n")

    floor_d, hp_d = [], []
    bc_b = bc_t = win_b = win_t = 0
    wins = losses = ties = 0
    for s in seeds:
        b, t = base[s], trained[s]
        df = t["final_floor"] - b["final_floor"]
        floor_d.append(df)
        hp_d.append(t["final_hp"] - b["final_hp"])
        bc_b += boss_clear(b); bc_t += boss_clear(t)
        win_b += "VICTORY" in b["outcome"]; win_t += "VICTORY" in t["outcome"]
        if df > 0: wins += 1
        elif df < 0: losses += 1
        else: ties += 1

    n = len(seeds)
    md = mean(floor_d)
    se = stdev(floor_d) / math.sqrt(n)
    t_stat = md / se if se else float("nan")
    print("=== floor (trained - base), paired ===")
    print(f"  mean base floor    : {mean([base[s]['final_floor'] for s in seeds]):.2f}")
    print(f"  mean trained floor : {mean([trained[s]['final_floor'] for s in seeds]):.2f}")
    print(f"  mean paired delta  : {md:+.2f}")
    print(f"  SE of delta        : {se:.2f}")
    print(f"  95% CI             : [{md - 1.96*se:+.2f}, {md + 1.96*se:+.2f}]")
    print(f"  paired t           : {t_stat:.2f}  (|t|>2 ~ p<0.05)")
    print(f"  per-seed: trained higher {wins}, lower {losses}, tie {ties}")
    # Sign test (binomial, ignoring ties)
    nz = wins + losses
    # two-sided p for >= wins under p=0.5
    from math import comb
    p_sign = 2 * sum(comb(nz, k) for k in range(wins, nz + 1)) / (2 ** nz) if nz else float("nan")
    print(f"  sign test (ignore ties): p = {min(p_sign,1.0):.3f}")
    print(f"\n=== final HP (trained - base), paired ===")
    print(f"  mean paired delta  : {mean(hp_d):+.2f}")
    print(f"\n=== boss-clear / wins ===")
    print(f"  boss-clear base {bc_b}/{n} ({bc_b/n:.0%})  trained {bc_t}/{n} ({bc_t/n:.0%})")
    print(f"  wins       base {win_b}/{n}             trained {win_t}/{n}")


if __name__ == "__main__":
    main()
