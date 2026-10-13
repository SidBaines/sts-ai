#!/usr/bin/env python3
"""Value calibration / advantage-bias analysis of rollout_probe output."""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

import numpy as np


def episodes(rows):
    """Split contiguous per-episode blocks (collect_rollouts appends whole trajectories)."""
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if r["terminated"] or r["truncated"]:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def ev(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    var = target.var()
    return float(1 - (target - pred).var() / var) if var > 0 else float("nan")


def main(path):
    data = pickle.load(open(path, "rb"))
    rows = data["rows"]
    eps = episodes(rows)
    for ep in eps:
        g = 0.0
        for r in reversed(ep):
            g += r["reward"]
            r["mc"] = g
        for i, r in enumerate(ep):
            nxt = ep[i + 1] if i + 1 < len(ep) else None
            r["next_phase"] = nxt["phase"] if nxt else "END"
            r["delta"] = r["reward"] + (0.0 if r["terminated"] else r["next_value"]) - r["value"]
    print(f"== {path.split('/')[-1]}  episodes={len(eps)} transitions={len(rows)}")
    by = defaultdict(list)
    for r in rows:
        by[r["phase"]].append(r)
        by["kind:" + r["kind"]].append(r)
        by[f"act{r['act']}:{r['phase']}"].append(r)
    print(f"{'group':28s} {'n':>7s} {'frac':>6s} {'forced':>6s} {'EV(V,MC)':>8s} {'corr(A,MC-V)':>12s} {'A_mean':>7s} {'A_std':>7s} {'|A|':>6s} {'d_mean':>7s}")
    total = len(rows)
    for key in sorted(by, key=lambda k: (not k.startswith(("combat", "out")), k)):
        g = by[key]
        v = np.array([r["value"] for r in g]); mc = np.array([r["mc"] for r in g])
        a = np.array([r["adv"] for r in g]); d = np.array([r["delta"] for r in g])
        forced = np.mean([r["n_legal"] == 1 for r in g])
        c = np.corrcoef(a, mc - v)[0, 1] if len(g) > 2 else float("nan")
        print(f"{key:28s} {len(g):7d} {len(g)/total:6.3f} {forced:6.2f} {ev(v, mc):8.3f} {c:12.3f} {a.mean():7.3f} {a.std():7.3f} {np.abs(a).mean():6.3f} {d.mean():7.4f}")
    # phase-boundary TD residual bias
    print("\nTD residual by transition type (phase -> next phase):")
    tb = defaultdict(list)
    for r in rows:
        tb[(r["phase"], r["next_phase"], r["kind"] if r["phase"] != "combat" else "combat")].append(r["delta"])
    for k, d in sorted(tb.items(), key=lambda kv: -len(kv[1])):
        d = np.array(d)
        print(f"  {str(k):60s} n={len(d):7d} mean_delta={d.mean():+.4f}  se={d.std()/np.sqrt(len(d)):.4f}")
    # Value at start vs realized return
    starts = np.array([ep[0]["value"] for ep in eps]); rets = np.array([ep[0]["mc"] for ep in eps])
    print(f"\nstart V mean {starts.mean():.3f} vs MC return mean {rets.mean():.3f}; corr {np.corrcoef(starts, rets)[0,1]:.3f}")
    # Mean value vs MC by floor bucket
    print("\nV vs MC by floor bucket (all phases):")
    fb = defaultdict(list)
    for r in rows:
        fb[(int(r['floor'] or 0) // 4) * 4].append((r["value"], r["mc"], r["phase"]))
    for f in sorted(fb):
        x = fb[f]
        v = np.mean([a for a, _, _ in x]); m = np.mean([b for _, b, _ in x])
        vc = np.mean([a for a, _, p in x if p == "combat"] or [np.nan]); vo = np.mean([a for a, _, p in x if p != "combat"] or [np.nan])
        mcc = np.mean([b for _, b, p in x if p == "combat"] or [np.nan]); mco = np.mean([b for _, b, p in x if p != "combat"] or [np.nan])
        print(f"  floor {f:2d}-{f+3:2d}: n={len(x):6d} V={v:+.3f} MC={m:+.3f} | combat V={vc:+.3f} MC={mcc:+.3f} | ooc V={vo:+.3f} MC={mco:+.3f}")
    # episode length & per-floor decisions
    lens = np.array([len(ep) for ep in eps]); floors = np.array([ep[-1]["floor"] or 0 for ep in eps])
    print(f"\nepisode length mean {lens.mean():.0f} median {np.median(lens):.0f} max {lens.max()}; final floor mean {floors.mean():.2f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
