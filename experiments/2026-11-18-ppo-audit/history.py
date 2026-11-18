"""Audit logged learning dynamics, paired evaluations and training world reuse."""

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import mean, median


def pairs(a, b):
    ids = sorted(a.keys() & b.keys())
    gains = sum(not a[k] and b[k] for k in ids)
    losses = sum(a[k] and not b[k] for k in ids)
    n = len(ids)
    d = (gains - losses) / n
    se = math.sqrt(((gains + losses) / n - d * d) / (n - 1))
    return dict(
        n=n,
        baseline=sum(a[k] for k in ids),
        wins=sum(b[k] for k in ids),
        gains=gains,
        losses=losses,
        delta=100 * d,
        ci=[100 * (d - 1.96 * se), 100 * (d + 1.96 * se)],
    )


def actual_worlds(path):
    with gzip.open(path) as f:
        r = json.load(f)
    return {t["seed"] for t in r["tasks"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    study = args.runs / "reward_ablation_20261023"
    result = {"arms": {}, "evaluations": []}
    for arm in ("floors", "wins"):
        rows = [
            r
            for r in map(json.loads, (study / arm / "log.jsonl").open())
            if r["event"] == "train"
        ]
        assert [r["iteration"] for r in rows] == list(range(1, 25))
        stats = {}
        for key in (
            "approx_kl",
            "clip_fraction",
            "grad_cosine_half",
            "value_loss",
            "return_scale",
            "explore_entropy",
            "collect_wins",
            "adv_raw_std",
            "steps",
        ):
            x = [r[key] for r in rows]
            stats[key] = dict(
                median=median(x),
                minimum=min(x),
                maximum=max(x),
                first_four=mean(x[:4]),
                last_four=mean(x[-4:]),
            )
        menus = {}
        for kind in ("card_reward", "map", "rest_site", "shop", "removal", "other"):
            nkey = "explore_n_" + kind

            def pooled(rows, metric):
                return sum(
                    (r["explore_" + metric + "_" + kind] or 0) * r[nkey] for r in rows
                ) / sum(r[nkey] for r in rows)

            menus[kind] = {
                "fraction": sum(r[nkey] for r in rows)
                / sum(r["explore_n"] for r in rows),
                "saturation_first4": pooled(rows[:4], "saturated"),
                "saturation_last4": pooled(rows[-4:], "saturated"),
                "entropy_first4": pooled(rows[:4], "entropy"),
                "entropy_last4": pooled(rows[-4:], "entropy"),
                "rare05_last4": pooled(rows[-4:], "rare05"),
            }
        stats["menus"] = menus
        stats["neg_cosines"] = sum(r["grad_cosine_half"] < 0 for r in rows)
        stats["early_stops"] = {
            r["iteration"]: r["steps"] for r in rows if r["early_stop"]
        }
        stats["faults"] = sum(r["faults"] for r in rows)
        stats["games"] = sum(r["games"] for r in rows)
        stats["choices"] = sum(r["ooc_choice_transitions"] for r in rows)
        result["arms"][arm] = stats
    for panel, iterations in [("e108", [4, 8, 12]), ("e109", [16, 20, 24])]:

        def load(name):
            data = json.loads((study / panel / name / "summary.json").read_bytes())
            return {
                r["world_seed"]: bool(r["win"])
                for r in data["records"]
                if r["status"] == "completed"
            }

        base = load("champion")
        for i in iterations:
            a, b = [load(f"{arm}_{i:04d}") for arm in ("floors", "wins")]
            result["evaluations"].append(
                dict(
                    panel=panel,
                    iteration=i,
                    floors=pairs(base, a),
                    wins=pairs(base, b),
                    wins_minus_floors=pairs(a, b),
                )
            )
    chosen = actual_worlds(study / "floors/collection_0001.json.gz")
    bench = actual_worlds(
        args.runs / "mac_code_performance_20261118/full_mps/0-b/collection_0001.json.gz"
    )
    result["bench_worlds"] = {
        "games": len(bench),
        "previously_used": len(chosen & bench),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
