"""Export every comparison and plot the difference between learning evidence and proxy scores."""

import argparse
import csv
import json
from pathlib import Path


def table(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def read(name):
        return json.loads((args.run / name).read_text())

    history = read("history.json")
    audit = read("results.json")
    checks = read("protocol_check/protocols.json")
    evo = []
    for v in history["evaluations"]:
        for arm in ("floors", "wins", "wins_minus_floors"):
            z = v[arm]
            evo.append(
                dict(
                    panel=v["panel"],
                    iteration=v["iteration"],
                    comparison=arm,
                    pairs=z["n"],
                    difference_pp=z["delta"],
                    low=z["ci"][0],
                    high=z["ci"][1],
                    gains=z["gains"],
                    losses=z["losses"],
                )
            )
    table(args.out / "game-outcomes.csv", evo)
    table(
        args.out / "menu-context.csv",
        [{"menu": k, **v} for k, v in audit["menus"].items()],
    )
    table(
        args.out / "frozen-critic.csv",
        [{"act": k, **v} for k, v in audit["critic"].items()],
    )
    folds = []
    for f in audit["folds"]:
        for arm, vals in f["arms"].items():
            folds.append(
                dict(
                    fold=f["fold"],
                    arm=arm,
                    train_rows=f["ntrain"],
                    test_rows=f["ntest"],
                    **vals,
                )
            )
    table(args.out / "optimizer-folds.csv", folds)
    table(
        args.out / "paired-surrogates.csv",
        [
            dict(arm=k, comparison=c, mean=v[0], low=v[1], high=v[2])
            for k, d in audit["paired_surrogate"].items()
            for c, v in d.items()
        ],
    )
    table(
        args.out / "protocols.csv",
        [
            dict(
                arm=k,
                pairs=v["n"],
                wins=v["wins"],
                up=v["gains"],
                down=v["losses"],
                delta=v["delta"],
                low=v["ci"][0],
                high=v["ci"][1],
            )
            for k, v in checks["comparisons"].items()
        ],
    )
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(9, 3.1),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.3, 1]},
    )
    for key, label, color in [
        ("floors", "Wins + floors", "#146B78"),
        ("wins", "Wins only", "#AD632C"),
    ]:
        x = [r for r in evo if r["comparison"] == key]
        for group in (x[:3], x[3:]):
            mid = [r["difference_pp"] for r in group]
            left.errorbar(
                [r["iteration"] for r in group],
                mid,
                yerr=[
                    [r["difference_pp"] - r["low"] for r in group],
                    [r["high"] - r["difference_pp"] for r in group],
                ],
                fmt="o-",
                capsize=2,
                color=color,
                label=label if group == x[:3] else None,
            )
    left.axhline(0, color="#939393", lw=0.8)
    left.set_xticks([4, 8, 12, 16, 20, 24])
    left.set_xlabel("Training iteration")
    left.set_ylabel("Win-rate gain (points)")
    left.set_title(
        "Paired games versus starting policy", loc="left", fontsize=10, weight="bold"
    )
    left.legend(frameon=False, fontsize=8)
    keys = ["card_with_items", "card_without_items"]
    vals = [audit["menus"][k]["saturated"] * 100 for k in keys]
    right.barh([1, 0], vals, color=["#50738E", "#62A996"], height=0.5)
    right.set_yticks([1, 0], ["Item offered", "No item offered"])
    for y, v in zip([1, 0], vals):
        right.text(
            v - 2, y, f"{v:.1f}%", ha="right", va="center", color="white", weight="bold"
        )
    right.set_xlim(0, 100)
    right.set_xlabel("Card-reward menus over 99% confidence")
    right.set_title(
        "Frozen champion on 512 worlds", loc="left", fontsize=10, weight="bold"
    )
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.out / "learning-review.png", dpi=140)


if __name__ == "__main__":
    main()
