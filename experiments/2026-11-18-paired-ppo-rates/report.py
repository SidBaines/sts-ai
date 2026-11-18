"""Export all scheduled checkpoint evaluations and the two frozen endpoint comparisons."""

import csv
import json
from statistics import mean

from launch import ARMS, ROOT, RUN


def dump_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = ROOT / "research/figures/2026-11-18-paired-ppo-rates"
    out.mkdir(parents=True, exist_ok=True)
    train, dev, decisions = [], [], []
    for arm in ARMS:
        events = [
            json.loads(s) for s in (RUN / arm / "log.jsonl").read_text().splitlines()
        ]
        # A recovery can repeat a written event. The final event for the saved iteration is authoritative.
        for kind, target, cols in [
            (
                "train",
                train,
                (
                    "games",
                    "collect_wins",
                    "collect_floor",
                    "faults",
                    "ooc_choice_transitions",
                    "steps",
                    "early_stop",
                    "approx_kl",
                    "clip_fraction",
                    "grad_cosine_half",
                    "explore_saturated_card_reward_with_items",
                    "explore_saturated_card_reward_without_items",
                    "collect_seconds",
                    "explore_seconds",
                    "update_seconds",
                    "iteration_seconds",
                ),
            ),
            (
                "eval",
                dev,
                (
                    "n",
                    "wins",
                    "mean_floor",
                    "act1_boss_pass",
                    "act2_boss_pass",
                    "act3_boss_pass",
                    "faults",
                ),
            ),
        ]:
            by_iter = {e["iteration"]: e for e in events if e["event"] == kind}
            expected = set(range(1, 25)) if kind == "train" else set(range(0, 25, 4))
            if set(by_iter) != expected:
                raise ValueError(f"Incomplete {kind} history for {arm}")
            for i in sorted(by_iter):
                target.append(
                    {"arm": arm, "iteration": i, **{k: by_iter[i][k] for k in cols}}
                )
    for phase in ("primary", "confirmation"):
        r = json.loads((RUN / f"{phase}.json").read_text())
        decisions.append(
            dict(
                phase=phase,
                first=r["a"],
                second=r["b"],
                assigned_pairs=r["n"],
                complete_pairs=r["paired_completed"],
                first_wins=r["first"]["wins"],
                second_wins=r["second"]["wins"],
                first_only=r["a_only"],
                second_only=r["b_only"],
                first_faults=r["first"]["faults"],
                second_faults=r["second"]["faults"],
                first_cut=r["first"]["cuts"],
                second_cut=r["second"]["cuts"],
                full_gain_low=r["full_cohort_gain_bounds_pp"][0],
                full_gain_high=r["full_cohort_gain_bounds_pp"][1],
                difference_pp=r.get("difference_pp"),
                ci_low=(r.get("ci95_pp") or [None, None])[0],
                ci_high=(r.get("ci95_pp") or [None, None])[1],
                p_two_sided=r.get("p_two_sided"),
            )
        )
    dump_csv(out / "training.csv", train)
    dump_csv(out / "development.csv", dev)
    dump_csv(out / "endpoints.csv", decisions)

    import matplotlib.pyplot as plt

    fig, (curve, est) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.4),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.3, 1]},
    )
    colors = {"control": "#12687A", "challenger": "#A85521"}
    for arm in ARMS:
        rows = [r for r in dev if r["arm"] == arm]
        curve.plot(
            [236 + r["iteration"] for r in rows],
            [100 * r["wins"] / r["n"] for r in rows],
            marker="o",
            label=f"Learning rate {ARMS[arm]:g}",
            color=colors[arm],
        )
    curve.set_xlabel("Cumulative training iteration")
    curve.set_ylabel("Win rate (%)")
    curve.set_xticks([236, 240, 244, 248, 252, 256, 260])
    curve.legend(frameon=False, fontsize=8)
    curve.set_title(
        "Development checks on 1,024 worlds", loc="left", fontsize=10, weight="bold"
    )
    est.axvline(0, color="#87949A", lw=0.8)
    for y, r in zip((1, 0), decisions):
        if r["difference_pp"] is None:
            est.plot(
                [r["full_gain_low"], r["full_gain_high"]], [y, y], lw=5, color="#748790"
            )
            est.text(
                mean((r["full_gain_low"], r["full_gain_high"])),
                y + 0.08,
                "incomplete cohort",
                ha="center",
                fontsize=8,
            )
        else:
            x = r["difference_pp"]
            est.errorbar(
                x,
                y,
                xerr=[[x - r["ci_low"]], [r["ci_high"] - x]],
                fmt="o",
                capsize=4,
                color="#176B75",
            )
    est.set_yticks([1, 0], ["Lower rate vs control", "Selected vs incumbent"])
    est.set_ylim(-0.45, 1.45)
    est.set_xlabel("Paired win-rate difference (points)")
    est.set_title(
        "Independent 16,384-world panels", loc="left", fontsize=10, weight="bold"
    )
    for ax in (curve, est):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "learning-rates.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    main()
