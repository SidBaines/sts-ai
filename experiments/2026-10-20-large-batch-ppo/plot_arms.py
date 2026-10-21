"""Plot each large-batch PPO arm's own evaluation and entropy against training games.

Usage: plot_arms.py RUN_DIR OUT_DIR
RUN_DIR holds one sub-directory per arm with the trainer's log.jsonl.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("recipe", "Arm 1: recipe, potions visible (different protocol)", "--"),
    ("recipe_hidden", "Arm 3: recipe with the potion filter", "-"),
    ("hygiene_hidden", "Arm 4: arm 3 plus hygiene options", "-"),
    ("small_hidden", "Arm 6: arm 3 at 768 games per iteration", "-"),
    ("round3_student", "Arm 5: round 3 (better combat head)", "-"),
]


def main(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, (ax_win, ax_ent) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, label, style in ARMS:
        records = [json.loads(line) for line in open(run_dir / arm / "log.jsonl")]
        train = [r for r in records if r["event"] == "train"]
        games = {0: 0}
        for r in train:
            games[r["iteration"]] = games[r["iteration"] - 1] + r["games"]
        evals = [r for r in records if r["event"] == "eval"]
        xs = [games[r["iteration"]] / 1000 for r in evals]
        ys = [100 * r["wins"] / r["n"] for r in evals]
        ax_win.plot(xs, ys, style, marker="o", markersize=3, label=label)
        ax_ent.plot([games[r["iteration"]] / 1000 for r in train], [r["entropy"] for r in train], style, label=label)
        for r in evals:
            rows.append({"arm": arm, "iteration": r["iteration"], "games": games[r["iteration"]],
                         "eval_wins": r["wins"], "eval_n": r["n"], "eval_mean_floor": round(r["mean_floor"], 3)})
    ax_win.set(xlabel="training games (thousands)", ylabel="trainer's evaluation win rate (%), 1,024 worlds",
               title="Own evaluation during training")
    ax_ent.set(xlabel="training games (thousands)", ylabel="out-of-combat policy entropy",
               title="Out-of-combat entropy")
    ax_win.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "trainer_eval_and_entropy.png", dpi=110)
    with open(out_dir / "trainer_eval.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
