"""Plot the fresh-world paired win-rate difference and export its source row."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(verdict, folder):
    r = json.loads(verdict.read_text())["comparison"]
    delta = r.get("difference_pp", r.get("complete_pair_difference_pp"))
    lower, upper = r.get("ci95_pp", r.get("complete_pair_ci95_pp"))
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "win-difference.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["assigned", "complete_pairs", "joint_only", "early_only", "difference_pp", "lower95", "upper95"])
        writer.writerow([r["n"], r["paired_completed"], r["b_only"], r["a_only"], delta, lower, upper])
    fig, ax = plt.subplots(figsize=(6.2, 1.9), layout="constrained")
    limit = max(abs(lower), abs(upper), 0.5) * 1.15
    ax.axvline(0, color="#72777c", linestyle="--", linewidth=1)
    ax.errorbar([delta], [0], xerr=[[delta - lower], [upper - delta]], fmt="o", color="#186789",
                linewidth=2, capsize=5)
    ax.set(xlim=(-limit, limit), ylim=(-1, 1), yticks=[],
           xlabel="Joint minus early-refit win rate (percentage points)",
           title=f"Paired 95% interval across {r['paired_completed']:,} completed worlds")
    ax.spines[["top", "left", "right"]].set_visible(False)
    fig.savefig(folder / "win-difference.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(sys.argv[1]), Path(sys.argv[2]))
