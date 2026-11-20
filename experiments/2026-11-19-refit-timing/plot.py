"""Draw the four final comparisons from their exported world-level estimates."""

import csv
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(folder):
    with (folder / "full-games.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(8, 3.3), layout="constrained")
    for n, r in enumerate(rows):
        if r["lower95"]:
            x = float(r["gain_pp"])
            ax.errorbar(
                x,
                n,
                xerr=[[max(0, x - float(r["lower95"]))], [max(0, float(r["upper95"]) - x)]],
                fmt="o",
                color="#1e607f",
                capsize=3,
            )
    ax.axvline(0, lw=0.8, color="#48565d")
    ax.set_yticks(
        range(len(rows)),
        labels=[f"{r['panel']} · {r['candidate']} vs {r['baseline']}" for r in rows],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Full-game win-rate gain (percentage points, paired 95% intervals)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "full-games.png", dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(sys.argv[1]))
