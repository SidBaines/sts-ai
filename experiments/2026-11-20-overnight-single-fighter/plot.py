import csv
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    with (folder / "paired.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["lower95"] and r["upper95"]]
    fig, ax = plt.subplots(figsize=(8.4, max(3, 0.43 * len(rows) + 1)), layout="constrained")
    for i, r in enumerate(rows):
        x = float(r["gain_pp"])
        lo = float(r["lower95"])
        hi = float(r["upper95"])
        ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", capsize=3, color="#186789")
    ax.axvline(0, color="#6c747a", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(rows)), [r["comparison"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate minus reference win rate (percentage points, paired 95% interval)")
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(folder / "paired.png", dpi=125)
    plt.close(fig)
