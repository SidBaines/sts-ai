"""Plot paired win-rate differences from the saved 27-configuration screening table."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


def draw(directory, destination):
    with (directory / "screen.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 27:
        raise ValueError("the complete 27-setting screening table is required")
    target = [0.05, 0.12, 0.22]
    rates = [1e-5, 3e-5, 6e-5]
    epochs = [1, 2, 4]
    data = {
        (float(r["exploration_target"]), float(r["combat_lr"]), int(r["passes"])): r
        for r in rows
    }
    limit = max(1.0, max(abs(float(r["paired_gain_pp"])) for r in rows))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.7))
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.22, top=0.69, wspace=0.25)
    for j, (ax, value) in enumerate(zip(axes, target)):
        z = np.array(
            [
                [float(data[value, rate, ep]["paired_gain_pp"]) for rate in rates]
                for ep in epochs
            ]
        )
        image = ax.imshow(z, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_title(f"{value:.0%} exploration", fontsize=12, loc="left")
        ax.set_xticks(range(3), ["0.00001", "0.00003", "0.00006"], fontsize=9)
        ax.set_yticks(range(3), epochs if j == 0 else ["", "", ""])
        if j == 0:
            ax.set_ylabel("Training passes")
        ax.tick_params(length=0, pad=4)
        for edge in ax.spines.values():
            edge.set_visible(False)
        for y, ep in enumerate(epochs):
            for x, rate in enumerate(rates):
                row = data[value, rate, ep]
                color = "white" if abs(z[y, x]) > 0.62 * limit else "#18202b"
                ax.text(
                    x,
                    y,
                    f"{z[y, x]:+.1f}" + ("" if row["eligible"] == "True" else "†"),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    weight="bold",
                )
                if value == 0.12 and rate == 3e-5 and ep == 2:
                    ax.add_patch(
                        Rectangle(
                            (x - 0.48, y - 0.47),
                            0.96,
                            0.94,
                            fill=False,
                            lw=1.7,
                            ec="#191d26",
                        )
                    )
    fig.text(0.08, 0.92, "Combat learning after one update", fontsize=18, weight="bold")
    fig.text(
        0.08,
        0.81,
        "Win-rate change in points on 768 matched worlds. The outline marks the original choice.",
        fontsize=10,
    )
    fig.text(0.48, 0.08, "Combat learning rate", ha="center", fontsize=10)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.936, 0.25, 0.014, 0.39]),
        format=FuncFormatter(lambda x, _: f"{x:+.0f}"),
    )
    cb.ax.tick_params(length=2, labelsize=8)
    fig.savefig(destination, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    draw(a.directory, a.destination)
