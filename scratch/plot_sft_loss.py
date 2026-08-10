"""Plot train/val loss for the Nob RWR-SFT runs from their MLX training logs.

Usage: .venv/bin/python scratch/plot_sft_loss.py
Writes scratch/plots/nob_sft_train_val_loss.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "data/local_curricula/gremlin_nob/logs"
RUNS = [
    # (label, log file, categorical slot hex)
    ("1400-iter retrain", "train_rwr_sft_won_native8k_stop_3ep_20260707.log", "#2a78d6"),
    ("200-iter first run", "train_rwr_sft_won_native8k_stop_20260707_151412.log", "#1baf7a"),
]
TRAIN_RE = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
CHOSEN = (1000, 0.396)  # fused checkpoint (best val)

TEXT = "#3d3d3a"
MUTED = "#8a8a85"


def series(log_path: Path, pattern: re.Pattern) -> tuple[list[int], list[float]]:
    pairs = [(int(m.group(1)), float(m.group(2))) for m in pattern.finditer(log_path.read_text())]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for label, log_name, color in RUNS:
        log_path = LOGS / log_name
        train_x, train_y = series(log_path, TRAIN_RE)
        val_x, val_y = series(log_path, VAL_RE)
        ax.plot(train_x, train_y, color=color, linewidth=1.2, alpha=0.45, zorder=2)
        ax.plot(
            val_x, val_y, color=color, linewidth=2.0, marker="o", markersize=6,
            zorder=3, label=f"{label} (val)",
        )
        # Direct labels at the right end of each run's curves.
        ax.annotate(
            f"{label} — val", (val_x[-1], val_y[-1]),
            xytext=(8, 4), textcoords="offset points", color=color, fontsize=9,
        )
        ax.annotate(
            "train", (train_x[-1], train_y[-1]),
            xytext=(8, -4), textcoords="offset points", color=color, fontsize=9, alpha=0.7,
        )

    # Ring the fused checkpoint (2px surface ring per mark spec).
    ax.scatter(*CHOSEN, s=170, facecolors="none", edgecolors=TEXT, linewidths=1.4, zorder=4)
    ax.annotate(
        f"fused checkpoint\niter {CHOSEN[0]}, val {CHOSEN[1]:.3f}",
        CHOSEN, xytext=(-118, 30), textcoords="offset points",
        color=TEXT, fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )

    ax.set_xlabel("training iteration (batch size 1)", color=TEXT)
    ax.set_ylabel("cross-entropy loss", color=TEXT)
    ax.set_title(
        "Gremlin Nob RWR-SFT (gemma-4-E4B native thinking): train vs val loss",
        color=TEXT, fontsize=11, loc="left",
    )
    ax.grid(True, color="#e6e6e2", linewidth=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)

    out = REPO / "scratch/plots/nob_sft_train_val_loss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
