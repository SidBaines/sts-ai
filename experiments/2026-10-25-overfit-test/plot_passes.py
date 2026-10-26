"""Per-pass curves for the stage-1 screens: teacher probability, agreement when the
teacher's top action is not shown first, and the first-displayed-option rate.

Usage: python plot_passes.py STATES_JSONL OUT_DIR LABEL=RUN_DIR [LABEL=RUN_DIR ...]
Writes passes.csv and passes.png. A label starting with "shared" is drawn as a
grey dashed reference.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PER_PASS = 8
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def passes(run: Path, top: dict) -> list[dict]:
    cells: dict[int, list[float]] = {}
    for path in sorted((run / "generations").glob("iter_*.jsonl")):
        for line in path.open():
            g = json.loads(line)
            k = (g["iteration"] - 1) // PER_PASS + 1
            c = cells.setdefault(k, [0, 0, 0, 0, 0, 0])
            c[0] += g["teacher_probability"] if g["valid"] else 0.0
            c[1] += 1
            if g["order"][0] != top[g["id"]]:
                c[2] += bool(g["valid"] and g["action_index"] == top[g["id"]])
                c[3] += 1
            c[4] += bool(g["valid"] and g["label"] == "A")
            c[5] += 1
    rows = []
    for k, c in sorted(cells.items()):
        if c[1] < 256:
            continue  # incomplete pass
        rows.append({"pass": k, "teacher_probability": c[0] / c[1],
                     "agreement_top_not_first": c[2] / max(c[3], 1), "first_displayed_rate": c[4] / c[5]})
    return rows


def main() -> None:
    states, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = [arg.split("=", 1) for arg in sys.argv[3:]]
    top = {json.loads(l)["id"]: json.loads(l)["teacher_top"] for l in states.open()}
    content_only = sum(1 / len(json.loads(l)["legal_indices"]) for l in states.open()) / len(top)
    out.mkdir(parents=True, exist_ok=True)
    data = {label: passes(Path(run), top) for label, run in runs}
    with (out / "passes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["run", "pass", "teacher_probability",
                                         "agreement_top_not_first", "first_displayed_rate"])
        writer.writeheader()
        for label, rows in data.items():
            for row in rows:
                writer.writerow({"run": label, **{k: round(v, 4) if isinstance(v, float) else v
                                                  for k, v in row.items()}})
    panels = [("teacher_probability", "Mean teacher probability"),
              ("agreement_top_not_first", "Matches teacher when its top\nchoice is not shown first"),
              ("first_displayed_rate", "Share of answers picking A")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    color_index = 0
    for label, rows in data.items():
        shared = label.startswith("shared")
        style = dict(color="#8a8984", linestyle="--", linewidth=1.5) if shared else \
            dict(color=COLORS[color_index], linewidth=2)
        if not shared:
            color_index += 1
        for ax, (key, _) in zip(axes, panels):
            ax.plot([r["pass"] for r in rows], [r[key] for r in rows], marker="o", markersize=4,
                    label=label, **style)
    axes[2].axhline(content_only, color="#52514e", linewidth=1, linestyle=":")
    axes[2].annotate(f"choosing by content: {content_only:.2f}", (1, content_only), xytext=(0, -12),
                     textcoords="offset points", fontsize=8, color="#52514e")
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("Pass over the 32 training states", fontsize=9, color="#52514e")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "passes.png", dpi=110)


if __name__ == "__main__":
    main()
