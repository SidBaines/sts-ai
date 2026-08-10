#!/usr/bin/env python3
"""SCRATCH: how far did rollouts get? Histograms of final floor, split by model
and by how the run ended (died / ran out of decisions / beat the act / other).

Throwaway-but-growing exploratory viz — edit freely. Reads the per-rollout
`*.meta.json` sidecars under a sweep dir (default: the Gemma H100 benchmark).
When this graduates to real viz code, fold it into rollout_view + add matplotlib
to the `[viz]` extra.

Usage:
    PYTHONPATH=src .venv/bin/python scratch/analyze_rollouts.py [sweep_dir] [--out-dir scratch]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402

# End-of-run categories, in stack/legend order, with stable colors.
CATEGORIES = ["died", "ran out of decisions", "beat the act", "other (invalid/error)"]
COLORS = {
    "died": "#d62728",                     # red
    "ran out of decisions": "#ff7f0e",     # orange
    "beat the act": "#2ca02c",             # green
    "other (invalid/error)": "#9467bd",    # purple
}


def categorize(meta: dict) -> str:
    """How did this rollout end? (death takes priority — a loss is a loss even at
    the decision cap; then act-clear; then the decision-budget cap.)"""
    outcome = str(meta.get("outcome", ""))
    if outcome.endswith("PLAYER_LOSS"):
        return "died"
    if outcome.endswith("PLAYER_VICTORY") or int(meta.get("final_act", 1)) >= 2:
        return "beat the act"
    if meta.get("stopped_reason") == "max_decisions":
        return "ran out of decisions"
    return "other (invalid/error)"


def model_short(model_id: str | None, agent: str) -> str:
    if not model_id:
        return agent or "?"
    return model_id.rsplit("/", 1)[-1]


def load(sweep_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "**", "*.meta.json"), recursive=True)):
        try:
            m = json.load(open(path))
        except Exception:  # noqa: BLE001 - skip unreadable sidecars in scratch
            continue
        rows.append(
            {
                "model": model_short(m.get("model_id"), m.get("agent", "")),
                "thinking": bool(m.get("thinking")),
                "floor": int(m.get("final_floor", 0)),
                "act": int(m.get("final_act", 1)),
                "decisions": int(m.get("n_decisions", 0)),
                "hp_frac": (m.get("final_hp", 0) / m["max_hp"]) if m.get("max_hp") else 0.0,
                "category": categorize(m),
            }
        )
    return rows


def arm_label(model: str, thinking: bool) -> str:
    return f"{model} · {'think' if thinking else 'no-think'}"


def print_summary(rows: list[dict]) -> None:
    arms = defaultdict(list)
    for r in rows:
        arms[(r["model"], r["thinking"])].append(r)
    print(f"\n{len(rows)} rollouts across {len(arms)} arms\n")
    hdr = f"{'arm':32s} {'n':>3} {'floor: mean':>11} {'med':>4} {'max':>4} | " + " ".join(f"{c.split()[0]:>5}" for c in CATEGORIES)
    print(hdr)
    print("-" * len(hdr))
    for (model, thinking), rs in sorted(arms.items()):
        floors = [r["floor"] for r in rs]
        cats = Counter(r["category"] for r in rs)
        line = (
            f"{arm_label(model, thinking):32s} {len(rs):>3} {statistics.mean(floors):>11.1f} "
            f"{int(statistics.median(floors)):>4} {max(floors):>4} | "
            + " ".join(f"{cats.get(c, 0):>5}" for c in CATEGORIES)
        )
        print(line)
    print("\n(category columns:", ", ".join(CATEGORIES), ")")


def plot_floor_hist(rows: list[dict], out_path: str) -> None:
    models = sorted({r["model"] for r in rows})
    thinks = [False, True]
    max_floor = max((r["floor"] for r in rows), default=1)
    bins = range(0, max_floor + 2)  # one bin per floor

    fig, axes = plt.subplots(
        len(models), len(thinks), figsize=(11, 3.2 * len(models)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, model in enumerate(models):
        for j, thinking in enumerate(thinks):
            ax = axes[i][j]
            cell = [r for r in rows if r["model"] == model and r["thinking"] == thinking]
            stacks = [[r["floor"] for r in cell if r["category"] == c] for c in CATEGORIES]
            present = [(s, c) for s, c in zip(stacks, CATEGORIES) if s]
            if present:
                ax.hist(
                    [s for s, _ in present], bins=list(bins), stacked=True,
                    color=[COLORS[c] for _, c in present], label=[c for _, c in present],
                    edgecolor="white", linewidth=0.3,
                )
            ax.set_title(f"{arm_label(model, thinking)}  (n={len(cell)})", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            if i == len(models) - 1:
                ax.set_xlabel("final floor reached")
            if j == 0:
                ax.set_ylabel("rollouts")

    # title on top, shared legend just below it, plots below that (no overlap)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CATEGORIES]
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle("How far did each model get? (final floor; stacked by how the run ended)", y=1.06, fontsize=12)
    fig.legend(handles, CATEGORIES, loc="upper center", bbox_to_anchor=(0.5, 1.00),
               ncol=len(CATEGORIES), fontsize=9, frameon=False)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", nargs="?", default="data/rollouts/gemma_bench")
    ap.add_argument("--out-dir", default="scratch")
    args = ap.parse_args()

    rows = load(args.sweep_dir)
    if not rows:
        raise SystemExit(f"no *.meta.json under {args.sweep_dir}")
    print_summary(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_floor_hist(rows, os.path.join(args.out_dir, "floor_hist_by_arm.png"))


if __name__ == "__main__":
    main()
