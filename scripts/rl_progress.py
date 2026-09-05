#!/usr/bin/env python
"""Live progress report + plots for a GRPO run directory (dry-run or production).

Walks ``<out-dir>/iter_*/rollouts/`` (tolerates in-flight, partially written
iterations), prints a per-iteration table, and writes PNGs to
``<out-dir>/progress_plots/``. Re-run any time; it re-reads everything.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/rl_progress.py \
        --out-dir data/rl/ooc_dryrun_v1
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iter_index(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _screen(record: dict[str, Any]) -> str:
    state = record.get("state") or {}
    screen = state.get("screen_state") or record.get("phase") or "unknown"
    return str(screen).replace("ScreenState.", "")


def collect_iteration(iter_dir: Path) -> dict[str, Any]:
    rollouts = iter_dir / "rollouts"
    metas = sorted(rollouts.glob("*.meta.json"))
    floors: list[int] = []
    stop_reasons: Counter[str] = Counter()
    groups: defaultdict[int, list[int]] = defaultdict(list)
    n_dec = inv = retried = 0
    resolution: Counter[str] = Counter()
    screens: Counter[str] = Counter()

    for meta_path in metas:
        meta = _load_json(meta_path)
        if meta is None:
            continue
        floor = int(meta.get("final_floor", 0) or 0)
        floors.append(floor)
        stop_reasons[str(meta.get("stopped_reason"))] += 1
        groups[int(meta.get("world_seed", -1))].append(floor)
        jsonl_path = Path(str(meta_path).removesuffix(".meta.json") + ".jsonl")
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            agent = record.get("agent") or {}
            md = agent.get("metadata") or {}
            n_dec += 1
            screens[_screen(record)] += 1
            if agent.get("retries", 0) > 0:
                retried += 1
            if not agent.get("valid", True):
                inv += 1
                continue
            resolution[
                md.get("semantic_match")
                or ("index_fallback" if md.get("semantic_fallback") else "exact")
            ] += 1

    group_stds = [statistics.pstdev(v) for v in groups.values() if len(v) > 1]
    trainer_metrics: dict[str, float] = {}
    for candidate in sorted(iter_dir.rglob("trainer_log.json")):
        log = _load_json(candidate)
        if not log:
            continue
        for key in ("loss", "mean_kl", "mean_ratio", "clip_fraction", "mean_advantage"):
            value = log.get(key) or (log.get("final") or {}).get(key)
            if isinstance(value, (int, float)):
                trainer_metrics[key] = float(value)
    denom = max(n_dec, 1)
    return {
        "iteration": _iter_index(iter_dir),
        "n_rollouts": len(floors),
        "floors": floors,
        "mean_floor": statistics.mean(floors) if floors else float("nan"),
        "median_floor": statistics.median(floors) if floors else float("nan"),
        "stop_reasons": dict(stop_reasons),
        "n_groups": len(groups),
        "n_zero_variance_groups": sum(
            1 for v in groups.values() if len(v) > 1 and statistics.pstdev(v) == 0.0
        ),
        "mean_group_std": statistics.mean(group_stds) if group_stds else float("nan"),
        "n_decisions": n_dec,
        "decision_invalid_rate": inv / denom,
        "decision_retry_rate": retried / denom,
        "resolution": dict(resolution),
        "screens": dict(screens),
        "trainer": trainer_metrics,
    }


def plot(iters: list[dict[str, Any]], plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    xs = [it["iteration"] for it in iters]

    # 1. Floors: per-rollout scatter + mean/median lines
    fig, ax = plt.subplots(figsize=(8, 5))
    for it in iters:
        ax.scatter([it["iteration"]] * len(it["floors"]), it["floors"], alpha=0.45, s=26, color="tab:blue")
    ax.plot(xs, [it["mean_floor"] for it in iters], "o-", color="tab:red", label="mean")
    ax.plot(xs, [it["median_floor"] for it in iters], "s--", color="tab:orange", label="median")
    ax.axhline(16, color="grey", lw=0.8, ls=":", label="act-1 boss (16)")
    ax.set_xlabel("iteration"); ax.set_ylabel("final floor"); ax.legend()
    ax.set_title("Reward: final floors per iteration")
    path = plots_dir / "floors.png"; fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # 2. Format health
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [it["decision_invalid_rate"] for it in iters], "o-", label="decision invalid rate")
    ax.plot(xs, [it["decision_retry_rate"] for it in iters], "s-", label="decision retry rate")
    keys = sorted({k for it in iters for k in it["resolution"]})
    for key in keys:
        rates = [it["resolution"].get(key, 0) / max(it["n_decisions"], 1) for it in iters]
        ax.plot(xs, rates, "--", alpha=0.8, label=f"resolution: {key}")
    ax.set_xlabel("iteration"); ax.set_ylabel("rate"); ax.legend(fontsize=8)
    ax.set_title("Format health (decision level)")
    path = plots_dir / "format_health.png"; fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # 3. Advantage signal
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [it["mean_group_std"] for it in iters], "o-", label="mean within-group floor std")
    ax.plot(xs, [it["n_zero_variance_groups"] for it in iters], "s-", label="zero-variance groups")
    ax.set_xlabel("iteration"); ax.legend()
    ax.set_title("Learning-signal availability")
    path = plots_dir / "advantage_signal.png"; fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # 4. Trainer metrics (when present)
    trainer_keys = sorted({k for it in iters for k in it["trainer"]})
    if trainer_keys:
        fig, axes = plt.subplots(1, len(trainer_keys), figsize=(4 * len(trainer_keys), 3.5), squeeze=False)
        for ax, key in zip(axes[0], trainer_keys):
            ax.plot(xs, [it["trainer"].get(key) for it in iters], "o-")
            ax.set_title(key); ax.set_xlabel("iteration")
        path = plots_dir / "trainer_metrics.png"; fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        written.append(path)

    # 5. Screen-type decision mix
    screen_keys = sorted({k for it in iters for k in it["screens"]})
    if screen_keys:
        fig, ax = plt.subplots(figsize=(8, 5))
        bottoms = [0.0] * len(iters)
        for key in screen_keys:
            vals = [it["screens"].get(key, 0) / max(it["n_decisions"], 1) for it in iters]
            ax.bar(xs, vals, bottom=bottoms, label=key)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xlabel("iteration"); ax.set_ylabel("share of decisions"); ax.legend(fontsize=8)
        ax.set_title("Decision mix by screen type")
        path = plots_dir / "screen_mix.png"; fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/rl/ooc_dryrun_v1"))
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args()

    iter_dirs = sorted(
        (d for d in args.out_dir.glob("iter_*") if d.is_dir()), key=_iter_index
    )
    if not iter_dirs:
        raise SystemExit(f"no iter_* directories under {args.out_dir}")
    iters = [collect_iteration(d) for d in iter_dirs]
    iters = [it for it in iters if it["n_rollouts"] > 0]

    header = (
        f"{'it':>3} {'n':>3} {'mean':>6} {'med':>5} {'grp_std':>7} {'zerovar':>7} "
        f"{'inv%':>5} {'retry%':>6}  stop_reasons / trainer"
    )
    print(header); print("-" * len(header))
    for it in iters:
        print(
            f"{it['iteration']:>3} {it['n_rollouts']:>3} {it['mean_floor']:>6.1f} "
            f"{it['median_floor']:>5.0f} {it['mean_group_std']:>7.2f} "
            f"{it['n_zero_variance_groups']:>7} {100*it['decision_invalid_rate']:>5.1f} "
            f"{100*it['decision_retry_rate']:>6.1f}  {it['stop_reasons']} {it['trainer']}"
        )
    plots_dir = args.plots_dir or (args.out_dir / "progress_plots")
    for path in plot(iters, plots_dir):
        print("wrote", path)


if __name__ == "__main__":
    main()
