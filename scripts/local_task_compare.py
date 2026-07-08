#!/usr/bin/env python
"""Compare two local-task eval arms."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from typing import Any

from sts_ai.eval_stats import bootstrap_ci, sign_test


def _load_metas(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(root).rglob("seed_*_r*.meta.json"))
    ]


def _task(meta: dict[str, Any]) -> dict[str, Any]:
    task = ((meta.get("extra") or {}).get("local_task") or {})
    return task if isinstance(task, dict) else {}


def _metric(meta: dict[str, Any], name: str) -> float:
    task = _task(meta)
    if name == "reward":
        return float(task.get("reward", 0.0))
    metrics = task.get("metrics") or {}
    return float(metrics.get(name, 0.0))


def _aggregate(metas: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(metas)
    rewards = [_metric(meta, "reward") for meta in metas]
    hp_losses = [_metric(meta, "hp_loss") for meta in metas]
    wins = [
        bool(((_task(meta).get("metrics") or {}).get("survived", False)))
        for meta in metas
    ]
    labels = [str(_task(meta).get("label", "")) for meta in metas]
    total_invalid = sum(int(meta.get("n_invalid", 0) or 0) for meta in metas)
    total_decisions = sum(int(meta.get("n_decisions", 0) or 0) for meta in metas)
    action_counts: Counter[str] = Counter()
    for meta in metas:
        counts = (((_task(meta).get("metrics") or {}).get("action_counts")) or {})
        action_counts.update({str(k): int(v) for k, v in counts.items()})
    total_actions = sum(action_counts.values()) or 1
    return {
        "n": n,
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "mean_hp_loss": statistics.mean(hp_losses) if hp_losses else 0.0,
        "win_rate": sum(wins) / n if n else 0.0,
        "convincing_rate": sum(1 for label in labels if label == "convincing") / n if n else 0.0,
        "invalid_rate": total_invalid / total_decisions if total_decisions else 0.0,
        "action_share": {
            kind: action_counts[kind] / total_actions
            for kind in sorted(action_counts)
        },
    }


def _grouped_by_window(metas: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    """All rollouts of each window, keyed by window_id.

    An eval arm can hold K > 1 sampled rollouts per window; pairing must use
    the per-window mean, never one arbitrary rollout per window.
    """
    grouped: dict[str, list[float]] = {}
    for meta in metas:
        grouped.setdefault(str(_task(meta).get("window_id")), []).append(
            _metric(meta, metric)
        )
    return grouped


def _rollout_count_range(grouped: dict[str, list[float]], window_ids: list[str]) -> dict[str, int]:
    counts = [len(grouped[window_id]) for window_id in window_ids]
    return {"min": min(counts), "max": max(counts)} if counts else {"min": 0, "max": 0}


def build_report(base_dir: Path, trained_dir: Path, *, metric: str) -> dict[str, Any]:
    base = _load_metas(base_dir)
    trained = _load_metas(trained_dir)
    base_grouped = _grouped_by_window(base, metric)
    trained_grouped = _grouped_by_window(trained, metric)
    paired_ids = sorted(set(base_grouped) & set(trained_grouped))
    deltas = [
        statistics.mean(trained_grouped[window_id]) - statistics.mean(base_grouped[window_id])
        for window_id in paired_ids
    ]
    return {
        "metric": metric,
        "base_dir": str(base_dir),
        "trained_dir": str(trained_dir),
        "arms": {"base": _aggregate(base), "trained": _aggregate(trained)},
        "paired": {
            "n": len(paired_ids),
            "mean_delta": statistics.mean(deltas) if deltas else 0.0,
            "bootstrap_ci_95": list(bootstrap_ci(deltas)) if deltas else [0.0, 0.0],
            "sign_test": sign_test(deltas),
            "rollouts_per_window": {
                "base": _rollout_count_range(base_grouped, paired_ids),
                "trained": _rollout_count_range(trained_grouped, paired_ids),
            },
            "window_ids": paired_ids,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local-task eval arms.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--metric", default="reward")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.base, args.trained, metric=args.metric)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
