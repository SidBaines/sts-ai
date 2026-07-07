#!/usr/bin/env python
"""Compare two local-task eval arms."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import statistics
from pathlib import Path
from typing import Any


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


def _by_window(metas: list[dict[str, Any]], metric: str) -> dict[str, float]:
    return {str(_task(meta).get("window_id")): _metric(meta, metric) for meta in metas}


def _sign_test(deltas: list[float]) -> dict[str, Any]:
    n_pos = sum(1 for delta in deltas if delta > 0)
    n_neg = sum(1 for delta in deltas if delta < 0)
    n = n_pos + n_neg
    if n == 0:
        p_value = 1.0
    else:
        k = min(n_pos, n_neg)
        p_value = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
    return {"n_pos": n_pos, "n_neg": n_neg, "n_zero": len(deltas) - n, "p_value": p_value}


def build_report(base_dir: Path, trained_dir: Path, *, metric: str) -> dict[str, Any]:
    base = _load_metas(base_dir)
    trained = _load_metas(trained_dir)
    base_by_window = _by_window(base, metric)
    trained_by_window = _by_window(trained, metric)
    paired_ids = sorted(set(base_by_window) & set(trained_by_window))
    deltas = [trained_by_window[window_id] - base_by_window[window_id] for window_id in paired_ids]
    return {
        "metric": metric,
        "base_dir": str(base_dir),
        "trained_dir": str(trained_dir),
        "arms": {"base": _aggregate(base), "trained": _aggregate(trained)},
        "paired": {
            "n": len(paired_ids),
            "mean_delta": statistics.mean(deltas) if deltas else 0.0,
            "sign_test": _sign_test(deltas),
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
