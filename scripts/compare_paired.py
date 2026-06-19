#!/usr/bin/env python
"""Paired base-vs-trained rollout comparison over per-rollout meta sidecars."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from sts_ai.eval_stats import paired_floor_summary
from sts_ai.train.reward import is_act_boss_clear


def load_metas(path: Path | str) -> list[dict[str, Any]]:
    # Recursive: run_until writes sidecars under an agent-label subdir
    # (output_dir/<label>/seed_*_r*.meta.json), so a non-recursive glob at the
    # arm root would silently find zero. rglob also matches flat layouts (the
    # pattern is checked at the root too), so callers may point at either the
    # arm root or the leaf label dir.
    root = Path(path)
    metas: list[dict[str, Any]] = []
    for meta_path in sorted(root.rglob("seed_*_r*.meta.json")):
        metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return metas


def group_by_world_seed(
    metas: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for meta in metas:
        seed = int(meta["world_seed"])
        grouped.setdefault(seed, []).append(meta)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def per_seed_metric_means(
    metas: Sequence[Mapping[str, Any]],
    metric: str = "final_floor",
) -> dict[int, float]:
    by_seed = group_by_world_seed(metas)
    return {
        seed: _mean([float(meta.get(metric, 0.0)) for meta in seed_metas])
        for seed, seed_metas in sorted(by_seed.items())
    }


def aggregate_arm(metas: Sequence[Mapping[str, Any]], *, min_act: int = 1) -> dict[str, float | int]:
    n_rollouts = len(metas)
    wins = sum(1 for meta in metas if "VICTORY" in str(meta.get("outcome", "")))
    floors = [float(meta.get("final_floor", 0.0)) for meta in metas]
    decisions = [int(meta.get("n_decisions", 0)) for meta in metas]
    total_decisions = sum(decisions)
    total_invalid = sum(int(meta.get("n_invalid", 0)) for meta in metas)
    budget_truncated = sum(
        1
        for meta in metas
        if str(meta.get("stopped_reason", "")) == "max_decisions"
    )
    act_boss_clears = sum(1 for meta in metas if is_act_boss_clear(meta, min_act))

    return {
        "n_rollouts": n_rollouts,
        "win_rate": wins / n_rollouts if n_rollouts else 0.0,
        "act_boss_clear_rate": act_boss_clears / n_rollouts if n_rollouts else 0.0,
        "mean_floor": _mean(floors),
        "agent_invalid_rate": total_invalid / total_decisions if total_decisions else 0.0,
        "mean_decisions": _mean([float(value) for value in decisions]),
        "budget_truncated_rate": budget_truncated / n_rollouts if n_rollouts else 0.0,
    }


def build_report(
    base_dir: Path | str,
    trained_dir: Path | str,
    *,
    metric: str = "final_floor",
    min_act: int = 1,
    bootstrap_seed: int = 0,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    base_metas = load_metas(base_dir)
    trained_metas = load_metas(trained_dir)
    base_by_seed = per_seed_metric_means(base_metas, metric)
    trained_by_seed = per_seed_metric_means(trained_metas, metric)

    return {
        "base_dir": str(Path(base_dir)),
        "trained_dir": str(Path(trained_dir)),
        "metric": metric,
        "min_act": min_act,
        "arms": {
            "base": aggregate_arm(base_metas, min_act=min_act),
            "trained": aggregate_arm(trained_metas, min_act=min_act),
        },
        "per_seed_means": {
            "base": base_by_seed,
            "trained": trained_by_seed,
        },
        "paired": paired_floor_summary(
            base_by_seed,
            trained_by_seed,
            bootstrap_seed=bootstrap_seed,
            n_resamples=n_resamples,
        ),
    }


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def format_report(report: Mapping[str, Any]) -> str:
    paired = report["paired"]
    sign = paired["sign_test"]
    ci = paired["bootstrap_ci"]
    rows = [
        (
            "n_rollouts",
            report["arms"]["base"]["n_rollouts"],
            report["arms"]["trained"]["n_rollouts"],
        ),
        ("win_rate", report["arms"]["base"]["win_rate"], report["arms"]["trained"]["win_rate"]),
        (
            "act_boss_clear_rate",
            report["arms"]["base"]["act_boss_clear_rate"],
            report["arms"]["trained"]["act_boss_clear_rate"],
        ),
        (
            "mean_floor",
            report["arms"]["base"]["mean_floor"],
            report["arms"]["trained"]["mean_floor"],
        ),
        (
            "agent_invalid_rate",
            report["arms"]["base"]["agent_invalid_rate"],
            report["arms"]["trained"]["agent_invalid_rate"],
        ),
        (
            "mean_decisions",
            report["arms"]["base"]["mean_decisions"],
            report["arms"]["trained"]["mean_decisions"],
        ),
        (
            "budget_truncated_rate",
            report["arms"]["base"]["budget_truncated_rate"],
            report["arms"]["trained"]["budget_truncated_rate"],
        ),
    ]
    metric_width = max(len("metric"), *(len(row[0]) for row in rows))
    base_values = [_fmt(row[1]) for row in rows]
    trained_values = [_fmt(row[2]) for row in rows]
    base_width = max(len("base"), *(len(value) for value in base_values))
    trained_width = max(len("trained"), *(len(value) for value in trained_values))

    lines = [
        "Paired base-vs-trained eval",
        f"metric: {report['metric']}",
        f"paired_seeds: {paired['n_seeds']}",
        f"paired_delta_mean: {_fmt(paired['mean_delta'])}",
        f"paired_delta_se: {_fmt(paired['se_delta'])}",
        f"bootstrap_ci_95: [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        (
            "sign-test: "
            f"n_pos={sign['n_pos']} n_neg={sign['n_neg']} "
            f"n_zero={sign['n_zero']} p_value={_fmt(sign['p_value'])}"
        ),
        "",
        f"{'metric'.ljust(metric_width)}  {'base'.rjust(base_width)}  {'trained'.rjust(trained_width)}",
        f"{'-' * metric_width}  {'-' * base_width}  {'-' * trained_width}",
    ]
    for (name, _, _), base_value, trained_value in zip(rows, base_values, trained_values):
        lines.append(
            f"{name.ljust(metric_width)}  "
            f"{base_value.rjust(base_width)}  "
            f"{trained_value.rjust(trained_width)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired base-vs-trained rollout comparison.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--metric", default="final_floor")
    parser.add_argument("--min-act", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=10000)
    args = parser.parse_args()

    report = build_report(
        args.base,
        args.trained,
        metric=args.metric,
        min_act=args.min_act,
        bootstrap_seed=args.bootstrap_seed,
        n_resamples=args.n_resamples,
    )
    # Fail loud rather than emit an all-zeros "no difference" report when an arm
    # has no rollouts (e.g. a wrong dir or a generation stage that wrote nothing).
    for arm in ("base", "trained"):
        if report["arms"][arm]["n_rollouts"] == 0:
            arm_dir = args.base if arm == "base" else args.trained
            print(
                f"ERROR: no seed_*_r*.meta.json found for the {arm} arm under {arm_dir}",
                file=sys.stderr,
            )
            sys.exit(2)
    print(format_report(report))
    if args.out is not None:
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
