"""Reward-derived labels for offline filtered behaviour cloning.

This module joins per-rollout outcome metadata to a binary competence label.
When act-boss-clear positives are too sparse, it falls back to keeping the top
floor quantile among non-crashed rollouts. Ties at the threshold may keep
slightly more than the nominal top fraction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import statistics
from typing import Any

__all__ = [
    "TrajectoryLabel",
    "is_act_boss_clear",
    "label_trajectories",
    "rwr_multiplicities",
]


@dataclass(frozen=True)
class TrajectoryLabel:
    stem: str
    world_seed: int
    rollout_index: int
    outcome: str
    final_act: int
    final_floor: int
    stopped_reason: str
    n_invalid: int
    n_decisions: int
    kept: bool
    keep_reason: str


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _str_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def is_act_boss_clear(meta: dict, min_act: int = 1) -> bool:
    """Return whether metadata proves the run cleared the Act-``min_act`` boss."""
    outcome = _str_value(meta.get("outcome", ""))
    if "VICTORY" in outcome:
        return True
    return _int_value(meta.get("final_act", 0)) > min_act


def _quantile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank-lower quantile over already sorted integer values."""
    if not sorted_values:
        raise ValueError("_quantile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0.0 and 1.0")
    idx = int(math.floor(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


def rwr_multiplicities(
    labels: list[TrajectoryLabel],
    *,
    beta: float,
    baseline: str = "median",
    max_multiplier: int = 8,
    exclude_simulator_errors: bool = True,
) -> tuple[dict[str, int], dict]:
    """Return deterministic RWR integer resampling multiplicities by stem.

    Weights are ``exp((final_floor - baseline_floor) / beta)`` and are rounded
    then clamped into ``[0, max_multiplier]``. As ``beta -> 0+``, below-baseline
    floors round to zero while above-baseline floors hit the cap, which is
    RWR's hard-filter limit but not identical to filter mode. As
    ``beta -> inf``, all weights approach one, giving near-uniform behaviour
    cloning over all valid trajectories.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if baseline not in {"median", "mean"}:
        raise ValueError("baseline must be 'median' or 'mean'")

    label_rows = list(labels)
    pool = [
        label
        for label in label_rows
        if not (
            exclude_simulator_errors
            and label.stopped_reason == "simulator_error"
        )
    ]
    n_simulator_error_excluded = len(label_rows) - len(pool)

    baseline_value: float | int | None
    if pool:
        floors = [int(label.final_floor) for label in pool]
        if baseline == "median":
            baseline_value = statistics.median(floors)
        else:
            baseline_value = statistics.mean(floors)
    else:
        baseline_value = None

    multiplicity_by_stem: dict[str, int] = {}
    multiplicity_histogram: Counter[int] = Counter()
    n_dropped_zero = 0
    n_capped = 0

    for label in label_rows:
        stem = str(label.stem)
        is_excluded = (
            exclude_simulator_errors
            and label.stopped_reason == "simulator_error"
        )
        if is_excluded or baseline_value is None:
            multiplicity = 0
        else:
            exponent = (int(label.final_floor) - baseline_value) / beta
            try:
                rounded = round(math.exp(exponent))
            except OverflowError:
                rounded = max_multiplier
            multiplicity = max(0, min(rounded, max_multiplier))
            if rounded == 0:
                n_dropped_zero += 1
            if max_multiplier > 0 and multiplicity == max_multiplier:
                n_capped += 1

        multiplicity_by_stem[stem] = int(multiplicity)
        multiplicity_histogram[int(multiplicity)] += 1

    report = {
        "mode": "rwr",
        "beta": beta,
        "baseline_kind": baseline,
        "baseline_value": baseline_value,
        "max_multiplier": max_multiplier,
        "n_pool": len(pool),
        "n_dropped_zero": n_dropped_zero,
        "n_capped": n_capped,
        "multiplicity_histogram": dict(multiplicity_histogram),
        "n_simulator_error_excluded": n_simulator_error_excluded,
        "total_trajectory_multiplicity": sum(multiplicity_by_stem.values()),
    }
    return multiplicity_by_stem, report


def label_trajectories(
    metas: list[dict],
    *,
    min_act: int = 1,
    fallback_floor_quantile: float = 0.8,
    min_positives: int = 20,
) -> tuple[list[TrajectoryLabel], dict]:
    rows: list[dict[str, Any]] = []
    positive_flags: list[bool] = []

    for meta in metas:
        world_seed = _int_value(meta.get("world_seed", 0))
        rollout_index = _int_value(meta.get("rollout_index", 0))
        row = {
            "stem": f"seed_{world_seed}_r{rollout_index}",
            "world_seed": world_seed,
            "rollout_index": rollout_index,
            "outcome": _str_value(meta.get("outcome", "")),
            "final_act": _int_value(meta.get("final_act", 0)),
            "final_floor": _int_value(meta.get("final_floor", 0)),
            "stopped_reason": _str_value(meta.get("stopped_reason", "")),
            "n_invalid": _int_value(meta.get("n_invalid", 0)),
            "n_decisions": _int_value(meta.get("n_decisions", 0)),
        }
        rows.append(row)
        positive_flags.append(is_act_boss_clear(meta, min_act))

    n_positives = sum(1 for is_positive in positive_flags if is_positive)
    fallback_engaged = n_positives < min_positives
    threshold_floor: int | None = None

    if fallback_engaged:
        eligible_floors = sorted(
            row["final_floor"] for row in rows if row["stopped_reason"] != "simulator_error"
        )
        if eligible_floors:
            threshold_floor = _quantile(eligible_floors, fallback_floor_quantile)

    labels: list[TrajectoryLabel] = []
    for row, is_positive in zip(rows, positive_flags):
        if fallback_engaged:
            kept = (
                threshold_floor is not None
                and row["stopped_reason"] != "simulator_error"
                and row["final_floor"] >= threshold_floor
            )
            keep_reason = "fallback_topq_floor" if kept else "not_kept"
        else:
            kept = is_positive
            if not kept:
                keep_reason = "not_kept"
            elif "VICTORY" in row["outcome"]:
                keep_reason = "victory"
            else:
                keep_reason = "act_boss_clear"

        labels.append(
            TrajectoryLabel(
                stem=row["stem"],
                world_seed=row["world_seed"],
                rollout_index=row["rollout_index"],
                outcome=row["outcome"],
                final_act=row["final_act"],
                final_floor=row["final_floor"],
                stopped_reason=row["stopped_reason"],
                n_invalid=row["n_invalid"],
                n_decisions=row["n_decisions"],
                kept=kept,
                keep_reason=keep_reason,
            )
        )

    stopped_reason_counts = Counter(row["stopped_reason"] for row in rows)
    kept_stopped_reason_counts = Counter(label.stopped_reason for label in labels if label.kept)
    total_invalid = sum(row["n_invalid"] for row in rows)
    total_decisions = sum(row["n_decisions"] for row in rows)

    report = {
        "n_total": len(rows),
        "n_positives": n_positives,
        "n_kept": sum(1 for label in labels if label.kept),
        "min_act": min_act,
        "min_positives": min_positives,
        "fallback_floor_quantile": fallback_floor_quantile,
        "fallback_engaged": fallback_engaged,
        "threshold_floor": threshold_floor,
        "stopped_reason_counts": dict(stopped_reason_counts),
        "kept_stopped_reason_counts": dict(kept_stopped_reason_counts),
        "n_victory": sum(1 for row in rows if "VICTORY" in row["outcome"]),
        "n_act_boss_clear": n_positives,
        "agent_invalid_rate": total_invalid / total_decisions if total_decisions else 0.0,
    }
    return labels, report
