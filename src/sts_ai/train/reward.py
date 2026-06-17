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
from typing import Any

__all__ = ["TrajectoryLabel", "is_act_boss_clear", "label_trajectories"]


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
