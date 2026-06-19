"""Advantage calculations for policy-gradient training labels."""
from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sts_ai.train.reward import TrajectoryLabel

__all__ = [
    "offline_advantages",
    "group_relative_advantages",
]


def _advantage_report_stats(advantages_by_stem: dict[str, float]) -> dict[str, float | None]:
    advantages = list(advantages_by_stem.values())
    if not advantages:
        return {
            "advantage_min": None,
            "advantage_max": None,
            "advantage_mean": None,
        }
    return {
        "advantage_min": min(advantages),
        "advantage_max": max(advantages),
        "advantage_mean": statistics.mean(advantages),
    }


def offline_advantages(
    labels: list[TrajectoryLabel],
    *,
    baseline: str = "median",
    exclude_simulator_errors: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return floor-minus-baseline advantages by trajectory stem."""
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
        floors = [label.final_floor for label in pool]
        if baseline == "median":
            baseline_value = statistics.median(floors)
        else:
            baseline_value = statistics.mean(floors)
    else:
        baseline_value = None

    advantages_by_stem: dict[str, float] = {}
    if baseline_value is not None:
        for label in pool:
            advantages_by_stem[label.stem] = float(label.final_floor) - baseline_value

    report = {
        "mode": "offline",
        "baseline_kind": baseline,
        "baseline_value": baseline_value,
        "n_pool": len(pool),
        "n_simulator_error_excluded": n_simulator_error_excluded,
        **_advantage_report_stats(advantages_by_stem),
    }
    return advantages_by_stem, report


def group_relative_advantages(
    labels: list[TrajectoryLabel],
    *,
    std_norm: bool = True,
    eps: float = 1e-6,
    exclude_simulator_errors: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return within-world-seed relative advantages by trajectory stem."""
    label_rows = list(labels)
    groups: defaultdict[int, list[TrajectoryLabel]] = defaultdict(list)
    n_simulator_error_excluded = 0

    for label in label_rows:
        if exclude_simulator_errors and label.stopped_reason == "simulator_error":
            n_simulator_error_excluded += 1
            continue
        groups[label.world_seed].append(label)

    advantages_by_stem: dict[str, float] = {}
    group_sizes: dict[int, int] = {}

    for world_seed, group in groups.items():
        if not group:
            continue
        group_sizes[world_seed] = len(group)
        floors = [label.final_floor for label in group]
        group_mean = statistics.mean(floors)
        group_std = statistics.pstdev(floors)

        for label in group:
            centered = float(label.final_floor) - group_mean
            if std_norm:
                if group_std == 0.0:
                    advantage = 0.0
                else:
                    advantage = centered / (group_std + eps)
            else:
                advantage = centered
            advantages_by_stem[label.stem] = float(advantage)

    report = {
        "mode": "group_relative",
        "std_norm": std_norm,
        "eps": eps,
        "n_groups": len(group_sizes),
        "group_sizes": group_sizes,
        "n_simulator_error_excluded": n_simulator_error_excluded,
        **_advantage_report_stats(advantages_by_stem),
    }
    return advantages_by_stem, report
