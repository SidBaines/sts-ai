"""Decision-level evaluation metrics for degenerate rollout loops."""
from __future__ import annotations

from typing import Any, Sequence

__all__ = ["aggregate_stall_metrics", "stall_metrics"]

MetricValue = int | float
Signature = tuple[Any, Any]


def _signature(record: dict[str, Any]) -> Signature:
    selected_action = record.get("selected_action", {})
    return (record.get("state_text", ""), selected_action.get("index"))


def stall_metrics(records: Sequence[dict[str, Any]]) -> dict[str, MetricValue]:
    """Compute consecutive-repeat stall proxies for one rollout."""
    n_decisions = len(records)
    if n_decisions == 0:
        return {
            "n_decisions": 0,
            "repeated_consecutive": 0,
            "longest_repeat_streak": 0,
            "distinct_state_fraction": 0.0,
        }

    repeated_consecutive = 0
    longest_repeat_streak = 1
    current_streak = 1
    previous_signature = _signature(records[0])

    for record in records[1:]:
        signature = _signature(record)
        if signature == previous_signature:
            repeated_consecutive += 1
            current_streak += 1
            longest_repeat_streak = max(longest_repeat_streak, current_streak)
        else:
            current_streak = 1
        previous_signature = signature

    distinct_states = {record.get("state_text", "") for record in records}
    distinct_state_fraction = len(distinct_states) / n_decisions

    return {
        "n_decisions": n_decisions,
        "repeated_consecutive": repeated_consecutive,
        "longest_repeat_streak": longest_repeat_streak,
        "distinct_state_fraction": distinct_state_fraction,
    }


def aggregate_stall_metrics(
    rollouts: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, MetricValue]:
    """Average stall proxies over non-empty rollouts."""
    metrics = [stall_metrics(records) for records in rollouts if records]
    n_rollouts = len(metrics)
    if n_rollouts == 0:
        return {
            "n_rollouts": 0,
            "mean_repeated_consecutive": 0.0,
            "mean_longest_repeat_streak": 0.0,
            "mean_distinct_state_fraction": 0.0,
            "mean_decisions": 0.0,
        }

    return {
        "n_rollouts": n_rollouts,
        "mean_repeated_consecutive": sum(
            metric["repeated_consecutive"] for metric in metrics
        )
        / n_rollouts,
        "mean_longest_repeat_streak": sum(
            metric["longest_repeat_streak"] for metric in metrics
        )
        / n_rollouts,
        "mean_distinct_state_fraction": sum(
            metric["distinct_state_fraction"] for metric in metrics
        )
        / n_rollouts,
        "mean_decisions": sum(metric["n_decisions"] for metric in metrics) / n_rollouts,
    }
