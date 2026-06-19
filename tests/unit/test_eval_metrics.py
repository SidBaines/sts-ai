from __future__ import annotations

import unittest
from typing import Any

from sts_ai.eval_metrics import aggregate_stall_metrics, stall_metrics


def _record(state_text: str, action_index: int, decision_index: int = 0) -> dict[str, Any]:
    return {
        "state_text": state_text,
        "selected_action": {"index": action_index},
        "phase": "combat",
        "decision_index": decision_index,
    }


class StallMetricsTest(unittest.TestCase):
    def test_counts_consecutive_repeats_and_longest_streak(self) -> None:
        records = [
            _record("same state", 2, 0),
            _record("same state", 2, 1),
            _record("same state", 2, 2),
            _record("other state", 2, 3),
            _record("other state", 3, 4),
        ]

        result = stall_metrics(records)

        self.assertEqual(result["n_decisions"], 5)
        self.assertEqual(result["repeated_consecutive"], 2)
        self.assertEqual(result["longest_repeat_streak"], 3)
        self.assertEqual(result["distinct_state_fraction"], 2 / 5)

    def test_all_distinct_states_have_no_consecutive_repeats(self) -> None:
        records = [
            _record("state 1", 0, 0),
            _record("state 2", 0, 1),
            _record("state 3", 0, 2),
        ]

        result = stall_metrics(records)

        self.assertEqual(result["repeated_consecutive"], 0)
        self.assertEqual(result["longest_repeat_streak"], 1)
        self.assertEqual(result["distinct_state_fraction"], 1.0)

    def test_empty_rollout_returns_zeros(self) -> None:
        result = stall_metrics([])

        self.assertEqual(
            result,
            {
                "n_decisions": 0,
                "repeated_consecutive": 0,
                "longest_repeat_streak": 0,
                "distinct_state_fraction": 0.0,
            },
        )

    def test_missing_selected_action_is_tolerated(self) -> None:
        records: list[dict[str, Any]] = [
            {"state_text": "same state", "phase": "combat", "decision_index": 0},
            {"state_text": "same state", "phase": "combat", "decision_index": 1},
        ]

        result = stall_metrics(records)

        self.assertEqual(result["repeated_consecutive"], 1)
        self.assertEqual(result["longest_repeat_streak"], 2)
        self.assertEqual(result["distinct_state_fraction"], 0.5)


class AggregateStallMetricsTest(unittest.TestCase):
    def test_averages_non_empty_rollouts_and_skips_empty_ones(self) -> None:
        repeat_rollout = [
            _record("same state", 2, 0),
            _record("same state", 2, 1),
            _record("same state", 2, 2),
            _record("other state", 2, 3),
            _record("other state", 3, 4),
        ]
        distinct_rollout = [
            _record("state 1", 0, 0),
            _record("state 2", 0, 1),
            _record("state 3", 0, 2),
        ]

        result = aggregate_stall_metrics([repeat_rollout, [], distinct_rollout])

        self.assertEqual(result["n_rollouts"], 2)
        self.assertEqual(result["mean_repeated_consecutive"], 1.0)
        self.assertEqual(result["mean_longest_repeat_streak"], 2.0)
        self.assertEqual(result["mean_distinct_state_fraction"], 0.7)
        self.assertEqual(result["mean_decisions"], 4.0)

    def test_empty_input_returns_zeros(self) -> None:
        result = aggregate_stall_metrics([])

        self.assertEqual(
            result,
            {
                "n_rollouts": 0,
                "mean_repeated_consecutive": 0.0,
                "mean_longest_repeat_streak": 0.0,
                "mean_distinct_state_fraction": 0.0,
                "mean_decisions": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
