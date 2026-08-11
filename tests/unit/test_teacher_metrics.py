from __future__ import annotations

from dataclasses import replace
import unittest

from sts_ai.teacher_metrics import (
    build_metrics_report,
    score_choice,
    state_visit_stats,
)


def _query(
    visits: dict[str, int],
    sequence: list[tuple[str, int]],
    *,
    abstained: bool = False,
) -> dict:
    return {
        "teacher_vote": {
            "abstained": abstained,
            "displayed_action_visits": visits,
        },
        "search": {
            "best_sequence": [
                {"bits": offset, "description": description, "turn": turn}
                for offset, (description, turn) in enumerate(sequence)
            ]
        },
    }


def _audit_row(
    *,
    public_state_hash: str = "hash-a",
    window_id: str = "window-a",
    actions: tuple[str, ...] = ("strike", "end turn", "defend"),
    queries: list[dict] | None = None,
) -> dict:
    if queries is None:
        queries = [
            _query(
                {"0": 50, "1": 40, "2": 10},
                [("strike", 0), ("end turn", 0), ("future action", 1)],
            ),
            _query(
                {"0": 10, "1": 20, "2": 0},
                [("end turn", 0)],
            ),
            _query(
                {"0": 0, "1": 1000, "2": 0},
                [("defend", 0)],
                abstained=True,
            ),
        ]
    return {
        "public_state_hash": public_state_hash,
        "window_id": window_id,
        "turn": 0,
        "legal_actions": [
            {"index": index, "bits": index + 10, "description": description}
            for index, description in enumerate(actions)
        ],
        "teacher_queries": queries,
    }


class StateVisitStatsTest(unittest.TestCase):
    def test_aggregates_non_abstaining_visits_and_turn_set(self):
        stats = state_visit_stats(_audit_row())

        self.assertEqual(stats.total_visits, 130)
        self.assertEqual(stats.consensus_action_index, 0)
        self.assertEqual(stats.top_set, (0, 1))
        self.assertAlmostEqual(stats.visit_share[0], 60 / 130)
        self.assertAlmostEqual(stats.visit_share[1], 60 / 130)
        self.assertAlmostEqual(stats.visit_share[2], 10 / 130)
        self.assertEqual(stats.margin, 0.0)
        self.assertEqual(
            stats.turn_set_descriptions,
            frozenset({"strike", "end turn"}),
        )
        self.assertNotIn("defend", stats.turn_set_descriptions)

    def test_skips_collector_shaped_draw_index_abstention(self):
        abstention = {
            "teacher_vote": {
                "action_index": None,
                "abstained": True,
                "reason": "unsupported_draw_index_hidden_order_intervention",
            },
            "search": {"error": "x", "selection_method": "abstain"},
        }
        row = _audit_row(
            actions=("strike", "end turn"),
            queries=[
                _query(
                    {"0": 60, "1": 40},
                    [("strike", 0), ("end turn", 0)],
                ),
                abstention,
            ],
        )

        stats = state_visit_stats(row)

        self.assertEqual(stats.total_visits, 100)
        self.assertEqual(stats.consensus_action_index, 0)
        self.assertEqual(stats.top_set, (0,))
        self.assertEqual(
            stats.turn_set_descriptions,
            frozenset({"strike", "end turn"}),
        )

    def test_tie_ratio_boundary_is_in_top_set(self):
        row = _audit_row(
            actions=("first", "second"),
            queries=[_query({"0": 50, "1": 40}, [])],
        )
        stats = state_visit_stats(row, tie_ratio=0.8)
        self.assertEqual(stats.top_set, (0, 1))
        self.assertAlmostEqual(stats.margin, 1 / 9)

    def test_tie_ratio_ulp_boundary_is_in_top_set(self):
        row = _audit_row(
            actions=("first", "second"),
            queries=[_query({"0": 100, "1": 7}, [])],
        )

        stats = state_visit_stats(row, tie_ratio=0.07)

        self.assertIn(1, stats.top_set)

    def test_single_legal_action_has_zero_margin(self):
        row = _audit_row(
            actions=("end turn",),
            queries=[_query({"0": 7}, [("end turn", 0)])],
        )
        stats = state_visit_stats(row)
        self.assertEqual(stats.margin, 0.0)
        self.assertEqual(stats.top_set, (0,))

    def test_score_choice_resolves_exact_legal_description(self):
        stats = state_visit_stats(_audit_row())
        end_turn = score_choice(stats, 1)
        defend = score_choice(stats, 2)

        self.assertFalse(end_turn["strict_top1"])
        self.assertTrue(end_turn["in_top_set"])
        self.assertTrue(end_turn["in_turn_set"])
        self.assertEqual(end_turn["regret_visit_share"], 0.0)
        self.assertFalse(defend["in_turn_set"])
        self.assertAlmostEqual(defend["regret_visit_share"], 50 / 130)

    def test_matching_supplied_description_resolves_and_scores(self):
        stats = state_visit_stats(_audit_row())

        supplied = score_choice(stats, 1, chosen_description="end turn")

        self.assertEqual(supplied, score_choice(stats, 1))

    def test_detached_stats_without_description_fail_closed(self):
        stats = replace(state_visit_stats(_audit_row()))

        supplied = score_choice(stats, 1, chosen_description="end turn")
        self.assertTrue(supplied["in_turn_set"])
        with self.assertRaisesRegex(
            ValueError,
            "chosen_description_unavailable",
        ):
            score_choice(stats, 1)

    def test_zero_non_abstaining_visits_fails_closed(self):
        row = _audit_row(
            actions=("end turn",),
            queries=[_query({"0": 10}, [], abstained=True)],
        )
        with self.assertRaisesRegex(ValueError, "zero_non_abstaining_visits"):
            state_visit_stats(row)


class MetricsReportTest(unittest.TestCase):
    def test_supplied_description_must_match_audit_legal_action(self):
        row = _audit_row()

        with self.assertRaisesRegex(
            ValueError,
            "chosen_description_mismatch:",
        ):
            build_metrics_report(
                [
                    {
                        "public_state_hash": "hash-a",
                        "chosen_index": 1,
                        "chosen_description": "strike",
                    }
                ],
                {"hash-a": row},
            )

    def test_quarantine_changes_clean_but_not_all(self):
        clean = _audit_row(public_state_hash="clean", window_id="clean-window")
        quarantined = _audit_row(
            public_state_hash="bad",
            window_id="bad-window",
        )
        choices = [
            {"public_state_hash": "clean", "chosen_index": 0},
            {"public_state_hash": "bad", "chosen_index": 2},
        ]
        audit = {"clean": clean, "bad": quarantined}

        unfiltered = build_metrics_report(choices, audit)
        filtered = build_metrics_report(
            choices,
            audit,
            quarantined_windows=frozenset({"bad-window"}),
        )

        self.assertEqual(unfiltered["overall"]["all"], filtered["overall"]["all"])
        self.assertEqual(filtered["overall"]["all"]["n"], 2)
        self.assertEqual(filtered["overall"]["clean"]["n"], 1)
        self.assertEqual(filtered["overall"]["clean"]["strict_top1_rate"], 1.0)
        self.assertEqual(set(filtered["per_window"]), {"bad-window", "clean-window"})
        self.assertEqual(filtered["rows"][1]["window_id"], "bad-window")

    def test_all_quarantined_produces_defined_empty_aggregate(self):
        row = _audit_row()
        report = build_metrics_report(
            [{"public_state_hash": "hash-a", "chosen_index": 0}],
            {"hash-a": row},
            quarantined_windows=frozenset({"window-a"}),
        )
        self.assertEqual(
            report["overall"]["clean"],
            {
                "n": 0,
                "mean_regret_visit_share": None,
                "top_set_rate": None,
                "turn_set_rate": None,
                "strict_top1_rate": None,
                "mean_margin": None,
            },
        )

    def test_missing_hash_join_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "audit_row_missing_for_hash"):
            build_metrics_report(
                [{"public_state_hash": "missing", "chosen_index": 0}],
                {},
            )


if __name__ == "__main__":
    unittest.main()
