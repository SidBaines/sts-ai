from __future__ import annotations

import unittest

from scripts.report_search_teacher import build_report


def _row(
    *,
    five_k_action: int | None = 1,
    reference_action: int | None = 1,
    reference_tied: bool = False,
    five_k_fraction: float = 1.0,
    reference_fraction: float = 1.0,
    hidden_action: int | None = 1,
    hidden_fraction: float = 2 / 3,
) -> dict:
    return {
        "observation_version": "combat_public_v2",
        "teacher_privilege": "simulator_full_state",
        "teacher_selection_rule": "aggregated_root_visits",
        "window_id": "seed_1_r0_w0",
        "turn": 3,
        "public_state_hash": "public-hash",
        "base_action": {"display_index": 0},
        "teacher_queries": [
            {
                "observation_version": "combat_public_v2",
                "teacher_privilege": "simulator_full_state",
                "teacher_selection_rule": "aggregated_root_visits",
                "search": {"selection_method": "winning_sequence"},
            }
        ],
        "reference": {
            "budget_consensus": {
                "5000": {
                    "consensus_action_index": five_k_action,
                    "consensus_fraction": five_k_fraction,
                    "tied": five_k_action is None,
                    "unanimous": five_k_action is not None,
                },
                "50000": {
                    "consensus_action_index": reference_action,
                    "consensus_fraction": reference_fraction,
                    "tied": reference_tied,
                    "unanimous": reference_action is not None,
                },
            },
            "hidden_order_consensus": {
                "consensus_action_index": hidden_action,
                "consensus_fraction": hidden_fraction,
            },
        },
    }


class SearchTeacherReportTest(unittest.TestCase):
    def test_hidden_majority_for_different_action_is_not_eligible(self):
        row = _row(hidden_action=2)

        report = build_report([row])

        self.assertEqual(report["eligible_for_5k_collection"]["n"], 0)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 0)
        state = report["per_state"][0]
        self.assertEqual(state["hidden_consensus_action_index"], 2)
        self.assertFalse(state["hidden_action_matches_reference"])
        self.assertFalse(state["eligible_for_5k_collection"])
        self.assertFalse(state["eligible_for_direct_50k_collection"])

    def test_budget_disagreement_only_excludes_5k_collection(self):
        report = build_report([_row(five_k_action=2)])

        self.assertEqual(report["eligible_for_5k_collection"]["n"], 0)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 1)
        state = report["per_state"][0]
        self.assertFalse(state["eligible_for_5k_collection"])
        self.assertTrue(state["eligible_for_direct_50k_collection"])

    def test_stable_matching_row_is_eligible_for_both_collections(self):
        report = build_report([_row()])

        self.assertEqual(report["observation_version"], "combat_public_v2")
        self.assertEqual(
            report["teacher_selection_rule"],
            "aggregated_root_visits",
        )
        self.assertEqual(
            report["native_search_selection_method_counts"],
            {"winning_sequence": 1},
        )
        self.assertEqual(report["eligible_for_5k_collection"]["n"], 1)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 1)

    def test_tied_reference_is_not_eligible_even_if_action_is_populated(self):
        report = build_report([_row(reference_tied=True)])

        self.assertEqual(report["eligible_for_5k_collection"]["n"], 0)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 0)

    def test_hidden_consensus_below_threshold_is_not_eligible(self):
        report = build_report([_row(hidden_fraction=0.65)])

        self.assertEqual(report["eligible_for_5k_collection"]["n"], 0)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 0)

    def test_reference_consensus_below_threshold_is_not_eligible(self):
        report = build_report([_row(reference_fraction=0.5)])

        self.assertEqual(report["eligible_for_5k_collection"]["n"], 0)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 0)

    def test_low_5k_consensus_only_excludes_5k_collection(self):
        report = build_report([_row(five_k_fraction=0.5)])

        self.assertEqual(report["eligible_for_5k_collection"]["n"], 0)
        self.assertEqual(report["eligible_for_direct_50k_collection"]["n"], 1)

    def test_direct_50k_eligibility_does_not_follow_a_larger_budget(self):
        row = _row()
        row["reference"]["budget_consensus"]["100000"] = {
            "consensus_action_index": 2,
            "tied": False,
            "unanimous": True,
        }

        report = build_report([row])

        state = report["per_state"][0]
        self.assertEqual(state["reference_action_index"], 2)
        self.assertEqual(state["fifty_k_action_index"], 1)
        self.assertTrue(state["eligible_for_direct_50k_collection"])

    def test_rejects_mixed_observation_or_selection_contract(self):
        v1 = _row()
        v1["observation_version"] = "combat_public_v1"
        with self.assertRaisesRegex(ValueError, "observation_version"):
            build_report([v1])

        native = _row()
        native["teacher_selection_rule"] = "native_selected_action"
        native["teacher_queries"][0][
            "teacher_selection_rule"
        ] = "native_selected_action"
        with self.assertRaisesRegex(ValueError, "mix selection rules"):
            build_report([_row(), native])


if __name__ == "__main__":
    unittest.main()
