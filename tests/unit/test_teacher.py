from __future__ import annotations

import unittest

from sts_ai.schemas import LegalAction
from sts_ai.teacher import (
    consensus_summary,
    displayed_action_index,
    public_observation_hash,
    teacher_label,
)


class TeacherBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = [
            LegalAction(index=0, bits=7, description="play Strike -> NOB"),
            LegalAction(index=1, bits=99, description="end turn"),
        ]

    def test_public_hash_uses_visible_text_and_order_but_not_native_bits(self):
        base = public_observation_hash("state", self.actions)
        same_visible = [
            LegalAction(index=0, bits=1234, description="play Strike -> NOB"),
            LegalAction(index=1, bits=5678, description="end turn"),
        ]
        self.assertEqual(base, public_observation_hash("state", same_visible))
        self.assertNotEqual(base, public_observation_hash("different", self.actions))
        self.assertNotEqual(base, public_observation_hash("state", list(reversed(self.actions))))

    def test_display_mapping_prefers_description_for_deduplicated_card_copy(self):
        self.assertEqual(
            displayed_action_index(
                self.actions,
                bits=444,  # search selected a display-equivalent second Strike
                description="play Strike -> NOB",
            ),
            0,
        )

    def test_display_mapping_falls_back_to_bits_and_fails_closed(self):
        self.assertEqual(
            displayed_action_index(self.actions, bits=99, description="old wording"),
            1,
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            displayed_action_index(self.actions, bits=55, description="missing")

    def test_teacher_label_is_json_serializable_and_action_only(self):
        row = teacher_label(
            state_text="public state",
            legal_actions=self.actions,
            search_result={
                "action": object(),
                "bits": 7,
                "description": "play Strike -> NOB",
                "simulations": 5000,
                "search_seed": 42,
                "root_edges": [{"action": object(), "bits": 7, "visits": 4000}],
            },
        )
        self.assertEqual(row["teacher_action_index"], 0)
        self.assertEqual(row["target"], '{"action_index":0}')
        self.assertNotIn("action", row["search"])
        self.assertNotIn("action", row["search"]["root_edges"][0])
        self.assertEqual(row["teacher_privilege"], "simulator_full_state")

    def test_consensus_reports_ties_and_unanimity(self):
        rows = [
            {"public_state_hash": "h", "teacher_action_index": 0},
            {"public_state_hash": "h", "teacher_action_index": 0},
            {"public_state_hash": "h", "teacher_action_index": 1},
        ]
        summary = consensus_summary(rows)
        self.assertEqual(summary["consensus_action_index"], 0)
        self.assertEqual(summary["consensus_fraction"], 2 / 3)
        self.assertFalse(summary["unanimous"])

        tie = consensus_summary(rows[1:])
        self.assertIsNone(tie["consensus_action_index"])
        self.assertTrue(tie["tied"])

    def test_root_vote_override_and_abstentions_are_auditable(self):
        selected = teacher_label(
            state_text="public state",
            legal_actions=self.actions,
            search_result={
                "bits": 7,
                "description": "play Strike -> NOB",
                "root_edges": [],
            },
            teacher_vote={
                "action_index": 1,
                "abstained": False,
                "reason": None,
                "displayed_action_visits": {"0": 2, "1": 8},
            },
            selection_rule="aggregated_root_visits",
        )
        self.assertEqual(selected["teacher_action_index"], 1)
        self.assertEqual(selected["teacher_action_bits"], 99)
        self.assertEqual(selected["target"], '{"action_index":1}')
        self.assertEqual(selected["teacher_selection_rule"], "aggregated_root_visits")
        # Raw native selection remains intact in search provenance rather than
        # being rewritten to look like the root-visit decision.
        self.assertEqual(selected["search"]["bits"], 7)

        abstained = teacher_label(
            state_text="public state",
            legal_actions=self.actions,
            search_result={"bits": 7, "description": "play Strike -> NOB"},
            teacher_vote={
                "action_index": None,
                "abstained": True,
                "reason": "tied_max_visits",
            },
            selection_rule="aggregated_root_visits",
        )
        self.assertIsNone(abstained["teacher_action_index"])
        self.assertIsNone(abstained["target"])
        summary = consensus_summary([selected, abstained, selected])
        self.assertEqual(summary["consensus_action_index"], 1)
        self.assertEqual(summary["consensus_fraction"], 2 / 3)
        self.assertEqual(summary["n_abstentions"], 1)
        self.assertFalse(summary["unanimous"])

    def test_consensus_rejects_different_public_states(self):
        with self.assertRaisesRegex(ValueError, "one public state"):
            consensus_summary(
                [
                    {"public_state_hash": "a", "teacher_action_index": 0},
                    {"public_state_hash": "b", "teacher_action_index": 0},
                ]
            )


if __name__ == "__main__":
    unittest.main()
