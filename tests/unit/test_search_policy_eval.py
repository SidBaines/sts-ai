from __future__ import annotations

import unittest

from sts_ai.search_policy_eval import action_vote_consensus, root_visit_vote


class SearchPolicyEvalTest(unittest.TestCase):
    def test_root_visits_aggregate_display_equivalent_native_edges(self):
        legal = [
            {"index": 0, "bits": 10, "description": "play Defend (cost 1)"},
            {"index": 1, "bits": 20, "description": "play Bash -> NOB"},
        ]
        result = {
            "root_edges": [
                {
                    "bits": 10,
                    "description": "play Defend (cost 1)",
                    "valid": True,
                    "visits": 6,
                },
                {
                    "bits": 11,
                    "description": "play Defend (cost 1)",
                    "valid": True,
                    "visits": 5,
                },
                {
                    "bits": 20,
                    "description": "play Bash -> NOB",
                    "valid": True,
                    "visits": 10,
                },
                {
                    "bits": 30,
                    "description": "invalid",
                    "valid": False,
                    "visits": 999,
                },
            ]
        }

        vote = root_visit_vote(result, legal)

        self.assertEqual(vote["action_index"], 0)
        self.assertEqual(vote["displayed_action_visits"], {"0": 11, "1": 10})
        self.assertEqual(vote["displayed_action_edge_counts"], {"0": 2, "1": 1})
        self.assertEqual(vote["n_invalid_edges"], 1)
        self.assertFalse(vote["abstained"])

    def test_root_visit_exact_tie_abstains(self):
        legal = [
            {"index": 0, "bits": 10, "description": "A"},
            {"index": 1, "bits": 20, "description": "B"},
        ]
        result = {
            "root_edges": [
                {"bits": 10, "description": "A", "valid": True, "visits": 5},
                {"bits": 20, "description": "B", "valid": True, "visits": 5},
            ]
        }

        vote = root_visit_vote(result, legal)

        self.assertIsNone(vote["action_index"])
        self.assertTrue(vote["abstained"])
        self.assertEqual(vote["reason"], "tied_max_visits")
        self.assertEqual(vote["tied_action_indices"], [0, 1])

    def test_unmapped_valid_edge_fails_closed(self):
        legal = [{"index": 0, "bits": 10, "description": "A"}]
        result = {
            "root_edges": [
                {"bits": 10, "description": "A", "valid": True, "visits": 9},
                {"bits": 20, "description": "B", "valid": True, "visits": 1},
            ]
        }

        vote = root_visit_vote(result, legal)

        self.assertIsNone(vote["action_index"])
        self.assertEqual(vote["reason"], "unmapped_valid_root_edge")
        self.assertEqual(len(vote["unmapped_edges"]), 1)

    def test_consensus_two_votes_and_abstention_passes(self):
        consensus = action_vote_consensus([2, 2, None])

        self.assertTrue(consensus["eligible"])
        self.assertEqual(consensus["action_index"], 2)
        self.assertEqual(consensus["consensus_fraction"], 2 / 3)
        self.assertEqual(consensus["n_abstentions"], 1)
        self.assertFalse(consensus["unanimous"])

    def test_consensus_tie_or_single_vote_fails_closed(self):
        tied = action_vote_consensus([0, 1, None])
        single = action_vote_consensus([0, None, None])

        self.assertFalse(tied["eligible"])
        self.assertEqual(tied["reason"], "tied_seed_vote")
        self.assertFalse(single["eligible"])
        self.assertEqual(single["reason"], "below_consensus_threshold")


if __name__ == "__main__":
    unittest.main()
