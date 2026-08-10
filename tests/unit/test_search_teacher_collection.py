from __future__ import annotations

import unittest

from scripts.collect_search_teacher import _source_action, _validate_independent_seeds
from sts_ai.interactive.replay import ReplayError
from sts_ai.schemas import LegalAction


class _FakeEnv:
    def __init__(self, actions):
        self._actions = actions

    def legal_actions(self):
        return self._actions


class SearchTeacherCollectionTest(unittest.TestCase):
    def test_rejects_duplicate_votes(self):
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            _validate_independent_seeds([2, 2, 3], name="--search-seeds")

    def test_rejects_zero_one_search_rng_alias(self):
        with self.assertRaisesRegex(ValueError, "0 and 1 alias"):
            _validate_independent_seeds(
                [0, 1, 2],
                name="--search-seeds",
                default_random_engine=True,
            )

    def test_accepts_empirically_distinct_search_seeds(self):
        _validate_independent_seeds(
            [1, 2, 3],
            name="--search-seeds",
            default_random_engine=True,
        )

    def test_source_action_rejects_same_bits_with_different_semantics(self):
        actions = [LegalAction(index=0, bits=1, description="play Strike (cost 1)")]
        record = {
            "selected_action": {
                "index": 0,
                "bits": 1,
                "description": "play Defend (cost 1)",
            }
        }
        with self.assertRaises(ReplayError):
            _source_action(_FakeEnv(actions), record, actions)

    def test_source_action_allows_historical_upgrade_adornment_and_records_live(self):
        actions = [
            LegalAction(
                index=0,
                bits=3,
                description="play Bash+ [Attack] (cost 2) -> NOB",
            )
        ]
        record = {
            "selected_action": {
                "index": 0,
                "bits": 3,
                "description": "play Bash (cost 2) -> NOB",
            }
        }

        index, provenance = _source_action(_FakeEnv(actions), record, actions)

        self.assertEqual(index, 0)
        self.assertEqual(
            provenance["description"],
            "play Bash+ [Attack] (cost 2) -> NOB",
        )
        self.assertEqual(provenance["recorded_description"], "play Bash (cost 2) -> NOB")

    def test_source_action_allows_v2_selection_type_and_cost_under_exact_bits(self):
        actions = [
            LegalAction(
                index=0,
                bits=1073741824,
                description="select card for DUAL_WIELD: Strike+ [Attack] (cost 0, base 1)",
            )
        ]
        record = {
            "selected_action": {
                "index": 0,
                "bits": 1073741824,
                "description": "select card for DUAL_WIELD: Strike",
            }
        }

        index, provenance = _source_action(_FakeEnv(actions), record, actions)

        self.assertEqual(index, 0)
        self.assertIn("[Attack]", provenance["description"])


if __name__ == "__main__":
    unittest.main()
