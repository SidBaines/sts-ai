from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts.collect_search_bank import SearchAgent, search_json
from sts_ai.schemas import LegalAction


class SearchBankAgentTest(unittest.TestCase):
    def test_native_duplicate_uses_v3_description_to_choose_display_index(self):
        calls = []

        class NativeAction:
            def describe(self, context, **kwargs):
                calls.append((context, kwargs))
                return "Strike (Attack) -> Jaw Worm (6 damage)"

        def search(simulations, **kwargs):
            self.assertEqual((simulations, kwargs), (2000, {"search_seed": 1}))
            return {"action": NativeAction(), "bits": 99, "description": "v2 description",
                    "root_edges": [{"action": NativeAction(), "bits": 99, "visits": 4}]}

        env = SimpleNamespace(bc=object(), search_best_combat_action=search)
        agent = SearchAgent(env, 2000)
        menu = [LegalAction(0, 0, "End Turn"),
                LegalAction(1, 17, "Strike (Attack) -> Jaw Worm (6 damage)")]
        result = agent.choose_action("", menu, phase="combat")
        self.assertEqual(result.action_index, 1)
        self.assertEqual(calls, [(env.bc, {"include_card_type": True, "include_computed_damage": True})])
        self.assertEqual(result.metadata["search"]["root_edges"], [{"bits": 99, "visits": 4}])
        self.assertNotIn("action", result.metadata["search"])

    def test_ooc_does_not_query_search(self):
        agent = SearchAgent(None, 2000)
        result = agent.choose_action("", [LegalAction(0, 1, "Skip"),
                                          LegalAction(1, 2, "Take gold")], phase="out_of_combat")
        self.assertEqual(result.action_index, 1)

    def test_unmatched_search_action_fails_closed(self):
        env = SimpleNamespace(bc=None, search_best_combat_action=lambda *a, **k: {
            "action": SimpleNamespace(describe=lambda *a, **k: "missing"), "bits": 9})
        with self.assertRaisesRegex(ValueError, "absent"):
            SearchAgent(env, 2000).choose_action("", [LegalAction(0, 0, "End Turn")], phase="combat")

    def test_search_json_preserves_nested_diagnostics(self):
        self.assertEqual(search_json({"best_sequence": [{"action": object(), "turn": 2}],
                                      "provenance": {"search_seed": 1}}),
                         {"best_sequence": [{"turn": 2}], "provenance": {"search_seed": 1}})
