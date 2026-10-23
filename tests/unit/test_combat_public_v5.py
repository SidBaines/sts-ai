from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import Mock

from sts_env._backend.combat_public_v5 import CombatPublicV5Env, selection_label

from sts_env._backend.schemas import LegalAction


class CombatPublicV5Test(unittest.TestCase):
    def test_multiselect_labels_use_actual_mask_and_public_card_instances(self):
        hand = [{"index": 0, "label": "Defend [Skill] (cost 1)"},
                {"index": 2, "label": "Defend+ [Skill] (cost 1)"}]
        for task, verb in (("EXHAUST_MANY", "exhaust"), ("GAMBLE", "discard and redraw")):
            action = LegalAction(0, 1610612741, "{ " + task + " broken native indices }")
            self.assertEqual(selection_label(action, hand),
                             f"{verb}: hand [0] Defend [Skill] (cost 1); hand [2] Defend+ [Skill] (cost 1)")
        self.assertEqual(selection_label(LegalAction(0, 1610612736, "{ EXHAUST_MANY none"), []), "exhaust no cards")

    def test_all_selection_surfaces_and_execution_keep_identity(self):
        base = Mock(combat_observation="combat_public_v3")
        base.phase.return_value = "combat"
        base.legal_actions.return_value = [LegalAction(0, 1610612736, "{ EXHAUST_MANY none")]
        base.step.side_effect = lambda i: base.legal_actions()[i]
        base.public_combat_cards.return_value = {"hand": [], "selection": {"type": "EXHAUST_MANY", "candidates": ["{ EXHAUST_MANY none"]}}
        base.summary.return_value = {"ascension": 0, "combat_observation": "combat_public_v3", "combat": {"cards": base.public_combat_cards(), "recent_actions": ["{ EXHAUST_MANY none"]}}
        base.describe_state.return_value = "Selection type: EXHAUST_MANY\n  - { EXHAUST_MANY none\nRecent actions this turn: { EXHAUST_MANY none"
        original = deepcopy(base.summary())
        env = CombatPublicV5Env(base)
        action = env.legal_actions()[0]
        self.assertEqual(action.description, "exhaust no cards")
        self.assertEqual(env.step(0), action)
        self.assertEqual(env.public_combat_cards()["selection"]["candidates"], [action.description])
        self.assertEqual(env.summary()["combat"]["recent_actions"], [action.description])
        self.assertNotIn("{ EXHAUST_MANY", env.describe_state())
        self.assertEqual(base.summary(), original)
        expected = {"phase": "combat", "state": original, "legal_actions": [vars(base.legal_actions()[0])], "state_text": "old"}
        actual = {"phase": "combat", "state": env.summary(), "legal_actions": [vars(action)], "state_text": "new"}
        self.assertEqual([(x["index"], x["bits"]) for x in actual["legal_actions"]],
                         [(x["index"], x["bits"]) for x in expected["legal_actions"]])
        actual["legal_actions"][0]["bits"] += 1
        self.assertNotEqual(actual["legal_actions"][0]["bits"], expected["legal_actions"][0]["bits"])
