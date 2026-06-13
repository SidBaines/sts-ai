import unittest

from sts_ai.agents import parse_json_action
from sts_ai.schemas import LegalAction


class ParseJsonActionTest(unittest.TestCase):
    def setUp(self):
        self.actions = [
            LegalAction(index=0, bits=1, description="first"),
            LegalAction(index=1, bits=2, description="second"),
        ]

    def test_parses_exact_json(self):
        decision = parse_json_action('{"reasoning": "take second", "action_index": 1}', self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.reasoning, "take second")

    def test_falls_back_on_invalid_index(self):
        decision = parse_json_action('{"reasoning": "bad", "action_index": 99}', self.actions)
        self.assertFalse(decision.valid)
        self.assertEqual(decision.action_index, 0)

    def test_extracts_json_from_extra_text(self):
        decision = parse_json_action('Answer: {"reasoning": "ok", "action_index": 0}', self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)


if __name__ == "__main__":
    unittest.main()
