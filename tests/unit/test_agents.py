import unittest

from sts_ai.agents import MlxQwenJsonAgent, RandomLegalAgent, parse_json_action
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

    def test_extracts_json_after_think_block_with_braces(self):
        text = '<think>{"not": "the answer"}</think>\n{"reasoning": "final", "action_index": 1}'
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.reasoning, "final")

    def test_extracts_last_balanced_json_object(self):
        text = '{"debug": true}\nfinal answer: {"reasoning": "choose second", "action_index": 1}'
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)

    def test_ignores_braces_inside_json_strings(self):
        text = 'prefix {"reasoning": "this string has {braces}", "action_index": 0} suffix'
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)

    def test_captures_thinking_chain_of_thought(self):
        text = (
            "<think>\nThe second option preserves HP, which matters at low health.\n</think>\n\n"
            '{"reasoning": "preserve hp", "action_index": 1}'
        )
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.reasoning, "preserve hp")  # brief JSON field
        self.assertIn("preserves HP", decision.thinking)  # full CoT captured separately

    def test_captures_truncated_unclosed_thinking(self):
        # thinking-mode generation that ran out of budget mid-<think>: no JSON,
        # but the partial chain-of-thought must still be retained.
        text = "<think>\nLet me weigh the options. The first action is risky because"
        decision = parse_json_action(text, self.actions)
        self.assertFalse(decision.valid)
        self.assertEqual(decision.metadata["error"], "no json object")
        self.assertIn("weigh the options", decision.thinking)

    def test_no_thinking_block_leaves_thinking_empty(self):
        decision = parse_json_action('{"reasoning": "ok", "action_index": 0}', self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.thinking, "")


class RandomLegalAgentSeedTest(unittest.TestCase):
    def setUp(self):
        self.actions = [
            LegalAction(index=0, bits=1, description="first"),
            LegalAction(index=1, bits=2, description="second"),
            LegalAction(index=2, bits=4, description="third"),
        ]

    def test_reseed_resets_random_sequence(self):
        agent = RandomLegalAgent()
        agent.reseed(12345)
        seq_a = [agent.choose_action("state", self.actions).action_index for _ in range(8)]
        agent.reseed(12345)
        seq_b = [agent.choose_action("state", self.actions).action_index for _ in range(8)]
        agent.reseed(67890)
        seq_c = [agent.choose_action("state", self.actions).action_index for _ in range(8)]

        self.assertEqual(seq_a, seq_b)
        self.assertNotEqual(seq_a, seq_c)


class MlxQwenJsonAgentRetryTest(unittest.TestCase):
    def test_retries_after_invalid_json(self):
        actions = [LegalAction(index=0, bits=1, description="first")]
        agent = object.__new__(MlxQwenJsonAgent)
        agent.framing = "neutral"
        agent.max_retries = 1
        # Stub the tokenizer-backed seams so this stays a pure unit test (no mlx).
        agent._apply_chat_template = lambda prompt: prompt
        agent._count_tokens = lambda text: 0
        responses = iter(["not json", '{"reasoning": "fixed", "action_index": 0}'])
        agent._generate_chat = lambda chat_prompt: next(responses)

        decision = agent.choose_action("state", actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.reasoning, "fixed")
        self.assertEqual(decision.retries, 1)
        self.assertGreaterEqual(decision.latency_s, 0.0)  # timing populated


if __name__ == "__main__":
    unittest.main()
