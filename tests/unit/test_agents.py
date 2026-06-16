import time
import types
import unittest

from sts_ai.agents import MlxQwenJsonAgent, RandomLegalAgent, VllmJsonAgent, parse_json_action
from sts_ai.prompting import NEUTRAL_FRAME
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
        self.assertTrue(decision.metadata["thinking_closed"])
        start, end = decision.metadata["json_span"]
        self.assertEqual(text[start:end], '{"reasoning": "final", "action_index": 1}')

    def test_unclosed_think_with_final_json_does_not_pollute_thinking(self):
        text = (
            "<think>Prefer the second action because it is stronger.\n"
            '{"reasoning": "final", "action_index": 1}'
        )
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertIn("Prefer the second", decision.thinking)
        self.assertNotIn('"action_index"', decision.thinking)
        self.assertFalse(decision.metadata["thinking_closed"])
        self.assertTrue(decision.metadata["json_inside_unclosed_think"])

    def test_stray_closing_think_is_flagged_but_json_parses(self):
        text = '{"reasoning": "ok", "action_index": 0}\n</think>'
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
        self.assertTrue(decision.metadata["stray_think_close"])

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
        self.assertTrue(decision.metadata["thinking_truncated"])
        self.assertIn("weigh the options", decision.thinking)

    def test_classifies_max_token_no_json_as_truncated_before_json(self):
        decision = parse_json_action(
            "<think>still thinking",
            self.actions,
            completion_tokens=256,
            max_tokens=256,
        )
        self.assertFalse(decision.valid)
        self.assertEqual(decision.metadata["error"], "truncated_before_json")
        self.assertEqual(decision.metadata["parse_error"], "truncated_before_json")

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


class VllmJsonAgentTest(unittest.TestCase):
    def setUp(self):
        self.actions = [
            LegalAction(index=0, bits=1, description="first"),
            LegalAction(index=1, bits=2, description="second"),
        ]

    def test_reasoning_mode_resolution(self):
        agent = object.__new__(VllmJsonAgent)

        agent.enable_thinking = False
        agent._native_thinking = False
        self.assertEqual(agent.reasoning_mode, "none")

        agent.enable_thinking = True
        agent._native_thinking = True
        self.assertEqual(agent.reasoning_mode, "native")

        agent.enable_thinking = True
        agent._native_thinking = False
        self.assertEqual(agent.reasoning_mode, "prompted")

    def test_probe_native_thinking_detects_changed_template(self):
        class ThinkingTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                return "thinking" if enable_thinking else "plain"

        agent = object.__new__(VllmJsonAgent)
        agent.tokenizer = ThinkingTokenizer()

        self.assertTrue(agent._probe_native_thinking())

    def test_probe_native_thinking_rejects_ignored_kwarg(self):
        class ConstantTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                return "constant"

        agent = object.__new__(VllmJsonAgent)
        agent.tokenizer = ConstantTokenizer()

        self.assertFalse(agent._probe_native_thinking())

    def test_probe_native_thinking_rejects_unsupported_kwarg(self):
        class NoThinkingKwargTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "constant"

        agent = object.__new__(VllmJsonAgent)
        agent.tokenizer = NoThinkingKwargTokenizer()

        self.assertFalse(agent._probe_native_thinking())

    def test_render_prompt_adds_prompted_thinking_instruction(self):
        agent = object.__new__(VllmJsonAgent)
        agent.framing = NEUTRAL_FRAME
        agent._apply_chat_template = lambda prompt: prompt

        agent.enable_thinking = True
        agent._native_thinking = False
        prompt = agent._render_prompt("state", self.actions)
        self.assertIn("<think>...</think>", prompt)

        agent.enable_thinking = False
        agent._native_thinking = False
        prompt = agent._render_prompt("state", self.actions)
        self.assertNotIn("<think>...</think>", prompt)

    def test_choose_actions_batch_fails_soft_when_generate_raises(self):
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class RaisingLlm:
            def __init__(self):
                self.calls = []

            def generate(self, prompts, params):
                self.calls.append((prompts, params.kwargs))
                raise RuntimeError("boom")

        agent = object.__new__(VllmJsonAgent)
        agent._render_prompt = lambda state_text, legal_actions: f"prompt: {state_text}"
        agent._SamplingParams = FakeSamplingParams
        agent._seed = 123
        agent.temperature = 0.2
        agent.max_tokens = 4096
        agent.llm = RaisingLlm()

        self.assertIsNone(agent._generate(["probe prompt"]))
        self.assertEqual(agent.llm.calls[0][0], ["probe prompt"])
        self.assertEqual(agent.llm.calls[0][1]["seed"], 123)

        decisions = agent.choose_actions_batch([("state 1", self.actions), ("state 2", self.actions)])

        self.assertEqual(len(decisions), 2)
        for decision in decisions:
            self.assertFalse(decision.valid)
            self.assertEqual(decision.action_index, 0)
            self.assertEqual(decision.metadata["error"], "vllm generation failed")
            self.assertGreaterEqual(decision.latency_s, 0.0)

    def test_choose_actions_batch_parses_results_and_sets_token_counts(self):
        agent = object.__new__(VllmJsonAgent)
        agent._render_prompt = lambda state_text, legal_actions: f"prompt: {state_text}"
        agent._count_tokens = lambda text: len(text.split()) if text else 0
        agent._generate = lambda prompts: [
            {
                "text": '{"reasoning": "first ok", "action_index": 0}',
                "prompt_tokens": 11,
                "completion_tokens": 7,
            },
            {
                "text": '<think>short thought</think>\n{"reasoning": "second ok", "action_index": 1}',
                "prompt_tokens": 13,
                "completion_tokens": 9,
            },
        ]

        decisions = agent.choose_actions_batch([("state 1", self.actions), ("state 2", self.actions)])

        self.assertEqual(len(decisions), 2)
        self.assertTrue(decisions[0].valid)
        self.assertEqual(decisions[0].action_index, 0)
        self.assertEqual(decisions[0].prompt_tokens, 11)
        self.assertEqual(decisions[0].completion_tokens, 7)
        self.assertEqual(decisions[0].thinking_tokens, 0)
        self.assertTrue(decisions[1].valid)
        self.assertEqual(decisions[1].action_index, 1)
        self.assertEqual(decisions[1].prompt_tokens, 13)
        self.assertEqual(decisions[1].completion_tokens, 9)
        self.assertEqual(decisions[1].thinking_tokens, 2)
        self.assertEqual(decisions[1].retries, 0)
        self.assertGreaterEqual(decisions[1].latency_s, 0.0)

    def test_build_decision_from_text_sets_token_counts_and_falls_back_on_invalid_text(self):
        agent = object.__new__(VllmJsonAgent)
        agent._count_tokens = lambda text: len(text.split()) if text else 0

        decision = agent.build_decision_from_text(
            '<think>short thought</think>\n{"reasoning": "second ok", "action_index": 1}',
            13,
            9,
            self.actions,
        )

        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.prompt_tokens, 13)
        self.assertEqual(decision.completion_tokens, 9)
        self.assertEqual(decision.retries, 0)
        self.assertEqual(decision.thinking_tokens, 2)

        invalid_decision = agent.build_decision_from_text("not json", 3, 2, self.actions)

        self.assertFalse(invalid_decision.valid)
        self.assertEqual(invalid_decision.action_index, 0)

    def test_choose_action_retries_after_invalid_json(self):
        agent = object.__new__(VllmJsonAgent)
        agent.framing = NEUTRAL_FRAME
        agent.max_retries = 1
        agent.enable_thinking = False
        agent._native_thinking = False
        agent._apply_chat_template = lambda prompt: prompt
        agent._count_tokens = lambda text: 0
        responses = iter(
            [
                [{"text": "not json", "prompt_tokens": 3, "completion_tokens": 2}],
                [{"text": '{"reasoning": "fixed", "action_index": 0}', "prompt_tokens": 5, "completion_tokens": 4}],
            ]
        )
        agent._generate = lambda prompts: next(responses)

        decision = agent.choose_action("state", self.actions)

        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.reasoning, "fixed")
        self.assertEqual(decision.prompt_tokens, 5)
        self.assertEqual(decision.completion_tokens, 4)
        self.assertEqual(decision.retries, 1)
        self.assertGreaterEqual(decision.latency_s, 0.0)

    def test_stream_poll_reports_submit_to_finish_latency(self):
        # Streaming path: stream_submit records submit time; stream_poll reports the
        # request's submit->finish wall-time (the per-decision latency) in its output.
        agent = object.__new__(VllmJsonAgent)
        agent._submit_ts = {"7:0:0:a0": time.perf_counter() - 0.02}
        finished = types.SimpleNamespace(
            finished=True,
            request_id="7:0:0:a0",
            prompt_token_ids=[1, 2, 3],
            outputs=[types.SimpleNamespace(text='{"action_index": 0}', token_ids=[1, 2])],
        )
        agent.llm = types.SimpleNamespace(
            llm_engine=types.SimpleNamespace(step=lambda: [finished])
        )

        polled = agent.stream_poll()

        self.assertEqual(len(polled), 1)
        rid, payload = polled[0]
        self.assertEqual(rid, "7:0:0:a0")
        self.assertGreaterEqual(payload["latency_s"], 0.02)
        self.assertEqual(payload["prompt_tokens"], 3)
        self.assertEqual(payload["completion_tokens"], 2)
        self.assertNotIn("7:0:0:a0", agent._submit_ts)  # consumed


if __name__ == "__main__":
    unittest.main()
