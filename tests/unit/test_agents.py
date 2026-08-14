import time
import types
import unittest

from sts_ai.agents import (
    MlxQwenJsonAgent,
    RandomLegalAgent,
    VllmJsonAgent,
    parse_json_action,
    resolve_output_contract,
)
from sts_ai.prompting import NEUTRAL_FRAME, retry_instruction
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

    def test_parses_action_only_json_without_inventing_reasoning(self):
        decision = parse_json_action('{"action_index": 1}', self.actions)

        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.reasoning, "")

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

    def test_gemma_thought_stripped_form_captured(self):
        # Gemma-4 native thinking with the <|channel> tokens stripped (vLLM default):
        # the completion starts at the `thought` role label, then reasoning, then the
        # fenced JSON answer.
        text = (
            "thought\nThe second action is stronger here.\n"
            'I will pick it.```json\n{"reasoning": "final", "action_index": 1}\n```'
        )
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata.get("reasoning_format"), "gemma_thought")
        self.assertIn("second action is stronger", decision.thinking)
        self.assertNotIn('"action_index"', decision.thinking)  # JSON not stored as CoT
        self.assertNotIn("```", decision.thinking)             # trailing answer fence trimmed

    def test_gemma_thought_channel_tokens_captured(self):
        # skip_special_tokens=False: the channel markers survive and bound the thought.
        text = (
            "<|channel>thought\nReasoning body here.<channel|>"
            '{"reasoning": "final", "action_index": 0}<|end_of_turn|>'
        )
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata.get("reasoning_format"), "gemma_thought")
        self.assertEqual(decision.thinking, "Reasoning body here.")
        self.assertTrue(decision.metadata["thinking_closed"])

    def test_plain_json_answer_has_no_gemma_thought(self):
        # A no-thinking completion (starts with the fenced JSON) must not be mistaken
        # for a Gemma thought channel.
        text = '```json\n{"reasoning": "ok", "action_index": 0}\n```'
        decision = parse_json_action(text, self.actions)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.thinking, "")
        self.assertIsNone(decision.metadata.get("reasoning_format"))

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


class ParseSemanticActionTest(unittest.TestCase):
    """Live-play resolution for the semantic output contracts."""

    def setUp(self):
        self.actions = [
            LegalAction(index=0, bits=1, description="play Strike (cost 1) -> Gremlin Nob (deal 9)"),
            LegalAction(index=1, bits=2, description="play Defend (cost 1)"),
            LegalAction(index=2, bits=4, description="end turn"),
        ]

    def test_action_text_exact_match_resolves_index(self):
        decision = parse_json_action(
            '{"action": "play Defend (cost 1)"}',
            self.actions,
            output_contract="action_text",
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata["legal_action"]["description"], "play Defend (cost 1)")

    def test_action_text_unmatched_text_is_invalid(self):
        decision = parse_json_action(
            '{"action": "drink potion Fire Potion"}',
            self.actions,
            output_contract="action_text",
        )
        self.assertFalse(decision.valid)
        self.assertEqual(decision.metadata["parse_error"], "unmatched action text")

    def test_action_text_unique_prefix_resolves(self):
        decision = parse_json_action(
            '{"action": "play Strike (cost 1)"}',
            self.actions,
            output_contract="action_text",
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["semantic_match"], "unique_prefix")

    def test_action_text_ambiguous_prefix_is_invalid(self):
        actions = [
            LegalAction(index=0, bits=1, description="play Strike (cost 1) -> Sentry [enemy 0]"),
            LegalAction(index=1, bits=2, description="play Strike (cost 1) -> Sentry [enemy 2]"),
        ]
        decision = parse_json_action(
            '{"action": "play Strike (cost 1)"}', actions, output_contract="action_text"
        )
        self.assertFalse(decision.valid)

    def test_action_text_empty_string_is_invalid(self):
        decision = parse_json_action(
            '{"action": ""}', self.actions, output_contract="action_text"
        )
        self.assertFalse(decision.valid)

    def test_action_text_tolerates_extra_keys(self):
        decision = parse_json_action(
            '{"reasoning": "block", "action": "play Defend (cost 1)"}',
            self.actions,
            output_contract="action_text",
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.reasoning, "block")

    def test_action_text_falls_back_to_valid_action_index(self):
        decision = parse_json_action(
            '{"action_index": 2}',
            self.actions,
            output_contract="action_text",
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 2)
        self.assertEqual(decision.metadata["semantic_fallback"], "action_index")

    def test_action_text_rejects_bool_action_index_fallback(self):
        decision = parse_json_action(
            '{"action_index": true}',
            self.actions,
            output_contract="action_text",
        )
        self.assertFalse(decision.valid)

    def test_turn_plan_resolves_action_and_keeps_plan(self):
        decision = parse_json_action(
            '{"plan": ["play Defend (cost 1)", "end turn"], "action": "play Defend (cost 1)"}',
            self.actions,
            output_contract="turn_plan",
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata["plan"], ["play Defend (cost 1)", "end turn"])

    def test_default_contract_still_rejects_action_text_payload(self):
        decision = parse_json_action('{"action": "play Defend (cost 1)"}', self.actions)
        self.assertFalse(decision.valid)
        self.assertEqual(decision.metadata["parse_error"], "invalid action_index")

    def test_duplicate_descriptions_resolve_to_first_position(self):
        actions = [
            LegalAction(index=0, bits=1, description="end turn"),
            LegalAction(index=1, bits=2, description="end turn"),
        ]
        decision = parse_json_action(
            '{"action": "end turn"}', actions, output_contract="action_text"
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)


class ResolveOutputContractTest(unittest.TestCase):
    def _agent(self, combat="action_text", ooc=None):
        return types.SimpleNamespace(output_contract=combat, ooc_output_contract=ooc)

    def test_combat_and_phaseless_use_combat_contract(self):
        agent = self._agent(ooc="reasoning_action")
        self.assertEqual(resolve_output_contract(agent, "combat"), "action_text")
        self.assertEqual(resolve_output_contract(agent, None), "action_text")

    def test_ooc_phase_switches_when_override_set(self):
        agent = self._agent(ooc="reasoning_action")
        for phase in ("event_screen", "map_screen", "rewards", "shop"):
            self.assertEqual(resolve_output_contract(agent, phase), "reasoning_action")

    def test_ooc_phase_without_override_keeps_uniform_contract(self):
        agent = self._agent(ooc=None)
        self.assertEqual(resolve_output_contract(agent, "event_screen"), "action_text")

    def test_agent_without_attrs_defaults_to_reasoning_action(self):
        self.assertEqual(resolve_output_contract(object(), "combat"), "reasoning_action")


class StreamingRequestContractTest(unittest.TestCase):
    """The per-request contract stored at submit time must drive parse time."""

    def _agent(self):
        agent = object.__new__(VllmJsonAgent)
        agent.output_contract = "action_text"
        agent.ooc_output_contract = "reasoning_action"
        agent.max_tokens = 4096
        agent.tokenizer = None
        return agent

    def test_build_decision_uses_stored_request_contract(self):
        agent = self._agent()
        agent._request_contracts = {"r1": "reasoning_action", "r2": "action_text"}
        actions = [LegalAction(index=0, bits=1, description="take gold 25g")]
        semantic_payload = '{"action": "take gold 25g"}'
        # r1 was submitted as an OOC decision: index contract, so exact-text
        # payloads are invalid there...
        d1 = agent.build_decision_from_text(semantic_payload, 1, 1, actions, request_id="r1")
        self.assertFalse(d1.valid)
        # ...while r2 was a combat submit: action_text resolves the same payload.
        d2 = agent.build_decision_from_text(semantic_payload, 1, 1, actions, request_id="r2")
        self.assertTrue(d2.valid)
        self.assertEqual(agent._request_contracts, {})

    def test_explicit_contract_beats_store_and_missing_id_falls_back(self):
        agent = self._agent()
        actions = [LegalAction(index=0, bits=1, description="take gold 25g")]
        d = agent.build_decision_from_text(
            '{"action": "take gold 25g"}', 1, 1, actions, output_contract="action_text"
        )
        self.assertTrue(d.valid)
        # No request_id, no explicit contract -> uniform (combat) contract.
        d2 = agent.build_decision_from_text('{"action": "take gold 25g"}', 1, 1, actions)
        self.assertTrue(d2.valid)


class CompositeAgentTest(unittest.TestCase):
    class _Sub:
        def __init__(self, name):
            self.name = name
            self.calls = []
            self.config = {"model_id": name, "output_contract": "x"}

        def reseed(self, policy_seed):
            self.calls.append(("reseed", policy_seed))

        def choose_action(self, state_text, legal_actions, phase=None):
            self.calls.append(("choose", phase))
            from sts_ai.schemas import AgentDecision
            return AgentDecision(action_index=0, raw_response=self.name)

        def choose_actions_batch(self, items, retry_flags=None, phases=None):
            from sts_ai.schemas import AgentDecision
            self.calls.append(("batch", tuple(phases or ())))
            return [AgentDecision(action_index=0, raw_response=self.name) for _ in items]

    def setUp(self):
        from sts_ai.agents import CompositeAgent
        self.combat = self._Sub("combat")
        self.ooc = self._Sub("ooc")
        self.agent = CompositeAgent(self.combat, self.ooc)
        self.actions = [LegalAction(index=0, bits=1, description="end turn")]

    def test_routes_by_phase_and_phaseless_goes_to_combat(self):
        self.assertEqual(
            self.agent.choose_action("s", self.actions, phase="combat").raw_response, "combat"
        )
        self.assertEqual(
            self.agent.choose_action("s", self.actions, phase="event_screen").raw_response, "ooc"
        )
        self.assertEqual(
            self.agent.choose_action("s", self.actions, phase=None).raw_response, "combat"
        )

    def test_batch_split_preserves_item_order(self):
        items = [("a", self.actions), ("b", self.actions), ("c", self.actions)]
        phases = ["combat", "map_screen", "combat"]
        out = self.agent.choose_actions_batch(items, phases=phases)
        self.assertEqual([d.raw_response for d in out], ["combat", "ooc", "combat"])

    def test_reseed_reaches_both(self):
        self.agent.reseed(7)
        self.assertIn(("reseed", 7), self.combat.calls)
        self.assertIn(("reseed", 7), self.ooc.calls)

    def test_config_carries_both_identities(self):
        cfg = self.agent.config
        self.assertTrue(cfg["composite"])
        self.assertEqual(cfg["combat_agent"]["model_id"], "combat")
        self.assertEqual(cfg["ooc_agent"]["model_id"], "ooc")


class RetryInstructionTest(unittest.TestCase):
    def test_default_contract_keeps_frozen_literal(self):
        self.assertEqual(
            retry_instruction("reasoning_action"),
            "\n\nYour previous response was invalid. Return only one JSON object "
            "with a legal integer action_index from the listed actions. Do not include "
            "a <think> block, markdown fence, or any other text.",
        )
        self.assertEqual(retry_instruction("action_only"), retry_instruction("reasoning_action"))

    def test_semantic_contracts_name_their_schema(self):
        self.assertIn('{"action": ', retry_instruction("action_text"))
        self.assertNotIn("action_index", retry_instruction("action_text"))
        self.assertIn('"plan"', retry_instruction("turn_plan"))

    def test_unknown_contract_rejected(self):
        with self.assertRaises(ValueError):
            retry_instruction("bogus")


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
    def test_config_records_action_only_output_contract(self):
        agent = object.__new__(MlxQwenJsonAgent)
        agent.model_id = "model"
        agent.framing = NEUTRAL_FRAME
        agent.temperature = 0.2
        agent.max_tokens = 64
        agent.enable_thinking = False
        agent.max_retries = 1
        agent.adapter_path = None
        agent.output_contract = "action_only"

        self.assertEqual(agent.config["output_contract"], "action_only")

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

    def test_choose_action_uses_action_only_prompt_contract(self):
        actions = [LegalAction(index=0, bits=1, description="first")]
        agent = object.__new__(MlxQwenJsonAgent)
        agent.framing = NEUTRAL_FRAME
        agent.output_contract = "action_only"
        agent.max_retries = 0
        agent.max_tokens = 64
        agent._apply_chat_template = lambda prompt: prompt
        agent._count_tokens = lambda text: 0
        seen = {}

        def generate(prompt):
            seen["prompt"] = prompt
            return '{"action_index":0}'

        agent._generate_chat = generate
        decision = agent.choose_action("state", actions)

        self.assertTrue(decision.valid)
        self.assertIn('{"action_index": 0}', seen["prompt"])
        self.assertNotIn('"reasoning"', seen["prompt"])

    def test_prompt_override_bypasses_render_action_prompt(self):
        # The Interactive Studio advanced-template path: choose_action must send
        # the override verbatim and never call render_action_prompt's framing path.
        actions = [LegalAction(index=0, bits=1, description="first")]
        agent = object.__new__(MlxQwenJsonAgent)
        agent.framing = "SHOULD NOT APPEAR"
        agent.max_retries = 0
        agent.max_tokens = 64
        agent._count_tokens = lambda text: 0
        seen = {}
        agent._apply_chat_template = lambda prompt: prompt

        def _gen(chat_prompt):
            seen["prompt"] = chat_prompt
            return '{"reasoning":"r","action_index":0}'

        agent._generate_chat = _gen

        decision = agent.choose_action("state", actions, prompt_override="CUSTOM PROMPT {x}")
        self.assertTrue(decision.valid)
        self.assertEqual(seen["prompt"], "CUSTOM PROMPT {x}")
        self.assertNotIn("SHOULD NOT APPEAR", seen["prompt"])


class MlxStreamChooseActionTest(unittest.TestCase):
    def test_stream_yields_segments_then_returns_decision(self):
        actions = [LegalAction(index=0, bits=1, description="first"),
                   LegalAction(index=1, bits=2, description="second")]
        agent = object.__new__(MlxQwenJsonAgent)
        agent.framing = "neutral"
        agent.max_tokens = 64
        agent._sampler = None
        agent._count_tokens = lambda text: len(text.split()) if text else 0
        agent._apply_chat_template = lambda prompt: prompt

        class _Resp:
            def __init__(self, text):
                self.text = text

        segments = ['{"reasoning"', ': "go", ', '"action_index": 1}']

        def fake_stream(model, tokenizer, prompt, **kwargs):
            for s in segments:
                yield _Resp(s)

        agent.model = None
        agent.tokenizer = None
        agent._stream_generate = fake_stream

        gen = agent.stream_choose_action("state", actions)
        streamed = []
        decision = None
        try:
            while True:
                streamed.append(next(gen))
        except StopIteration as stop:
            decision = stop.value

        self.assertEqual(streamed, segments)  # incremental segments surfaced
        self.assertIsNotNone(decision)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.reasoning, "go")


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

    def test_render_prompt_uses_action_only_contract(self):
        agent = object.__new__(VllmJsonAgent)
        agent.framing = NEUTRAL_FRAME
        agent.output_contract = "action_only"
        agent.enable_thinking = False
        agent._native_thinking = False
        agent._apply_chat_template = lambda prompt: prompt

        prompt = agent._render_prompt("state", self.actions)

        self.assertIn('{"action_index": 0}', prompt)
        self.assertNotIn('"reasoning"', prompt)

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
        agent._render_prompt = lambda state_text, legal_actions, output_contract=None: f"prompt: {state_text}"
        agent._SamplingParams = FakeSamplingParams
        agent._seed = 123
        agent.temperature = 0.2
        agent.top_p = 0.95
        agent.top_k = 64
        agent.max_tokens = 4096
        agent.llm = RaisingLlm()

        self.assertIsNone(agent._generate(["probe prompt"]))
        self.assertEqual(agent.llm.calls[0][0], ["probe prompt"])
        self.assertEqual(agent.llm.calls[0][1]["seed"], 123)
        # top_p / top_k flow through to the vLLM SamplingParams
        self.assertEqual(agent.llm.calls[0][1]["top_p"], 0.95)
        self.assertEqual(agent.llm.calls[0][1]["top_k"], 64)
        self.assertTrue(agent.llm.calls[0][1]["skip_special_tokens"])

        decisions = agent.choose_actions_batch([("state 1", self.actions), ("state 2", self.actions)])

        self.assertEqual(len(decisions), 2)
        for decision in decisions:
            self.assertFalse(decision.valid)
            self.assertEqual(decision.action_index, 0)
            self.assertEqual(decision.metadata["error"], "vllm generation failed")
            self.assertGreaterEqual(decision.latency_s, 0.0)

    def test_choose_actions_batch_parses_results_and_sets_token_counts(self):
        agent = object.__new__(VllmJsonAgent)
        agent._render_prompt = lambda state_text, legal_actions, output_contract=None: f"prompt: {state_text}"
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

    def test_generate_preserves_special_tokens_when_enabled(self):
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Output:
            prompt_token_ids = [1, 2]
            outputs = [types.SimpleNamespace(text='{"action_index": 0}', token_ids=[3])]

        class CapturingLlm:
            def __init__(self):
                self.calls = []

            def generate(self, prompts, params):
                self.calls.append((prompts, params.kwargs))
                return [Output()]

        agent = object.__new__(VllmJsonAgent)
        agent._SamplingParams = FakeSamplingParams
        agent._seed = 123
        agent.temperature = 0.2
        agent.top_p = 1.0
        agent.top_k = -1
        agent.max_tokens = 4096
        agent.preserve_special_tokens = True
        agent.llm = CapturingLlm()

        result = agent._generate(["probe prompt"])

        self.assertIsNotNone(result)
        self.assertFalse(agent.llm.calls[0][1]["skip_special_tokens"])

    def test_stream_submit_threads_skip_special_tokens(self):
        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CapturingEngine:
            def __init__(self):
                self.calls = []

            def add_request(self, request_id, prompt, params):
                self.calls.append((request_id, prompt, params.kwargs))

        engine = CapturingEngine()
        agent = object.__new__(VllmJsonAgent)
        agent._base_prompt = lambda state_text, legal_actions, output_contract=None: "base prompt"
        agent._apply_chat_template = lambda prompt: f"chat: {prompt}"
        agent._SamplingParams = FakeSamplingParams
        agent.temperature = 0.2
        agent.top_p = 0.95
        agent.top_k = 64
        agent.max_tokens = 4096
        agent.preserve_special_tokens = True
        agent.llm = types.SimpleNamespace(llm_engine=engine)

        agent.stream_submit("req-1", "state", self.actions, seed=77)

        self.assertEqual(engine.calls[0][0], "req-1")
        self.assertEqual(engine.calls[0][1], "chat: base prompt")
        self.assertFalse(engine.calls[0][2]["skip_special_tokens"])

    def test_config_includes_preserve_special_tokens(self):
        agent = object.__new__(VllmJsonAgent)
        agent.model_id = "model"
        agent.framing = NEUTRAL_FRAME
        agent.temperature = 0.2
        agent.top_p = 1.0
        agent.top_k = -1
        agent.max_tokens = 4096
        agent.enable_thinking = True
        agent.enable_prefix_caching = True
        agent._native_thinking = True
        agent.preserve_special_tokens = True
        agent.max_retries = 1
        agent.dtype = "auto"
        agent.gpu_memory_utilization = 0.9
        agent.adapter_path = None
        agent.output_contract = "action_only"

        self.assertTrue(agent.config["preserve_special_tokens"])
        self.assertEqual(agent.config["output_contract"], "action_only")

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


class StrippedMenuIndexTest(unittest.TestCase):
    def test_menu_line_copy_with_index_prefix_resolves(self):
        actions = [
            LegalAction(index=0, bits=1, description="play Strike [Attack] (cost 1) -> CULTIST (deal 9)"),
            LegalAction(index=1, bits=2, description="play Defend [Skill] (cost 1)"),
        ]
        decision = parse_json_action(
            '{"action": "0: play Strike [Attack] (cost 1) -> CULTIST (deal 9)"}',
            actions,
            output_contract="action_text",
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.metadata["semantic_match"], "stripped_menu_index")

    def test_stripped_prefix_then_unique_prefix_composes(self):
        actions = [
            LegalAction(index=0, bits=1, description="play Strike [Attack] (cost 1) -> CULTIST (deal 9)"),
            LegalAction(index=1, bits=2, description="play Defend [Skill] (cost 1)"),
        ]
        decision = parse_json_action(
            '{"action": "1: play Strike [Attack] (cost 1)"}', actions, output_contract="action_text"
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.action_index, 0)
