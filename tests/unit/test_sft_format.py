"""Unit tests for skew-free SFT prompt/completion formatting."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_ai.train.sft_format import (
    assistant_turn_terminator,
    assistant_turn_content,
    build_example,
    completion_text,
    loss_mask_token_accounting,
    reconstruct_prompt,
    resolve_loss_mask_mode,
    tokenize_example,
    user_content,
)
from sts_ai.prompting import ACTION_TEXT_OUTPUT, TURN_PLAN_OUTPUT


FRAMING = "Test framing: choose the most defensible legal action."


def _record(**overrides):
    record = {
        "world_seed": 5,
        "decision_index": 7,
        "phase": "combat",
        "state_text": "Act 1, floor 0, screen EVENT_SCREEN. Stored text only.",
        "legal_actions": [
            {"index": 0, "bits": 0, "description": "event option zero"},
            {"index": 1, "bits": 8, "description": "event option one"},
        ],
        "agent": {
            "raw_response": '{"reasoning": "keep exact braces", "action_index": 1}'
        },
    }
    record.update(overrides)
    return record


class RecordingTokenizer:
    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return (
            f"wrapped(thinking={enable_thinking},gen={add_generation_prompt})\n"
            f"{messages[0]['content']}"
        )


class LegacyTokenizer:
    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return f"legacy(gen={add_generation_prompt})\n{messages[0]['content']}"


class EncodingTokenizer:
    def __init__(self):
        self.encode_calls = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append(
            {"text": text, "add_special_tokens": add_special_tokens}
        )
        prefix = "special" if add_special_tokens else "plain"
        return [f"{prefix}:{token}" for token in text.split()]


class LegacyEncodingTokenizer:
    def __init__(self):
        self.encode_calls = []

    def encode(self, text):
        self.encode_calls.append({"text": text})
        return text.split()


class OffsetCharTokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=True):
        prefix = [999_999] if add_special_tokens else []
        return prefix + [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if add_special_tokens:
            raise AssertionError("completion tokenization must not add special tokens")
        if not return_offsets_mapping:
            raise AssertionError("action masking must request offsets")
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
    ):
        del tokenize, enable_thinking
        rendered = f"<user>{messages[0]['content']}<turn><model>"
        if len(messages) == 2:
            rendered += messages[1]["content"] + "<turn>"
        elif not add_generation_prompt:
            rendered = f"<user>{messages[0]['content']}<turn>"
        return rendered


class ThoughtBoundaryMergingTokenizer(OffsetCharTokenizer):
    def __init__(self):
        self.offsets = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens, return_offsets_mapping
        reasoning_start = text.index('"secret"')
        merged_end = reasoning_start + len('"sec')
        offsets = [(index, index + 1) for index in range(reasoning_start)]
        offsets.append((reasoning_start, merged_end))
        offsets.extend((index, index + 1) for index in range(merged_end, len(text)))
        self.offsets = offsets
        return {
            "input_ids": list(range(10_000, 10_000 + len(offsets))),
            "offset_mapping": offsets,
        }


class ChannelAwareFakeTokenizer:
    assistant_header = "<assistant_header>"
    turn_end = "<turn_end>"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
    ):
        if tokenize:
            raise AssertionError("unit fake only renders text templates")

        user_prefix = f"<user_header>{messages[0]['content']}{self.turn_end}"
        if len(messages) == 1:
            if add_generation_prompt:
                return f"{user_prefix}{self.assistant_header}"
            return user_prefix

        if len(messages) == 2:
            return (
                f"{user_prefix}{self.assistant_header}"
                f"{self._assistant_content(messages[1]['content'])}{self.turn_end}"
            )

        raise AssertionError(f"unexpected message count: {len(messages)}")

    def _assistant_content(self, content):
        return content


class DoubleWrappingFakeTokenizer(ChannelAwareFakeTokenizer):
    def _assistant_content(self, content):
        return f"<|channel|>thought\n{content}"


def _gemma_thought_record(raw_response):
    return _record(
        agent={
            "raw_response": raw_response,
            "metadata": {"reasoning_format": "gemma_thought"},
        }
    )


def _assistant_remainder(tokenizer, messages, *, enable_thinking=True):
    inference_prompt = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_turn = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if not full_turn.startswith(inference_prompt):
        return None
    return full_turn[len(inference_prompt) :]


class SftFormatTest(unittest.TestCase):
    def test_user_content_returns_raw_rendered_action_prompt(self):
        rendered = user_content(_record(), FRAMING)

        self.assertIn(FRAMING, rendered)
        self.assertIn("Act 1, floor 0, screen EVENT_SCREEN. Stored text only.", rendered)
        self.assertIn("0: event option zero", rendered)
        self.assertIn("1: event option one", rendered)
        self.assertNotIn("wrapped(", rendered)
        self.assertNotIn("legacy(", rendered)

    def test_reconstruct_prompt_renders_body_and_applies_chat_template(self):
        tokenizer = RecordingTokenizer()

        rendered = reconstruct_prompt(
            _record(),
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=False,
        )

        self.assertTrue(rendered.startswith("wrapped(thinking=False,gen=True)\n"))
        self.assertIn(FRAMING, rendered)
        self.assertIn("Act 1, floor 0, screen EVENT_SCREEN. Stored text only.", rendered)
        self.assertIn("0: event option zero", rendered)
        self.assertIn("1: event option one", rendered)
        self.assertNotIn("<think>...</think>", rendered)
        self.assertEqual(len(tokenizer.chat_calls), 1)
        self.assertEqual(tokenizer.chat_calls[0]["messages"][0]["role"], "user")
        self.assertFalse(tokenizer.chat_calls[0]["tokenize"])
        self.assertTrue(tokenizer.chat_calls[0]["add_generation_prompt"])
        self.assertFalse(tokenizer.chat_calls[0]["enable_thinking"])

    def test_reconstruct_prompt_supports_induced_reasoning_and_native_thinking(self):
        tokenizer = RecordingTokenizer()

        rendered = reconstruct_prompt(
            _record(),
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
            induce_reasoning=True,
        )

        self.assertTrue(rendered.startswith("wrapped(thinking=True,gen=True)\n"))
        self.assertIn("<think>...</think>", rendered)
        self.assertTrue(tokenizer.chat_calls[0]["enable_thinking"])

    def test_reconstruct_prompt_falls_back_for_legacy_chat_templates(self):
        tokenizer = LegacyTokenizer()

        rendered = reconstruct_prompt(
            _record(),
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
        )

        self.assertTrue(rendered.startswith("legacy(gen=True)\n"))
        self.assertIn(FRAMING, rendered)
        self.assertIn("0: event option zero", rendered)
        self.assertEqual(len(tokenizer.chat_calls), 1)
        self.assertEqual(tokenizer.chat_calls[0]["messages"][0]["role"], "user")
        self.assertFalse(tokenizer.chat_calls[0]["tokenize"])
        self.assertTrue(tokenizer.chat_calls[0]["add_generation_prompt"])

    def test_completion_text_returns_raw_response_verbatim(self):
        raw_response = '{"reasoning": "{unchanged}", "action_index": 0}\n'

        self.assertEqual(
            completion_text(_record(agent={"raw_response": raw_response})),
            raw_response,
        )

    def test_assistant_turn_content_returns_raw_response_for_all_formats(self):
        raw_response = "<|channel|>thought\nkeep native markers\n"

        self.assertEqual(
            assistant_turn_content(
                _gemma_thought_record(raw_response),
                reasoning_format="gemma_thought",
            ),
            raw_response,
        )
        self.assertEqual(
            assistant_turn_content(
                _record(agent={"raw_response": raw_response}),
                reasoning_format=None,
            ),
            raw_response,
        )

    def test_build_example_returns_canonical_text_pair_and_metadata(self):
        tokenizer = RecordingTokenizer()
        record = _record()

        example = build_example(
            record,
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=False,
        )

        self.assertEqual(
            set(example),
            {
                "messages",
                "prompt",
                "completion",
                "world_seed",
                "decision_index",
                "phase",
            },
        )
        self.assertEqual(
            example["messages"],
            [
                {"role": "user", "content": user_content(record, FRAMING)},
                {
                    "role": "assistant",
                    "content": '{"reasoning": "keep exact braces", "action_index": 1}',
                },
            ],
        )
        self.assertEqual(example["messages"][0]["role"], "user")
        self.assertNotIn("wrapped(", example["messages"][0]["content"])
        self.assertEqual(
            example["messages"][1],
            {
                "role": "assistant",
                "content": '{"reasoning": "keep exact braces", "action_index": 1}',
            },
        )
        self.assertTrue(example["prompt"].startswith("wrapped(thinking=False,gen=True)"))
        self.assertEqual(
            example["completion"],
            '{"reasoning": "keep exact braces", "action_index": 1}',
        )
        self.assertEqual(example["world_seed"], 5)
        self.assertEqual(example["decision_index"], 7)
        self.assertEqual(example["phase"], "combat")

    def test_build_example_defaults_missing_phase_to_out_of_combat(self):
        tokenizer = RecordingTokenizer()
        record = _record()
        del record["phase"]

        example = build_example(
            record,
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=False,
        )

        self.assertEqual(example["phase"], "out_of_combat")

    def test_build_example_preserves_laundered_gemma_thought_completion(self):
        tokenizer = ChannelAwareFakeTokenizer()
        laundered_raw_response = (
            "<|channel|>thought\n"
            "Laundered native reasoning survives into the target.\n"
            "<|channel|>final\n"
            '{"reasoning": "kept after laundering", "action_index": 1}'
        )
        record = _gemma_thought_record(laundered_raw_response)

        example = build_example(
            record,
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
        )

        self.assertEqual(
            record["agent"]["metadata"]["reasoning_format"],
            "gemma_thought",
        )
        self.assertIn("<|channel|>thought\n", example["messages"][1]["content"])
        self.assertIn("<|channel|>final\n", example["messages"][1]["content"])
        self.assertEqual(example["messages"][1]["content"], laundered_raw_response)
        self.assertEqual(example["completion"], laundered_raw_response)

    def test_gemma_thought_assistant_span_round_trips_with_channel_aware_template(self):
        tokenizer = ChannelAwareFakeTokenizer()
        raw_response = (
            "<|channel|>thought\n"
            "Native thought tokens emitted by inference.\n"
            "<|channel|>final\n"
            '{"action_index": 1}'
        )
        example = build_example(
            _gemma_thought_record(raw_response),
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
        )

        inference_prompt = tokenizer.apply_chat_template(
            example["messages"][:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_turn = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )

        # Real tokenizer token-id round-trip coverage belongs in tests/integration/.
        self.assertTrue(full_turn.startswith(inference_prompt))
        self.assertEqual(
            full_turn[len(inference_prompt) :],
            raw_response + tokenizer.turn_end,
        )

    def test_gemma_thought_round_trip_check_catches_double_wrapping_template(self):
        tokenizer = DoubleWrappingFakeTokenizer()
        raw_response = (
            "<|channel|>thought\n"
            "Already wrapped by inference.\n"
            "<|channel|>final\n"
            '{"action_index": 0}'
        )
        example = build_example(
            _gemma_thought_record(raw_response),
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
        )

        self.assertNotEqual(
            _assistant_remainder(tokenizer, example["messages"]),
            raw_response + tokenizer.turn_end,
        )

    def test_tokenize_example_masks_prompt_and_avoids_completion_special_tokens(self):
        tokenizer = EncodingTokenizer()
        example = {"prompt": "prompt token", "completion": "completion token"}

        tokenized = tokenize_example(example, tokenizer)

        prompt_ids = ["special:prompt", "special:token"]
        completion_ids = ["plain:completion", "plain:token"]
        self.assertEqual(tokenized["input_ids"], prompt_ids + completion_ids)
        self.assertEqual(tokenized["labels"], [-100, -100] + completion_ids)
        self.assertEqual(tokenized["n_prompt_tokens"], 2)
        self.assertEqual(tokenized["n_completion_tokens"], 2)
        self.assertEqual(
            tokenizer.encode_calls,
            [
                {"text": "prompt token", "add_special_tokens": True},
                {"text": "completion token", "add_special_tokens": False},
            ],
        )

    def test_tokenize_example_falls_back_for_legacy_encode(self):
        tokenizer = LegacyEncodingTokenizer()
        example = {"prompt": "prompt token", "completion": "completion token"}

        tokenized = tokenize_example(example, tokenizer)

        self.assertEqual(
            tokenized["input_ids"],
            ["prompt", "token", "completion", "token"],
        )
        self.assertEqual(tokenized["labels"], [-100, -100, "completion", "token"])
        self.assertEqual(
            tokenizer.encode_calls,
            [{"text": "prompt token"}, {"text": "completion token"}],
        )

    def test_action_mask_preserves_native_markers_action_and_turn_end(self):
        tokenizer = OffsetCharTokenizer()
        completion = (
            "<|channel>thought\nprivate plan\n<channel|>"
            '{"action_index": 12}'
        )
        tokenized = tokenize_example(
            {
                "prompt": "prompt",
                "completion": completion,
                "target_action_index": 12,
                "assistant_turn_terminator": "<turn>",
            },
            tokenizer,
            loss_mask_mode="action",
        )

        completion_ids = tokenized["input_ids"][tokenized["n_prompt_tokens"] :]
        completion_labels = tokenized["labels"][tokenized["n_prompt_tokens"] :]
        rendered = "".join(chr(token_id) for token_id in completion_ids)
        supervised = "".join(
            chr(token_id) if label != -100 else "·"
            for token_id, label in zip(completion_ids, completion_labels)
        )
        self.assertEqual(rendered, completion + "<turn>")
        self.assertIn("<|channel>thought", supervised)
        self.assertIn("<channel|>", supervised)
        self.assertIn('·······', supervised)
        self.assertNotIn("private plan", supervised)
        self.assertIn('"action_index"', supervised)
        self.assertIn("12", supervised)
        self.assertTrue(supervised.endswith("<turn>"))
        self.assertEqual(tokenized["n_supervised_thought_tokens"], 0)
        self.assertGreater(tokenized["n_supervised_format_tokens"], 0)
        self.assertEqual(tokenized["n_supervised_action_tokens"], 2)

    def test_action_mask_excludes_reasoning_inside_nonthinking_json(self):
        tokenizer = OffsetCharTokenizer()
        completion = (
            '{"reasoning": "do not imitate me", "confidence": 0.9, '
            '"action_index": 3}'
        )
        tokenized = tokenize_example(
            {
                "prompt": "prompt",
                "completion": completion,
                "target_action_index": 3,
                "assistant_turn_terminator": "<turn>",
            },
            tokenizer,
            loss_mask_mode="action",
        )
        start = tokenized["n_prompt_tokens"]
        completion_ids = tokenized["input_ids"][start:]
        completion_labels = tokenized["labels"][start:]
        supervised = "".join(
            chr(token_id) if label != -100 else "·"
            for token_id, label in zip(completion_ids, completion_labels)
        )
        self.assertNotIn("do not imitate me", supervised)
        self.assertNotIn("0.9", supervised)
        self.assertNotIn("confidence", supervised)
        self.assertNotIn("reasoning", supervised)
        self.assertIn('"action_index": 3', supervised)
        self.assertEqual(tokenized["n_supervised_thought_tokens"], 0)
        self.assertEqual(tokenized["n_supervised_action_tokens"], 1)

    def test_token_straddling_reasoning_and_json_syntax_is_masked(self):
        tokenizer = ThoughtBoundaryMergingTokenizer()
        completion = '{"reasoning": "secret", "action_index": 3}'
        tokenized = tokenize_example(
            {
                "prompt": "prompt",
                "completion": completion,
                "target_action_index": 3,
                "assistant_turn_terminator": "<turn>",
            },
            tokenizer,
            loss_mask_mode="action",
        )
        merged_offset = (completion.index('"secret"'), completion.index('"secret"') + 4)
        merged_index = tokenizer.offsets.index(merged_offset)
        completion_labels = tokenized["labels"][tokenized["n_prompt_tokens"] :]
        self.assertEqual(completion_labels[merged_index], -100)

    def test_action_mask_supports_think_blocks_and_markdown_fence(self):
        tokenizer = OffsetCharTokenizer()
        completion = (
            "<think>private</think>\n```json\n"
            '{"action_index": 0}\n```'
        )
        tokenized = tokenize_example(
            {
                "prompt": "prompt",
                "completion": completion,
                "target_action_index": 0,
                "assistant_turn_terminator": "<turn>",
            },
            tokenizer,
            loss_mask_mode="action",
        )
        self.assertGreater(tokenized["n_format_tokens"], 0)
        self.assertGreater(tokenized["n_thought_tokens"], 0)
        self.assertEqual(tokenized["n_action_tokens"], 1)
        self.assertEqual(tokenized["n_supervised_thought_tokens"], 0)

    def test_action_mask_rejects_missing_malformed_or_mismatched_action(self):
        tokenizer = OffsetCharTokenizer()
        base = {
            "prompt": "prompt",
            "assistant_turn_terminator": "<turn>",
        }
        for completion, target in (
            ("no object", 0),
            ('{"action_index":', 0),
            ('{"action_index": true}', 1),
            ('{"action_index": 2}', 1),
        ):
            with self.subTest(completion=completion):
                with self.assertRaises(ValueError):
                    tokenize_example(
                        {
                            **base,
                            "completion": completion,
                            "target_action_index": target,
                        },
                        tokenizer,
                        loss_mask_mode="action",
                    )

    def test_build_action_example_records_auditable_counts_and_terminator(self):
        tokenizer = OffsetCharTokenizer()
        record = _record(
            agent={
                "action_index": 1,
                "raw_response": (
                    "<|channel>thought\nthink\n<channel|>"
                    '{"action_index": 1}'
                ),
            }
        )
        example = build_example(
            record,
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
            loss_mask_mode="action",
        )
        self.assertEqual(example["loss_mask_mode"], "action")
        self.assertEqual(example["target_action_index"], 1)
        self.assertEqual(example["assistant_turn_terminator"], "<turn>")
        self.assertGreater(example["token_counts"]["n_action_tokens"], 0)
        self.assertEqual(
            example["token_counts"]["n_supervised_thought_tokens"],
            0,
        )

    def test_action_text_mask_targets_exact_json_string_value(self):
        tokenizer = OffsetCharTokenizer()
        description = "play Strike -> Jaw Worm (deal 9)"
        completion = json.dumps({"action": description}, separators=(",", ":"))
        tokenized = tokenize_example(
            {
                "prompt": "prompt",
                "completion": completion,
                "output_contract": ACTION_TEXT_OUTPUT,
                "target_action_description": description,
                "assistant_turn_terminator": "<turn>",
            },
            tokenizer,
            loss_mask_mode="action",
        )

        start = tokenized["n_prompt_tokens"]
        action_text = "".join(
            chr(token_id)
            for token_id, is_action in zip(
                tokenized["input_ids"][start:],
                tokenized["action_mask"][start:],
            )
            if is_action
        )
        self.assertEqual(action_text, json.dumps(description))
        self.assertEqual(tokenized["n_supervised_thought_tokens"], 0)
        self.assertGreater(tokenized["n_supervised_format_tokens"], 0)

    def test_action_text_contract_rejects_mismatched_or_missing_action(self):
        tokenizer = OffsetCharTokenizer()
        base = {
            "prompt": "prompt",
            "output_contract": ACTION_TEXT_OUTPUT,
            "target_action_description": "play Strike -> Jaw Worm (deal 9)",
            "assistant_turn_terminator": "<turn>",
        }
        for completion in (
            '{"action":"end turn"}',
            '{"not_action":"play Strike -> Jaw Worm (deal 9)"}',
        ):
            with self.subTest(completion=completion):
                with self.assertRaises(ValueError):
                    tokenize_example(
                        {**base, "completion": completion},
                        tokenizer,
                        loss_mask_mode="action",
                    )

    def test_turn_plan_array_is_format_and_first_action_value_is_target(self):
        tokenizer = OffsetCharTokenizer()
        first = "play Bash -> Gremlin Nob (deal 8)"
        plan = [first, "play Strike -> Gremlin Nob (deal 9)", "end turn"]
        completion = json.dumps(
            {"plan": plan, "action": first},
            separators=(",", ":"),
        )
        tokenized = tokenize_example(
            {
                "prompt": "prompt",
                "completion": completion,
                "output_contract": TURN_PLAN_OUTPUT,
                "target_action_description": first,
                "assistant_turn_terminator": "<turn>",
            },
            tokenizer,
            loss_mask_mode="action",
        )

        start = tokenized["n_prompt_tokens"]
        action_text = "".join(
            chr(token_id)
            for token_id, is_action in zip(
                tokenized["input_ids"][start:],
                tokenized["action_mask"][start:],
            )
            if is_action
        )
        self.assertEqual(action_text, json.dumps(first))
        self.assertEqual(tokenized["n_thought_tokens"], 0)
        self.assertGreater(tokenized["n_format_tokens"], len(json.dumps(plan)))

    def test_turn_plan_contract_rejects_mismatched_or_missing_plan(self):
        tokenizer = OffsetCharTokenizer()
        action = "play Bash -> Gremlin Nob (deal 8)"
        base = {
            "prompt": "prompt",
            "output_contract": TURN_PLAN_OUTPUT,
            "target_action_description": action,
            "assistant_turn_terminator": "<turn>",
        }
        for completion in (
            json.dumps(
                {"plan": ["end turn"], "action": action},
                separators=(",", ":"),
            ),
            json.dumps({"action": action}, separators=(",", ":")),
        ):
            with self.subTest(completion=completion):
                with self.assertRaises(ValueError):
                    tokenize_example(
                        {**base, "completion": completion},
                        tokenizer,
                        loss_mask_mode="action",
                    )

    def test_semantic_build_examples_round_trip_and_account(self):
        tokenizer = OffsetCharTokenizer()
        examples = []
        for output_contract, completion in (
            (
                ACTION_TEXT_OUTPUT,
                '{"action":"event option one"}',
            ),
            (
                TURN_PLAN_OUTPUT,
                '{"plan":["event option one","end turn"],'
                '"action":"event option one"}',
            ),
        ):
            with self.subTest(output_contract=output_contract):
                example = build_example(
                    _record(
                        legal_actions=[
                            {"index": 0, "bits": 0, "description": "event option zero"},
                            {"index": 1, "bits": 8, "description": "event option one"},
                            {"index": 2, "bits": 9, "description": "end turn"},
                        ],
                        agent={
                            "action_index": 1,
                            "raw_response": completion,
                        },
                    ),
                    FRAMING,
                    tokenizer=tokenizer,
                    enable_thinking=False,
                    loss_mask_mode="action",
                    output_contract=output_contract,
                )
                tokenized = tokenize_example(
                    example,
                    tokenizer,
                    loss_mask_mode="action",
                )
                self.assertEqual(example["completion"], completion)
                self.assertEqual(example["target_action_description"], "event option one")
                self.assertGreater(tokenized["n_action_tokens"], 0)
                self.assertGreater(example["token_counts"]["n_supervised_tokens"], 0)
                examples.append(example)

        accounting = loss_mask_token_accounting(examples)
        self.assertEqual(accounting["n_examples"], 2)
        self.assertEqual(accounting["n_examples_counted"], 2)
        self.assertGreater(accounting["totals"]["n_action_tokens"], 0)

    def test_assistant_turn_terminator_is_derived_from_template(self):
        self.assertEqual(assistant_turn_terminator(OffsetCharTokenizer()), "<turn>")

    def test_loss_mask_auto_preserves_legacy_and_enforces_new_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy.json"
            action = root / "action.json"
            legacy.write_text("{}", encoding="utf-8")
            action.write_text(
                json.dumps({"loss_mask_mode": "action"}),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_loss_mask_mode("auto", manifest_path=legacy),
                "completion",
            )
            self.assertEqual(
                resolve_loss_mask_mode("auto", manifest_path=action),
                "action",
            )
            with self.assertRaisesRegex(ValueError, "disagrees"):
                resolve_loss_mask_mode("completion", manifest_path=action)


if __name__ == "__main__":
    unittest.main()


class EmittedSemanticVariantMaskTest(unittest.TestCase):
    """On-policy action-mask supervision of lenient-parser variants."""

    def _record(self, raw_response: str):
        return {
            "world_seed": 1,
            "decision_index": 0,
            "phase": "out_of_combat",
            "state_text": "state",
            "legal_actions": [
                {"index": 0, "bits": 1, "description": "take gold 25g"},
                {"index": 1, "bits": 2, "description": "skip rewards / proceed"},
            ],
            "agent": {
                "action_index": 0,
                "raw_response": raw_response,
                "valid": True,
                "retries": 0,
                "metadata": {},
            },
            "action_executed": True,
        }

    def _build(self, raw_response: str):
        from sts_ai.train.sft_format import build_example
        from tests.unit.test_pg_dataset import FakeTokenizer

        return build_example(
            self._record(raw_response),
            "frame",
            tokenizer=FakeTokenizer(),
            enable_thinking=False,
            loss_mask_mode="action",
            output_contract="action_text",
        )

    def test_menu_prefixed_emission_is_supervised_as_emitted(self):
        example = self._build('{"action": "0: take gold 25g"}')
        self.assertEqual(example["target_action_description"], "0: take gold 25g")
        self.assertGreater(example["token_counts"]["n_supervised_action_tokens"], 0)

    def test_exact_emission_keeps_canonical_description(self):
        example = self._build('{"action": "take gold 25g"}')
        self.assertEqual(example["target_action_description"], "take gold 25g")

    def test_unresolvable_emission_still_fails_closed(self):
        with self.assertRaises(ValueError):
            self._build('{"action": "drink potion Fire Potion"}')

    def test_wrong_action_resolution_fails_closed(self):
        # Emitted text resolves to a DIFFERENT action than the recorded index.
        with self.assertRaises(ValueError):
            self._build('{"action": "skip rewards / proceed"}')
