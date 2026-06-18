"""Unit tests for skew-free SFT prompt/completion formatting."""
from __future__ import annotations

import unittest

from sts_ai.train.sft_format import (
    assistant_turn_content,
    build_example,
    completion_text,
    reconstruct_prompt,
    tokenize_example,
    user_content,
)


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

    def test_build_example_gemma_thought_preserves_channel_marked_completion(self):
        tokenizer = ChannelAwareFakeTokenizer()
        raw_response = (
            "<|channel|>thought\n"
            "Retain the native reasoning channel.\n"
            "<|channel|>final\n"
            '{"action_index": 1}'
        )
        record = _gemma_thought_record(raw_response)

        example = build_example(
            record,
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=True,
        )

        self.assertEqual(example["messages"][1]["content"], raw_response)
        self.assertEqual(example["completion"], example["messages"][1]["content"])

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


if __name__ == "__main__":
    unittest.main()
