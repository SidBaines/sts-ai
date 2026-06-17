"""Unit tests for skew-free SFT prompt/completion formatting."""
from __future__ import annotations

import unittest

from sts_ai.train.sft_format import (
    build_example,
    completion_text,
    reconstruct_prompt,
    tokenize_example,
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


class SftFormatTest(unittest.TestCase):
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

    def test_build_example_returns_canonical_text_pair_and_metadata(self):
        tokenizer = RecordingTokenizer()

        example = build_example(
            _record(),
            FRAMING,
            tokenizer=tokenizer,
            enable_thinking=False,
        )

        self.assertEqual(
            set(example),
            {"prompt", "completion", "world_seed", "decision_index", "phase"},
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
