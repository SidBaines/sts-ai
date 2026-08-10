from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from sts_ai.prompting import (
    ACTION_ONLY_OUTPUT,
    NEUTRAL_FRAME,
    REASONING_ACTION_OUTPUT,
    render_action_prompt,
)
from sts_ai.schemas import LegalAction
from sts_ai.teacher import PUBLIC_OBSERVATION_VERSION, public_observation_hash
from sts_ai.visible_reasoning_eval import (
    ACTION_ONLY_SCHEMA,
    REASONING_ACTION_SCHEMA,
    GeneratedText,
    build_contract_prompts,
    build_model_comparison,
    evaluate_generated_text,
)


STATE_TEXT = "Battle turn 2\nPlayer HP: 50/80, block: 0, energy: 3/3"
ACTIONS = [
    LegalAction(index=0, bits=11, description="play Strike [Attack] (cost 1)"),
    LegalAction(index=1, bits=12, description="play Defend [Skill] (cost 1)"),
    LegalAction(index=2, bits=13, description="end turn"),
]


def _row() -> dict:
    user = render_action_prompt(
        STATE_TEXT,
        ACTIONS,
        framing=NEUTRAL_FRAME,
        induce_reasoning=False,
        output_contract=ACTION_ONLY_OUTPUT,
    )
    completion = '{"action_index":1}'
    prompt = f"<chat>{user}</chat><assistant>"
    descriptions = [{"description": action.description} for action in ACTIONS]
    return {
        "loss_mask_mode": "action",
        "output_contract": "action_only",
        "prompt": prompt,
        "completion": completion,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": completion},
        ],
        "target_action_index": 1,
        "teacher_action_index": 1,
        "window_id": "seed_10_r0_w0",
        "world_seed": 10,
        "decision_index": 22,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "public_state_hash": public_observation_hash(
            STATE_TEXT,
            descriptions,
            observation_version=PUBLIC_OBSERVATION_VERSION,
        ),
    }


class _FakeGenerator:
    def __init__(self) -> None:
        self.action_prompt = _row()["messages"][0]["content"]

    def render_chat_prompt(self, user_prompt: str) -> str:
        return f"<chat>{user_prompt}</chat><assistant>"

    def generate(
        self,
        chat_prompt: str,
        *,
        max_tokens: int,
        seed: int,
    ) -> GeneratedText:
        del max_tokens, seed
        if REASONING_ACTION_SCHEMA in chat_prompt:
            text = '{"reasoning":"block the incoming hit","action_index":1}'
        else:
            text = '{"action_index":1}'
        return GeneratedText(
            text=text,
            prompt_tokens=101,
            completion_tokens=9,
            latency_s=0.25,
        )


class PromptRebuildTests(unittest.TestCase):
    def test_action_only_rebuild_is_byte_identical_to_stored_user_content(self) -> None:
        row = _row()

        prompts = build_contract_prompts(row)

        self.assertEqual(
            prompts.user_prompts[ACTION_ONLY_OUTPUT],
            row["messages"][0]["content"],
        )
        self.assertEqual(
            hashlib.sha256(
                prompts.user_prompts[ACTION_ONLY_OUTPUT].encode("utf-8")
            ).hexdigest(),
            hashlib.sha256(
                row["messages"][0]["content"].encode("utf-8")
            ).hexdigest(),
        )

    def test_reasoning_contract_changes_exactly_one_schema_line(self) -> None:
        prompts = build_contract_prompts(_row())
        action_only = prompts.user_prompts[ACTION_ONLY_OUTPUT]
        reasoning = prompts.user_prompts[REASONING_ACTION_OUTPUT]

        self.assertEqual(
            reasoning,
            action_only.replace(
                ACTION_ONLY_SCHEMA,
                REASONING_ACTION_SCHEMA,
                1,
            ),
        )
        self.assertEqual(action_only.count(ACTION_ONLY_SCHEMA), 1)
        self.assertEqual(reasoning.count(REASONING_ACTION_SCHEMA), 1)

    def test_rebuild_rejects_non_neutral_or_other_stored_prompt_drift(self) -> None:
        row = _row()
        row["messages"][0]["content"] = row["messages"][0]["content"].replace(
            NEUTRAL_FRAME,
            "Choose recklessly.",
        )
        row["prompt"] = f"<chat>{row['messages'][0]['content']}</chat><assistant>"

        with self.assertRaisesRegex(
            ValueError,
            "action_only_user_prompt_rebuild_mismatch",
        ):
            build_contract_prompts(row)


class OutputParsingTests(unittest.TestCase):
    def test_action_only_and_reasoning_outputs_have_exact_contract_validity(self) -> None:
        action_only = evaluate_generated_text(
            GeneratedText('{"action_index":1}', 10, 4, 0.1),
            legal_actions=ACTIONS,
            output_contract=ACTION_ONLY_OUTPUT,
            teacher_action_index=1,
        )
        reasoning = evaluate_generated_text(
            GeneratedText(
                '{"reasoning":"Defend avoids damage","action_index":1}',
                10,
                10,
                0.2,
            ),
            legal_actions=ACTIONS,
            output_contract=REASONING_ACTION_OUTPUT,
            teacher_action_index=1,
        )

        self.assertTrue(action_only["harness_action_valid"])
        self.assertTrue(action_only["exact_contract_valid"])
        self.assertTrue(action_only["teacher_agreement"])
        self.assertTrue(reasoning["harness_action_valid"])
        self.assertTrue(reasoning["exact_contract_valid"])
        self.assertEqual(reasoning["visible_reasoning"], "Defend avoids damage")

    def test_harness_valid_is_distinct_from_exact_contract_valid(self) -> None:
        fenced = evaluate_generated_text(
            GeneratedText('answer: {"action_index":1}', 10, 6, 0.1),
            legal_actions=ACTIONS,
            output_contract=ACTION_ONLY_OUTPUT,
            teacher_action_index=1,
        )
        missing_reasoning = evaluate_generated_text(
            GeneratedText('{"action_index":1}', 10, 4, 0.1),
            legal_actions=ACTIONS,
            output_contract=REASONING_ACTION_OUTPUT,
            teacher_action_index=1,
        )

        self.assertTrue(fenced["harness_action_valid"])
        self.assertFalse(fenced["exact_contract_valid"])
        self.assertTrue(missing_reasoning["harness_action_valid"])
        self.assertFalse(missing_reasoning["exact_contract_valid"])

    def test_out_of_range_action_is_invalid_and_disagrees(self) -> None:
        result = evaluate_generated_text(
            GeneratedText('{"action_index":99}', 10, 4, 0.1),
            legal_actions=ACTIONS,
            output_contract=ACTION_ONLY_OUTPUT,
            teacher_action_index=1,
        )

        self.assertFalse(result["harness_action_valid"])
        self.assertFalse(result["strict_action_legal"])
        self.assertFalse(result["teacher_agreement"])


class PairedReportTests(unittest.TestCase):
    def test_model_comparison_pairs_seeds_and_records_exact_prompts(self) -> None:
        comparison = build_model_comparison(
            [_row()],
            _FakeGenerator(),
            model_label="base",
            max_tokens=64,
            seed=17,
        )

        self.assertEqual(comparison["n_generations"], 2)
        action, reasoning = comparison["rows"]
        self.assertEqual(action["sample_seed"], 17)
        self.assertEqual(reasoning["sample_seed"], 17)
        self.assertEqual(action["output_contract"], ACTION_ONLY_OUTPUT)
        self.assertEqual(reasoning["output_contract"], REASONING_ACTION_OUTPUT)
        self.assertEqual(
            action["chat_prompt"],
            _row()["prompt"],
        )
        self.assertTrue(
            comparison["contract_summaries"][ACTION_ONLY_OUTPUT][
                "teacher_agreement_rate"
            ]
        )
        self.assertTrue(
            comparison["contract_summaries"][REASONING_ACTION_OUTPUT][
                "teacher_agreement_rate"
            ]
        )

    def test_runtime_action_prompt_must_match_frozen_chat_prompt(self) -> None:
        class _Drifted(_FakeGenerator):
            def render_chat_prompt(self, user_prompt: str) -> str:
                return "DRIFT" + super().render_chat_prompt(user_prompt)

        with self.assertRaisesRegex(
            ValueError,
            "runtime_action_only_chat_prompt_mismatch",
        ):
            build_model_comparison(
                [deepcopy(_row())],
                _Drifted(),
                model_label="base",
                max_tokens=64,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
