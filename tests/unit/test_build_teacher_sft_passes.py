from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_teacher_sft import build_teacher_examples, main
from sts_ai.prompting import ACTION_ONLY_OUTPUT, ACTION_TEXT_OUTPUT


class _OffsetCharTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        rendered = f"<user>{messages[0]['content']}<turn><model>"
        if len(messages) == 2:
            return rendered + messages[1]["content"] + "<turn>"
        return rendered if add_generation_prompt else rendered.removesuffix("<model>")

    def encode(self, text, add_special_tokens=True):
        prefix = [999_999] if add_special_tokens else []
        return prefix + [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if add_special_tokens or not return_offsets_mapping:
            raise AssertionError("action masking must request plain offsets")
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _row(state_hash: str, *, visits: tuple[int, int] = (3, 1)) -> dict:
    query = {
        "observation_version": "combat_public_v2",
        "teacher_privilege": "simulator_full_state",
        "teacher_selection_rule": "aggregated_root_visits",
        "query": {
            "requested_simulations": sum(visits),
            "search_seed": 1,
            "draw_order_seed": None,
        },
        "teacher_vote": {
            "action_index": 0,
            "abstained": False,
            "displayed_action_visits": {
                "0": visits[0],
                "1": visits[1],
            },
        },
        "search": {
            "best_evaluation": 1.0,
            "best_sequence": [
                {"bits": 3, "description": "play Strike -> NOB", "turn": 0},
                {"bits": 99, "description": "end turn", "turn": 0},
            ],
        },
    }
    return {
        "observation_version": "combat_public_v2",
        "teacher_privilege": "simulator_full_state",
        "teacher_selection_rule": "aggregated_root_visits",
        "teacher_queries": [query],
        "public_state_hash": state_hash,
        "world_seed": 17,
        "source_decision_index": 4,
        "turn": 0,
        "window_id": f"window-{state_hash}",
        "source_stem": "seed_17_r0",
        "state_text": f"public combat state {state_hash}",
        "legal_actions": [
            {"index": 0, "bits": 3, "description": "play Strike -> NOB"},
            {"index": 1, "bits": 99, "description": "end turn"},
        ],
        "base_action": {"display_index": 0},
        "reference": {
            "consensus_action_index": 0,
            "consensus_fraction": 1.0,
        },
    }


def _build(rows: list[dict], **overrides):
    arguments = {
        "tokenizer": _OffsetCharTokenizer(),
        "tokenizer_id": "fake/tokenizer",
        "enable_thinking": False,
        "min_consensus": 2 / 3,
        "require_hidden_consensus": False,
    }
    arguments.update(overrides)
    return build_teacher_examples(rows, **arguments)


def _labels_manifest(n_rows: int) -> dict:
    return {
        "kind": "search_teacher_labels",
        "version": 2,
        "observation_version": "combat_public_v2",
        "teacher_privilege": "simulator_full_state",
        "teacher_selection_rule": "aggregated_root_visits",
        "n_rows": n_rows,
        "source_manifest": "source.validated_v3.json",
        "source_manifest_sha256": "a" * 64,
    }


class BuildTeacherSftPassesTest(unittest.TestCase):
    def test_expanded_pass_schedule_is_complete_deterministic_and_auditable(self):
        rows = [_row("state-c"), _row("state-a"), _row("state-b")]

        first, manifest = _build(rows, passes=4, schedule_seed=19)
        second, _ = _build(rows, passes=4, schedule_seed=19)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(
            [example["schedule_step"] for example in first],
            list(range(12)),
        )
        for pass_index in range(4):
            self.assertEqual(
                {
                    example["public_state_hash"]
                    for example in first
                    if example["pass_index"] == pass_index
                },
                {"state-a", "state-b", "state-c"},
            )
        self.assertTrue(
            all(
                example["source_identity"] == example["public_state_hash"]
                for example in first
            )
        )
        self.assertEqual(manifest["output_contract"], ACTION_ONLY_OUTPUT)
        self.assertEqual(manifest["passes"], 4)
        self.assertEqual(manifest["schedule_seed"], 19)
        self.assertEqual(manifest["target_sampling"], "consensus")
        self.assertEqual(manifest["n_examples"], 12)

    def test_single_pass_default_and_explicit_legacy_cli_are_byte_identical(self):
        rows = [_row("state-b"), _row("state-a")]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = root / "labels.jsonl"
            labels.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            labels.with_suffix(".manifest.json").write_text(
                json.dumps(_labels_manifest(len(rows))),
                encoding="utf-8",
            )
            outputs = []
            extra_arguments = (
                [],
                [
                    "--output-contract",
                    "action_only",
                    "--passes",
                    "1",
                    "--schedule-seed",
                    "0",
                    "--target-sampling",
                    "consensus",
                ],
            )
            for index, extra in enumerate(extra_arguments):
                output = root / f"teacher-{index}.jsonl"
                argv = [
                    "build_teacher_sft.py",
                    "--labels",
                    str(labels),
                    "--tokenizer",
                    "fake/tokenizer",
                    "--out",
                    str(output),
                    *extra,
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch(
                        "scripts.build_teacher_sft._load_tokenizer",
                        return_value=_OffsetCharTokenizer(),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    main()
                outputs.append(output.read_bytes())

        self.assertEqual(outputs[0], outputs[1])
        decoded = [json.loads(line) for line in outputs[0].splitlines()]
        self.assertTrue(all("pass_index" not in row for row in decoded))
        self.assertTrue(all("schedule_step" not in row for row in decoded))
        self.assertTrue(all("target_source" not in row for row in decoded))
        self.assertEqual(
            set(decoded[0]),
            {
                "assistant_turn_terminator",
                "base_action_index",
                "completion",
                "decision_index",
                "loss_mask_mode",
                "messages",
                "observation_version",
                "output_contract",
                "phase",
                "prompt",
                "public_state_hash",
                "search_reference",
                "source_stem",
                "target_action_index",
                "teacher_action_index",
                "teacher_privilege",
                "teacher_selection_rule",
                "token_counts",
                "window_id",
                "world_seed",
            },
        )
        user_content = (
            "You are playing Slay the Spire. Choose one legal action from the "
            "list. Use the game state and action descriptions to make the "
            "strongest choice you can.\n\n"
            "Return exactly one JSON object with this schema:\n"
            '{"action_index": 0}\n\n'
            "Valid action_index values are: 0, 1. Use only these LEGAL ACTIONS "
            "indices; do not use hand, enemy, deck, or map indices as "
            "action_index.\n\n"
            "GAME STATE\npublic combat state state-a\n\n"
            "LEGAL ACTIONS\n0: play Strike -> NOB\n1: end turn\n"
        )
        self.assertEqual(
            decoded[0],
            {
                "assistant_turn_terminator": "<turn>",
                "base_action_index": 0,
                "completion": '{"action_index":0}',
                "decision_index": 4,
                "loss_mask_mode": "action",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": '{"action_index":0}'},
                ],
                "observation_version": "combat_public_v2",
                "output_contract": "action_only",
                "phase": "combat",
                "prompt": f"<user>{user_content}<turn><model>",
                "public_state_hash": "state-a",
                "search_reference": {
                    "consensus_action_index": 0,
                    "consensus_fraction": 1.0,
                },
                "source_stem": "seed_17_r0",
                "target_action_index": 0,
                "teacher_action_index": 0,
                "teacher_privilege": "simulator_full_state",
                "teacher_selection_rule": "aggregated_root_visits",
                "token_counts": {
                    "n_action_tokens": 1,
                    "n_completion_tokens": 24,
                    "n_format_tokens": 23,
                    "n_prompt_tokens": 469,
                    "n_supervised_action_tokens": 1,
                    "n_supervised_format_tokens": 23,
                    "n_supervised_thought_tokens": 0,
                    "n_supervised_tokens": 24,
                    "n_thought_tokens": 0,
                },
                "window_id": "window-state-a",
                "world_seed": 17,
            },
        )

    def test_visit_sampling_is_deterministic_and_tracks_empirical_shares(self):
        row = _row("sample-state", visits=(3, 1))

        first, manifest = _build(
            [row],
            output_contract=ACTION_TEXT_OUTPUT,
            passes=2000,
            schedule_seed=23,
            target_sampling="visit_sampled",
        )
        second, _ = _build(
            [row],
            output_contract=ACTION_TEXT_OUTPUT,
            passes=2000,
            schedule_seed=23,
            target_sampling="visit_sampled",
        )

        first_targets = [example["teacher_action_index"] for example in first]
        self.assertEqual(first_targets, [row["teacher_action_index"] for row in second])
        self.assertEqual(set(first_targets), {0, 1})
        observed_zero_share = first_targets.count(0) / len(first_targets)
        self.assertAlmostEqual(observed_zero_share, 0.75, delta=0.04)
        for example in first:
            expected_share = 0.75 if example["teacher_action_index"] == 0 else 0.25
            self.assertEqual(
                example["target_source"],
                {"kind": "visit_sampled", "share": expected_share},
            )
        self.assertEqual(manifest["target_sampling"], "visit_sampled")

    def test_visit_sampling_rejects_incompatible_contract_or_single_pass(self):
        with self.assertRaisesRegex(ValueError, "action_text.*passes > 1"):
            _build([_row("state")], passes=2, target_sampling="visit_sampled")
        with self.assertRaisesRegex(ValueError, "action_text.*passes > 1"):
            _build(
                [_row("state")],
                output_contract=ACTION_TEXT_OUTPUT,
                passes=1,
                target_sampling="visit_sampled",
            )


if __name__ == "__main__":
    unittest.main()
