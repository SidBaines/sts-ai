from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_teacher_sft import (
    _load_labels_manifest,
    build_teacher_examples,
    main,
)
from sts_ai.prompting import (
    ACTION_ONLY_OUTPUT,
    ACTION_TEXT_OUTPUT,
    NEUTRAL_FRAME,
    TURN_PLAN_OUTPUT,
    render_action_prompt,
)
from sts_ai.schemas import LegalAction


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


def _teacher_row(state_hash: str, action_index: int) -> dict:
    return {
        "observation_version": "combat_public_v2",
        "teacher_privilege": "simulator_full_state",
        "teacher_selection_rule": "aggregated_root_visits",
        "teacher_queries": [
            {
                "observation_version": "combat_public_v2",
                "teacher_privilege": "simulator_full_state",
                "teacher_selection_rule": "aggregated_root_visits",
            }
        ],
        "public_state_hash": state_hash,
        "world_seed": 17,
        "source_decision_index": 4,
        "turn": 0,
        "window_id": "seed_17_r0_w0",
        "source_stem": "seed_17_r0",
        "state_text": f"public combat state {state_hash}",
        "legal_actions": [
            {"index": 0, "bits": 3, "description": "play Strike -> NOB"},
            {"index": 1, "bits": 99, "description": "end turn"},
        ],
        "base_action": {"display_index": 0},
        "reference": {
            "consensus_action_index": action_index,
            "consensus_fraction": 1.0,
        },
    }


def _teacher_query(
    action_index: int,
    *,
    evaluation: float,
    search_seed: int,
    draw_order_seed: int | None,
    sequence: list[tuple[str, int]],
    abstained: bool = False,
) -> dict:
    return {
        "observation_version": "combat_public_v2",
        "teacher_privilege": "simulator_full_state",
        "teacher_selection_rule": "aggregated_root_visits",
        "query": {
            "search_seed": search_seed,
            "draw_order_seed": draw_order_seed,
        },
        "teacher_vote": {
            "action_index": action_index,
            "abstained": abstained,
            "displayed_action_visits": {"0": 70, "1": 30},
        },
        "search": {
            "best_evaluation": evaluation,
            "best_sequence": [
                {"bits": index, "description": description, "turn": turn}
                for index, (description, turn) in enumerate(sequence)
            ],
        },
    }


class TeacherSftOutputContractTest(unittest.TestCase):
    def test_action_only_teacher_rejects_thinking_prefix(self):
        with self.assertRaisesRegex(ValueError, "enable_thinking=False"):
            build_teacher_examples(
                [],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=True,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
            )

    def test_dataset_prompt_matches_action_only_inference_prompt(self):
        legal_action_dicts = [
            {"index": 0, "bits": 3, "description": "play Strike -> NOB"},
            {"index": 1, "bits": 99, "description": "end turn"},
        ]
        row = _teacher_row("public-hash", 1)
        row["state_text"] = "public combat state"
        row["legal_actions"] = legal_action_dicts
        tokenizer = _OffsetCharTokenizer()

        examples, manifest = build_teacher_examples(
            [row],
            tokenizer=tokenizer,
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
        )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        expected_user_prompt = render_action_prompt(
            row["state_text"],
            [LegalAction(**action) for action in legal_action_dicts],
            NEUTRAL_FRAME,
            output_contract=ACTION_ONLY_OUTPUT,
        )
        expected_chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": expected_user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        self.assertEqual(example["messages"][0]["content"], expected_user_prompt)
        self.assertEqual(example["prompt"], expected_chat_prompt)
        self.assertEqual(example["completion"], '{"action_index":1}')
        self.assertEqual(example["output_contract"], ACTION_ONLY_OUTPUT)
        self.assertNotIn('"reasoning"', example["prompt"])
        self.assertEqual(manifest["version"], 3)
        self.assertEqual(manifest["observation_version"], "combat_public_v2")
        self.assertEqual(
            manifest["teacher_selection_rule"],
            "aggregated_root_visits",
        )
        self.assertEqual(manifest["output_contract"], ACTION_ONLY_OUTPUT)
        self.assertEqual(manifest["loss_mask_mode"], "action")
        self.assertEqual(manifest["n_examples_before_limit"], 1)
        self.assertIsNone(manifest["max_examples"])
        self.assertEqual(manifest["n_examples"], 1)

    def test_action_text_target_is_canonical_consensus_description(self):
        row = _teacher_row("semantic-state", 0)

        examples, manifest = build_teacher_examples(
            [row],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            output_contract=ACTION_TEXT_OUTPUT,
        )

        self.assertEqual(examples[0]["completion"], '{"action":"play Strike -> NOB"}')
        self.assertEqual(
            examples[0]["messages"][1]["content"],
            examples[0]["completion"],
        )
        self.assertEqual(examples[0]["target_action_index"], 0)
        self.assertEqual(
            examples[0]["target_action_description"],
            "play Strike -> NOB",
        )
        self.assertEqual(examples[0]["target_source"], {"kind": "consensus"})
        self.assertEqual(manifest["output_contract"], ACTION_TEXT_OUTPUT)

    def test_turn_plan_tie_break_is_deterministic_and_appends_end_turn(self):
        row = _teacher_row("plan-state", 0)
        row["legal_actions"].insert(
            1,
            {"index": 2, "bits": 4, "description": "play Defend"},
        )
        queries = [
            _teacher_query(
                2,
                evaluation=100.0,
                search_seed=0,
                draw_order_seed=None,
                sequence=[
                    ("play Strike -> NOB", 0),
                    ("losing non-eligible query", 0),
                ],
            ),
            _teacher_query(
                0,
                evaluation=20.0,
                search_seed=4,
                draw_order_seed=None,
                sequence=[("play Strike -> NOB", 0), ("losing draw tie", 0)],
            ),
            _teacher_query(
                0,
                evaluation=20.0,
                search_seed=3,
                draw_order_seed=9,
                sequence=[("play Strike -> NOB", 0), ("losing seeded draw", 0)],
            ),
            _teacher_query(
                0,
                evaluation=20.0,
                search_seed=3,
                draw_order_seed=None,
                sequence=[
                    ("play Strike -> NOB", 0),
                    ("play Defend", 0),
                    ("future turn", 1),
                ],
            ),
            _teacher_query(
                0,
                evaluation=19.0,
                search_seed=1,
                draw_order_seed=None,
                sequence=[("play Strike -> NOB", 0), ("losing evaluation", 0)],
            ),
        ]
        # Visit dictionaries must cover the expanded legal menu.
        for query in queries:
            query["teacher_vote"]["displayed_action_visits"] = {
                "0": 70,
                "1": 20,
                "2": 10,
            }

        completions = []
        for ordered_queries in (queries, list(reversed(queries))):
            candidate = copy.deepcopy(row)
            candidate["teacher_queries"] = ordered_queries
            examples, manifest = build_teacher_examples(
                [candidate],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
                output_contract=TURN_PLAN_OUTPUT,
            )
            completions.append(examples[0]["completion"])
            self.assertEqual(examples[0]["plan_source"], "eligible_query")
            self.assertEqual(manifest["output_contract"], TURN_PLAN_OUTPUT)
            self.assertEqual(
                manifest["turn_plan_skips"],
                {
                    "n_skipped": 0,
                    "reason_counts": {},
                    "skipped_public_state_hashes": [],
                },
            )

        self.assertEqual(completions[0], completions[1])
        self.assertEqual(
            completions[0],
            '{"plan":["play Strike -> NOB","play Defend","end turn"],'
            '"action":"play Strike -> NOB"}',
        )

    def test_turn_plan_falls_back_to_non_eligible_anchored_query(self):
        row = _teacher_row("fallback-plan-hash", 0)
        row["teacher_queries"] = [
            _teacher_query(
                0,
                evaluation=100.0,
                search_seed=1,
                draw_order_seed=None,
                sequence=[("end turn", 0)],
            ),
            _teacher_query(
                1,
                evaluation=2.0,
                search_seed=3,
                draw_order_seed=None,
                sequence=[("play Strike -> NOB", 0), ("end turn", 0)],
            ),
            _teacher_query(
                1,
                evaluation=3.0,
                search_seed=4,
                draw_order_seed=None,
                sequence=[("play Strike -> NOB", 0), ("fallback winner", 0)],
            ),
        ]

        examples, manifest = build_teacher_examples(
            [row],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            output_contract=TURN_PLAN_OUTPUT,
        )

        self.assertEqual(
            examples[0]["completion"],
            '{"plan":["play Strike -> NOB","fallback winner","end turn"],'
            '"action":"play Strike -> NOB"}',
        )
        self.assertEqual(examples[0]["plan_source"], "non_eligible_query")
        self.assertEqual(manifest["turn_plan_skips"]["n_skipped"], 0)

    def test_turn_plan_unanchorable_states_are_skipped_and_manifested(self):
        anchorable = _teacher_row("m-anchorable", 0)
        anchorable["teacher_queries"] = [
            _teacher_query(
                0,
                evaluation=1.0,
                search_seed=1,
                draw_order_seed=None,
                sequence=[("play Strike -> NOB", 0), ("end turn", 0)],
            )
        ]
        mismatch = _teacher_row("z-mismatch", 0)
        mismatch["teacher_queries"] = [
            _teacher_query(
                0,
                evaluation=2.0,
                search_seed=1,
                draw_order_seed=None,
                sequence=[("end turn", 0)],
            )
        ]
        empty_this_turn = _teacher_row("a-empty-this-turn", 0)
        empty_this_turn["teacher_queries"] = [
            _teacher_query(
                0,
                evaluation=3.0,
                search_seed=1,
                draw_order_seed=None,
                sequence=[("future turn", 1)],
            )
        ]

        examples, manifest = build_teacher_examples(
            [mismatch, anchorable, empty_this_turn],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            output_contract=TURN_PLAN_OUTPUT,
        )

        self.assertEqual(
            [example["public_state_hash"] for example in examples],
            ["m-anchorable"],
        )
        self.assertEqual(
            manifest["turn_plan_skips"],
            {
                "n_skipped": 2,
                "reason_counts": {"no_candidate_query": 2},
                "skipped_public_state_hashes": [
                    "a-empty-this-turn",
                    "z-mismatch",
                ],
            },
        )

    def test_hidden_order_majority_must_match_unmodified_teacher_action(self):
        row = _teacher_row("public-hash", 0)
        row["reference"]["hidden_order_consensus"] = {
            "consensus_action_index": 1,
            "consensus_fraction": 2 / 3,
        }

        examples, manifest = build_teacher_examples(
            [row],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=True,
        )

        self.assertEqual(examples, [])
        self.assertEqual(
            manifest["skipped_record_counts"],
            {"hidden_order_action_conflict": 1},
        )

    def test_max_examples_selects_stable_hash_prefix_after_filtering(self):
        rejected = _teacher_row("0-rejected", 0)
        rejected["reference"]["consensus_fraction"] = 0.5
        rows = [
            _teacher_row("z-state", 0),
            _teacher_row("a-state", 1),
            _teacher_row("m-state", 1),
            _teacher_row("z-state", 0),
            rejected,
        ]

        examples, manifest = build_teacher_examples(
            rows,
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            max_examples=2,
        )

        self.assertEqual(
            [example["public_state_hash"] for example in examples],
            ["a-state", "m-state"],
        )
        self.assertEqual(manifest["n_examples_before_limit"], 3)
        self.assertEqual(manifest["n_examples"], 2)
        self.assertEqual(manifest["max_examples"], 2)
        self.assertEqual(
            manifest["selection_method"],
            "stable_public_state_hash_prefix_after_filtering_and_dedup",
        )
        self.assertEqual(manifest["n_examples_omitted_by_limit"], 1)
        self.assertEqual(manifest["skipped_record_counts"], {"low_search_consensus": 1})
        self.assertEqual(manifest["teacher_action_counts"], {"1": 2})
        self.assertEqual(manifest["token_accounting"]["n_examples"], 2)
        self.assertEqual(manifest["token_accounting"]["n_examples_counted"], 2)

    def test_max_examples_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_examples must be positive"):
            build_teacher_examples(
                [_teacher_row("a-state", 1)],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
                max_examples=0,
            )

    def test_first_per_turn_selects_earliest_source_state_before_filtering(self):
        first = _teacher_row("first-turn-zero", 1)
        first["source_decision_index"] = 4
        later = _teacher_row("later-turn-zero", 0)
        later["source_decision_index"] = 5
        second_turn = _teacher_row("first-turn-one", 2)
        second_turn["turn"] = 1
        second_turn["source_decision_index"] = 6

        examples, manifest = build_teacher_examples(
            [later, second_turn, first],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            state_selection="first_per_turn",
        )

        self.assertEqual(
            {example["public_state_hash"] for example in examples},
            {"first-turn-zero", "first-turn-one"},
        )
        self.assertEqual(manifest["source_state_selection"], "first_per_turn")
        self.assertEqual(manifest["n_input_rows"], 3)
        self.assertEqual(manifest["n_source_rows_selected"], 2)
        self.assertEqual(manifest["n_source_rows_omitted_by_selection"], 1)

    def test_first_per_turn_does_not_fall_forward_after_filtering(self):
        first = _teacher_row("ambiguous-first", 1)
        first["reference"]["consensus_fraction"] = 0.5
        later = _teacher_row("eligible-later", 0)
        later["source_decision_index"] = 5

        examples, manifest = build_teacher_examples(
            [later, first],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            state_selection="first_per_turn",
        )

        self.assertEqual(examples, [])
        self.assertEqual(
            manifest["skipped_record_counts"],
            {"low_search_consensus": 1},
        )
        self.assertEqual(manifest["n_source_rows_selected"], 1)
        self.assertEqual(manifest["n_source_rows_omitted_by_selection"], 1)

    def test_first_per_turn_keeps_same_coordinates_in_distinct_windows(self):
        first_window = _teacher_row("first-window", 0)
        second_window = _teacher_row("second-window", 1)
        second_window["window_id"] = "seed_18_r0_w0"
        second_window["source_stem"] = "seed_18_r0"

        examples, manifest = build_teacher_examples(
            [second_window, first_window],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            state_selection="first_per_turn",
        )

        self.assertEqual(
            {example["public_state_hash"] for example in examples},
            {"first-window", "second-window"},
        )
        self.assertEqual(manifest["n_source_rows_selected"], 2)

    def test_first_per_turn_selection_is_stable_under_input_shuffle(self):
        turn_zero_first = _teacher_row("turn-zero-first", 0)
        turn_zero_later = _teacher_row("turn-zero-later", 1)
        turn_zero_later["source_decision_index"] = 5
        turn_one_first = _teacher_row("turn-one-first", 1)
        turn_one_first["turn"] = 1
        turn_one_first["source_decision_index"] = 6
        rows = [turn_zero_first, turn_zero_later, turn_one_first]

        selected_hash_sets = []
        for ordered_rows in (rows, list(reversed(rows))):
            examples, _ = build_teacher_examples(
                ordered_rows,
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
                state_selection="first_per_turn",
            )
            selected_hash_sets.append(
                {example["public_state_hash"] for example in examples}
            )

        self.assertEqual(
            selected_hash_sets,
            [
                {"turn-zero-first", "turn-one-first"},
                {"turn-zero-first", "turn-one-first"},
            ],
        )

    def test_first_per_turn_deduplicates_exact_source_coordinate_rows(self):
        row = _teacher_row("exact-duplicate", 1)

        examples, manifest = build_teacher_examples(
            [row, copy.deepcopy(row)],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
            state_selection="first_per_turn",
        )

        self.assertEqual(
            [example["public_state_hash"] for example in examples],
            ["exact-duplicate"],
        )
        self.assertEqual(manifest["n_input_rows"], 2)
        self.assertEqual(manifest["n_source_rows_selected"], 1)
        self.assertEqual(manifest["n_source_rows_omitted_by_selection"], 1)

    def test_first_per_turn_rejects_conflicting_source_coordinate_rows(self):
        first = _teacher_row("first-state", 0)
        conflicting = _teacher_row("different-state", 1)

        with self.assertRaisesRegex(
            ValueError,
            "conflicting rows at source coordinate",
        ):
            build_teacher_examples(
                [first, conflicting],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
                state_selection="first_per_turn",
            )

    def test_missing_reference_action_is_not_an_alias_conflict(self):
        row = _teacher_row("ambiguous-reference", 0)
        row["reference"]["consensus_action_index"] = None
        row["reference"]["consensus_fraction"] = 1 / 3

        examples, manifest = build_teacher_examples(
            [row],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
        )

        self.assertEqual(examples, [])
        self.assertEqual(
            manifest["skipped_record_counts"],
            {"missing_or_ambiguous_reference_action": 1},
        )

    def test_distinct_reference_actions_are_a_public_alias_conflict(self):
        first = _teacher_row("aliased-state", 0)
        second = _teacher_row("aliased-state", 1)

        examples, manifest = build_teacher_examples(
            [first, second],
            tokenizer=_OffsetCharTokenizer(),
            tokenizer_id="fake/tokenizer",
            enable_thinking=False,
            min_consensus=2 / 3,
            require_hidden_consensus=False,
        )

        self.assertEqual(examples, [])
        self.assertEqual(
            manifest["skipped_record_counts"],
            {"public_alias_teacher_conflict": 2},
        )

    def test_first_per_turn_requires_auditable_source_coordinates(self):
        row = _teacher_row("missing-turn", 0)
        del row["turn"]
        with self.assertRaisesRegex(ValueError, "integer turn"):
            build_teacher_examples(
                [row],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
                state_selection="first_per_turn",
            )

    def test_cli_max_examples_writes_auditable_subset_manifest(self):
        rows = [
            _teacher_row("z-state", 0),
            _teacher_row("a-state", 1),
            _teacher_row("m-state", 1),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels_path = root / "labels.jsonl"
            out_path = root / "teacher.jsonl"
            labels_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            labels_path.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "search_teacher_labels",
                        "version": 2,
                        "observation_version": "combat_public_v2",
                        "teacher_privilege": "simulator_full_state",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "n_rows": len(rows),
                        "source_manifest": "source.validated_v3.json",
                        "source_manifest_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "build_teacher_sft.py",
                "--labels",
                str(labels_path),
                "--tokenizer",
                "fake/tokenizer",
                "--out",
                str(out_path),
                "--max-examples",
                "2",
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

            examples = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(
                out_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            [example["public_state_hash"] for example in examples],
            ["a-state", "m-state"],
        )
        self.assertEqual(manifest["n_examples_before_limit"], 3)
        self.assertEqual(manifest["n_examples"], 2)
        self.assertEqual(manifest["max_examples"], 2)
        self.assertEqual(manifest["teacher_action_counts"], {"1": 2})
        self.assertEqual(manifest["token_accounting"]["n_examples"], 2)
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        self.assertEqual(len(manifest["source_labels"]["sha256"]), 64)
        self.assertEqual(
            len(manifest["source_labels"]["manifest_sha256"]),
            64,
        )

    def test_cli_first_per_turn_writes_source_selection_audit(self):
        first = _teacher_row("first-state", 1)
        later = _teacher_row("later-state", 0)
        later["source_decision_index"] = 5
        rows = [later, first]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels_path = root / "labels.jsonl"
            out_path = root / "teacher.jsonl"
            labels_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            labels_path.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "search_teacher_labels",
                        "version": 2,
                        "observation_version": "combat_public_v2",
                        "teacher_privilege": "simulator_full_state",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "n_rows": len(rows),
                        "source_manifest": "source.validated_v3.json",
                        "source_manifest_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "build_teacher_sft.py",
                "--labels",
                str(labels_path),
                "--tokenizer",
                "fake/tokenizer",
                "--out",
                str(out_path),
                "--state-selection",
                "first_per_turn",
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

            examples = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(
                out_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            [example["public_state_hash"] for example in examples],
            ["first-state"],
        )
        self.assertEqual(manifest["source_state_selection"], "first_per_turn")
        self.assertEqual(manifest["n_source_rows_selected"], 1)
        self.assertEqual(manifest["n_source_rows_omitted_by_selection"], 1)

    def test_adjacent_label_manifest_is_required_and_must_match_rows(self):
        rows = [_teacher_row("state", 0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            labels_path = Path(temp_dir) / "labels.jsonl"
            labels_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adjacent manifest"):
                _load_labels_manifest(labels_path, rows)

            labels_path.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "search_teacher_labels",
                        "version": 2,
                        "observation_version": "combat_public_v2",
                        "teacher_privilege": "simulator_full_state",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "n_rows": 2,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "disagrees"):
                _load_labels_manifest(labels_path, rows)

    def test_rejects_v1_or_native_selection_rows_before_formatting(self):
        v1 = _teacher_row("v1", 0)
        v1["observation_version"] = "combat_public_v1"
        with self.assertRaisesRegex(ValueError, "observation_version"):
            build_teacher_examples(
                [v1],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
            )

        native = _teacher_row("native", 0)
        native["teacher_selection_rule"] = "native_selected_action"
        native["teacher_queries"][0][
            "teacher_selection_rule"
        ] = "native_selected_action"
        with self.assertRaisesRegex(ValueError, "expected 'aggregated_root_visits'"):
            build_teacher_examples(
                [native],
                tokenizer=_OffsetCharTokenizer(),
                tokenizer_id="fake/tokenizer",
                enable_thinking=False,
                min_consensus=2 / 3,
                require_hidden_consensus=False,
            )


if __name__ == "__main__":
    unittest.main()
