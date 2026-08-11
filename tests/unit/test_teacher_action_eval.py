from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.score_teacher_actions import (
    _adapter_provenance,
    _manifest_provenance,
    main as score_main,
)
from sts_ai.teacher_action_eval import (
    ACTION_ONLY_OUTPUT_CONTRACT,
    ACTION_TEXT_OUTPUT_CONTRACT,
    CandidateSequenceScore,
    MlxCandidateScorer,
    MlxGreedyGenerator,
    TURN_PLAN_OUTPUT_CONTRACT,
    action_completion,
    action_text_completion,
    build_teacher_action_report,
    build_teacher_generation_report,
    declared_action_descriptions,
    declared_action_indices,
    evaluate_generated_action,
    rerender_turn_plan_prompt_for_action_text,
    turn_plan_completion,
)
from sts_ai.prompting import render_action_prompt
from sts_ai.schemas import LegalAction


SEMANTIC_ASSISTANT_TURN_TERMINATOR = "<turn|>\n"


def _prompt(indices: str = "0, 1, 2") -> str:
    return (
        "<bos>exact stored bytes\n"
        f"Valid action_index values are: {indices}. Use only these LEGAL ACTIONS "
        "indices; do not use hand indices.\n"
        "LEGAL ACTIONS\n0: a\n1: b\n2: c\n"
    )


def _row(
    *,
    teacher: int = 1,
    window_id: str = "seed_1_r0_w0",
    prompt: str | None = None,
) -> dict:
    return {
        "loss_mask_mode": "action",
        "output_contract": "action_only",
        "prompt": _prompt() if prompt is None else prompt,
        "completion": action_completion(teacher),
        "target_action_index": teacher,
        "teacher_action_index": teacher,
        "window_id": window_id,
        "world_seed": 1,
        "decision_index": 7,
        "public_state_hash": "abc",
    }


def _semantic_prompt(output_contract: str) -> str:
    user_prompt = render_action_prompt(
        "Player HP: 40/80\nGAME STATE sentinel",
        [
            LegalAction(index=0, bits=10, description="play Bash -> Nob"),
            LegalAction(index=1, bits=11, description="play Strike -> Nob"),
            LegalAction(index=2, bits=12, description="end turn"),
        ],
        output_contract=output_contract,
    )
    return (
        "<bos>"
        + user_prompt.removesuffix("\n")
        + SEMANTIC_ASSISTANT_TURN_TERMINATOR
        + "<|turn>model\n"
    )


def _semantic_row(
    output_contract: str,
    *,
    teacher: int = 0,
) -> dict:
    descriptions = ["play Bash -> Nob", "play Strike -> Nob", "end turn"]
    completion = (
        action_text_completion(descriptions[teacher])
        if output_contract == ACTION_TEXT_OUTPUT_CONTRACT
        else turn_plan_completion([descriptions[teacher], "end turn"])
    )
    return {
        **_row(teacher=teacher, prompt=_semantic_prompt(output_contract)),
        "output_contract": output_contract,
        "completion": completion,
        "target_action_description": descriptions[teacher],
        "assistant_turn_terminator": SEMANTIC_ASSISTANT_TURN_TERMINATOR,
    }


class FakeScorer:
    def __init__(self, log_probabilities: dict[str, float]):
        self.log_probabilities = log_probabilities
        self.calls: list[tuple[str, list[str]]] = []

    def score_candidates(self, prompt, completions):
        completions = list(completions)
        self.calls.append((prompt, completions))
        return [
            CandidateSequenceScore(self.log_probabilities[value], len(value))
            for value in completions
        ]


class SizedFakeScorer:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def score_candidates(self, prompt, completions):
        completions = list(completions)
        self.calls.append((prompt, completions))
        return [CandidateSequenceScore(*self.values[value]) for value in completions]


class FakeGenerator:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def generate(self, prompt, *, max_tokens):
        self.calls.append((prompt, max_tokens))
        return next(self.values)


class DeclaredActionIndicesTest(unittest.TestCase):
    def test_parses_exact_zero_based_contiguous_declaration(self):
        self.assertEqual(declared_action_indices(_prompt()), [0, 1, 2])

    def test_rejects_missing_duplicate_malformed_and_noncontiguous_declarations(self):
        bad_prompts = [
            "no declaration",
            _prompt() + _prompt(),
            _prompt("0, 2"),
            _prompt("1, 2"),
            _prompt("0, 01"),
            _prompt("0, 1, x"),
        ]
        for prompt in bad_prompts:
            with self.subTest(prompt=prompt[:40]):
                with self.assertRaises(ValueError):
                    declared_action_indices(prompt)


class DeclaredActionDescriptionsTest(unittest.TestCase):
    def test_parses_exact_numbered_menu(self):
        self.assertEqual(
            declared_action_descriptions(
                _semantic_prompt(ACTION_TEXT_OUTPUT_CONTRACT),
                assistant_turn_terminator=SEMANTIC_ASSISTANT_TURN_TERMINATOR,
            ),
            ["play Bash -> Nob", "play Strike -> Nob", "end turn"],
        )

    def test_realistic_template_tail_does_not_pollute_last_description(self):
        prompt = _semantic_prompt(ACTION_TEXT_OUTPUT_CONTRACT)

        self.assertTrue(
            prompt.endswith("\n2: end turn<turn|>\n<|turn>model\n")
        )
        self.assertEqual(
            declared_action_descriptions(
                prompt,
                assistant_turn_terminator=SEMANTIC_ASSISTANT_TURN_TERMINATOR,
            )[-1],
            "end turn",
        )

    def test_rejects_empty_menu_and_junk_before_menu(self):
        bad_prompts = (
            "<bos>LEGAL ACTIONS<turn|>\n<|turn>model\n",
            (
                "<bos>LEGAL ACTIONS\nnot a menu entry\n0: end turn"
                "<turn|>\n<|turn>model\n"
            ),
        )
        for prompt in bad_prompts:
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    declared_action_descriptions(
                        prompt,
                        assistant_turn_terminator=(
                            SEMANTIC_ASSISTANT_TURN_TERMINATOR
                        ),
                    )

    def test_rejects_noncontiguous_duplicate_and_duplicated_menus(self):
        base = _semantic_prompt(ACTION_TEXT_OUTPUT_CONTRACT)
        bad_prompts = (
            base.replace("1: play Strike -> Nob", "3: play Strike -> Nob"),
            base.replace("1: play Strike -> Nob", "0: play Strike -> Nob"),
            base.replace(
                SEMANTIC_ASSISTANT_TURN_TERMINATOR,
                "\nLEGAL ACTIONS\n0: duplicate"
                + SEMANTIC_ASSISTANT_TURN_TERMINATOR,
            ),
        )
        for prompt in bad_prompts:
            with self.subTest(prompt=prompt[-80:]):
                with self.assertRaises(ValueError):
                    declared_action_descriptions(
                        prompt,
                        assistant_turn_terminator=(
                            SEMANTIC_ASSISTANT_TURN_TERMINATOR
                        ),
                    )


class SemanticCandidateReportTest(unittest.TestCase):
    def test_last_entry_teacher_action_scores_with_realistic_template_tail(self):
        row = _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT, teacher=2)
        candidates = [
            action_text_completion("play Bash -> Nob"),
            action_text_completion("play Strike -> Nob"),
            action_text_completion("end turn"),
        ]
        scorer = SizedFakeScorer(
            {
                candidate: (-float(index + 1), 2)
                for index, candidate in enumerate(candidates)
            }
        )

        report = build_teacher_action_report(
            [row],
            scorer,
            output_contract=ACTION_TEXT_OUTPUT_CONTRACT,
        )

        self.assertEqual(report["n_scored_rows"], 1)
        self.assertEqual(report["n_skipped_rows"], 0)
        self.assertEqual(scorer.calls, [(row["prompt"], candidates)])
        self.assertEqual(report["rows"][0]["teacher_action_index"], 2)

    def test_semantic_contract_invariants_raise_instead_of_skipping(self):
        cases = []
        missing_terminator = _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT)
        del missing_terminator["assistant_turn_terminator"]
        cases.append((missing_terminator, "missing_assistant_turn_terminator"))
        empty_terminator = _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT)
        empty_terminator["assistant_turn_terminator"] = ""
        cases.append((empty_terminator, "missing_assistant_turn_terminator"))
        drifted_target = _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT)
        drifted_target["target_action_description"] = "play Strike -> Nob"
        cases.append((drifted_target, "target_action_description_mismatch"))

        for row, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    build_teacher_action_report(
                        [row],
                        SizedFakeScorer({}),
                        output_contract=ACTION_TEXT_OUTPUT_CONTRACT,
                    )

    def test_action_text_candidates_are_canonical_and_primary_uses_raw_sum(self):
        row = _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT, teacher=0)
        candidates = [
            action_text_completion("play Bash -> Nob"),
            action_text_completion("play Strike -> Nob"),
            action_text_completion("end turn"),
        ]
        scorer = SizedFakeScorer(
            {
                candidates[0]: (-1.0, 1),
                candidates[1]: (-2.0, 20),
                candidates[2]: (-3.0, 2),
            }
        )

        report = build_teacher_action_report(
            [row],
            scorer,
            output_contract=ACTION_TEXT_OUTPUT_CONTRACT,
        )

        scored = report["rows"][0]
        self.assertEqual(scorer.calls, [(row["prompt"], candidates)])
        self.assertEqual(scored["top1_action_index"], 0)
        self.assertEqual(scored["top1_by_mean_token_index"], 1)
        self.assertTrue(scored["top1_agreement"])
        self.assertEqual(
            [candidate["action_description"] for candidate in scored["candidates"]],
            ["play Bash -> Nob", "play Strike -> Nob", "end turn"],
        )

    def test_turn_plan_scoring_swaps_only_instruction_and_uses_action_candidates(self):
        row = _semantic_row(TURN_PLAN_OUTPUT_CONTRACT, teacher=1)
        candidates = [
            action_text_completion("play Bash -> Nob"),
            action_text_completion("play Strike -> Nob"),
            action_text_completion("end turn"),
        ]
        scorer = SizedFakeScorer(
            {candidate: (-float(index + 1), 3) for index, candidate in enumerate(candidates)}
        )

        report = build_teacher_action_report(
            [row],
            scorer,
            output_contract=TURN_PLAN_OUTPUT_CONTRACT,
        )

        scoring_prompt = scorer.calls[0][0]
        self.assertEqual(scorer.calls[0][1], candidates)
        self.assertNotEqual(scoring_prompt, row["prompt"])
        self.assertEqual(
            scoring_prompt.partition("GAME STATE\n")[2],
            row["prompt"].partition("GAME STATE\n")[2],
        )
        self.assertEqual(report["rows"][0]["scoring_contract"], "action_text")

    def test_rerender_rejects_missing_or_duplicate_instruction(self):
        prompt = _semantic_prompt(TURN_PLAN_OUTPUT_CONTRACT)
        rendered = rerender_turn_plan_prompt_for_action_text(prompt)
        self.assertEqual(
            rendered.partition("GAME STATE\n")[2],
            prompt.partition("GAME STATE\n")[2],
        )
        with self.assertRaisesRegex(ValueError, "instruction_count"):
            rerender_turn_plan_prompt_for_action_text("no instruction")

    def test_noncanonical_semantic_completions_are_skipped(self):
        action_rows = []
        for completion in (
            '{"action": "play Bash -> Nob"}',
            '{"action":"not declared"}',
            '{"action":"play Bash -> Nob","extra":1}',
        ):
            row = _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT)
            row["completion"] = completion
            action_rows.append(row)
        action_report = build_teacher_action_report(
            action_rows,
            FakeScorer({}),
            output_contract=ACTION_TEXT_OUTPUT_CONTRACT,
        )
        self.assertEqual(
            action_report["skipped_record_counts"],
            {"noncanonical_action_text_completion": 3},
        )

        plan_rows = []
        for completion in (
            '{"action":"play Bash -> Nob","plan":["play Bash -> Nob","end turn"]}',
            '{"plan":["play Bash -> Nob","end turn"],"action":"end turn"}',
            '{"plan":[],"action":"play Bash -> Nob"}',
            '{"plan":["play Bash -> Nob"],"action":"play Bash -> Nob"}',
        ):
            row = _semantic_row(TURN_PLAN_OUTPUT_CONTRACT)
            row["completion"] = completion
            plan_rows.append(row)
        plan_report = build_teacher_action_report(
            plan_rows,
            FakeScorer({}),
            output_contract=TURN_PLAN_OUTPUT_CONTRACT,
        )
        self.assertEqual(
            plan_report["skipped_record_counts"],
            {"noncanonical_turn_plan_completion": 4},
        )


class GenerativeTeacherActionEvalTest(unittest.TestCase):
    def test_generated_parse_edges_are_fail_closed(self):
        cases = (
            ("not json", False, False, None),
            ('{"action":"play Bash -> Nob","extra":1}', True, False, None),
            ('{"action":"unknown"}', True, False, None),
            ('prefix {"action":"play Bash -> Nob"}', False, False, None),
            ('{"action":"play Bash -> Nob"} trailing', False, False, None),
            ('{"action":"play Bash -> Nob"}', True, True, 0),
        )
        for text, valid_json, matched, chosen_index in cases:
            with self.subTest(text=text):
                result = evaluate_generated_action(
                    text,
                    output_contract=ACTION_TEXT_OUTPUT_CONTRACT,
                    action_indices=[0, 1],
                    action_descriptions=["play Bash -> Nob", "end turn"],
                )
                self.assertEqual(result["valid_json"], valid_json)
                self.assertEqual(result["matched"], matched)
                self.assertEqual(result["chosen_index"], chosen_index)

    def test_generation_report_supports_all_contracts_and_aggregates(self):
        cases = (
            (ACTION_ONLY_OUTPUT_CONTRACT, _row(teacher=1), '{"action_index":1}'),
            (
                ACTION_TEXT_OUTPUT_CONTRACT,
                _semantic_row(ACTION_TEXT_OUTPUT_CONTRACT, teacher=1),
                '{"action":"play Strike -> Nob"}',
            ),
            (
                TURN_PLAN_OUTPUT_CONTRACT,
                _semantic_row(TURN_PLAN_OUTPUT_CONTRACT, teacher=1),
                '{"plan":["play Strike -> Nob","end turn"],'
                '"action":"play Strike -> Nob"}',
            ),
        )
        for output_contract, row, generated in cases:
            with self.subTest(output_contract=output_contract):
                generator = FakeGenerator([generated])
                report = build_teacher_generation_report(
                    [row],
                    generator,
                    output_contract=output_contract,
                )
                self.assertEqual(report["kind"], "teacher_action_greedy_generation")
                self.assertEqual(report["overall"]["valid_json_rate"], 1.0)
                self.assertEqual(report["overall"]["matched_rate"], 1.0)
                self.assertEqual(report["overall"]["top1_rate"], 1.0)
                self.assertEqual(report["rows"][0]["chosen_index"], 1)
                self.assertEqual(generator.calls, [(row["prompt"], 128)])

    def test_mlx_generator_construction_is_lazy(self):
        generator = MlxGreedyGenerator("fake/model", adapter_path="fake/adapter")
        self.assertIsNone(generator._model)
        self.assertIsNone(generator._tokenizer)


class TeacherActionReportTest(unittest.TestCase):
    def test_mlx_scorer_construction_is_lazy(self):
        scorer = MlxCandidateScorer("fake/model", adapter_path="fake/adapter")
        self.assertIsNone(scorer._model)
        self.assertIsNone(scorer._tokenizer)

    def test_scores_all_canonical_candidates_on_exact_stored_prompt(self):
        probabilities = {
            action_completion(0): -3.0,
            action_completion(1): -1.0,
            action_completion(2): -2.0,
        }
        scorer = FakeScorer(probabilities)
        row = _row(teacher=1)
        report = build_teacher_action_report(
            [row],
            scorer,
            provenance={"model_id": "fake"},
        )

        self.assertEqual(scorer.calls, [(row["prompt"], list(probabilities))])
        self.assertEqual(report["n_scored_rows"], 1)
        self.assertEqual(report["n_skipped_rows"], 0)
        self.assertFalse(report["candidate_includes_assistant_turn_terminator"])
        scored = report["rows"][0]
        self.assertEqual(len(scored["prompt_sha256"]), 64)
        self.assertEqual(scored["top1_action_index"], 1)
        self.assertTrue(scored["top1_agreement"])
        expected_probability = math.exp(-1.0) / sum(
            math.exp(value) for value in (-3.0, -1.0, -2.0)
        )
        self.assertAlmostEqual(
            scored["teacher_normalized_probability"],
            expected_probability,
        )
        self.assertAlmostEqual(
            scored["teacher_candidate_nll"],
            -math.log(expected_probability),
        )
        self.assertAlmostEqual(report["overall"]["top1_agreement"], 1.0)
        self.assertEqual(report["overall"]["mean_n_candidates"], 3.0)
        self.assertEqual(report["per_window"][row["window_id"]]["n"], 1)
        self.assertEqual(report["provenance"], {"model_id": "fake"})

    def test_reports_per_window_summary_and_deterministic_top_tie(self):
        values = {action_completion(index): -1.0 for index in range(3)}
        report = build_teacher_action_report(
            [_row(teacher=0), _row(teacher=1)],
            FakeScorer(values),
        )
        self.assertEqual(report["overall"]["n"], 2)
        self.assertEqual(report["overall"]["top1_agreement"], 0.5)
        self.assertTrue(report["rows"][0]["top1_tied"])
        self.assertEqual(report["rows"][0]["top_action_indices"], [0, 1, 2])
        self.assertEqual(report["rows"][0]["top1_action_index"], 0)

    def test_invalid_rows_are_skipped_with_explicit_counts(self):
        missing_prompt = _row()
        missing_prompt["prompt"] = ""
        wrong_contract = _row()
        wrong_contract["output_contract"] = "reasoning_action"
        noncanonical = _row()
        noncanonical["completion"] = '{"action_index": 1}'
        noncontiguous = _row(prompt=_prompt("0, 2"))
        report = build_teacher_action_report(
            [missing_prompt, wrong_contract, noncanonical, noncontiguous],
            FakeScorer({}),
        )
        self.assertEqual(report["n_scored_rows"], 0)
        self.assertEqual(report["n_invalid_rows"], 4)
        self.assertEqual(report["n_skipped_rows"], 4)
        self.assertEqual(
            report["skipped_record_counts"],
            {
                "missing_prompt": 1,
                "noncanonical_action_only_completion": 1,
                "output_contract_not_action_only": 1,
                "valid_action_indices_not_zero_based_contiguous": 1,
            },
        )
        self.assertEqual(
            [row["row_index"] for row in report["skipped_rows"]],
            [0, 1, 2, 3],
        )
        self.assertIsNone(report["overall"]["top1_agreement"])

    def test_target_metadata_must_be_consistent_and_in_range(self):
        mismatch = _row()
        mismatch["teacher_action_index"] = 2
        out_of_range = _row(teacher=4)
        report = build_teacher_action_report(
            [mismatch, out_of_range],
            FakeScorer({}),
        )
        self.assertEqual(
            report["skipped_record_counts"],
            {"teacher_action_index_mismatch": 1, "teacher_action_out_of_range": 1},
        )

    def test_scorer_contract_errors_do_not_produce_a_misleading_row(self):
        class ShortScorer:
            def score_candidates(self, prompt, completions):
                return [CandidateSequenceScore(-1.0, 1)]

        with self.assertRaisesRegex(ValueError, "candidate_score_count_mismatch"):
            build_teacher_action_report([_row()], ShortScorer())


class ScoringProvenanceTest(unittest.TestCase):
    def test_manifest_and_adapter_are_content_addressed_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "teacher.manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "kind": "search_teacher_sft",
                        "version": 3,
                        "loss_mask_mode": "action",
                        "output_contract": "action_only",
                        "enable_thinking": False,
                        "tokenizer_id": "fake/model",
                        "observation_version": "combat_public_v2",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "teacher_privilege": "simulator_full_state",
                        "dataset_sha256": "d" * 64,
                        "source_labels": {
                            "sha256": "a" * 64,
                            "manifest_sha256": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter_path = root / "adapter"
            adapter_path.mkdir()
            (adapter_path / "adapters.safetensors").write_bytes(b"adapter")
            (adapter_path / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )

            manifest = _manifest_provenance(manifest_path)
            adapter = _adapter_provenance(adapter_path)

            self.assertEqual(manifest["output_contract"], "action_only")
            self.assertEqual(manifest["observation_version"], "combat_public_v2")
            self.assertEqual(
                manifest["teacher_selection_rule"],
                "aggregated_root_visits",
            )
            self.assertEqual(len(manifest["sha256"]), 64)
            self.assertEqual(len(adapter["identity_sha256"]), 64)
            self.assertIn("adapters.safetensors", adapter["files"])

    def test_manifest_rejects_reasoning_or_thinking_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            base = {
                "kind": "search_teacher_sft",
                "version": 3,
                "loss_mask_mode": "action",
                "output_contract": "reasoning_action",
                "enable_thinking": False,
                "observation_version": "combat_public_v2",
                "teacher_selection_rule": "aggregated_root_visits",
                "teacher_privilege": "simulator_full_state",
                "dataset_sha256": "d" * 64,
                "source_labels": {
                    "sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                },
            }
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output_contract"):
                _manifest_provenance(path)
            base["output_contract"] = "action_only"
            base["enable_thinking"] = True
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "enable_thinking=false"):
                _manifest_provenance(path)

    def test_manifest_rejects_v1_or_missing_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            value = {
                "kind": "search_teacher_sft",
                "version": 3,
                "loss_mask_mode": "action",
                "output_contract": "action_only",
                "enable_thinking": False,
                "observation_version": "combat_public_v1",
                "teacher_selection_rule": "aggregated_root_visits",
                "teacher_privilege": "simulator_full_state",
                "dataset_sha256": "d" * 64,
                "source_labels": {
                    "sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observation_version"):
                _manifest_provenance(path)
            value["observation_version"] = "combat_public_v2"
            value.pop("source_labels")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_labels"):
                _manifest_provenance(path)


class ScoreTeacherActionsCliTest(unittest.TestCase):
    def test_writes_requested_per_row_jsonl_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.jsonl"
            manifest = root / "teacher.manifest.json"
            output = root / "report.json"
            per_row = root / "rows.jsonl"
            row = {
                **_row(),
                "observation_version": "combat_public_v2",
                "teacher_selection_rule": "aggregated_root_visits",
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "search_teacher_sft",
                        "version": 3,
                        "loss_mask_mode": "action",
                        "output_contract": "action_only",
                        "enable_thinking": False,
                        "tokenizer_id": "fake/model",
                        "observation_version": "combat_public_v2",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "teacher_privilege": "simulator_full_state",
                        "n_examples": 1,
                        "dataset_sha256": dataset_sha256,
                        "source_labels": {
                            "sha256": "a" * 64,
                            "manifest_sha256": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            values = {
                action_completion(index): -float(index) for index in range(3)
            }
            argv = [
                str(dataset),
                "--manifest",
                str(manifest),
                "--out",
                str(output),
                "--per-row-out",
                str(per_row),
            ]
            with mock.patch(
                "scripts.score_teacher_actions.MlxCandidateScorer",
                return_value=FakeScorer(values),
            ), redirect_stdout(io.StringIO()):
                score_main(argv)

            report = json.loads(output.read_text(encoding="utf-8"))
            per_row_values = [
                json.loads(line)
                for line in per_row.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(per_row_values, report["rows"])

            with self.assertRaisesRegex(ValueError, "output_must_be_fresh"):
                score_main(argv)

    def test_generate_mode_supports_turn_plan_and_per_row_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.jsonl"
            manifest = root / "teacher.manifest.json"
            output = root / "report.json"
            per_row = root / "rows.jsonl"
            row = {
                **_semantic_row(TURN_PLAN_OUTPUT_CONTRACT, teacher=1),
                "observation_version": "combat_public_v2",
                "teacher_selection_rule": "aggregated_root_visits",
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "search_teacher_sft",
                        "version": 3,
                        "loss_mask_mode": "action",
                        "output_contract": TURN_PLAN_OUTPUT_CONTRACT,
                        "enable_thinking": False,
                        "tokenizer_id": "fake/model",
                        "observation_version": "combat_public_v2",
                        "teacher_selection_rule": "aggregated_root_visits",
                        "teacher_privilege": "simulator_full_state",
                        "n_examples": 1,
                        "dataset_sha256": dataset_sha256,
                        "source_labels": {
                            "sha256": "a" * 64,
                            "manifest_sha256": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            generator = FakeGenerator(
                [
                    '{"plan":["play Strike -> Nob","end turn"],'
                    '"action":"play Strike -> Nob"}'
                ]
            )
            argv = [
                str(dataset),
                "--manifest",
                str(manifest),
                "--out",
                str(output),
                "--per-row-out",
                str(per_row),
                "--contract",
                TURN_PLAN_OUTPUT_CONTRACT,
                "--mode",
                "generate",
            ]
            with mock.patch(
                "scripts.score_teacher_actions.MlxGreedyGenerator",
                return_value=generator,
            ), redirect_stdout(io.StringIO()):
                score_main(argv)

            report = json.loads(output.read_text(encoding="utf-8"))
            sidecar = [json.loads(line) for line in per_row.read_text().splitlines()]
            self.assertEqual(report["kind"], "teacher_action_greedy_generation")
            self.assertEqual(report["output_contract"], TURN_PLAN_OUTPUT_CONTRACT)
            self.assertEqual(sidecar, report["rows"])
            self.assertEqual(report["overall"]["top1_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
