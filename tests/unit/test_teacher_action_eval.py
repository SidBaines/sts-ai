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
    CandidateSequenceScore,
    MlxCandidateScorer,
    action_completion,
    build_teacher_action_report,
    declared_action_indices,
)


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


if __name__ == "__main__":
    unittest.main()
