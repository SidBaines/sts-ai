from __future__ import annotations

from copy import deepcopy
import math
import unittest

from sts_ai.permutation_sft import build_paired_datasets
from sts_ai.sft_loss_decomposition import (
    TeacherForcedLossScore,
    build_loss_decomposition_report,
    validate_exact_schedule,
)
from tests.unit.test_permutation_sft import (
    MODEL_ID,
    _CharacterTokenizer,
    _source,
)


def _paired_fixture(*, repetitions: int = 3):
    tokenizer = _CharacterTokenizer()
    rows, manifest, dataset_sha, manifest_sha = _source(tokenizer)
    return build_paired_datasets(
        rows,
        manifest,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        repetitions=repetitions,
        seed=19,
        source_dataset_sha256=dataset_sha,
        source_manifest_sha256=manifest_sha,
        require_paired_token_counts=True,
    )


class _StructuredLossScorer:
    def score_row(self, row):
        metadata = row["permutation_augmentation"]
        n_format = row["token_counts"]["n_supervised_format_tokens"]
        n_action = row["token_counts"]["n_supervised_action_tokens"]
        format_loss = float(metadata["pass_index"] + 1)
        action_loss = float(metadata["assigned_teacher_action_index"] + 10)
        return TeacherForcedLossScore(
            categories=("format",) * n_format + ("action",) * n_action,
            negative_log_probabilities=(
                (format_loss,) * n_format + (action_loss,) * n_action
            ),
            n_input_tokens=(
                row["token_counts"]["n_prompt_tokens"]
                + row["token_counts"]["n_completion_tokens"]
            ),
            n_prompt_tokens=row["token_counts"]["n_prompt_tokens"],
            n_completion_tokens=row["token_counts"]["n_completion_tokens"],
            computation_dtype="float32",
        )


class _BadLossScorer(_StructuredLossScorer):
    def __init__(self, *, mode):
        self.mode = mode

    def score_row(self, row):
        result = super().score_row(row)
        if self.mode == "nan":
            return TeacherForcedLossScore(
                categories=result.categories,
                negative_log_probabilities=(
                    math.nan,
                    *result.negative_log_probabilities[1:],
                ),
                n_input_tokens=result.n_input_tokens,
                n_prompt_tokens=result.n_prompt_tokens,
                n_completion_tokens=result.n_completion_tokens,
                computation_dtype=result.computation_dtype,
            )
        if self.mode == "dtype":
            return TeacherForcedLossScore(
                categories=result.categories,
                negative_log_probabilities=result.negative_log_probabilities,
                n_input_tokens=result.n_input_tokens,
                n_prompt_tokens=result.n_prompt_tokens,
                n_completion_tokens=result.n_completion_tokens,
                computation_dtype="float16",
            )
        return TeacherForcedLossScore(
            categories=result.categories[:-1],
            negative_log_probabilities=result.negative_log_probabilities[:-1],
            n_input_tokens=result.n_input_tokens,
            n_prompt_tokens=result.n_prompt_tokens,
            n_completion_tokens=result.n_completion_tokens,
            computation_dtype=result.computation_dtype,
        )


class _ContentLossScorer:
    def __init__(self):
        self.calls = 0

    def score_row(self, row):
        self.calls += 1
        n_format = row["token_counts"]["n_supervised_format_tokens"]
        n_action = row["token_counts"]["n_supervised_action_tokens"]
        action_loss = float(row["target_action_index"] + 1)
        return TeacherForcedLossScore(
            categories=("format",) * n_format + ("action",) * n_action,
            negative_log_probabilities=(
                (0.25,) * n_format + (action_loss,) * n_action
            ),
            n_input_tokens=(
                row["token_counts"]["n_prompt_tokens"]
                + row["token_counts"]["n_completion_tokens"]
            ),
            n_prompt_tokens=row["token_counts"]["n_prompt_tokens"],
            n_completion_tokens=row["token_counts"]["n_completion_tokens"],
            computation_dtype="float32",
        )


class _CachedContentLossScorer(_ContentLossScorer):
    def scoring_key(self, row):
        return (
            row["prompt"],
            row["completion"],
            row["target_action_index"],
            row["assistant_turn_terminator"],
        )


class ExactScheduleValidationTests(unittest.TestCase):
    def test_accepts_both_paired_arms_and_recomputes_hashes(self):
        built = _paired_fixture()

        control = validate_exact_schedule(
            built.control_rows,
            built.control_manifest,
        )
        augmented = validate_exact_schedule(
            built.augmented_rows,
            built.augmented_manifest,
        )

        self.assertEqual(control["arm"], "control_identity")
        self.assertEqual(augmented["arm"], "augmented_cyclic")
        self.assertEqual(control["n_rows"], 6)
        self.assertEqual(control["n_source_rows"], 2)
        self.assertEqual(control["repetitions_per_source"], 3)
        self.assertEqual(
            control["ordered_step_source_sha256"],
            built.control_manifest["augmentation"][
                "ordered_step_source_sha256"
            ],
        )

    def test_rejects_reordered_or_partially_tampered_schedule(self):
        built = _paired_fixture()
        rows = deepcopy(built.augmented_rows)
        rows[1]["schedule_step"] = 7

        with self.assertRaisesRegex(ValueError, "schedule_not_contiguous"):
            validate_exact_schedule(rows, built.augmented_manifest)

        rows = deepcopy(built.augmented_rows)
        rows[0]["permutation_augmentation"]["original_to_assigned"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "original_to_assigned"):
            validate_exact_schedule(rows, built.augmented_manifest)

    def test_rejects_stale_manifest_aggregates_and_token_counts(self):
        built = _paired_fixture()
        manifest = deepcopy(built.augmented_manifest)
        manifest["augmentation"]["menu_size_counts"]["2"] += 1
        with self.assertRaisesRegex(ValueError, "menu_size_counts_mismatch"):
            validate_exact_schedule(built.augmented_rows, manifest)

        rows = deepcopy(built.augmented_rows)
        rows[0]["token_counts"]["n_supervised_action_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "supervised_token_partition"):
            validate_exact_schedule(rows, built.augmented_manifest)


class LossDecompositionTests(unittest.TestCase):
    def test_reports_float32_format_action_and_requested_slices(self):
        built = _paired_fixture()
        report = build_loss_decomposition_report(
            built.augmented_rows,
            built.augmented_manifest,
            _StructuredLossScorer(),
            provenance={"test": True},
        )

        self.assertEqual(report["n_scored_rows"], 6)
        self.assertEqual(report["n_skipped_rows"], 0)
        self.assertEqual(report["loss_computation_dtypes"], ["float32"])
        self.assertEqual(set(report["by_pass"]), {"0", "1", "2"})
        self.assertEqual(set(report["by_menu_size"]), {"2", "3"})
        self.assertEqual(
            report["by_pass"]["0"]["format"]["mean_token_nll"],
            1.0,
        )
        self.assertEqual(
            report["by_pass"]["2"]["format"]["mean_token_nll"],
            3.0,
        )
        for position, summary in report["by_assigned_position"].items():
            self.assertEqual(
                summary["action"]["mean_token_nll"],
                10.0 + int(position),
            )
        self.assertTrue(report["outside_candidate_mass_included"])
        self.assertFalse(report["candidate_normalization"])
        self.assertEqual(len(report["rows"]), 6)
        self.assertEqual(report["provenance"], {"test": True})
        self.assertEqual(report["score_cache"]["n_model_forwards"], 6)
        self.assertEqual(report["score_cache"]["n_cache_hits"], 0)

    def test_exact_duplicate_cache_preserves_every_row_and_group_result(self):
        built = _paired_fixture(repetitions=5)
        uncached_scorer = _ContentLossScorer()
        cached_scorer = _CachedContentLossScorer()

        uncached = build_loss_decomposition_report(
            built.control_rows,
            built.control_manifest,
            uncached_scorer,
        )
        cached = build_loss_decomposition_report(
            built.control_rows,
            built.control_manifest,
            cached_scorer,
        )

        self.assertEqual(uncached_scorer.calls, 10)
        self.assertEqual(cached_scorer.calls, 2)
        self.assertEqual(cached["score_cache"]["n_unique_scoring_inputs"], 2)
        self.assertEqual(cached["score_cache"]["n_model_forwards"], 2)
        self.assertEqual(cached["score_cache"]["n_cache_hits"], 8)
        for key in (
            "overall",
            "by_pass",
            "by_assigned_position",
            "by_menu_size",
            "rows",
        ):
            self.assertEqual(cached[key], uncached[key])

    def test_operational_or_numeric_errors_abort_without_skips(self):
        built = _paired_fixture()
        for mode, pattern in (
            ("nan", "negative_log_probability"),
            ("shape", "action_token_count"),
            ("dtype", "score_not_float32"),
        ):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, pattern):
                    build_loss_decomposition_report(
                        built.control_rows,
                        built.control_manifest,
                        _BadLossScorer(mode=mode),
                    )


if __name__ == "__main__":
    unittest.main()
