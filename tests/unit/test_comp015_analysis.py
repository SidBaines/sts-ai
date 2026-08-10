from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import tempfile
from pathlib import Path
import unittest

from scripts.analyze_comp015 import _publish_fresh
from sts_ai.comp015_analysis import (
    ANALYSIS_CONFIG_KIND,
    ANALYSIS_CONFIG_VERSION,
    LoadedArtifact,
    _metric_summary,
    _source_macro,
    build_comp015_analysis,
    canonical_sha256,
    declared_artifact_specs,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


TRAIN_DATASET = {
    "sha256": _sha("train-dataset"),
    "manifest_sha256": _sha("train-manifest"),
    "n_rows": 150,
}
DEV_DATASET = {
    "sha256": _sha("dev-dataset"),
    "manifest_sha256": _sha("dev-manifest"),
    "n_rows": 57,
}


def _checkpoint(seed: int, arm: str, step: int) -> dict[str, object]:
    return {
        "identity_sha256": _sha(f"checkpoint-{seed}-{arm}-{step}"),
        "files": {
            "adapter_config.json": _sha(f"config-{seed}-{arm}-{step}"),
            "adapters.safetensors": _sha(f"weights-{seed}-{arm}-{step}"),
        },
    }


def _candidate_row(
    index: int,
    *,
    split: str,
    correct: bool,
    correct_probability: float = 0.8,
    incorrect_probability: float = 0.2,
) -> dict[str, object]:
    teacher_probability = (
        correct_probability if correct else incorrect_probability
    )
    probabilities = [teacher_probability, 1.0 - teacher_probability]
    scores = [math.log(value) for value in probabilities]
    top = 0 if probabilities[0] > probabilities[1] else 1
    window = (
        f"dev_window_{index % 11:02d}"
        if split == "development"
        else f"train_window_{index % 10:02d}"
    )
    return {
        "row_index": index,
        "public_state_hash": _sha(f"{split}-state-{index}"),
        "prompt_sha256": _sha(f"{split}-prompt-{index}"),
        "window_id": window,
        "world_seed": 100 + index // 8,
        "decision_index": index,
        "teacher_action_index": 0,
        "n_candidates": 2,
        "candidates": [
            {
                "action_index": candidate_index,
                "completion": json.dumps(
                    {"action_index": candidate_index},
                    separators=(",", ":"),
                ),
                "n_tokens": 1,
                "normalized_probability": probability,
                "sequence_log_probability": scores[candidate_index],
                "mean_token_log_probability": scores[candidate_index],
                "normalized_log_probability": math.log(probability),
            }
            for candidate_index, probability in enumerate(probabilities)
        ],
        "top_action_indices": [top],
        "top1_action_index": top,
        "top1_tied": False,
        "top1_agreement": correct,
        "teacher_normalized_probability": teacher_probability,
        "teacher_normalized_log_probability": math.log(teacher_probability),
        "teacher_candidate_nll": -math.log(teacher_probability),
        "teacher_sequence_log_probability": scores[0],
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    return {
        "n": count,
        "top1_agreement": sum(bool(row["top1_agreement"]) for row in rows)
        / count,
        "mean_n_candidates": 2.0,
        "mean_teacher_candidate_nll": sum(
            float(row["teacher_candidate_nll"]) for row in rows
        )
        / count,
        "mean_teacher_normalized_probability": sum(
            float(row["teacher_normalized_probability"]) for row in rows
        )
        / count,
        "mean_teacher_sequence_log_probability": sum(
            float(row["teacher_sequence_log_probability"]) for row in rows
        )
        / count,
    }


def _teacher_report(
    *,
    split: str,
    count: int,
    correct: int,
    checkpoint: dict[str, object],
    correct_probability: float = 0.8,
    incorrect_probability: float = 0.2,
) -> dict[str, object]:
    rows = [
        _candidate_row(
            index,
            split=split,
            correct=index < correct,
            correct_probability=correct_probability,
            incorrect_probability=incorrect_probability,
        )
        for index in range(count)
    ]
    by_window: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_window.setdefault(str(row["window_id"]), []).append(row)
    dataset = TRAIN_DATASET if split == "train" else DEV_DATASET
    return {
        "kind": "teacher_action_candidate_likelihood",
        "version": 1,
        "candidate_format": "compact_action_json_v1",
        "candidate_includes_assistant_turn_terminator": False,
        "output_contract": "action_only",
        "prompt_source": "exact_stored_prompt",
        "n_input_rows": count,
        "n_scored_rows": count,
        "n_invalid_rows": 0,
        "n_skipped_rows": 0,
        "skipped_rows": [],
        "skipped_record_counts": {},
        "overall": _summary(rows),
        "per_window": {
            window: _summary(window_rows)
            for window, window_rows in sorted(by_window.items())
        },
        "rows": rows,
        "provenance": {
            "dataset": {"sha256": dataset["sha256"]},
            "manifest": {
                "sha256": dataset["manifest_sha256"],
                "dataset_sha256": dataset["sha256"],
                "n_examples": count,
            },
            "adapter": checkpoint,
        },
    }


def _cyclic_report(
    dev_report: dict[str, object],
    *,
    checkpoint: dict[str, object],
    checkpoint_label: str,
    reference_sha256: str,
    invariant: bool,
    tv: float,
    rotated_correct: bool,
    rotated_teacher_probability: float,
) -> dict[str, object]:
    variants: list[dict[str, object]] = []
    for source_index, reference_value in enumerate(dev_report["rows"]):
        reference = dict(reference_value)
        for rotation in (0, 1):
            is_identity = rotation == 0
            probability = (
                float(reference["teacher_normalized_probability"])
                if is_identity
                else rotated_teacher_probability
            )
            top1_agreement = (
                bool(reference["top1_agreement"])
                if is_identity
                else rotated_correct
            )
            variants.append(
                {
                    "source_row_index": source_index,
                    "source_public_state_hash": reference["public_state_hash"],
                    "prompt_sha256": (
                        reference["prompt_sha256"]
                        if is_identity
                        else _sha(f"rotated-{source_index}")
                    ),
                    "transformation_sha256": _sha(
                        f"transform-{source_index}-{rotation}"
                    ),
                    "window_id": reference["window_id"],
                    "world_seed": reference["world_seed"],
                    "decision_index": reference["decision_index"],
                    "rotation": rotation,
                    "menu_size": 2,
                    "original_teacher_action_index": 0,
                    "top1_agreement": top1_agreement,
                    "top1_tied": False,
                    "semantic_top1_invariant": True if is_identity else invariant,
                    "semantic_top_set_invariant": True if is_identity else invariant,
                    "semantic_probability_total_variation": 0.0 if is_identity else tv,
                    "teacher_probability_delta_from_unpermuted": (
                        0.0
                        if is_identity
                        else probability
                        - float(reference["teacher_normalized_probability"])
                    ),
                    "teacher_normalized_probability": probability,
                    "teacher_candidate_nll": -math.log(probability),
                    "teacher_sequence_log_probability": math.log(probability),
                }
            )
    identity = [row for row in variants if row["rotation"] == 0]
    rotated = [row for row in variants if row["rotation"] != 0]
    transformation = canonical_sha256(
        [
            {
                "source_row_index": row["source_row_index"],
                "rotation": row["rotation"],
                "prompt_sha256": row["prompt_sha256"],
                "transformation_sha256": row["transformation_sha256"],
            }
            for row in variants
        ]
    )
    by_source: dict[int, list[dict[str, object]]] = {}
    for row in variants:
        by_source.setdefault(int(row["source_row_index"]), []).append(row)
    checkpoint_report = {
        "checkpoint_label": checkpoint_label,
        "n_source_rows": 57,
        "n_unpermuted_rows": 57,
        "n_synthetic_rotated_rows": 57,
        "n_all_variants": 114,
        "n_candidate_sequence_scores": 114,
        "transformation_set_sha256": transformation,
        "unpermuted": _metric_summary(identity),
        "rotated": _metric_summary(rotated),
        "all_variants": _metric_summary(variants),
        "rotated_source_macro": _source_macro(rotated),
        "all_variants_source_macro": _source_macro(variants),
        "per_source_row": [
            {
                "source_row_index": index,
                "source_public_state_hash": dev_report["rows"][index][
                    "public_state_hash"
                ],
                "window_id": dev_report["rows"][index]["window_id"],
                "menu_size": 2,
                "all_rotations_top1_invariant": invariant,
                "all_rotations_top_set_invariant": invariant,
                "unpermuted_top1_action_index": dev_report["rows"][index][
                    "top1_action_index"
                ],
                "metrics": _metric_summary(by_source[index]),
            }
            for index in range(57)
        ],
        "rows": variants,
        "provenance": {
            "checkpoint": checkpoint,
            "reference_report": {"sha256": reference_sha256},
        },
    }
    return {
        "kind": "teacher_action_order_equivariance",
        "version": 1,
        "checkpoint_labels": [checkpoint_label],
        "n_source_rows": 57,
        "transformation_set_sha256": transformation,
        "checkpoints": {checkpoint_label: checkpoint_report},
        "provenance": {
            "dataset": {"sha256": DEV_DATASET["sha256"]},
            "manifest": {
                "sha256": DEV_DATASET["manifest_sha256"],
                "dataset_sha256": DEV_DATASET["sha256"],
            },
            "checkpoints": {checkpoint_label: checkpoint},
            "reference_reports": {
                checkpoint_label: {"sha256": reference_sha256}
            },
        },
    }


class _Fixture:
    def __init__(self, *, optional: bool = False):
        self.artifacts: dict[str, LoadedArtifact] = {}
        self.config: dict[str, object] = {
            "kind": ANALYSIS_CONFIG_KIND,
            "version": ANALYSIS_CONFIG_VERSION,
            "experiment_id": "COMP-015",
            "seeds": [0, 1, 2],
            "arms": ["control", "augmented"],
            "datasets": {
                "train": TRAIN_DATASET,
                "development": DEV_DATASET,
            },
            "thresholds": {
                "primary_step": 1500,
                "checkpoint_grid": [750, 1500, 2250, 3000],
                "sensitivity_invariance_min": 0.75,
                "sensitivity_tv_max": 0.20,
                "strong_stability_invariance_min": 0.90,
                "strong_stability_tv_max": 0.10,
                "minimum_pairs_ceasing_sensitivity": 2,
                "mean_invariance_delta_min": 0.20,
                "mean_tv_delta_max": -0.20,
                "mean_all_variant_macro_agreement_delta_min": 0.10,
                "mean_all_variant_macro_nll_delta_max": -0.25,
                "selection_dev_correct_min": 28,
                "behavior_train_correct_min": 135,
                "behavior_dev_correct_min": 32,
                "behavior_teacher_probability_min": 0.372162,
                "behavior_teacher_nll_max": 1.776904,
                "behavior_qualifying_seeds_min": 2,
            },
            "bootstrap": {
                "seed": 17,
                "n_resamples": 200,
                "alpha": 0.05,
                "expected_window_clusters": 11,
            },
            "replicates": {},
        }
        for seed in (0, 1, 2):
            replicate: dict[str, object] = {}
            for arm in ("control", "augmented"):
                checkpoint = _checkpoint(seed, arm, 1500)
                train_correct = 100 if arm == "control" else 140
                dev_correct = (
                    30 if arm == "control" else (33 if seed < 2 else 31)
                )
                train = _teacher_report(
                    split="train",
                    count=150,
                    correct=train_correct,
                    checkpoint=checkpoint,
                )
                dev = _teacher_report(
                    split="development",
                    count=57,
                    correct=dev_correct,
                    checkpoint=checkpoint,
                    correct_probability=0.85 if arm == "augmented" and seed == 1 else 0.8,
                )
                prefix = f"seed{seed}_{arm}_1500"
                train_spec = self._put(f"{prefix}_train.json", train)
                dev_spec = self._put(f"{prefix}_dev.json", dev)
                label = f"seed{seed}_{arm}_step1500"
                cyclic = _cyclic_report(
                    dev,
                    checkpoint=checkpoint,
                    checkpoint_label=label,
                    reference_sha256=dev_spec["sha256"],
                    invariant=arm == "augmented",
                    tv=0.1 if arm == "augmented" else 0.5,
                    rotated_correct=arm == "augmented",
                    rotated_teacher_probability=0.8 if arm == "augmented" else 0.1,
                )
                cyclic_spec = self._put(f"{prefix}_cyclic.json", cyclic)
                arm_spec: dict[str, object] = {
                    "fixed_step1500": {
                        "checkpoint": checkpoint,
                        "train_report": train_spec,
                        "development_report": dev_spec,
                        "cyclic_report": cyclic_spec,
                        "cyclic_checkpoint_label": label,
                    }
                }
                if optional:
                    grid: dict[str, object] = {}
                    for step in (750, 1500, 2250, 3000):
                        if step == 1500:
                            grid[str(step)] = {
                                "checkpoint": checkpoint,
                                "development_report": dev_spec,
                            }
                            continue
                        other_checkpoint = _checkpoint(seed, arm, step)
                        other_report = _teacher_report(
                            split="development",
                            count=57,
                            correct=27,
                            checkpoint=other_checkpoint,
                        )
                        grid[str(step)] = {
                            "checkpoint": other_checkpoint,
                            "development_report": self._put(
                                f"seed{seed}_{arm}_{step}_dev.json",
                                other_report,
                            ),
                        }
                    arm_spec["checkpoint_dev_reports"] = grid
                replicate[arm] = arm_spec
            if optional:
                behavior: dict[str, object] = {"step": 1500}
                for arm in ("control", "augmented"):
                    behavior[arm] = {
                        "checkpoint": replicate[arm]["fixed_step1500"][
                            "checkpoint"
                        ],
                        "train_report": replicate[arm]["fixed_step1500"][
                            "train_report"
                        ],
                        "cyclic_report": replicate[arm]["fixed_step1500"][
                            "cyclic_report"
                        ],
                        "cyclic_checkpoint_label": replicate[arm][
                            "fixed_step1500"
                        ]["cyclic_checkpoint_label"],
                    }
                replicate["behavior_step_reports"] = behavior
            self.config["replicates"][str(seed)] = replicate

    def _put(
        self,
        path: str,
        value: dict[str, object],
    ) -> dict[str, str]:
        digest = _sha(path)
        self.artifacts[path] = LoadedArtifact(digest, value)
        return {"path": path, "sha256": digest}


class Comp015AnalysisTests(unittest.TestCase):
    def test_fixed_primary_rules_and_bootstrap_are_paired_and_deterministic(
        self,
    ) -> None:
        fixture = _Fixture()

        first = build_comp015_analysis(fixture.config, fixture.artifacts)
        second = build_comp015_analysis(fixture.config, fixture.artifacts)

        primary = first["fixed_step1500_primary_decision"]
        self.assertTrue(primary["augmentation_primary_claim_pass"])
        self.assertEqual(primary["n_paired_replicates"], 3)
        self.assertTrue(primary["rotations_are_not_independent"])
        self.assertEqual(
            primary["treatment_seeds_ceasing_comp014_sensitivity"],
            3,
        )
        self.assertEqual(
            first["secondary_hierarchical_bootstrap"],
            second["secondary_hierarchical_bootstrap"],
        )
        self.assertEqual(
            first["secondary_hierarchical_bootstrap"]["n_window_clusters"],
            11,
        )
        self.assertEqual(first["behavior_gate"]["status"], "not_available")

    def test_checkpoint_selection_behavior_gate_and_deployment_tie_break(
        self,
    ) -> None:
        fixture = _Fixture(optional=True)

        report = build_comp015_analysis(fixture.config, fixture.artifacts)

        for seed in ("0", "1", "2"):
            self.assertEqual(
                report["checkpoint_selection"][seed]["augmented"]["selected"][
                    "step"
                ],
                1500,
            )
        behavior = report["behavior_gate"]
        self.assertTrue(behavior["behavior_gate_pass"])
        self.assertEqual(behavior["n_qualifying_treatment_seeds"], 2)
        self.assertEqual(behavior["selected_deployment"]["seed"], 1)
        self.assertFalse(
            behavior["qualifications_by_seed"]["2"]["qualifies_for_behavior"]
        )

    def test_rejects_partial_grid_row_mismatch_and_reference_hash_mismatch(
        self,
    ) -> None:
        fixture = _Fixture(optional=True)
        partial = deepcopy(fixture.config)
        del partial["replicates"]["2"]["control"]["checkpoint_dev_reports"]
        with self.assertRaisesRegex(
            ValueError,
            "partial_checkpoint_grid",
        ):
            declared_artifact_specs(partial)

        row_mismatch = deepcopy(fixture.artifacts)
        path = fixture.config["replicates"]["1"]["control"][
            "fixed_step1500"
        ]["development_report"]["path"]
        changed = deepcopy(row_mismatch[path].value)
        changed["rows"][0]["public_state_hash"] = _sha("wrong-state")
        row_mismatch[path] = LoadedArtifact(row_mismatch[path].sha256, changed)
        with self.assertRaisesRegex(ValueError, "reference_identity"):
            build_comp015_analysis(fixture.config, row_mismatch)

        bad_reference = deepcopy(fixture.artifacts)
        cyclic_path = fixture.config["replicates"]["0"]["control"][
            "fixed_step1500"
        ]["cyclic_report"]["path"]
        changed_cyclic = deepcopy(bad_reference[cyclic_path].value)
        label = changed_cyclic["checkpoint_labels"][0]
        changed_cyclic["provenance"]["reference_reports"][label][
            "sha256"
        ] = _sha("wrong-reference")
        bad_reference[cyclic_path] = LoadedArtifact(
            bad_reference[cyclic_path].sha256,
            changed_cyclic,
        )
        with self.assertRaisesRegex(ValueError, "reference_sha256_mismatch"):
            build_comp015_analysis(fixture.config, bad_reference)

    def test_requires_exact_artifact_set_and_frozen_thresholds(self) -> None:
        fixture = _Fixture()
        missing = dict(fixture.artifacts)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "artifact_set_mismatch"):
            build_comp015_analysis(fixture.config, missing)

        changed = deepcopy(fixture.config)
        changed["thresholds"]["mean_invariance_delta_min"] = 0.19
        with self.assertRaisesRegex(ValueError, "frozen_thresholds"):
            build_comp015_analysis(changed, fixture.artifacts)

    def test_fresh_atomic_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            _publish_fresh(path, b"one")
            self.assertEqual(path.read_bytes(), b"one")
            with self.assertRaisesRegex(ValueError, "output_must_be_fresh"):
                _publish_fresh(path, b"two")
            self.assertEqual(path.read_bytes(), b"one")


if __name__ == "__main__":
    unittest.main()
