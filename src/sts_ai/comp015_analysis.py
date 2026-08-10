"""Fail-closed paired analysis for the COMP-015 augmentation experiment.

The module is deliberately model- and filesystem-free.  Callers provide a
frozen JSON config plus already loaded, content-addressed report artifacts.
Training seeds are the three paired replicates; cyclic rotations and the 57
development states are never treated as independent replicates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from sts_ai.action_order_diagnostic import (
    DIAGNOSTIC_KIND,
    DIAGNOSTIC_VERSION,
)
ANALYSIS_CONFIG_KIND = "comp015_paired_analysis_config"
ANALYSIS_CONFIG_VERSION = 1
ANALYSIS_REPORT_KIND = "comp015_paired_analysis"
ANALYSIS_REPORT_VERSION = 1
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_ARMS = ("control", "augmented")
EXPECTED_STEPS = (750, 1500, 2250, 3000)
PRIMARY_STEP = 1500
TRAIN_ROWS = 150
DEV_ROWS = 57
DEV_WINDOWS = 11
TEACHER_REPORT_KIND = "teacher_action_candidate_likelihood"
TEACHER_REPORT_VERSION = 1
_FLOAT_TOLERANCE = 1e-11


@dataclass(frozen=True)
class LoadedArtifact:
    """One JSON report and the SHA-256 of its exact bytes."""

    sha256: str
    value: dict[str, Any]


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _object(value: Any, reason: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(reason)
    return value


def _integer(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(reason)
    return value


def _number(value: Any, reason: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(reason)
    return float(value)


def _artifact_spec(value: Any, reason: str) -> dict[str, Any]:
    spec = _object(value, reason)
    if not isinstance(spec.get("path"), str) or not spec["path"]:
        raise ValueError(f"{reason}_path")
    if not _is_sha256(spec.get("sha256")):
        raise ValueError(f"{reason}_sha256")
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"{reason}_unexpected_keys")
    return spec


def _checkpoint_spec(value: Any, reason: str) -> dict[str, Any]:
    spec = _object(value, reason)
    if not _is_sha256(spec.get("identity_sha256")):
        raise ValueError(f"{reason}_identity_sha256")
    allowed = {"identity_sha256", "files"}
    if not set(spec).issubset(allowed):
        raise ValueError(f"{reason}_unexpected_keys")
    if "files" in spec:
        files = _object(spec["files"], f"{reason}_files")
        if not files or any(
            not isinstance(name, str)
            or not name
            or not _is_sha256(digest)
            for name, digest in files.items()
        ):
            raise ValueError(f"{reason}_files")
    return spec


def _expected_thresholds() -> dict[str, Any]:
    return {
        "primary_step": PRIMARY_STEP,
        "checkpoint_grid": list(EXPECTED_STEPS),
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
    }


def _validate_config(config: dict[str, Any]) -> tuple[bool, bool]:
    if set(config) != {
        "kind",
        "version",
        "experiment_id",
        "seeds",
        "arms",
        "datasets",
        "thresholds",
        "bootstrap",
        "replicates",
    }:
        raise ValueError("analysis_config_keys_mismatch")
    if config.get("kind") != ANALYSIS_CONFIG_KIND:
        raise ValueError("analysis_config_kind_mismatch")
    if config.get("version") != ANALYSIS_CONFIG_VERSION:
        raise ValueError("analysis_config_version_mismatch")
    if config.get("experiment_id") != "COMP-015":
        raise ValueError("analysis_config_experiment_id_mismatch")
    if config.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError("analysis_config_seeds_mismatch")
    if config.get("arms") != list(EXPECTED_ARMS):
        raise ValueError("analysis_config_arms_mismatch")
    if config.get("thresholds") != _expected_thresholds():
        raise ValueError("analysis_config_frozen_thresholds_mismatch")
    bootstrap = _object(config.get("bootstrap"), "analysis_config_bootstrap")
    if set(bootstrap) != {
        "seed",
        "n_resamples",
        "alpha",
        "expected_window_clusters",
    }:
        raise ValueError("analysis_config_bootstrap_keys")
    _integer(bootstrap["seed"], "analysis_config_bootstrap_seed")
    _integer(
        bootstrap["n_resamples"],
        "analysis_config_bootstrap_n_resamples",
        minimum=1,
    )
    alpha = _number(bootstrap["alpha"], "analysis_config_bootstrap_alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("analysis_config_bootstrap_alpha")
    if bootstrap["expected_window_clusters"] != DEV_WINDOWS:
        raise ValueError("analysis_config_bootstrap_window_count")
    datasets = _object(config.get("datasets"), "analysis_config_datasets")
    if set(datasets) != {"train", "development"}:
        raise ValueError("analysis_config_dataset_splits")
    for split, count in (("train", TRAIN_ROWS), ("development", DEV_ROWS)):
        dataset = _object(
            datasets[split],
            f"analysis_config_{split}_dataset",
        )
        if set(dataset) != {"sha256", "manifest_sha256", "n_rows"}:
            raise ValueError(f"analysis_config_{split}_dataset_keys")
        if not _is_sha256(dataset["sha256"]):
            raise ValueError(f"analysis_config_{split}_dataset_sha256")
        if not _is_sha256(dataset["manifest_sha256"]):
            raise ValueError(f"analysis_config_{split}_manifest_sha256")
        if dataset["n_rows"] != count:
            raise ValueError(f"analysis_config_{split}_row_count")

    replicates = _object(config.get("replicates"), "analysis_config_replicates")
    if set(replicates) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("analysis_config_replicate_seeds")
    grid_presence: list[bool] = []
    behavior_presence: list[bool] = []
    for seed in EXPECTED_SEEDS:
        replicate = _object(
            replicates[str(seed)],
            f"analysis_config_seed_{seed}",
        )
        if not set(replicate).issubset(
            {"control", "augmented", "behavior_step_reports"}
        ):
            raise ValueError(f"analysis_config_seed_{seed}_unexpected_keys")
        if not {"control", "augmented"}.issubset(replicate):
            raise ValueError(f"analysis_config_seed_{seed}_arms")
        behavior_presence.append("behavior_step_reports" in replicate)
        for arm in EXPECTED_ARMS:
            arm_spec = _object(
                replicate[arm],
                f"analysis_config_seed_{seed}_{arm}",
            )
            if not set(arm_spec).issubset(
                {"fixed_step1500", "checkpoint_dev_reports"}
            ):
                raise ValueError(
                    f"analysis_config_seed_{seed}_{arm}_unexpected_keys"
                )
            fixed = _object(
                arm_spec.get("fixed_step1500"),
                f"analysis_config_seed_{seed}_{arm}_fixed",
            )
            if set(fixed) != {
                "checkpoint",
                "train_report",
                "development_report",
                "cyclic_report",
                "cyclic_checkpoint_label",
            }:
                raise ValueError(
                    f"analysis_config_seed_{seed}_{arm}_fixed_keys"
                )
            _checkpoint_spec(
                fixed["checkpoint"],
                f"analysis_config_seed_{seed}_{arm}_fixed_checkpoint",
            )
            for key in ("train_report", "development_report", "cyclic_report"):
                _artifact_spec(
                    fixed[key],
                    f"analysis_config_seed_{seed}_{arm}_{key}",
                )
            if (
                not isinstance(fixed["cyclic_checkpoint_label"], str)
                or not fixed["cyclic_checkpoint_label"]
            ):
                raise ValueError(
                    f"analysis_config_seed_{seed}_{arm}_cyclic_label"
                )
            has_grid = "checkpoint_dev_reports" in arm_spec
            grid_presence.append(has_grid)
            if has_grid:
                grid = _object(
                    arm_spec["checkpoint_dev_reports"],
                    f"analysis_config_seed_{seed}_{arm}_grid",
                )
                if set(grid) != {str(step) for step in EXPECTED_STEPS}:
                    raise ValueError(
                        f"analysis_config_seed_{seed}_{arm}_grid_steps"
                    )
                for step in EXPECTED_STEPS:
                    entry = _object(
                        grid[str(step)],
                        f"analysis_config_seed_{seed}_{arm}_step_{step}",
                    )
                    if set(entry) != {"checkpoint", "development_report"}:
                        raise ValueError(
                            f"analysis_config_seed_{seed}_{arm}_step_{step}_keys"
                        )
                    _checkpoint_spec(
                        entry["checkpoint"],
                        f"analysis_config_seed_{seed}_{arm}_step_{step}_checkpoint",
                    )
                    _artifact_spec(
                        entry["development_report"],
                        f"analysis_config_seed_{seed}_{arm}_step_{step}_report",
                    )
    if any(grid_presence) and not all(grid_presence):
        raise ValueError("analysis_config_partial_checkpoint_grid")
    if any(behavior_presence) and not all(behavior_presence):
        raise ValueError("analysis_config_partial_behavior_reports")
    if any(behavior_presence) and not all(grid_presence):
        raise ValueError("analysis_config_behavior_requires_checkpoint_grid")
    if all(behavior_presence):
        for seed in EXPECTED_SEEDS:
            behavior = _object(
                replicates[str(seed)]["behavior_step_reports"],
                f"analysis_config_seed_{seed}_behavior",
            )
            if set(behavior) != {"step", "control", "augmented"}:
                raise ValueError(
                    f"analysis_config_seed_{seed}_behavior_keys"
                )
            if behavior["step"] not in EXPECTED_STEPS:
                raise ValueError(
                    f"analysis_config_seed_{seed}_behavior_step"
                )
            for arm in EXPECTED_ARMS:
                entry = _object(
                    behavior[arm],
                    f"analysis_config_seed_{seed}_behavior_{arm}",
                )
                if set(entry) != {
                    "checkpoint",
                    "train_report",
                    "cyclic_report",
                    "cyclic_checkpoint_label",
                }:
                    raise ValueError(
                        f"analysis_config_seed_{seed}_behavior_{arm}_keys"
                    )
                _checkpoint_spec(
                    entry["checkpoint"],
                    f"analysis_config_seed_{seed}_behavior_{arm}_checkpoint",
                )
                _artifact_spec(
                    entry["train_report"],
                    f"analysis_config_seed_{seed}_behavior_{arm}_train",
                )
                _artifact_spec(
                    entry["cyclic_report"],
                    f"analysis_config_seed_{seed}_behavior_{arm}_cyclic",
                )
                if (
                    not isinstance(entry["cyclic_checkpoint_label"], str)
                    or not entry["cyclic_checkpoint_label"]
                ):
                    raise ValueError(
                        f"analysis_config_seed_{seed}_behavior_{arm}_label"
                    )
    return all(grid_presence), all(behavior_presence)


def declared_artifact_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    """Return every report declaration after validating the config schema."""

    has_grid, has_behavior = _validate_config(config)
    result: list[dict[str, str]] = []
    replicates = config["replicates"]
    for seed in EXPECTED_SEEDS:
        replicate = replicates[str(seed)]
        for arm in EXPECTED_ARMS:
            arm_spec = replicate[arm]
            fixed = arm_spec["fixed_step1500"]
            for key in ("train_report", "development_report", "cyclic_report"):
                result.append(dict(fixed[key]))
            if has_grid:
                for step in EXPECTED_STEPS:
                    result.append(
                        dict(
                            arm_spec["checkpoint_dev_reports"][str(step)][
                                "development_report"
                            ]
                        )
                    )
        if has_behavior:
            for arm in EXPECTED_ARMS:
                behavior = replicate["behavior_step_reports"][arm]
                result.append(dict(behavior["train_report"]))
                result.append(dict(behavior["cyclic_report"]))
    by_path: dict[str, str] = {}
    unique: list[dict[str, str]] = []
    for spec in result:
        prior = by_path.get(spec["path"])
        if prior is not None and prior != spec["sha256"]:
            raise ValueError("analysis_config_conflicting_hash_for_same_path")
        if prior is None:
            by_path[spec["path"]] = spec["sha256"]
            unique.append(spec)
    return unique


def _artifact(
    artifacts: Mapping[str, LoadedArtifact],
    spec: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    loaded = artifacts.get(spec["path"])
    if loaded is None:
        raise ValueError(f"{reason}_missing")
    if loaded.sha256 != spec["sha256"]:
        raise ValueError(f"{reason}_sha256_mismatch")
    return _object(loaded.value, f"{reason}_not_object")


def _assert_close(actual: Any, expected: float, reason: str) -> None:
    value = _number(actual, reason)
    if not math.isclose(
        value,
        float(expected),
        rel_tol=_FLOAT_TOLERANCE,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError(reason)


def _assert_summary(
    summary: Any,
    rows: Sequence[dict[str, Any]],
    reason: str,
) -> None:
    summary = _object(summary, reason)
    if not rows:
        raise ValueError(f"{reason}_empty")
    count = len(rows)
    expected = {
        "n": count,
        "top1_agreement": sum(bool(row["top1_agreement"]) for row in rows)
        / count,
        "mean_n_candidates": sum(int(row["n_candidates"]) for row in rows)
        / count,
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
    if summary.get("n") != count:
        raise ValueError(reason)
    for key, value in expected.items():
        if key != "n":
            _assert_close(summary.get(key), value, reason)


def _checkpoint_provenance(report: dict[str, Any]) -> dict[str, Any]:
    provenance = _object(report.get("provenance"), "report_provenance")
    adapter = provenance.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("report_adapter_provenance_missing")
    return adapter


def _validate_checkpoint(
    actual: Any,
    expected: dict[str, Any],
    reason: str,
) -> None:
    actual = _object(actual, f"{reason}_actual")
    expected = _checkpoint_spec(expected, f"{reason}_expected")
    if actual.get("identity_sha256") != expected["identity_sha256"]:
        raise ValueError(f"{reason}_identity_mismatch")
    if "files" in expected and actual.get("files") != expected["files"]:
        raise ValueError(f"{reason}_files_mismatch")


def _validate_dataset_provenance(
    report: dict[str, Any],
    dataset: dict[str, Any],
    reason: str,
) -> None:
    provenance = _object(report.get("provenance"), f"{reason}_provenance")
    actual_dataset = _object(
        provenance.get("dataset"),
        f"{reason}_dataset_provenance",
    )
    actual_manifest = _object(
        provenance.get("manifest"),
        f"{reason}_manifest_provenance",
    )
    if actual_dataset.get("sha256") != dataset["sha256"]:
        raise ValueError(f"{reason}_dataset_sha256_mismatch")
    if actual_manifest.get("sha256") != dataset["manifest_sha256"]:
        raise ValueError(f"{reason}_manifest_sha256_mismatch")
    if actual_manifest.get("dataset_sha256") != dataset["sha256"]:
        raise ValueError(f"{reason}_manifest_dataset_sha256_mismatch")
    if actual_manifest.get("n_examples") != dataset["n_rows"]:
        raise ValueError(f"{reason}_manifest_row_count_mismatch")


def _validate_candidate_row(row: Any, row_index: int, reason: str) -> dict[str, Any]:
    row = _object(row, reason)
    if row.get("row_index") != row_index:
        raise ValueError(f"{reason}_row_index")
    n_candidates = _integer(
        row.get("n_candidates"),
        f"{reason}_candidate_count",
        minimum=2,
    )
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != n_candidates:
        raise ValueError(f"{reason}_candidates")
    probabilities: list[float] = []
    scores: list[float] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate = _object(candidate, f"{reason}_candidate")
        if candidate.get("action_index") != candidate_index:
            raise ValueError(f"{reason}_candidate_index")
        expected_completion = json.dumps(
            {"action_index": candidate_index},
            separators=(",", ":"),
        )
        if candidate.get("completion") != expected_completion:
            raise ValueError(f"{reason}_candidate_completion")
        n_tokens = _integer(
            candidate.get("n_tokens"),
            f"{reason}_candidate_n_tokens",
            minimum=1,
        )
        probability = _number(
            candidate.get("normalized_probability"),
            f"{reason}_candidate_probability",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{reason}_candidate_probability")
        probabilities.append(probability)
        score = _number(
            candidate.get("sequence_log_probability"),
            f"{reason}_candidate_score",
        )
        scores.append(score)
        _assert_close(
            candidate.get("mean_token_log_probability"),
            score / n_tokens,
            f"{reason}_candidate_mean_score",
        )
        _assert_close(
            candidate.get("normalized_log_probability"),
            math.log(probability),
            f"{reason}_candidate_normalized_log_probability",
        )
    if not math.isclose(
        sum(probabilities),
        1.0,
        rel_tol=_FLOAT_TOLERANCE,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError(f"{reason}_candidate_probability_sum")
    maximum_score = max(scores)
    normalizer = maximum_score + math.log(
        sum(math.exp(score - maximum_score) for score in scores)
    )
    for candidate_index, (probability, score) in enumerate(
        zip(probabilities, scores)
    ):
        _assert_close(
            math.log(probability),
            score - normalizer,
            f"{reason}_candidate_{candidate_index}_softmax",
        )
    teacher_index = _integer(
        row.get("teacher_action_index"),
        f"{reason}_teacher_index",
    )
    if teacher_index >= n_candidates:
        raise ValueError(f"{reason}_teacher_index")
    max_score = max(scores)
    top_indices = [
        index
        for index, score in enumerate(scores)
        if score == max_score
    ]
    if row.get("top_action_indices") != top_indices:
        raise ValueError(f"{reason}_top_action_indices")
    if row.get("top1_action_index") != top_indices[0]:
        raise ValueError(f"{reason}_top1_action_index")
    if row.get("top1_tied") is not (len(top_indices) > 1):
        raise ValueError(f"{reason}_top1_tied")
    if row.get("top1_agreement") is not (top_indices[0] == teacher_index):
        raise ValueError(f"{reason}_top1_agreement")
    teacher_probability = probabilities[teacher_index]
    if teacher_probability <= 0.0:
        raise ValueError(f"{reason}_teacher_probability")
    _assert_close(
        row.get("teacher_normalized_probability"),
        teacher_probability,
        f"{reason}_teacher_probability",
    )
    _assert_close(
        row.get("teacher_candidate_nll"),
        -math.log(teacher_probability),
        f"{reason}_teacher_nll",
    )
    _assert_close(
        row.get("teacher_normalized_log_probability"),
        math.log(teacher_probability),
        f"{reason}_teacher_normalized_log_probability",
    )
    _assert_close(
        row.get("teacher_sequence_log_probability"),
        scores[teacher_index],
        f"{reason}_teacher_score",
    )
    for key in ("public_state_hash", "prompt_sha256"):
        if not _is_sha256(row.get(key)):
            raise ValueError(f"{reason}_{key}")
    if not isinstance(row.get("window_id"), str) or not row["window_id"]:
        raise ValueError(f"{reason}_window_id")
    _integer(row.get("world_seed"), f"{reason}_world_seed")
    _integer(row.get("decision_index"), f"{reason}_decision_index")
    return row


def _teacher_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["row_index"],
        row["public_state_hash"],
        row["prompt_sha256"],
        row["window_id"],
        row["world_seed"],
        row["decision_index"],
        row["teacher_action_index"],
        row["n_candidates"],
    )


def _validate_teacher_report(
    report: dict[str, Any],
    *,
    dataset: dict[str, Any],
    checkpoint: dict[str, Any],
    expected_rows: int,
    reason: str,
) -> list[dict[str, Any]]:
    if report.get("kind") != TEACHER_REPORT_KIND:
        raise ValueError(f"{reason}_kind")
    if report.get("version") != TEACHER_REPORT_VERSION:
        raise ValueError(f"{reason}_version")
    expected_contract = {
        "candidate_format": "compact_action_json_v1",
        "candidate_includes_assistant_turn_terminator": False,
        "output_contract": "action_only",
        "prompt_source": "exact_stored_prompt",
    }
    for key, expected in expected_contract.items():
        if report.get(key) != expected:
            raise ValueError(f"{reason}_{key}")
    if report.get("n_input_rows") != expected_rows:
        raise ValueError(f"{reason}_input_count")
    if report.get("n_scored_rows") != expected_rows:
        raise ValueError(f"{reason}_scored_count")
    if report.get("n_invalid_rows") != 0:
        raise ValueError(f"{reason}_invalid_rows")
    if report.get("n_skipped_rows") != 0:
        raise ValueError(f"{reason}_skipped_rows")
    if report.get("skipped_rows") != []:
        raise ValueError(f"{reason}_skipped_row_details")
    if report.get("skipped_record_counts") not in ({}, None):
        raise ValueError(f"{reason}_skipped_record_counts")
    rows_value = report.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != expected_rows:
        raise ValueError(f"{reason}_rows")
    rows = [
        _validate_candidate_row(row, index, f"{reason}_row_{index}")
        for index, row in enumerate(rows_value)
    ]
    if len({_teacher_identity(row) for row in rows}) != expected_rows:
        raise ValueError(f"{reason}_duplicate_row_identity")
    _assert_summary(report.get("overall"), rows, f"{reason}_overall")
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[row["window_id"]].append(row)
    per_window = report.get("per_window")
    if not isinstance(per_window, dict) or set(per_window) != set(by_window):
        raise ValueError(f"{reason}_per_window")
    for window_id, window_rows in by_window.items():
        _assert_summary(
            per_window[window_id],
            window_rows,
            f"{reason}_per_window_{window_id}",
        )
    _validate_dataset_provenance(report, dataset, reason)
    _validate_checkpoint(
        _checkpoint_provenance(report),
        checkpoint,
        f"{reason}_checkpoint",
    )
    return rows


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("cyclic_summary_empty")
    result = {
        "n": count,
        "correct": sum(bool(row["top1_agreement"]) for row in rows),
        "top1_agreement": sum(bool(row["top1_agreement"]) for row in rows)
        / count,
        "mean_teacher_normalized_probability": sum(
            float(row["teacher_normalized_probability"]) for row in rows
        )
        / count,
        "mean_teacher_candidate_nll": sum(
            float(row["teacher_candidate_nll"]) for row in rows
        )
        / count,
        "mean_teacher_sequence_log_probability": sum(
            float(row["teacher_sequence_log_probability"]) for row in rows
        )
        / count,
        "top1_tied_count": sum(bool(row["top1_tied"]) for row in rows),
    }
    rotated = [row for row in rows if row["rotation"] != 0]
    if rotated:
        result.update(
            {
                "semantic_top1_invariant_count": sum(
                    bool(row["semantic_top1_invariant"]) for row in rotated
                ),
                "semantic_top1_invariance": sum(
                    bool(row["semantic_top1_invariant"]) for row in rotated
                )
                / len(rotated),
                "semantic_top_set_invariant_count": sum(
                    bool(row["semantic_top_set_invariant"]) for row in rotated
                ),
                "semantic_top_set_invariance": sum(
                    bool(row["semantic_top_set_invariant"]) for row in rotated
                )
                / len(rotated),
                "mean_semantic_probability_total_variation": sum(
                    float(row["semantic_probability_total_variation"])
                    for row in rotated
                )
                / len(rotated),
                "mean_absolute_teacher_probability_delta": sum(
                    abs(float(row["teacher_probability_delta_from_unpermuted"]))
                    for row in rotated
                )
                / len(rotated),
            }
        )
    return result


def _source_macro(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_row_index"]].append(row)
    summaries = [_metric_summary(group) for group in grouped.values()]
    count = len(summaries)
    return {
        "n_source_rows": count,
        "mean_source_top1_agreement": sum(
            summary["top1_agreement"] for summary in summaries
        )
        / count,
        "mean_source_teacher_probability": sum(
            summary["mean_teacher_normalized_probability"]
            for summary in summaries
        )
        / count,
        "mean_source_teacher_nll": sum(
            summary["mean_teacher_candidate_nll"] for summary in summaries
        )
        / count,
        "all_rotations_top1_invariant_source_count": sum(
            all(bool(row["semantic_top1_invariant"]) for row in group)
            for group in grouped.values()
        ),
        "all_rotations_top1_invariant_source_fraction": sum(
            all(bool(row["semantic_top1_invariant"]) for row in group)
            for group in grouped.values()
        )
        / count,
        "all_rotations_top_set_invariant_source_count": sum(
            all(bool(row["semantic_top_set_invariant"]) for row in group)
            for group in grouped.values()
        ),
        "all_rotations_top_set_invariant_source_fraction": sum(
            all(bool(row["semantic_top_set_invariant"]) for row in group)
            for group in grouped.values()
        )
        / count,
    }


def _assert_mapping_close(actual: Any, expected: Mapping[str, Any], reason: str) -> None:
    actual = _object(actual, reason)
    if set(actual) != set(expected):
        raise ValueError(reason)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            _assert_close(actual_value, expected_value, reason)
        elif actual_value != expected_value:
            raise ValueError(reason)


def _validate_cyclic_variant(
    row: Any,
    *,
    source_index: int,
    reason: str,
) -> dict[str, Any]:
    row = _object(row, reason)
    if row.get("source_row_index") != source_index:
        raise ValueError(f"{reason}_source_index")
    menu_size = _integer(row.get("menu_size"), f"{reason}_menu_size", minimum=2)
    rotation = _integer(row.get("rotation"), f"{reason}_rotation")
    if rotation >= menu_size:
        raise ValueError(f"{reason}_rotation")
    for key in ("source_public_state_hash", "prompt_sha256", "transformation_sha256"):
        if not _is_sha256(row.get(key)):
            raise ValueError(f"{reason}_{key}")
    for key in (
        "top1_agreement",
        "top1_tied",
        "semantic_top1_invariant",
        "semantic_top_set_invariant",
    ):
        if not isinstance(row.get(key), bool):
            raise ValueError(f"{reason}_{key}")
    for key in (
        "teacher_normalized_probability",
        "teacher_candidate_nll",
        "teacher_sequence_log_probability",
        "semantic_probability_total_variation",
        "teacher_probability_delta_from_unpermuted",
    ):
        _number(row.get(key), f"{reason}_{key}")
    if not 0.0 <= float(row["semantic_probability_total_variation"]) <= 1.0:
        raise ValueError(f"{reason}_tv")
    if not isinstance(row.get("window_id"), str) or not row["window_id"]:
        raise ValueError(f"{reason}_window")
    _integer(row.get("world_seed"), f"{reason}_world_seed")
    _integer(row.get("decision_index"), f"{reason}_decision_index")
    return row


def _validate_cyclic_report(
    report: dict[str, Any],
    *,
    label: str,
    dataset: dict[str, Any],
    checkpoint: dict[str, Any],
    reference_report_sha256: str,
    reference_rows: Sequence[dict[str, Any]],
    reason: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if report.get("kind") != DIAGNOSTIC_KIND:
        raise ValueError(f"{reason}_kind")
    if report.get("version") != DIAGNOSTIC_VERSION:
        raise ValueError(f"{reason}_version")
    if report.get("n_source_rows") != DEV_ROWS:
        raise ValueError(f"{reason}_source_count")
    labels = report.get("checkpoint_labels")
    checkpoints = report.get("checkpoints")
    if not isinstance(labels, list) or not isinstance(checkpoints, dict):
        raise ValueError(f"{reason}_checkpoint_wrapper")
    if labels != sorted(checkpoints) or label not in checkpoints:
        raise ValueError(f"{reason}_checkpoint_label")
    checkpoint_report = _object(
        checkpoints[label],
        f"{reason}_checkpoint_report",
    )
    if checkpoint_report.get("checkpoint_label") != label:
        raise ValueError(f"{reason}_checkpoint_label_mismatch")
    if checkpoint_report.get("n_source_rows") != DEV_ROWS:
        raise ValueError(f"{reason}_checkpoint_source_count")
    rows_value = checkpoint_report.get("rows")
    if not isinstance(rows_value, list):
        raise ValueError(f"{reason}_rows")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows_value):
        source_index = _integer(
            _object(row, f"{reason}_row_{row_number}").get("source_row_index"),
            f"{reason}_row_{row_number}_source_index",
        )
        if source_index >= DEV_ROWS:
            raise ValueError(f"{reason}_row_{row_number}_source_index")
        grouped[source_index].append(
            _validate_cyclic_variant(
                row,
                source_index=source_index,
                reason=f"{reason}_row_{row_number}",
            )
        )
    if set(grouped) != set(range(DEV_ROWS)):
        raise ValueError(f"{reason}_source_indices")
    for source_index, group in grouped.items():
        reference = reference_rows[source_index]
        menu_size = reference["n_candidates"]
        rotations = sorted(row["rotation"] for row in group)
        if rotations != list(range(menu_size)):
            raise ValueError(f"{reason}_source_{source_index}_rotations")
        for row in group:
            expected_identity = (
                reference["public_state_hash"],
                reference["window_id"],
                reference["world_seed"],
                reference["decision_index"],
                reference["teacher_action_index"],
                reference["n_candidates"],
            )
            actual_identity = (
                row["source_public_state_hash"],
                row["window_id"],
                row["world_seed"],
                row["decision_index"],
                row.get("original_teacher_action_index"),
                row["menu_size"],
            )
            if actual_identity != expected_identity:
                raise ValueError(
                    f"{reason}_source_{source_index}_reference_identity"
                )
        identity = next(row for row in group if row["rotation"] == 0)
        if identity["prompt_sha256"] != reference["prompt_sha256"]:
            raise ValueError(f"{reason}_source_{source_index}_identity_prompt")
        for key in (
            "top1_agreement",
            "teacher_normalized_probability",
            "teacher_candidate_nll",
            "teacher_sequence_log_probability",
        ):
            left = identity[key]
            right = reference[key]
            if isinstance(right, float):
                _assert_close(
                    left,
                    right,
                    f"{reason}_source_{source_index}_identity_metric",
                )
            elif left != right:
                raise ValueError(
                    f"{reason}_source_{source_index}_identity_metric"
                )
    variants = [row for index in range(DEV_ROWS) for row in grouped[index]]
    identity_rows = [row for row in variants if row["rotation"] == 0]
    rotated_rows = [row for row in variants if row["rotation"] != 0]
    if checkpoint_report.get("n_unpermuted_rows") != DEV_ROWS:
        raise ValueError(f"{reason}_unpermuted_count")
    if checkpoint_report.get("n_synthetic_rotated_rows") != len(rotated_rows):
        raise ValueError(f"{reason}_rotated_count")
    if checkpoint_report.get("n_all_variants") != len(variants):
        raise ValueError(f"{reason}_variant_count")
    expected_candidate_scores = sum(row["menu_size"] for row in rotated_rows)
    if checkpoint_report.get("n_candidate_sequence_scores") != expected_candidate_scores:
        raise ValueError(f"{reason}_candidate_score_count")
    transformation_hash = canonical_sha256(
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
    if checkpoint_report.get("transformation_set_sha256") != transformation_hash:
        raise ValueError(f"{reason}_transformation_hash")
    if report.get("transformation_set_sha256") != transformation_hash:
        raise ValueError(f"{reason}_wrapper_transformation_hash")
    for key, selected_rows in (
        ("unpermuted", identity_rows),
        ("rotated", rotated_rows),
        ("all_variants", variants),
    ):
        _assert_mapping_close(
            checkpoint_report.get(key),
            _metric_summary(selected_rows),
            f"{reason}_{key}_summary",
        )
    for key, selected_rows in (
        ("rotated_source_macro", rotated_rows),
        ("all_variants_source_macro", variants),
    ):
        _assert_mapping_close(
            checkpoint_report.get(key),
            _source_macro(selected_rows),
            f"{reason}_{key}_summary",
        )
    per_source = checkpoint_report.get("per_source_row")
    if not isinstance(per_source, list) or len(per_source) != DEV_ROWS:
        raise ValueError(f"{reason}_per_source")
    source_metrics: dict[int, dict[str, Any]] = {}
    for source_index, item in enumerate(per_source):
        item = _object(item, f"{reason}_per_source_{source_index}")
        if item.get("source_row_index") != source_index:
            raise ValueError(f"{reason}_per_source_index")
        if (
            item.get("source_public_state_hash")
            != reference_rows[source_index]["public_state_hash"]
            or item.get("window_id") != reference_rows[source_index]["window_id"]
            or item.get("menu_size") != reference_rows[source_index]["n_candidates"]
        ):
            raise ValueError(f"{reason}_per_source_identity")
        expected_metrics = _metric_summary(grouped[source_index])
        _assert_mapping_close(
            item.get("metrics"),
            expected_metrics,
            f"{reason}_per_source_metrics",
        )
        source_metrics[source_index] = expected_metrics
    provenance = _object(report.get("provenance"), f"{reason}_provenance")
    dataset_provenance = _object(
        provenance.get("dataset"),
        f"{reason}_dataset_provenance",
    )
    manifest_provenance = _object(
        provenance.get("manifest"),
        f"{reason}_manifest_provenance",
    )
    if dataset_provenance.get("sha256") != dataset["sha256"]:
        raise ValueError(f"{reason}_dataset_sha256_mismatch")
    if manifest_provenance.get("sha256") != dataset["manifest_sha256"]:
        raise ValueError(f"{reason}_manifest_sha256_mismatch")
    if manifest_provenance.get("dataset_sha256") != dataset["sha256"]:
        raise ValueError(f"{reason}_manifest_dataset_sha256_mismatch")
    checkpoints_provenance = _object(
        provenance.get("checkpoints"),
        f"{reason}_checkpoints_provenance",
    )
    references_provenance = _object(
        provenance.get("reference_reports"),
        f"{reason}_references_provenance",
    )
    _validate_checkpoint(
        checkpoints_provenance.get(label),
        checkpoint,
        f"{reason}_checkpoint",
    )
    checkpoint_local_provenance = _object(
        checkpoint_report.get("provenance"),
        f"{reason}_checkpoint_local_provenance",
    )
    _validate_checkpoint(
        checkpoint_local_provenance.get("checkpoint"),
        checkpoint,
        f"{reason}_checkpoint_local",
    )
    reference = _object(
        references_provenance.get(label),
        f"{reason}_reference_provenance",
    )
    if reference.get("sha256") != reference_report_sha256:
        raise ValueError(f"{reason}_reference_sha256_mismatch")
    local_reference = _object(
        checkpoint_local_provenance.get("reference_report"),
        f"{reason}_local_reference_provenance",
    )
    if local_reference.get("sha256") != reference_report_sha256:
        raise ValueError(f"{reason}_local_reference_sha256_mismatch")
    return checkpoint_report, source_metrics


def _report_metrics(
    dev_report: dict[str, Any],
    cyclic: dict[str, Any],
) -> dict[str, float | int | bool]:
    dev = dev_report["overall"]
    rotated = cyclic["rotated"]
    macro = cyclic["all_variants_source_macro"]
    invariance = float(rotated["semantic_top1_invariance"])
    tv = float(rotated["mean_semantic_probability_total_variation"])
    return {
        "semantic_top1_invariance": invariance,
        "mean_semantic_probability_total_variation": tv,
        "all_variant_source_macro_agreement": float(
            macro["mean_source_top1_agreement"]
        ),
        "all_variant_source_macro_nll": float(
            macro["mean_source_teacher_nll"]
        ),
        "unpermuted_dev_correct": sum(
            bool(row["top1_agreement"]) for row in dev_report["rows"]
        ),
        "unpermuted_dev_accuracy": float(dev["top1_agreement"]),
        "unpermuted_dev_probability": float(
            dev["mean_teacher_normalized_probability"]
        ),
        "unpermuted_dev_nll": float(dev["mean_teacher_candidate_nll"]),
        "ceases_comp014_sensitivity": invariance >= 0.75 and tv <= 0.20,
        "meets_strong_stability": invariance >= 0.90 and tv <= 0.10,
    }


def _deltas(
    control: Mapping[str, float | int | bool],
    augmented: Mapping[str, float | int | bool],
) -> dict[str, float]:
    keys = (
        "semantic_top1_invariance",
        "mean_semantic_probability_total_variation",
        "all_variant_source_macro_agreement",
        "all_variant_source_macro_nll",
        "unpermuted_dev_accuracy",
        "unpermuted_dev_probability",
        "unpermuted_dev_nll",
    )
    return {
        key: float(augmented[key]) - float(control[key])
        for key in keys
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean_requires_values")
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _hierarchical_bootstrap(
    values: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    seed: int,
    n_resamples: int,
    alpha: float,
) -> dict[str, Any]:
    """Bootstrap paired deltas over replicate, then window cluster.

    For each sampled training replicate, development windows are independently
    resampled and their within-window state means are averaged.  The three
    replicate means are then averaged.  This is a secondary uncertainty
    summary; the fixed three-pair thresholds remain the primary decision rule.
    """

    seed_ids = sorted(values)
    if seed_ids != list(EXPECTED_SEEDS):
        raise ValueError("bootstrap_seed_replicates_mismatch")
    windows = sorted(values[seed_ids[0]])
    if len(windows) != DEV_WINDOWS:
        raise ValueError("bootstrap_window_count_mismatch")
    if any(sorted(values[item]) != windows for item in seed_ids):
        raise ValueError("bootstrap_window_id_mismatch")
    metrics = sorted(next(iter(values[seed_ids[0]].values())))
    for item in seed_ids:
        for window in windows:
            if sorted(values[item][window]) != metrics:
                raise ValueError("bootstrap_metric_mismatch")
    observed = {
        metric: _mean(
            [
                _mean(
                    [values[item][window][metric] for window in windows]
                )
                for item in seed_ids
            ]
        )
        for metric in metrics
    }
    rng = random.Random(seed)
    samples = {metric: [] for metric in metrics}
    for _ in range(n_resamples):
        sampled_seeds = [
            seed_ids[rng.randrange(len(seed_ids))]
            for _ in range(len(seed_ids))
        ]
        replicate_values = {metric: [] for metric in metrics}
        for sampled_seed in sampled_seeds:
            sampled_windows = [
                windows[rng.randrange(len(windows))]
                for _ in range(len(windows))
            ]
            for metric in metrics:
                replicate_values[metric].append(
                    _mean(
                        [
                            values[sampled_seed][window][metric]
                            for window in sampled_windows
                        ]
                    )
                )
        for metric in metrics:
            samples[metric].append(_mean(replicate_values[metric]))
    result: dict[str, Any] = {}
    for metric in metrics:
        ordered = sorted(samples[metric])
        result[metric] = {
            "observed_window_macro_delta": observed[metric],
            "percentile_ci": [
                _percentile(ordered, alpha / 2.0),
                _percentile(ordered, 1.0 - alpha / 2.0),
            ],
        }
    return {
        "secondary_only": True,
        "does_not_replace_fixed_three_pair_thresholds": True,
        "resampling_unit_order": [
            "paired_training_seed_replicate",
            "development_window_cluster",
        ],
        "n_training_seed_replicates": len(seed_ids),
        "n_window_clusters": len(windows),
        "window_ids": windows,
        "n_resamples": n_resamples,
        "seed": seed,
        "alpha": alpha,
        "metrics": result,
    }


def _state_deltas(
    validated: Mapping[int, Mapping[str, dict[str, Any]]],
) -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, list[float]]]] = {}
    for seed in EXPECTED_SEEDS:
        control = validated[seed]["control"]
        augmented = validated[seed]["augmented"]
        by_window: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for control_row, augmented_row in zip(
            control["dev_rows"],
            augmented["dev_rows"],
        ):
            window = control_row["window_id"]
            by_window[window]["unpermuted_dev_top1"].append(
                float(bool(augmented_row["top1_agreement"]))
                - float(bool(control_row["top1_agreement"]))
            )
            by_window[window]["unpermuted_dev_nll"].append(
                float(augmented_row["teacher_candidate_nll"])
                - float(control_row["teacher_candidate_nll"])
            )
            by_window[window]["unpermuted_dev_probability"].append(
                float(augmented_row["teacher_normalized_probability"])
                - float(control_row["teacher_normalized_probability"])
            )
        for source_index in range(DEV_ROWS):
            control_source = control["cyclic_source_metrics"][source_index]
            augmented_source = augmented["cyclic_source_metrics"][source_index]
            window = control["dev_rows"][source_index]["window_id"]
            by_window[window]["semantic_top1_invariance"].append(
                float(augmented_source["semantic_top1_invariance"])
                - float(control_source["semantic_top1_invariance"])
            )
            by_window[window]["semantic_probability_tv"].append(
                float(
                    augmented_source[
                        "mean_semantic_probability_total_variation"
                    ]
                )
                - float(
                    control_source[
                        "mean_semantic_probability_total_variation"
                    ]
                )
            )
            by_window[window]["all_variant_top1_agreement"].append(
                float(augmented_source["top1_agreement"])
                - float(control_source["top1_agreement"])
            )
            by_window[window]["all_variant_nll"].append(
                float(augmented_source["mean_teacher_candidate_nll"])
                - float(control_source["mean_teacher_candidate_nll"])
            )
        result[seed] = {
            window: {
                metric: _mean(values)
                for metric, values in sorted(metrics.items())
            }
            for window, metrics in sorted(by_window.items())
        }
    return result


def _select_checkpoint(
    reports: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for step in EXPECTED_STEPS:
        report = reports[step]
        correct = sum(bool(row["top1_agreement"]) for row in report["rows"])
        candidates.append(
            {
                "step": step,
                "correct": correct,
                "accuracy": float(report["overall"]["top1_agreement"]),
                "nll": float(report["overall"]["mean_teacher_candidate_nll"]),
                "eligible": correct >= 28,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = min(
        eligible,
        key=lambda candidate: (candidate["nll"], candidate["step"]),
        default=None,
    )
    return {
        "rule": (
            "retain >=28/57 development correct, then lowest development "
            "NLL, then earlier checkpoint"
        ),
        "candidates": candidates,
        "selected": selected,
    }


def build_comp015_analysis(
    config: dict[str, Any],
    artifacts: Mapping[str, LoadedArtifact],
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate every paired artifact and build the frozen-rule report."""

    has_grid, has_behavior = _validate_config(config)
    if config_sha256 is not None and not _is_sha256(config_sha256):
        raise ValueError("analysis_config_file_sha256_invalid")
    declarations = declared_artifact_specs(config)
    declared_paths = {spec["path"] for spec in declarations}
    if set(artifacts) != declared_paths:
        raise ValueError("analysis_artifact_set_mismatch")
    datasets = config["datasets"]
    replicates = config["replicates"]
    validated: dict[int, dict[str, dict[str, Any]]] = {}
    transformation_hash: str | None = None
    canonical_train_identities: list[tuple[Any, ...]] | None = None
    canonical_dev_identities: list[tuple[Any, ...]] | None = None
    for seed in EXPECTED_SEEDS:
        validated[seed] = {}
        for arm in EXPECTED_ARMS:
            fixed = replicates[str(seed)][arm]["fixed_step1500"]
            checkpoint = fixed["checkpoint"]
            train_report = _artifact(
                artifacts,
                fixed["train_report"],
                f"seed_{seed}_{arm}_fixed_train",
            )
            dev_report = _artifact(
                artifacts,
                fixed["development_report"],
                f"seed_{seed}_{arm}_fixed_dev",
            )
            cyclic_report = _artifact(
                artifacts,
                fixed["cyclic_report"],
                f"seed_{seed}_{arm}_fixed_cyclic",
            )
            train_rows = _validate_teacher_report(
                train_report,
                dataset=datasets["train"],
                checkpoint=checkpoint,
                expected_rows=TRAIN_ROWS,
                reason=f"seed_{seed}_{arm}_fixed_train",
            )
            dev_rows = _validate_teacher_report(
                dev_report,
                dataset=datasets["development"],
                checkpoint=checkpoint,
                expected_rows=DEV_ROWS,
                reason=f"seed_{seed}_{arm}_fixed_dev",
            )
            cyclic, source_metrics = _validate_cyclic_report(
                cyclic_report,
                label=fixed["cyclic_checkpoint_label"],
                dataset=datasets["development"],
                checkpoint=checkpoint,
                reference_report_sha256=fixed["development_report"]["sha256"],
                reference_rows=dev_rows,
                reason=f"seed_{seed}_{arm}_fixed_cyclic",
            )
            current_train = [_teacher_identity(row) for row in train_rows]
            current_dev = [_teacher_identity(row) for row in dev_rows]
            if canonical_train_identities is None:
                canonical_train_identities = current_train
                canonical_dev_identities = current_dev
            elif (
                current_train != canonical_train_identities
                or current_dev != canonical_dev_identities
            ):
                raise ValueError("paired_report_row_identity_mismatch")
            windows = {row["window_id"] for row in dev_rows}
            if len(windows) != DEV_WINDOWS:
                raise ValueError("development_window_count_mismatch")
            current_transformation = cyclic["transformation_set_sha256"]
            if transformation_hash is None:
                transformation_hash = current_transformation
            elif current_transformation != transformation_hash:
                raise ValueError("paired_cyclic_transformation_mismatch")
            validated[seed][arm] = {
                "train_report": train_report,
                "train_rows": train_rows,
                "dev_report": dev_report,
                "dev_rows": dev_rows,
                "cyclic": cyclic,
                "cyclic_source_metrics": source_metrics,
                "metrics": _report_metrics(dev_report, cyclic),
            }
    pairs: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        control_metrics = validated[seed]["control"]["metrics"]
        augmented_metrics = validated[seed]["augmented"]["metrics"]
        pairs[str(seed)] = {
            "control": control_metrics,
            "augmented": augmented_metrics,
            "augmented_minus_control": _deltas(
                control_metrics,
                augmented_metrics,
            ),
        }
    delta_keys = list(pairs["0"]["augmented_minus_control"])
    mean_deltas = {
        key: _mean(
            [
                pairs[str(seed)]["augmented_minus_control"][key]
                for seed in EXPECTED_SEEDS
            ]
        )
        for key in delta_keys
    }
    directional_by_pair = {
        str(seed): {
            "invariance_better": pairs[str(seed)]["augmented_minus_control"][
                "semantic_top1_invariance"
            ]
            > 0.0,
            "tv_better": pairs[str(seed)]["augmented_minus_control"][
                "mean_semantic_probability_total_variation"
            ]
            < 0.0,
            "macro_agreement_better": pairs[str(seed)][
                "augmented_minus_control"
            ]["all_variant_source_macro_agreement"]
            > 0.0,
            "macro_nll_better": pairs[str(seed)]["augmented_minus_control"][
                "all_variant_source_macro_nll"
            ]
            < 0.0,
            "unpermuted_dev_agreement_no_regression": pairs[str(seed)][
                "augmented_minus_control"
            ]["unpermuted_dev_accuracy"]
            >= 0.0,
            "unpermuted_dev_nll_no_regression": pairs[str(seed)][
                "augmented_minus_control"
            ]["unpermuted_dev_nll"]
            <= 0.0,
        }
        for seed in EXPECTED_SEEDS
    }
    all_directional = all(
        all(values.values()) for values in directional_by_pair.values()
    )
    ceasing_count = sum(
        bool(pairs[str(seed)]["augmented"]["ceases_comp014_sensitivity"])
        for seed in EXPECTED_SEEDS
    )
    magnitude_checks = {
        "mean_invariance_delta_at_least_0_20": mean_deltas[
            "semantic_top1_invariance"
        ]
        >= 0.20,
        "mean_tv_delta_at_most_minus_0_20": mean_deltas[
            "mean_semantic_probability_total_variation"
        ]
        <= -0.20,
        "mean_macro_agreement_delta_at_least_0_10": mean_deltas[
            "all_variant_source_macro_agreement"
        ]
        >= 0.10,
        "mean_macro_nll_delta_at_most_minus_0_25": mean_deltas[
            "all_variant_source_macro_nll"
        ]
        <= -0.25,
    }
    primary = {
        "replicate_unit": "paired_training_seed",
        "n_paired_replicates": 3,
        "rotations_are_not_independent": True,
        "development_states_are_not_training_replicates": True,
        "directional_checks_by_pair": directional_by_pair,
        "all_directional_and_no_regression_checks_pass": all_directional,
        "treatment_seeds_ceasing_comp014_sensitivity": ceasing_count,
        "minimum_treatment_seeds_ceasing_sensitivity_pass": ceasing_count >= 2,
        "mean_paired_deltas": mean_deltas,
        "magnitude_checks": magnitude_checks,
        "augmentation_primary_claim_pass": (
            all_directional
            and ceasing_count >= 2
            and all(magnitude_checks.values())
        ),
    }
    bootstrap_config = config["bootstrap"]
    bootstrap = _hierarchical_bootstrap(
        _state_deltas(validated),
        seed=bootstrap_config["seed"],
        n_resamples=bootstrap_config["n_resamples"],
        alpha=bootstrap_config["alpha"],
    )

    selections: dict[str, Any] | None = None
    grid_reports: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    if has_grid:
        selections = {}
        for seed in EXPECTED_SEEDS:
            selections[str(seed)] = {}
            grid_reports[seed] = {}
            for arm in EXPECTED_ARMS:
                grid_reports[seed][arm] = {}
                arm_spec = replicates[str(seed)][arm]
                for step in EXPECTED_STEPS:
                    entry = arm_spec["checkpoint_dev_reports"][str(step)]
                    report = _artifact(
                        artifacts,
                        entry["development_report"],
                        f"seed_{seed}_{arm}_grid_{step}",
                    )
                    rows = _validate_teacher_report(
                        report,
                        dataset=datasets["development"],
                        checkpoint=entry["checkpoint"],
                        expected_rows=DEV_ROWS,
                        reason=f"seed_{seed}_{arm}_grid_{step}",
                    )
                    if [_teacher_identity(row) for row in rows] != canonical_dev_identities:
                        raise ValueError("checkpoint_grid_row_identity_mismatch")
                    if step == PRIMARY_STEP and (
                        entry["development_report"]["sha256"]
                        != replicates[str(seed)][arm]["fixed_step1500"][
                            "development_report"
                        ]["sha256"]
                        or entry["checkpoint"]
                        != replicates[str(seed)][arm]["fixed_step1500"][
                            "checkpoint"
                        ]
                    ):
                        raise ValueError("checkpoint_grid_step1500_mismatch")
                    grid_reports[seed][arm][step] = report
                selections[str(seed)][arm] = _select_checkpoint(
                    grid_reports[seed][arm]
                )

    behavior: dict[str, Any] = {
        "status": "not_available",
        "reason": (
            "complete checkpoint grids and paired selected-treatment-step "
            "train/cyclic reports were not supplied"
        ),
    }
    if has_behavior:
        assert selections is not None
        qualifications: dict[str, Any] = {}
        qualifying: list[dict[str, Any]] = []
        for seed in EXPECTED_SEEDS:
            selection = selections[str(seed)]["augmented"]["selected"]
            if selection is None:
                raise ValueError(
                    f"seed_{seed}_behavior_reports_but_no_selected_treatment"
                )
            behavior_spec = replicates[str(seed)]["behavior_step_reports"]
            step = behavior_spec["step"]
            if step != selection["step"]:
                raise ValueError(f"seed_{seed}_behavior_step_not_selected")
            selected_validated: dict[str, Any] = {}
            for arm in EXPECTED_ARMS:
                entry = behavior_spec[arm]
                grid_entry = replicates[str(seed)][arm][
                    "checkpoint_dev_reports"
                ][str(step)]
                if entry["checkpoint"] != grid_entry["checkpoint"]:
                    raise ValueError(
                        f"seed_{seed}_{arm}_behavior_checkpoint_mismatch"
                    )
                train_report = _artifact(
                    artifacts,
                    entry["train_report"],
                    f"seed_{seed}_{arm}_behavior_train",
                )
                train_rows = _validate_teacher_report(
                    train_report,
                    dataset=datasets["train"],
                    checkpoint=entry["checkpoint"],
                    expected_rows=TRAIN_ROWS,
                    reason=f"seed_{seed}_{arm}_behavior_train",
                )
                if [_teacher_identity(row) for row in train_rows] != canonical_train_identities:
                    raise ValueError("behavior_train_row_identity_mismatch")
                dev_report = grid_reports[seed][arm][step]
                cyclic_report = _artifact(
                    artifacts,
                    entry["cyclic_report"],
                    f"seed_{seed}_{arm}_behavior_cyclic",
                )
                cyclic, _ = _validate_cyclic_report(
                    cyclic_report,
                    label=entry["cyclic_checkpoint_label"],
                    dataset=datasets["development"],
                    checkpoint=entry["checkpoint"],
                    reference_report_sha256=grid_entry["development_report"][
                        "sha256"
                    ],
                    reference_rows=dev_report["rows"],
                    reason=f"seed_{seed}_{arm}_behavior_cyclic",
                )
                selected_validated[arm] = {
                    "train_correct": sum(
                        bool(row["top1_agreement"]) for row in train_rows
                    ),
                    "metrics": _report_metrics(dev_report, cyclic),
                }
            treatment = selected_validated["augmented"]
            control = selected_validated["control"]
            treatment_metrics = treatment["metrics"]
            control_metrics = control["metrics"]
            selected_deltas = _deltas(control_metrics, treatment_metrics)
            checks = {
                "train_at_least_135_of_150": treatment["train_correct"] >= 135,
                "development_at_least_32_of_57": treatment_metrics[
                    "unpermuted_dev_correct"
                ]
                >= 32,
                "teacher_probability_at_least_0_372162": treatment_metrics[
                    "unpermuted_dev_probability"
                ]
                >= 0.372162,
                "teacher_nll_at_most_1_776904": treatment_metrics[
                    "unpermuted_dev_nll"
                ]
                <= 1.776904,
                "invariance_at_least_0_75": treatment_metrics[
                    "semantic_top1_invariance"
                ]
                >= 0.75,
                "tv_at_most_0_20": treatment_metrics[
                    "mean_semantic_probability_total_variation"
                ]
                <= 0.20,
                "beats_control_dev_agreement": selected_deltas[
                    "unpermuted_dev_accuracy"
                ]
                > 0.0,
                "beats_control_dev_nll": selected_deltas[
                    "unpermuted_dev_nll"
                ]
                < 0.0,
                "beats_control_invariance": selected_deltas[
                    "semantic_top1_invariance"
                ]
                > 0.0,
                "beats_control_tv": selected_deltas[
                    "mean_semantic_probability_total_variation"
                ]
                < 0.0,
            }
            qualifies = all(checks.values())
            qualifications[str(seed)] = {
                "step": step,
                "control": control,
                "augmented": treatment,
                "augmented_minus_control": selected_deltas,
                "checks": checks,
                "qualifies_for_behavior": qualifies,
            }
            if qualifies:
                qualifying.append(
                    {
                        "seed": seed,
                        "step": step,
                        "development_nll": treatment_metrics[
                            "unpermuted_dev_nll"
                        ],
                        "development_correct": treatment_metrics[
                            "unpermuted_dev_correct"
                        ],
                        "checkpoint_identity_sha256": behavior_spec[
                            "augmented"
                        ]["checkpoint"]["identity_sha256"],
                    }
                )
        deployment = min(
            qualifying,
            key=lambda item: (
                item["development_nll"],
                -item["development_correct"],
                item["seed"],
            ),
            default=None,
        )
        behavior_gate_pass = len(qualifying) >= 2
        behavior = {
            "status": "complete",
            "required_qualifying_treatment_seeds": 2,
            "n_qualifying_treatment_seeds": len(qualifying),
            "behavior_gate_pass": behavior_gate_pass,
            "qualifications_by_seed": qualifications,
            "deployment_tie_break": (
                "lowest development NLL, then higher development agreement, "
                "then lower training seed"
            ),
            "selected_deployment": deployment if behavior_gate_pass else None,
        }
    return {
        "kind": ANALYSIS_REPORT_KIND,
        "version": ANALYSIS_REPORT_VERSION,
        "experiment_id": "COMP-015",
        "analysis_config_sha256": config_sha256,
        "frozen_thresholds": _expected_thresholds(),
        "artifact_sha256": {
            spec["path"]: spec["sha256"] for spec in declarations
        },
        "n_paired_training_seed_replicates": 3,
        "n_development_states": DEV_ROWS,
        "n_development_window_clusters": DEV_WINDOWS,
        "cyclic_transformation_set_sha256": transformation_hash,
        "fixed_step1500_pairs": pairs,
        "fixed_step1500_primary_decision": primary,
        "secondary_hierarchical_bootstrap": bootstrap,
        "checkpoint_selection": selections,
        "behavior_gate": behavior,
    }
