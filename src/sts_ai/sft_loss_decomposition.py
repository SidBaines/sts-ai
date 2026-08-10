"""Audited teacher-forced loss decomposition for COMP-015 SFT schedules.

The pure report core accepts a small scorer protocol so schedule validation and
aggregation remain unit-testable without MLX.  The MLX scorer deliberately
reuses the training tokenizer/mask implementation and computes the selected
token log-softmax in float32.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Protocol, Sequence

from sts_ai.action_order_diagnostic import (
    cyclic_index_maps,
    parse_teacher_menu,
)
from sts_ai.permutation_sft import (
    PAIRED_AUGMENTATION_KIND,
    PAIRED_AUGMENTATION_VERSION,
    SOURCE_MANIFEST_KIND,
    SOURCE_MANIFEST_VERSION,
    canonical_sha256,
)
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import (
    ACTION_ONLY_OUTPUT_CONTRACT,
    MlxCandidateScorer,
    validate_teacher_row,
)
from sts_ai.train.sft_format import tokenize_example


_TOKEN_COUNT_KEYS = (
    "n_prompt_tokens",
    "n_completion_tokens",
    "n_format_tokens",
    "n_thought_tokens",
    "n_action_tokens",
    "n_supervised_format_tokens",
    "n_supervised_thought_tokens",
    "n_supervised_action_tokens",
    "n_supervised_tokens",
)
_SHA256_KEYS = (
    "pair_id",
    "source_row_sha256",
    "source_public_state_hash",
    "source_prompt_sha256",
    "transformation_sha256",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_integer(
    value: Any,
    *,
    reason: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(reason)
    if minimum is not None and value < minimum:
        raise ValueError(reason)
    return value


def _require_sha256(value: Any, *, reason: str) -> str:
    if not _is_sha256(value):
        raise ValueError(reason)
    return str(value)


def _count_dict(counter: Counter[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items())
    }


def _nested_count_dict(
    counters: dict[int, Counter[int]],
) -> dict[str, dict[str, int]]:
    return {
        str(outer): _count_dict(counter)
        for outer, counter in sorted(counters.items())
    }


def _validate_token_counts(row: dict[str, Any], *, row_index: int) -> None:
    counts = row.get("token_counts")
    if not isinstance(counts, dict) or set(counts) != set(_TOKEN_COUNT_KEYS):
        raise ValueError(f"row_{row_index}_token_counts_shape")
    for key in _TOKEN_COUNT_KEYS:
        _strict_integer(
            counts.get(key),
            reason=f"row_{row_index}_token_counts_{key}",
            minimum=0,
        )
    if counts["n_completion_tokens"] != (
        counts["n_format_tokens"]
        + counts["n_thought_tokens"]
        + counts["n_action_tokens"]
    ):
        raise ValueError(f"row_{row_index}_completion_token_partition")
    if counts["n_supervised_tokens"] != (
        counts["n_supervised_format_tokens"]
        + counts["n_supervised_thought_tokens"]
        + counts["n_supervised_action_tokens"]
    ):
        raise ValueError(f"row_{row_index}_supervised_token_partition")
    if counts["n_supervised_thought_tokens"] != 0:
        raise ValueError(f"row_{row_index}_thought_tokens_are_supervised")
    if counts["n_supervised_format_tokens"] <= 0:
        raise ValueError(f"row_{row_index}_missing_supervised_format_tokens")
    if counts["n_supervised_action_tokens"] <= 0:
        raise ValueError(f"row_{row_index}_missing_supervised_action_tokens")


def _validate_manifest_core(
    manifest: dict[str, Any],
    *,
    n_rows: int,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest_not_object")
    expected = {
        "kind": SOURCE_MANIFEST_KIND,
        "version": SOURCE_MANIFEST_VERSION,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "loss_mask_mode": "action",
        "output_contract": ACTION_ONLY_OUTPUT_CONTRACT,
        "enable_thinking": False,
        "target_contains_native_thought": False,
        "n_examples": n_rows,
    }
    mismatches = {
        key: {"stored": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "manifest_contract_mismatch:"
            + json.dumps(mismatches, separators=(",", ":"), sort_keys=True)
        )
    tokenizer_id = manifest.get("tokenizer_id")
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("manifest_tokenizer_id")
    template_hash = manifest.get("chat_template_hash")
    if (
        not isinstance(template_hash, str)
        or len(template_hash) != 16
        or any(character not in "0123456789abcdef" for character in template_hash)
    ):
        raise ValueError("manifest_chat_template_hash")
    source_labels = manifest.get("source_labels")
    if not isinstance(source_labels, dict):
        raise ValueError("manifest_source_labels")
    for key in ("sha256", "manifest_sha256"):
        _require_sha256(
            source_labels.get(key),
            reason=f"manifest_source_labels_{key}",
        )
    source_dataset = manifest.get("source_dataset")
    if not isinstance(source_dataset, dict):
        raise ValueError("manifest_source_dataset")
    for key in ("dataset_sha256", "manifest_sha256"):
        _require_sha256(
            source_dataset.get(key),
            reason=f"manifest_source_dataset_{key}",
        )
    augmentation = manifest.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError("manifest_augmentation")
    if (
        augmentation.get("kind") != PAIRED_AUGMENTATION_KIND
        or augmentation.get("version") != PAIRED_AUGMENTATION_VERSION
    ):
        raise ValueError("manifest_augmentation_contract")
    for key in (
        "source_schedule_sha256",
        "ordered_step_source_sha256",
        "paired_row_ids_sha256",
        "paired_semantic_targets_sha256",
        "rotation_assignment_sha256",
    ):
        _require_sha256(
            augmentation.get(key),
            reason=f"manifest_augmentation_{key}",
        )
    return augmentation


def validate_exact_schedule(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete COMP-015 schedule and its derived hashes.

    This is intentionally stricter than generic teacher-row validation: a
    partial pass, reordered row, malformed permutation map, or stale aggregate
    in the paired manifest aborts the diagnostic.
    """

    if not rows:
        raise ValueError("dataset_empty")
    augmentation = _validate_manifest_core(manifest, n_rows=len(rows))
    n_source_rows = _strict_integer(
        augmentation.get("n_source_rows"),
        reason="manifest_n_source_rows",
        minimum=1,
    )
    repetitions = _strict_integer(
        augmentation.get("repetitions_per_source"),
        reason="manifest_repetitions_per_source",
        minimum=1,
    )
    if augmentation.get("n_output_rows") != len(rows):
        raise ValueError("manifest_n_output_rows")
    if len(rows) != n_source_rows * repetitions:
        raise ValueError("schedule_size_not_source_rows_times_repetitions")
    arm = augmentation.get("arm")
    if arm not in ("control_identity", "augmented_cyclic"):
        raise ValueError("manifest_augmentation_arm")

    source_dataset_sha256 = manifest["source_dataset"]["dataset_sha256"]
    ordered_step_source: list[dict[str, Any]] = []
    source_schedule: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    semantic_targets: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    pair_id_set: set[str] = set()
    source_identity_by_index: dict[int, str] = {}
    source_indices_by_pass: dict[int, set[int]] = defaultdict(set)
    source_identities_by_pass: dict[int, set[str]] = defaultdict(set)
    rotations_by_source: dict[str, Counter[int]] = defaultdict(Counter)
    menu_size_by_source: dict[str, int] = {}
    rotation_counts: Counter[int] = Counter()
    menu_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    rotations_by_menu: dict[int, Counter[int]] = defaultdict(Counter)
    targets_by_menu: dict[int, Counter[int]] = defaultdict(Counter)
    token_totals: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row_{row_index}_not_object")
        validate_teacher_row(row)
        parsed = parse_teacher_menu(row)
        if row.get("observation_version") != PUBLIC_OBSERVATION_VERSION:
            raise ValueError(f"row_{row_index}_observation_version")
        if row.get("teacher_selection_rule") != AGGREGATED_ROOT_VISITS:
            raise ValueError(f"row_{row_index}_teacher_selection_rule")
        if row.get("teacher_privilege") != TEACHER_PRIVILEGE:
            raise ValueError(f"row_{row_index}_teacher_privilege")
        _validate_token_counts(row, row_index=row_index)
        for key in _TOKEN_COUNT_KEYS:
            token_totals[key] += int(row["token_counts"][key])

        schedule_step = _strict_integer(
            row.get("schedule_step"),
            reason=f"row_{row_index}_schedule_step",
            minimum=0,
        )
        if schedule_step != row_index:
            raise ValueError(f"row_{row_index}_schedule_not_contiguous")
        source_identity = _require_sha256(
            row.get("source_identity"),
            reason=f"row_{row_index}_source_identity",
        )
        metadata = row.get("permutation_augmentation")
        if not isinstance(metadata, dict):
            raise ValueError(f"row_{row_index}_augmentation")
        expected_metadata = {
            "kind": PAIRED_AUGMENTATION_KIND,
            "version": PAIRED_AUGMENTATION_VERSION,
            "arm": arm,
            "source_dataset_sha256": source_dataset_sha256,
            "schedule_position": row_index,
        }
        if any(
            metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise ValueError(f"row_{row_index}_augmentation_contract")
        for key in _SHA256_KEYS:
            _require_sha256(
                metadata.get(key),
                reason=f"row_{row_index}_augmentation_{key}",
            )
        if metadata["source_public_state_hash"] != source_identity:
            raise ValueError(f"row_{row_index}_source_identity_mismatch")

        pass_index = _strict_integer(
            metadata.get("pass_index"),
            reason=f"row_{row_index}_pass_index",
            minimum=0,
        )
        repetition_index = _strict_integer(
            metadata.get("repetition_index"),
            reason=f"row_{row_index}_repetition_index",
            minimum=0,
        )
        expected_pass = row_index // n_source_rows
        if pass_index != expected_pass or repetition_index != expected_pass:
            raise ValueError(f"row_{row_index}_pass_order")
        source_row_index = _strict_integer(
            metadata.get("source_row_index"),
            reason=f"row_{row_index}_source_row_index",
            minimum=0,
        )
        if source_row_index >= n_source_rows:
            raise ValueError(f"row_{row_index}_source_row_index")
        if source_row_index in source_indices_by_pass[pass_index]:
            raise ValueError(f"row_{row_index}_duplicate_source_in_pass")
        if source_identity in source_identities_by_pass[pass_index]:
            raise ValueError(f"row_{row_index}_duplicate_source_identity_in_pass")
        source_indices_by_pass[pass_index].add(source_row_index)
        source_identities_by_pass[pass_index].add(source_identity)
        prior_identity = source_identity_by_index.setdefault(
            source_row_index,
            source_identity,
        )
        if prior_identity != source_identity:
            raise ValueError(f"row_{row_index}_source_index_identity_drift")

        menu_size = _strict_integer(
            metadata.get("menu_size"),
            reason=f"row_{row_index}_menu_size",
            minimum=1,
        )
        if menu_size != len(parsed.action_descriptions):
            raise ValueError(f"row_{row_index}_menu_size_mismatch")
        prior_menu_size = menu_size_by_source.setdefault(source_identity, menu_size)
        if prior_menu_size != menu_size:
            raise ValueError(f"row_{row_index}_source_menu_size_drift")
        rotation = _strict_integer(
            metadata.get("rotation"),
            reason=f"row_{row_index}_rotation",
            minimum=0,
        )
        if rotation >= menu_size:
            raise ValueError(f"row_{row_index}_rotation")
        expected_original_to_assigned, expected_assigned_to_original = (
            cyclic_index_maps(menu_size, rotation)
        )
        if metadata.get("original_to_assigned") != list(
            expected_original_to_assigned
        ):
            raise ValueError(f"row_{row_index}_original_to_assigned")
        if metadata.get("assigned_to_original") != list(
            expected_assigned_to_original
        ):
            raise ValueError(f"row_{row_index}_assigned_to_original")
        source_teacher_action_index = _strict_integer(
            metadata.get("source_teacher_action_index"),
            reason=f"row_{row_index}_source_teacher_action_index",
            minimum=0,
        )
        if source_teacher_action_index >= menu_size:
            raise ValueError(f"row_{row_index}_source_teacher_action_index")
        assigned_teacher_action_index = _strict_integer(
            metadata.get("assigned_teacher_action_index"),
            reason=f"row_{row_index}_assigned_teacher_action_index",
            minimum=0,
        )
        expected_assigned = expected_original_to_assigned[
            source_teacher_action_index
        ]
        if (
            assigned_teacher_action_index != expected_assigned
            or assigned_teacher_action_index != parsed.teacher_action_index
            or assigned_teacher_action_index != row.get("target_action_index")
            or assigned_teacher_action_index != row.get("teacher_action_index")
        ):
            raise ValueError(f"row_{row_index}_assigned_teacher_action_mismatch")
        if arm == "control_identity" and rotation != 0:
            raise ValueError(f"row_{row_index}_control_has_nonidentity_rotation")

        search_provenance = metadata.get("search_reference_provenance")
        if not isinstance(search_provenance, dict):
            raise ValueError(f"row_{row_index}_search_reference_provenance")
        if (
            search_provenance.get("kind")
            != "permutation_derived_index_and_public_hash_remap"
            or search_provenance.get("source_query_public_state_hash")
            != source_identity
            or search_provenance.get("assigned_public_state_hash")
            != row.get("public_state_hash")
            or search_provenance.get("teacher_search_reexecuted_for_assigned_order")
            is not False
        ):
            raise ValueError(f"row_{row_index}_search_reference_provenance")
        source_search_sha = _require_sha256(
            row.get("source_search_reference_sha256"),
            reason=f"row_{row_index}_source_search_reference_sha256",
        )
        if search_provenance.get("source_search_reference_sha256") != source_search_sha:
            raise ValueError(f"row_{row_index}_source_search_reference_mismatch")

        pair_id = str(metadata["pair_id"])
        if pair_id in pair_id_set:
            raise ValueError(f"row_{row_index}_duplicate_pair_id")
        pair_id_set.add(pair_id)
        pair_ids.append(pair_id)
        ordered_step_source.append(
            {
                "schedule_step": schedule_step,
                "source_identity": source_identity,
            }
        )
        source_schedule.append(
            {
                "schedule_position": row_index,
                "pair_id": pair_id,
                "source_row_index": source_row_index,
                "source_row_sha256": metadata["source_row_sha256"],
                "repetition_index": repetition_index,
                "pass_index": pass_index,
            }
        )
        semantic_targets.append(
            {
                "pair_id": pair_id,
                "source_public_state_hash": source_identity,
                "source_teacher_action_index": source_teacher_action_index,
                "semantic_teacher_action_description": metadata.get(
                    "semantic_teacher_action_description"
                ),
            }
        )
        rotations.append({"pair_id": pair_id, "rotation": rotation})
        rotations_by_source[source_identity][rotation] += 1
        rotation_counts[rotation] += 1
        menu_counts[menu_size] += 1
        target_counts[assigned_teacher_action_index] += 1
        rotations_by_menu[menu_size][rotation] += 1
        targets_by_menu[menu_size][assigned_teacher_action_index] += 1

    expected_source_indices = set(range(n_source_rows))
    expected_source_identities = set(source_identity_by_index.values())
    if set(source_indices_by_pass) != set(range(repetitions)):
        raise ValueError("schedule_pass_indices")
    for pass_index in range(repetitions):
        if source_indices_by_pass[pass_index] != expected_source_indices:
            raise ValueError(f"pass_{pass_index}_source_indices")
        if source_identities_by_pass[pass_index] != expected_source_identities:
            raise ValueError(f"pass_{pass_index}_source_identities")

    expected_hashes = {
        "source_schedule_sha256": canonical_sha256(source_schedule),
        "ordered_step_source_sha256": canonical_sha256(ordered_step_source),
        "paired_row_ids_sha256": canonical_sha256(pair_ids),
        "paired_semantic_targets_sha256": canonical_sha256(semantic_targets),
        "rotation_assignment_sha256": canonical_sha256(rotations),
    }
    for key, expected_hash in expected_hashes.items():
        if augmentation.get(key) != expected_hash:
            raise ValueError(f"manifest_{key}_mismatch")

    expected_counts: dict[str, Any] = {
        "rotation_counts": _count_dict(rotation_counts),
        "menu_size_counts": _count_dict(menu_counts),
        "assigned_teacher_action_counts": _count_dict(target_counts),
        "rotation_counts_by_menu_size": _nested_count_dict(rotations_by_menu),
        "assigned_teacher_action_counts_by_menu_size": _nested_count_dict(
            targets_by_menu
        ),
    }
    for key, expected_value in expected_counts.items():
        if augmentation.get(key) != expected_value:
            raise ValueError(f"manifest_{key}_mismatch")
    if manifest.get("teacher_action_counts") != expected_counts[
        "assigned_teacher_action_counts"
    ]:
        raise ValueError("manifest_teacher_action_counts_mismatch")

    identity_included = all(
        counts.get(0, 0) > 0
        for counts in rotations_by_source.values()
    )
    maximum_imbalance = max(
        (
            max(
                counts.get(rotation, 0)
                for rotation in range(menu_size_by_source[source_identity])
            )
            - min(
                counts.get(rotation, 0)
                for rotation in range(menu_size_by_source[source_identity])
            )
        )
        for source_identity, counts in rotations_by_source.items()
    )
    if augmentation.get("identity_included_for_every_source") is not identity_included:
        raise ValueError("manifest_identity_included_mismatch")
    # The paired manifest records the treatment schedule's balance bound on
    # both arms.  The identity control intentionally does not itself realize
    # that balance; it is aligned to the balanced treatment row-for-row.
    stored_maximum_imbalance = augmentation.get(
        "maximum_within_source_rotation_count_difference"
    )
    if arm == "augmented_cyclic":
        if stored_maximum_imbalance != maximum_imbalance:
            raise ValueError("manifest_rotation_balance_mismatch")
    elif stored_maximum_imbalance != 1:
        raise ValueError("manifest_rotation_balance_contract")

    expected_accounting = {
        "n_examples": len(rows),
        "n_examples_counted": len(rows),
        "totals": dict(token_totals),
    }
    if manifest.get("token_accounting") != expected_accounting:
        raise ValueError("manifest_token_accounting_mismatch")

    return {
        "arm": arm,
        "n_rows": len(rows),
        "n_source_rows": n_source_rows,
        "repetitions_per_source": repetitions,
        "schedule_seed": augmentation.get("schedule_seed"),
        "source_schedule_sha256": expected_hashes["source_schedule_sha256"],
        "ordered_step_source_sha256": expected_hashes[
            "ordered_step_source_sha256"
        ],
        "rotation_assignment_sha256": expected_hashes[
            "rotation_assignment_sha256"
        ],
        "token_count_totals": dict(token_totals),
    }


@dataclass(frozen=True)
class TeacherForcedLossScore:
    """Token losses for one exact stored target."""

    categories: tuple[str, ...]
    negative_log_probabilities: tuple[float, ...]
    n_input_tokens: int
    n_prompt_tokens: int
    n_completion_tokens: int
    computation_dtype: str


class TeacherForcedLossScorer(Protocol):
    def score_row(self, row: dict[str, Any]) -> TeacherForcedLossScore:
        """Score every supervised format/action target token in one row."""

    def scoring_key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        """Return the exact model/tokenizer inputs that determine ``score_row``.

        Implementations that expose this method opt into exact duplicate
        caching.  The report core uses the raw tuple as the dictionary key, so
        no digest collision can merge distinct scoring inputs.
        """


def _validate_runtime_tokenization(
    row: dict[str, Any],
    tokenized: dict[str, Any],
) -> tuple[list[int], list[int], list[str]]:
    stored_counts = row.get("token_counts")
    if not isinstance(stored_counts, dict):
        raise ValueError("stored_token_counts_missing")
    observed_counts = {
        key: int(tokenized.get(key, 0))
        for key in _TOKEN_COUNT_KEYS
    }
    if observed_counts != stored_counts:
        raise ValueError(
            "runtime_token_counts_mismatch:"
            + json.dumps(
                {"stored": stored_counts, "observed": observed_counts},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    input_ids = [int(value) for value in tokenized["input_ids"]]
    labels = [int(value) for value in tokenized["labels"]]
    if len(input_ids) != len(labels) or not input_ids:
        raise ValueError("runtime_tokenization_shape")
    categories: list[str] = []
    supervised_positions: list[int] = []
    for position, label in enumerate(labels):
        if label == -100:
            continue
        is_format = bool(tokenized["format_mask"][position])
        is_action = bool(tokenized["action_mask"][position])
        is_thought = bool(tokenized["thought_mask"][position])
        if (is_format + is_action + is_thought) != 1:
            raise ValueError("runtime_supervised_category_partition")
        if is_thought:
            raise ValueError("runtime_thought_token_is_supervised")
        categories.append("format" if is_format else "action")
        supervised_positions.append(position)
    if not supervised_positions:
        raise ValueError("runtime_no_supervised_tokens")
    if supervised_positions[0] <= 0:
        raise ValueError("runtime_supervised_first_token")
    if supervised_positions != list(
        range(supervised_positions[0], supervised_positions[-1] + 1)
    ):
        raise ValueError("runtime_supervised_tokens_not_contiguous")
    return input_ids, supervised_positions, categories


class MlxTeacherForcedLossScorer:
    """Exact MLX scorer with float32 selected-token log-softmax."""

    def __init__(self, model_id: str, *, adapter_path: str):
        self._loader = MlxCandidateScorer(
            model_id,
            adapter_path=adapter_path,
        )

    def runtime_tokenizer(self) -> Any:
        return self._loader._load()[1]

    def scoring_key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        """Key every input on which the fixed-checkpoint score can depend."""

        return (
            "mlx_teacher_forced_action_mask_v1",
            row.get("prompt"),
            row.get("completion"),
            row.get("target_action_index"),
            row.get("assistant_turn_terminator"),
        )

    def score_row(self, row: dict[str, Any]) -> TeacherForcedLossScore:
        import mlx.core as mx

        model, tokenizer = self._loader._load()
        tokenized = tokenize_example(row, tokenizer, loss_mask_mode="action")
        input_ids, positions, categories = _validate_runtime_tokenization(
            row,
            tokenized,
        )
        inputs = mx.array([input_ids[:-1]], dtype=mx.int32)
        logits = model(inputs)
        first_position = positions[0]
        last_position = positions[-1]
        supervised_logits = logits[
            0,
            first_position - 1 : last_position,
            :,
        ].astype(mx.float32)
        target_ids = mx.array(
            input_ids[first_position : last_position + 1],
            dtype=mx.int32,
        )
        selected_logits = mx.take_along_axis(
            supervised_logits,
            target_ids[:, None],
            axis=-1,
        ).squeeze(-1)
        negative_log_probabilities = (
            mx.logsumexp(supervised_logits, axis=-1) - selected_logits
        ).astype(mx.float32)
        mx.eval(negative_log_probabilities)
        values = tuple(
            float(value)
            for value in negative_log_probabilities.tolist()
        )
        result = TeacherForcedLossScore(
            categories=tuple(categories),
            negative_log_probabilities=values,
            n_input_tokens=len(input_ids),
            n_prompt_tokens=int(tokenized["n_prompt_tokens"]),
            n_completion_tokens=int(tokenized["n_completion_tokens"]),
            computation_dtype="float32",
        )
        del (
            inputs,
            logits,
            supervised_logits,
            target_ids,
            selected_logits,
            negative_log_probabilities,
        )
        mx.clear_cache()
        return result

    def clear(self) -> None:
        self._loader._model = None
        self._loader._tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


def _finite_nonnegative(value: Any, *, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(reason)
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(reason)
    return result


def _row_loss_record(
    row: dict[str, Any],
    *,
    row_index: int,
    score: TeacherForcedLossScore,
) -> dict[str, Any]:
    if not isinstance(score, TeacherForcedLossScore):
        raise ValueError(f"row_{row_index}_score_type")
    if (
        len(score.categories) != len(score.negative_log_probabilities)
        or not score.categories
    ):
        raise ValueError(f"row_{row_index}_score_shape")
    if any(category not in ("format", "action") for category in score.categories):
        raise ValueError(f"row_{row_index}_score_category")
    if score.computation_dtype != "float32":
        raise ValueError(f"row_{row_index}_score_not_float32")
    losses = [
        _finite_nonnegative(
            value,
            reason=f"row_{row_index}_negative_log_probability",
        )
        for value in score.negative_log_probabilities
    ]
    expected_format = int(row["token_counts"]["n_supervised_format_tokens"])
    expected_action = int(row["token_counts"]["n_supervised_action_tokens"])
    if score.categories.count("format") != expected_format:
        raise ValueError(f"row_{row_index}_format_token_count")
    if score.categories.count("action") != expected_action:
        raise ValueError(f"row_{row_index}_action_token_count")
    n_input_tokens = _strict_integer(
        score.n_input_tokens,
        reason=f"row_{row_index}_score_n_input_tokens",
        minimum=1,
    )
    n_prompt_tokens = _strict_integer(
        score.n_prompt_tokens,
        reason=f"row_{row_index}_score_n_prompt_tokens",
        minimum=1,
    )
    n_completion_tokens = _strict_integer(
        score.n_completion_tokens,
        reason=f"row_{row_index}_score_n_completion_tokens",
        minimum=1,
    )
    expected_input = (
        int(row["token_counts"]["n_prompt_tokens"])
        + int(row["token_counts"]["n_completion_tokens"])
    )
    if (
        n_input_tokens != expected_input
        or n_prompt_tokens != row["token_counts"]["n_prompt_tokens"]
        or n_completion_tokens != row["token_counts"]["n_completion_tokens"]
    ):
        raise ValueError(f"row_{row_index}_score_tokenization_shape")
    metadata = row["permutation_augmentation"]
    format_losses = [
        loss
        for category, loss in zip(score.categories, losses)
        if category == "format"
    ]
    action_losses = [
        loss
        for category, loss in zip(score.categories, losses)
        if category == "action"
    ]
    return {
        "row_index": row_index,
        "schedule_step": row["schedule_step"],
        "pass_index": metadata["pass_index"],
        "source_identity": row["source_identity"],
        "pair_id": metadata["pair_id"],
        "window_id": row.get("window_id"),
        "world_seed": row.get("world_seed"),
        "decision_index": row.get("decision_index"),
        "public_state_hash": row.get("public_state_hash"),
        "prompt_sha256": hashlib.sha256(
            str(row["prompt"]).encode("utf-8")
        ).hexdigest(),
        "completion_sha256": hashlib.sha256(
            str(row["completion"]).encode("utf-8")
        ).hexdigest(),
        "menu_size": metadata["menu_size"],
        "rotation": metadata["rotation"],
        "assigned_teacher_action_index": metadata[
            "assigned_teacher_action_index"
        ],
        "n_input_tokens": n_input_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_completion_tokens": n_completion_tokens,
        "n_supervised_tokens": len(losses),
        "n_format_tokens": len(format_losses),
        "n_action_tokens": len(action_losses),
        "total_nll": math.fsum(losses),
        "format_nll": math.fsum(format_losses),
        "action_nll": math.fsum(action_losses),
        "mean_format_token_nll": (
            math.fsum(format_losses) / len(format_losses)
        ),
        "mean_action_token_nll": (
            math.fsum(action_losses) / len(action_losses)
        ),
        "computation_dtype": score.computation_dtype,
    }


def _perplexity(mean_nll: float | None) -> float | None:
    maximum_float_log = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
    if mean_nll is None or mean_nll > maximum_float_log:
        return None
    return math.exp(mean_nll)


def _component_summary(
    rows: Sequence[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    n_tokens = sum(int(row[f"n_{prefix}_tokens"]) for row in rows)
    total_nll = math.fsum(float(row[f"{prefix}_nll"]) for row in rows)
    mean_nll = total_nll / n_tokens if n_tokens else None
    row_means = [
        float(row[f"mean_{prefix}_token_nll"])
        for row in rows
        if int(row[f"n_{prefix}_tokens"]) > 0
    ]
    return {
        "n_tokens": n_tokens,
        "total_nll": total_nll,
        "mean_token_nll": mean_nll,
        "token_perplexity": _perplexity(mean_nll),
        "mean_row_token_nll": (
            math.fsum(row_means) / len(row_means)
            if row_means
            else None
        ),
    }


def _loss_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    format_summary = _component_summary(rows, prefix="format")
    action_summary = _component_summary(rows, prefix="action")
    n_tokens = format_summary["n_tokens"] + action_summary["n_tokens"]
    total_nll = format_summary["total_nll"] + action_summary["total_nll"]
    mean_nll = total_nll / n_tokens if n_tokens else None
    return {
        "n_rows": len(rows),
        "n_supervised_tokens": n_tokens,
        "total_nll": total_nll,
        "mean_supervised_token_nll": mean_nll,
        "supervised_token_perplexity": _perplexity(mean_nll),
        "mean_row_total_nll": (
            math.fsum(float(row["total_nll"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "format": format_summary,
        "action": action_summary,
        "action_fraction_of_total_nll": (
            action_summary["total_nll"] / total_nll
            if total_nll > 0.0
            else None
        ),
        "action_minus_format_mean_token_nll": (
            action_summary["mean_token_nll"] - format_summary["mean_token_nll"]
            if (
                action_summary["mean_token_nll"] is not None
                and format_summary["mean_token_nll"] is not None
            )
            else None
        ),
    }


def _grouped_summary(
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[key])].append(row)
    return {
        str(value): _loss_summary(group)
        for value, group in sorted(grouped.items())
    }


def build_loss_decomposition_report(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    scorer: TeacherForcedLossScorer,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score and aggregate every exact-schedule row without skip semantics."""

    schedule = validate_exact_schedule(rows, manifest)
    scoring_key = getattr(scorer, "scoring_key", None)
    cache_enabled = callable(scoring_key)
    cached_scores: dict[tuple[Any, ...], TeacherForcedLossScore] = {}
    n_model_forwards = 0
    n_cache_hits = 0
    scored_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if cache_enabled:
            key = scoring_key(row)
            if not isinstance(key, tuple):
                raise ValueError(f"row_{row_index}_scoring_key_not_tuple")
            try:
                hash(key)
            except TypeError as exc:
                raise ValueError(
                    f"row_{row_index}_scoring_key_not_hashable"
                ) from exc
            if key in cached_scores:
                score = cached_scores[key]
                n_cache_hits += 1
            else:
                score = scorer.score_row(row)
                cached_scores[key] = score
                n_model_forwards += 1
        else:
            score = scorer.score_row(row)
            n_model_forwards += 1
        scored_rows.append(
            _row_loss_record(
                row,
                row_index=row_index,
                score=score,
            )
        )
    dtypes = sorted({str(row["computation_dtype"]) for row in scored_rows})
    return {
        "kind": "comp015_teacher_forced_sft_loss_decomposition",
        "version": 1,
        "loss_definition": (
            "teacher-forced causal cross-entropy on the exact stored compact "
            "action completion plus assistant-turn terminator; prompt tokens are "
            "context only; the frozen mixed mask partitions every supervised "
            "target token into required-format or action-value"
        ),
        "aggregation_definition": (
            "token-weighted means use math.fsum over per-token NLL values; "
            "mean_row_token_nll is an unweighted row macro-average; by-pass "
            "statistics rescore the fixed checkpoint on rows assigned to each "
            "training pass and are not historical online training losses"
        ),
        "candidate_normalization": False,
        "outside_candidate_mass_included": True,
        "schedule": schedule,
        "n_input_rows": len(rows),
        "n_scored_rows": len(scored_rows),
        "n_skipped_rows": 0,
        "score_cache": {
            "enabled": cache_enabled,
            "key_contract": (
                "exact raw scorer-declared model/tokenizer input tuple; no "
                "digest-based equality"
                if cache_enabled
                else None
            ),
            "n_unique_scoring_inputs": (
                len(cached_scores) if cache_enabled else len(scored_rows)
            ),
            "n_model_forwards": n_model_forwards,
            "n_cache_hits": n_cache_hits,
        },
        "loss_computation_dtypes": dtypes,
        "overall": _loss_summary(scored_rows),
        "by_pass": _grouped_summary(scored_rows, key="pass_index"),
        "by_assigned_position": _grouped_summary(
            scored_rows,
            key="assigned_teacher_action_index",
        ),
        "by_menu_size": _grouped_summary(scored_rows, key="menu_size"),
        "rows": scored_rows,
        "provenance": dict(provenance or {}),
    }


__all__ = [
    "MlxTeacherForcedLossScorer",
    "TeacherForcedLossScore",
    "TeacherForcedLossScorer",
    "build_loss_decomposition_report",
    "validate_exact_schedule",
]
