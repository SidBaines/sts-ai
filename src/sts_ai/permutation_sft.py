"""Deterministic paired identity/rotation datasets for strict teacher SFT.

The source dataset remains immutable.  Each source row is repeated exactly
``repetitions`` times in a content-addressed schedule shared by both arms.  The
control keeps the original legal-action order; the augmented arm applies a
balanced deterministic schedule of cyclic rotations and remaps every
policy-visible or provenance action index.

This module performs no filesystem or model loading.  Callers supply an exact
tokenizer and publish the returned rows/manifests themselves.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

from sts_ai.action_order_diagnostic import (
    ParsedTeacherMenu,
    PromptRenderer,
    parse_teacher_menu,
    rotate_teacher_row,
    validate_prompt_round_trip,
)
from sts_ai.prompting import ACTION_ONLY_OUTPUT
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
)
from sts_ai.teacher_action_eval import action_completion, validate_teacher_row
from sts_ai.train.sft_format import (
    assistant_turn_terminator,
    chat_template_probe_hash,
    loss_mask_token_accounting,
    tokenize_example,
)


PAIRED_AUGMENTATION_KIND = "paired_cyclic_action_order_sft"
PAIRED_AUGMENTATION_VERSION = 1
SOURCE_MANIFEST_KIND = "search_teacher_sft"
SOURCE_MANIFEST_VERSION = 3
_LEGAL_ACTIONS_MARKER = "\nLEGAL ACTIONS\n"
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
_PAIRED_TOKEN_COUNT_KEYS = (
    "n_prompt_tokens",
    "n_completion_tokens",
    "n_total_tokens",
    "n_action_tokens",
    "n_supervised_format_tokens",
    "n_supervised_action_tokens",
    "n_supervised_tokens",
)


@dataclass(frozen=True)
class PairedDatasetBuild:
    """Rows and manifest cores for the two aligned training arms."""

    control_rows: list[dict[str, Any]]
    augmented_rows: list[dict[str, Any]]
    control_manifest: dict[str, Any]
    augmented_manifest: dict[str, Any]


@dataclass(frozen=True)
class _ScheduleItem:
    source_row_index: int
    repetition_index: int
    pair_id: str
    source_row_sha256: str


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value under one explicit canonical encoding."""

    rendered = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def render_jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    """Render deterministic compact JSONL bytes for content addressing."""

    return (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def finalize_manifest(
    manifest: dict[str, Any],
    dataset_bytes: bytes,
) -> dict[str, Any]:
    """Attach the exact serialized dataset digest to a manifest core."""

    finalized = deepcopy(manifest)
    finalized["dataset_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
    return finalized


def render_prompt(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    """Render one generation prompt with thinking explicitly disabled."""

    try:
        result = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        result = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(result, str) or not result:
        raise ValueError("runtime_chat_template_prompt_invalid")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(reason)
    return value


def _token_counts(example: dict[str, Any], tokenizer: Any) -> dict[str, int]:
    tokenized = tokenize_example(example, tokenizer, loss_mask_mode="action")
    return {key: int(tokenized[key]) for key in _TOKEN_COUNT_KEYS}


def _validate_source_manifest(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    model_id: str,
    source_dataset_sha256: str,
    source_manifest_sha256: str,
    tokenizer: Any,
) -> None:
    if not _is_sha256(source_dataset_sha256):
        raise ValueError("source_dataset_sha256_invalid")
    if not _is_sha256(source_manifest_sha256):
        raise ValueError("source_manifest_sha256_invalid")
    if not isinstance(manifest, dict):
        raise ValueError("source_manifest_not_object")
    expected = {
        "kind": SOURCE_MANIFEST_KIND,
        "version": SOURCE_MANIFEST_VERSION,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "loss_mask_mode": "action",
        "output_contract": ACTION_ONLY_OUTPUT,
        "enable_thinking": False,
        "tokenizer_id": model_id,
        "dataset_sha256": source_dataset_sha256,
        "n_examples": len(rows),
    }
    mismatches = {
        key: {"stored": manifest.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            "source_manifest_contract_mismatch:"
            + json.dumps(mismatches, separators=(",", ":"), sort_keys=True)
        )
    expected_probe = chat_template_probe_hash(tokenizer, enable_thinking=False)
    if manifest.get("chat_template_hash") != expected_probe:
        raise ValueError("runtime_chat_template_hash_mismatch")
    source_labels = manifest.get("source_labels")
    if not isinstance(source_labels, dict):
        raise ValueError("source_manifest_missing_source_labels")
    for key in ("sha256", "manifest_sha256"):
        if not _is_sha256(source_labels.get(key)):
            raise ValueError(f"source_manifest_source_labels_{key}_invalid")
    if manifest.get("augmentation") is not None:
        raise ValueError("source_manifest_already_augmented")


def _validate_source_rows(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    prompt_renderer: PromptRenderer,
) -> list[ParsedTeacherMenu]:
    if not rows:
        raise ValueError("source_dataset_empty")
    parsed_rows: list[ParsedTeacherMenu] = []
    public_hashes: set[str] = set()
    expected_terminator = assistant_turn_terminator(tokenizer)
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"source_row_{row_index}_not_object")
        if "permutation_augmentation" in row:
            raise ValueError(f"source_row_{row_index}_already_augmented")
        expected_fields = {
            "observation_version": PUBLIC_OBSERVATION_VERSION,
            "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
            "teacher_privilege": TEACHER_PRIVILEGE,
            "loss_mask_mode": "action",
            "output_contract": ACTION_ONLY_OUTPUT,
        }
        for key, expected_value in expected_fields.items():
            if row.get(key) != expected_value:
                raise ValueError(f"source_row_{row_index}_{key}_mismatch")
        try:
            validate_teacher_row(row)
            parsed = parse_teacher_menu(row)
            validate_prompt_round_trip(row, prompt_renderer)
        except ValueError as exc:
            raise ValueError(f"source_row_{row_index}_invalid:{exc}") from exc
        if parsed.source_public_state_hash in public_hashes:
            raise ValueError("source_public_state_hash_not_unique")
        public_hashes.add(parsed.source_public_state_hash)
        if row.get("assistant_turn_terminator") != expected_terminator:
            raise ValueError(
                f"source_row_{row_index}_assistant_turn_terminator_mismatch"
            )
        stored_token_counts = row.get("token_counts")
        if not isinstance(stored_token_counts, dict):
            raise ValueError(f"source_row_{row_index}_missing_token_counts")
        recomputed = _token_counts(row, tokenizer)
        if {
            key: stored_token_counts.get(key)
            for key in _TOKEN_COUNT_KEYS
        } != recomputed:
            raise ValueError(f"source_row_{row_index}_token_counts_mismatch")
        parsed_rows.append(parsed)
    return parsed_rows


def _remap_index(
    value: Any,
    original_to_assigned: Sequence[int],
    *,
    reason: str,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    index = _integer(value, reason)
    if index >= len(original_to_assigned):
        raise ValueError(reason)
    return int(original_to_assigned[index])


def _remap_action_counts(
    value: Any,
    original_to_assigned: Sequence[int],
    *,
    path: str,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}_not_object")
    remapped: dict[str, int] = {}
    for raw_index, raw_count in value.items():
        if not isinstance(raw_index, str):
            raise ValueError(f"{path}_index_not_string")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"{path}_index_invalid") from exc
        if str(index) != raw_index or index < 0 or index >= len(original_to_assigned):
            raise ValueError(f"{path}_index_invalid")
        count = _integer(raw_count, f"{path}_count_invalid")
        assigned = str(original_to_assigned[index])
        if assigned in remapped:
            raise ValueError(f"{path}_index_collision")
        remapped[assigned] = count
    return remapped


def _remap_reference_value(
    value: Any,
    original_to_assigned: Sequence[int],
    *,
    source_public_state_hash: str,
    assigned_public_state_hash: str,
    path: str,
) -> Any:
    if isinstance(value, dict):
        remapped: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "action_counts":
                remapped[key] = _remap_action_counts(
                    child,
                    original_to_assigned,
                    path=child_path,
                )
            elif key.endswith("action_index"):
                remapped[key] = _remap_index(
                    child,
                    original_to_assigned,
                    reason=f"{child_path}_invalid",
                    allow_none=True,
                )
            elif key == "public_state_hash":
                if child != source_public_state_hash:
                    raise ValueError(f"{child_path}_source_hash_mismatch")
                remapped[key] = assigned_public_state_hash
            else:
                remapped[key] = _remap_reference_value(
                    child,
                    original_to_assigned,
                    source_public_state_hash=source_public_state_hash,
                    assigned_public_state_hash=assigned_public_state_hash,
                    path=child_path,
                )
        return remapped
    if isinstance(value, list):
        return [
            _remap_reference_value(
                child,
                original_to_assigned,
                source_public_state_hash=source_public_state_hash,
                assigned_public_state_hash=assigned_public_state_hash,
                path=f"{path}[]",
            )
            for child in value
        ]
    return deepcopy(value)


def _rotated_user_content(
    parsed: ParsedTeacherMenu,
    assigned_descriptions: Sequence[str],
) -> str:
    action_block = "\n".join(
        f"{index}: {description}"
        for index, description in enumerate(assigned_descriptions)
    )
    return parsed.user_prefix + _LEGAL_ACTIONS_MARKER + action_block + "\n"


def _transform_row(
    row: dict[str, Any],
    parsed: ParsedTeacherMenu,
    *,
    rotation: int,
    arm: str,
    schedule_position: int,
    schedule_item: _ScheduleItem,
    source_dataset_sha256: str,
    tokenizer: Any,
    prompt_renderer: PromptRenderer,
) -> dict[str, Any]:
    rotated = rotate_teacher_row(
        row,
        rotation,
        prompt_renderer=prompt_renderer,
    )
    transformed = deepcopy(row)
    assigned_teacher_index = rotated.assigned_teacher_action_index
    assigned_completion = action_completion(assigned_teacher_index)
    assigned_user_content = _rotated_user_content(
        parsed,
        rotated.assigned_action_descriptions,
    )
    transformed.update(
        {
            "prompt": rotated.scoring_row["prompt"],
            "completion": assigned_completion,
            "public_state_hash": rotated.rotated_public_state_hash,
            "target_action_index": assigned_teacher_index,
            "teacher_action_index": assigned_teacher_index,
            "assistant_turn_terminator": assistant_turn_terminator(tokenizer),
        }
    )
    messages = transformed.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("messages_not_exact_user_assistant_pair")
    transformed["messages"] = [
        {"role": "user", "content": assigned_user_content},
        {"role": "assistant", "content": assigned_completion},
    ]
    transformed["base_action_index"] = _remap_index(
        row.get("base_action_index"),
        rotated.original_to_assigned,
        reason="base_action_index_invalid",
    )
    source_reference = row.get("search_reference")
    if not isinstance(source_reference, dict):
        raise ValueError("search_reference_not_object")
    transformed["search_reference"] = _remap_reference_value(
        source_reference,
        rotated.original_to_assigned,
        source_public_state_hash=parsed.source_public_state_hash,
        assigned_public_state_hash=rotated.rotated_public_state_hash,
        path="search_reference",
    )
    source_search_reference_sha256 = canonical_sha256(source_reference)
    teacher_description = parsed.action_descriptions[parsed.teacher_action_index]
    if (
        rotated.assigned_action_descriptions[assigned_teacher_index]
        != teacher_description
    ):
        raise RuntimeError("semantic_teacher_action_not_preserved")
    transformed["permutation_augmentation"] = {
        "kind": PAIRED_AUGMENTATION_KIND,
        "version": PAIRED_AUGMENTATION_VERSION,
        "arm": arm,
        "source_dataset_sha256": source_dataset_sha256,
        "source_row_index": schedule_item.source_row_index,
        "source_row_sha256": schedule_item.source_row_sha256,
        "source_public_state_hash": parsed.source_public_state_hash,
        "source_prompt_sha256": hashlib.sha256(
            parsed.prompt.encode("utf-8")
        ).hexdigest(),
        "source_teacher_action_index": parsed.teacher_action_index,
        "semantic_teacher_action_description": teacher_description,
        "repetition_index": schedule_item.repetition_index,
        "pass_index": schedule_item.repetition_index,
        "schedule_position": schedule_position,
        "pair_id": schedule_item.pair_id,
        "rotation": rotation,
        "menu_size": rotated.menu_size,
        "original_to_assigned": list(rotated.original_to_assigned),
        "assigned_to_original": list(rotated.assigned_to_original),
        "assigned_teacher_action_index": assigned_teacher_index,
        "transformation_sha256": rotated.transformation_sha256,
        "search_reference_provenance": {
            "kind": "permutation_derived_index_and_public_hash_remap",
            "source_search_reference_sha256": source_search_reference_sha256,
            "source_query_public_state_hash": parsed.source_public_state_hash,
            "assigned_public_state_hash": rotated.rotated_public_state_hash,
            "teacher_search_reexecuted_for_assigned_order": False,
        },
    }
    transformed["source_search_reference_sha256"] = (
        source_search_reference_sha256
    )
    transformed["schedule_step"] = schedule_position
    transformed["source_identity"] = parsed.source_public_state_hash
    transformed["token_counts"] = _token_counts(transformed, tokenizer)
    validate_teacher_row(transformed)
    reparsed = parse_teacher_menu(transformed)
    if (
        reparsed.action_descriptions != rotated.assigned_action_descriptions
        or reparsed.teacher_action_index != assigned_teacher_index
    ):
        raise RuntimeError("transformed_row_round_trip_mismatch")
    return transformed


def _source_schedule(
    rows: Sequence[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    source_dataset_sha256: str,
) -> list[_ScheduleItem]:
    schedule: list[_ScheduleItem] = []
    pair_ids: set[str] = set()
    source_row_hashes = [canonical_sha256(row) for row in rows]
    for repetition_index in range(repetitions):
        pass_rows: list[tuple[str, _ScheduleItem]] = []
        pass_sort_keys: set[str] = set()
        for source_row_index, source_row_sha256 in enumerate(source_row_hashes):
            identity = {
                "contract": PAIRED_AUGMENTATION_KIND,
                "version": PAIRED_AUGMENTATION_VERSION,
                "source_dataset_sha256": source_dataset_sha256,
                "source_row_index": source_row_index,
                "source_row_sha256": source_row_sha256,
                "repetition_index": repetition_index,
            }
            pair_id = canonical_sha256(identity)
            if pair_id in pair_ids:
                raise RuntimeError("pair_id_collision")
            pair_ids.add(pair_id)
            sort_key = canonical_sha256(
                {
                    "pair_id": pair_id,
                    "pass_index": repetition_index,
                    "schedule_seed": seed,
                    "source_dataset_sha256": source_dataset_sha256,
                }
            )
            if sort_key in pass_sort_keys:
                raise RuntimeError("schedule_sort_key_collision")
            pass_sort_keys.add(sort_key)
            pass_rows.append(
                (
                    sort_key,
                    _ScheduleItem(
                        source_row_index=source_row_index,
                        repetition_index=repetition_index,
                        pair_id=pair_id,
                        source_row_sha256=source_row_sha256,
                    ),
                )
            )
        schedule.extend(
            item for _, item in sorted(pass_rows, key=lambda value: value[0])
        )
    return schedule


def _rotation_order(
    *,
    menu_size: int,
    seed: int,
    source_dataset_sha256: str,
    source_row_sha256: str,
) -> list[int]:
    if menu_size <= 0:
        raise ValueError("menu_size_must_be_positive")
    nonidentity = list(range(1, menu_size))
    nonidentity.sort(
        key=lambda rotation: canonical_sha256(
            {
                "rotation": rotation,
                "schedule_seed": seed,
                "source_dataset_sha256": source_dataset_sha256,
                "source_row_sha256": source_row_sha256,
            }
        )
    )
    return [0, *nonidentity]


def _schedule_provenance(
    schedule: Sequence[_ScheduleItem],
    parsed_rows: Sequence[ParsedTeacherMenu],
) -> tuple[str, str, str]:
    source_schedule = [
        {
            "schedule_position": position,
            "pair_id": item.pair_id,
            "source_row_index": item.source_row_index,
            "source_row_sha256": item.source_row_sha256,
            "repetition_index": item.repetition_index,
            "pass_index": item.repetition_index,
        }
        for position, item in enumerate(schedule)
    ]
    pair_ids = [item.pair_id for item in schedule]
    semantic_targets = [
        {
            "pair_id": item.pair_id,
            "source_public_state_hash": (
                parsed_rows[item.source_row_index].source_public_state_hash
            ),
            "source_teacher_action_index": (
                parsed_rows[item.source_row_index].teacher_action_index
            ),
            "semantic_teacher_action_description": (
                parsed_rows[item.source_row_index].action_descriptions[
                    parsed_rows[item.source_row_index].teacher_action_index
                ]
            ),
        }
        for item in schedule
    ]
    return (
        canonical_sha256(source_schedule),
        canonical_sha256(pair_ids),
        canonical_sha256(semantic_targets),
    )


def _arm_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rotation_counts: Counter[str] = Counter()
    menu_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    rotations_by_menu: dict[str, Counter[str]] = defaultdict(Counter)
    targets_by_menu: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        metadata = row["permutation_augmentation"]
        rotation = str(metadata["rotation"])
        menu_size = str(metadata["menu_size"])
        target = str(row["teacher_action_index"])
        rotation_counts[rotation] += 1
        menu_counts[menu_size] += 1
        target_counts[target] += 1
        rotations_by_menu[menu_size][rotation] += 1
        targets_by_menu[menu_size][target] += 1
    return {
        "rotation_counts": dict(sorted(rotation_counts.items())),
        "menu_size_counts": dict(sorted(menu_counts.items())),
        "assigned_teacher_action_counts": dict(sorted(target_counts.items())),
        "rotation_counts_by_menu_size": {
            menu_size: dict(sorted(counts.items()))
            for menu_size, counts in sorted(rotations_by_menu.items())
        },
        "assigned_teacher_action_counts_by_menu_size": {
            menu_size: dict(sorted(counts.items()))
            for menu_size, counts in sorted(targets_by_menu.items())
        },
    }


def _paired_token_report(
    control_rows: Sequence[dict[str, Any]],
    augmented_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mismatch_counts: Counter[str] = Counter()
    padded_length_mismatches = 0
    mismatched_pair_ids: set[str] = set()
    for control, augmented in zip(control_rows, augmented_rows):
        control_meta = control["permutation_augmentation"]
        augmented_meta = augmented["permutation_augmentation"]
        if control_meta["pair_id"] != augmented_meta["pair_id"]:
            raise RuntimeError("paired_row_alignment_mismatch")
        for key in _PAIRED_TOKEN_COUNT_KEYS:
            if key == "n_total_tokens":
                control_value = (
                    control["token_counts"]["n_prompt_tokens"]
                    + control["token_counts"]["n_completion_tokens"]
                )
                augmented_value = (
                    augmented["token_counts"]["n_prompt_tokens"]
                    + augmented["token_counts"]["n_completion_tokens"]
                )
            else:
                control_value = control["token_counts"][key]
                augmented_value = augmented["token_counts"][key]
            if control_value != augmented_value:
                mismatch_counts[key] += 1
                mismatched_pair_ids.add(str(control_meta["pair_id"]))
        control_total = (
            control["token_counts"]["n_prompt_tokens"]
            + control["token_counts"]["n_completion_tokens"]
        )
        augmented_total = (
            augmented["token_counts"]["n_prompt_tokens"]
            + augmented["token_counts"]["n_completion_tokens"]
        )
        control_padded = 1 + 32 * ((control_total + 31) // 32)
        augmented_padded = 1 + 32 * ((augmented_total + 31) // 32)
        if control_padded != augmented_padded:
            padded_length_mismatches += 1
            mismatched_pair_ids.add(str(control_meta["pair_id"]))
    return {
        "compared_token_count_keys": list(_PAIRED_TOKEN_COUNT_KEYS),
        "n_pairs": len(control_rows),
        "n_pairs_with_any_mismatch": len(mismatched_pair_ids),
        "all_paired_token_counts_match": not mismatched_pair_ids,
        "batch1_padding_rule": "1 + 32 * ceil(n_total_tokens / 32)",
        "n_pairs_with_batch1_padded_length_mismatch": padded_length_mismatches,
        "all_batch1_padded_lengths_match": padded_length_mismatches == 0,
        "mismatch_counts_by_key": dict(sorted(mismatch_counts.items())),
        "mismatched_pair_ids_sha256": canonical_sha256(
            sorted(mismatched_pair_ids)
        ),
    }


def _build_manifest(
    *,
    arm: str,
    rows: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    source_dataset_sha256: str,
    source_manifest_sha256: str,
    repetitions: int,
    seed: int,
    source_schedule_sha256: str,
    pair_ids_sha256: str,
    semantic_targets_sha256: str,
    paired_token_report: dict[str, Any],
    require_paired_token_counts: bool,
    tokenizer: Any,
) -> dict[str, Any]:
    rotations = [
        {
            "pair_id": row["permutation_augmentation"]["pair_id"],
            "rotation": row["permutation_augmentation"]["rotation"],
        }
        for row in rows
    ]
    ordered_step_source = [
        {
            "schedule_step": row["schedule_step"],
            "source_identity": row["source_identity"],
        }
        for row in rows
    ]
    counts = _arm_counts(rows)
    return {
        "kind": SOURCE_MANIFEST_KIND,
        "version": SOURCE_MANIFEST_VERSION,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": AGGREGATED_ROOT_VISITS,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "loss_mask_mode": "action",
        "output_contract": ACTION_ONLY_OUTPUT,
        "target_contains_native_thought": False,
        "enable_thinking": False,
        "tokenizer_id": source_manifest["tokenizer_id"],
        "chat_template_hash": chat_template_probe_hash(
            tokenizer,
            enable_thinking=False,
        ),
        "n_input_rows": source_manifest["n_examples"],
        "n_source_examples": source_manifest["n_examples"],
        "n_examples": len(rows),
        "source_labels": deepcopy(source_manifest["source_labels"]),
        "source_dataset": {
            "kind": source_manifest["kind"],
            "version": source_manifest["version"],
            "dataset_sha256": source_dataset_sha256,
            "manifest_sha256": source_manifest_sha256,
            "n_examples": source_manifest["n_examples"],
            "tokenizer_id": source_manifest["tokenizer_id"],
            "chat_template_hash": source_manifest["chat_template_hash"],
        },
        "teacher_action_counts": counts["assigned_teacher_action_counts"],
        "token_accounting": loss_mask_token_accounting(rows),
        "augmentation": {
            "kind": PAIRED_AUGMENTATION_KIND,
            "version": PAIRED_AUGMENTATION_VERSION,
            "arm": arm,
            "schedule_seed": seed,
            "repetitions_per_source": repetitions,
            "n_source_rows": source_manifest["n_examples"],
            "n_output_rows": len(rows),
            "source_schedule_algorithm": (
                "explicit_passes_sha256_ranked_without_replacement_v1"
            ),
            "rotation_schedule_algorithm": (
                "identity_first_sha256_ranked_cyclic_rotations_modulo_menu_v1"
            ),
            "identity_included_for_every_source": True,
            "maximum_within_source_rotation_count_difference": 1,
            "source_schedule_sha256": source_schedule_sha256,
            "ordered_step_source_sha256": canonical_sha256(
                ordered_step_source
            ),
            "paired_row_ids_sha256": pair_ids_sha256,
            "paired_semantic_targets_sha256": semantic_targets_sha256,
            "rotation_assignment_sha256": canonical_sha256(rotations),
            "require_paired_token_counts": require_paired_token_counts,
            "paired_token_counts": deepcopy(paired_token_report),
            **counts,
        },
    }


def build_paired_datasets(
    rows: Sequence[dict[str, Any]],
    source_manifest: dict[str, Any],
    *,
    tokenizer: Any,
    model_id: str,
    repetitions: int,
    seed: int,
    source_dataset_sha256: str,
    source_manifest_sha256: str,
    require_paired_token_counts: bool = False,
) -> PairedDatasetBuild:
    """Build aligned identity and balanced cyclic-rotation teacher datasets."""

    repetitions = _integer(
        repetitions,
        "repetitions_must_be_positive",
        minimum=1,
    )
    seed = _integer(seed, "seed_must_be_non_negative")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id_invalid")
    if not isinstance(require_paired_token_counts, bool):
        raise ValueError("require_paired_token_counts_not_boolean")
    source_rows = list(rows)
    _validate_source_manifest(
        source_rows,
        source_manifest,
        model_id=model_id,
        source_dataset_sha256=source_dataset_sha256,
        source_manifest_sha256=source_manifest_sha256,
        tokenizer=tokenizer,
    )
    prompt_renderer = lambda messages: render_prompt(tokenizer, messages)
    parsed_rows = _validate_source_rows(
        source_rows,
        tokenizer=tokenizer,
        prompt_renderer=prompt_renderer,
    )
    schedule = _source_schedule(
        source_rows,
        repetitions=repetitions,
        seed=seed,
        source_dataset_sha256=source_dataset_sha256,
    )
    (
        source_schedule_sha256,
        pair_ids_sha256,
        semantic_targets_sha256,
    ) = _schedule_provenance(schedule, parsed_rows)
    per_source_rotation_orders = {
        source_row_index: _rotation_order(
            menu_size=len(parsed_rows[source_row_index].action_descriptions),
            seed=seed,
            source_dataset_sha256=source_dataset_sha256,
            source_row_sha256=canonical_sha256(source_rows[source_row_index]),
        )
        for source_row_index in range(len(source_rows))
    }

    control_rows: list[dict[str, Any]] = []
    augmented_rows: list[dict[str, Any]] = []
    augmented_rotation_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for schedule_position, item in enumerate(schedule):
        source_row = source_rows[item.source_row_index]
        parsed = parsed_rows[item.source_row_index]
        rotation_order = per_source_rotation_orders[item.source_row_index]
        augmented_rotation = rotation_order[
            item.repetition_index % len(rotation_order)
        ]
        control_rows.append(
            _transform_row(
                source_row,
                parsed,
                rotation=0,
                arm="control_identity",
                schedule_position=schedule_position,
                schedule_item=item,
                source_dataset_sha256=source_dataset_sha256,
                tokenizer=tokenizer,
                prompt_renderer=prompt_renderer,
            )
        )
        augmented_rows.append(
            _transform_row(
                source_row,
                parsed,
                rotation=augmented_rotation,
                arm="augmented_cyclic",
                schedule_position=schedule_position,
                schedule_item=item,
                source_dataset_sha256=source_dataset_sha256,
                tokenizer=tokenizer,
                prompt_renderer=prompt_renderer,
            )
        )
        augmented_rotation_counts[item.source_row_index][augmented_rotation] += 1

    for source_row_index, counts in augmented_rotation_counts.items():
        menu_size = len(parsed_rows[source_row_index].action_descriptions)
        realized = [counts.get(rotation, 0) for rotation in range(menu_size)]
        if max(realized) - min(realized) > 1:
            raise RuntimeError("rotation_schedule_not_balanced")
        if counts.get(0, 0) < 1:
            raise RuntimeError("rotation_schedule_missing_identity")
        if sum(realized) != repetitions:
            raise RuntimeError("source_repetition_count_mismatch")

    paired_token_report = _paired_token_report(control_rows, augmented_rows)
    if (
        require_paired_token_counts
        and not paired_token_report["all_paired_token_counts_match"]
    ):
        raise ValueError(
            "paired_token_count_requirement_failed:"
            + json.dumps(paired_token_report, separators=(",", ":"), sort_keys=True)
        )
    common_manifest_args = {
        "source_manifest": source_manifest,
        "source_dataset_sha256": source_dataset_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "repetitions": repetitions,
        "seed": seed,
        "source_schedule_sha256": source_schedule_sha256,
        "pair_ids_sha256": pair_ids_sha256,
        "semantic_targets_sha256": semantic_targets_sha256,
        "paired_token_report": paired_token_report,
        "require_paired_token_counts": require_paired_token_counts,
        "tokenizer": tokenizer,
    }
    return PairedDatasetBuild(
        control_rows=control_rows,
        augmented_rows=augmented_rows,
        control_manifest=_build_manifest(
            arm="control_identity",
            rows=control_rows,
            **common_manifest_args,
        ),
        augmented_manifest=_build_manifest(
            arm="augmented_cyclic",
            rows=augmented_rows,
            **common_manifest_args,
        ),
    )
