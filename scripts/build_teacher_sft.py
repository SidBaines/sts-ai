#!/usr/bin/env python
"""Build seed-split, deduplicated action-only SFT rows from search labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from sts_ai import provenance
from sts_ai.local_tasks import base
from sts_ai.prompting import ACTION_ONLY_OUTPUT, NEUTRAL_FRAME
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
    validate_search_teacher_manifest,
    validate_search_teacher_rows,
)
from sts_ai.train.sft_format import (
    build_example,
    chat_template_probe_hash,
    loss_mask_token_accounting,
)

SOURCE_STATE_SELECTIONS = ("every", "first_per_turn")


def _select_source_rows(
    rows: list[dict[str, Any]],
    *,
    state_selection: str,
) -> list[dict[str, Any]]:
    if state_selection not in SOURCE_STATE_SELECTIONS:
        raise ValueError(
            "state_selection must be one of "
            + ", ".join(repr(value) for value in SOURCE_STATE_SELECTIONS)
        )
    if state_selection == "every":
        return rows

    selected: dict[tuple[str, int], tuple[int, int, dict[str, Any]]] = {}
    source_coordinates: dict[
        tuple[str, int, int],
        tuple[int, dict[str, Any]],
    ] = {}
    for position, row in enumerate(rows):
        window_id = row.get("window_id")
        turn = row.get("turn")
        source_index = row.get("source_decision_index")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("first_per_turn selection requires a non-empty window_id")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise ValueError("first_per_turn selection requires a non-negative integer turn")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise ValueError(
                "first_per_turn selection requires a non-negative integer "
                "source_decision_index"
            )
        coordinate = (window_id, turn, source_index)
        previous = source_coordinates.get(coordinate)
        if previous is not None:
            previous_position, previous_row = previous
            if previous_row != row:
                raise ValueError(
                    "first_per_turn selection found conflicting rows at source "
                    f"coordinate {coordinate!r} (input positions "
                    f"{previous_position} and {position})"
                )
            continue
        source_coordinates[coordinate] = (position, row)
        key = (window_id, turn)
        candidate = (source_index, position, row)
        current = selected.get(key)
        if current is None or candidate[:2] < current[:2]:
            selected[key] = candidate
    return [
        value[2]
        for value in sorted(selected.values(), key=lambda value: value[:2])
    ]


def _load_tokenizer(tokenizer_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to build teacher SFT data") from exc
    return AutoTokenizer.from_pretrained(tokenizer_id)


def build_teacher_examples(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    tokenizer_id: str,
    enable_thinking: bool,
    min_consensus: float,
    require_hidden_consensus: bool,
    max_examples: int | None = None,
    state_selection: str = "every",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if enable_thinking:
        raise ValueError(
            "search-teacher action-only targets require enable_thinking=False; "
            "a future native-thinking arm must supply real thought-channel context"
        )
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    selection_rule = validate_search_teacher_rows(
        rows,
        expected_selection_rule=AGGREGATED_ROOT_VISITS,
    )
    selected_rows = _select_source_rows(
        rows,
        state_selection=state_selection,
    )
    skipped: Counter[str] = Counter()
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        state_hash = str(row.get("public_state_hash", ""))
        if not state_hash:
            skipped["missing_public_state_hash"] += 1
            continue
        by_hash[state_hash].append(row)

    examples: list[dict[str, Any]] = []
    for state_hash, variants in sorted(by_hash.items()):
        raw_reference_actions = [
            row.get("reference", {}).get("consensus_action_index")
            for row in variants
        ]
        reference_actions = {
            action for action in raw_reference_actions if action is not None
        }
        if len(reference_actions) > 1:
            skipped["public_alias_teacher_conflict"] += len(variants)
            continue
        if len(reference_actions) == 0 or any(
            action is None for action in raw_reference_actions
        ):
            skipped["missing_or_ambiguous_reference_action"] += len(variants)
            continue
        action_index = int(next(iter(reference_actions)))
        if any(
            float(row.get("reference", {}).get("consensus_fraction", 0.0))
            < min_consensus
            for row in variants
        ):
            skipped["low_search_consensus"] += len(variants)
            continue
        hidden = [
            row.get("reference", {}).get("hidden_order_consensus") for row in variants
        ]
        if require_hidden_consensus and any(not isinstance(value, dict) for value in hidden):
            skipped["missing_hidden_order_audit"] += len(variants)
            continue
        if any(
            isinstance(value, dict)
            and float(value.get("consensus_fraction", 0.0)) < min_consensus
            for value in hidden
        ):
            skipped["low_hidden_order_consensus"] += len(variants)
            continue
        if any(
            isinstance(value, dict)
            and value.get("consensus_action_index") != action_index
            for value in hidden
        ):
            skipped["hidden_order_action_conflict"] += len(variants)
            continue

        row = variants[0]
        target = json.dumps({"action_index": action_index}, separators=(",", ":"))
        pseudo_record = {
            "world_seed": row.get("world_seed"),
            "decision_index": row.get("source_decision_index"),
            "phase": "combat",
            "state_text": row["state_text"],
            "legal_actions": row["legal_actions"],
            "agent": {
                "action_index": action_index,
                "raw_response": target,
                "metadata": {},
            },
        }
        try:
            example = build_example(
                pseudo_record,
                NEUTRAL_FRAME,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                loss_mask_mode="action",
                output_contract=ACTION_ONLY_OUTPUT,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped[f"format_error:{exc.__class__.__name__}"] += 1
            continue
        example.update(
            {
                "public_state_hash": state_hash,
                "window_id": row.get("window_id"),
                "source_stem": row.get("source_stem"),
                "base_action_index": row.get("base_action", {}).get("display_index"),
                "teacher_action_index": action_index,
                "observation_version": PUBLIC_OBSERVATION_VERSION,
                "teacher_privilege": TEACHER_PRIVILEGE,
                "teacher_selection_rule": selection_rule,
                "search_reference": row.get("reference"),
            }
        )
        examples.append(example)

    n_examples_before_limit = len(examples)
    if max_examples is not None:
        examples = examples[:max_examples]
    action_counts = Counter(
        str(example["teacher_action_index"])
        for example in examples
    )

    manifest = {
        "kind": "search_teacher_sft",
        "version": 3,
        "observation_version": PUBLIC_OBSERVATION_VERSION,
        "teacher_selection_rule": selection_rule,
        "loss_mask_mode": "action",
        "output_contract": ACTION_ONLY_OUTPUT,
        "target_contains_native_thought": False,
        "teacher_privilege": TEACHER_PRIVILEGE,
        "tokenizer_id": tokenizer_id,
        "enable_thinking": enable_thinking,
        "chat_template_hash": chat_template_probe_hash(
            tokenizer,
            enable_thinking=enable_thinking,
        ),
        "min_consensus": min_consensus,
        "require_hidden_consensus": require_hidden_consensus,
        "n_input_rows": len(rows),
        "source_state_selection": state_selection,
        "n_source_rows_selected": len(selected_rows),
        "n_source_rows_omitted_by_selection": len(rows) - len(selected_rows),
        "n_unique_public_states": len(by_hash),
        "n_examples_before_limit": n_examples_before_limit,
        "n_examples": len(examples),
        "max_examples": max_examples,
        "selection_method": "stable_public_state_hash_prefix_after_filtering_and_dedup",
        "n_examples_omitted_by_limit": n_examples_before_limit - len(examples),
        "skipped_record_counts": dict(skipped),
        "teacher_action_counts": dict(action_counts),
        "token_accounting": loss_mask_token_accounting(examples),
    }
    return examples, manifest


def _load_labels_manifest(
    labels_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = labels_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(
            f"search-teacher labels require adjacent manifest: {manifest_path}"
        )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("search-teacher label manifest is not a JSON object")
    selection_rule = validate_search_teacher_rows(
        rows,
        expected_selection_rule=AGGREGATED_ROOT_VISITS,
    )
    validate_search_teacher_manifest(
        value,
        n_rows=len(rows),
        selection_rule=selection_rule,
    )
    return manifest_path, value


def _source_labels_identity(
    labels_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(labels_path.resolve()),
        "sha256": provenance.file_sha256(labels_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": provenance.file_sha256(manifest_path),
        "manifest_version": manifest["version"],
        "source_manifest": manifest.get("source_manifest"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must remain off for this action-only builder: an immediate JSON target "
        "under a thinking prefix would train the adapter to skip thought. A future "
        "native arm needs real thought/channel context and a separate data contract.",
    )
    parser.add_argument("--min-consensus", type=float, default=2 / 3)
    parser.add_argument("--require-hidden-consensus", action="store_true")
    parser.add_argument(
        "--state-selection",
        choices=SOURCE_STATE_SELECTIONS,
        default="every",
        help="Select every collected micro-action state or only the earliest "
        "source decision for each (window, turn) before confidence filtering.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Keep the first N eligible examples in stable public-state-hash order, "
        "after confidence filtering and deduplication (default: keep all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.min_consensus <= 1.0:
        raise ValueError("--min-consensus must be in (0, 1]")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive when provided")
    labels_manifest_candidate = args.labels.with_suffix(".manifest.json")
    before_read = {
        "labels": provenance.file_sha256(args.labels),
        "manifest": provenance.file_sha256(labels_manifest_candidate),
    }
    rows = base.load_jsonl(args.labels)
    labels_manifest_path, labels_manifest = _load_labels_manifest(args.labels, rows)
    source_labels = _source_labels_identity(
        args.labels,
        labels_manifest_path,
        labels_manifest,
    )
    if before_read != {
        "labels": source_labels["sha256"],
        "manifest": source_labels["manifest_sha256"],
    }:
        raise RuntimeError(
            "search-teacher labels or manifest changed while they were loaded"
        )
    tokenizer = _load_tokenizer(args.tokenizer)
    examples, manifest = build_teacher_examples(
        rows,
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer,
        enable_thinking=args.thinking,
        min_consensus=args.min_consensus,
        require_hidden_consensus=args.require_hidden_consensus,
        max_examples=args.max_examples,
        state_selection=args.state_selection,
    )
    if not examples:
        raise ValueError("no teacher examples survived confidence/dedup filters")
    final_source_labels = _source_labels_identity(
        args.labels,
        labels_manifest_path,
        labels_manifest,
    )
    if final_source_labels != source_labels:
        raise RuntimeError(
            "search-teacher labels or manifest changed while the dataset was built"
        )
    manifest["source_labels"] = final_source_labels
    base.write_jsonl(args.out, examples)
    manifest["dataset_sha256"] = provenance.file_sha256(args.out)
    base.write_json(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
