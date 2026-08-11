#!/usr/bin/env python
"""Build seed-split, deduplicated action-only SFT rows from search labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from sts_ai import provenance
from sts_ai.local_tasks import base
from sts_ai.prompting import (
    ACTION_ONLY_OUTPUT,
    ACTION_TEXT_OUTPUT,
    NEUTRAL_FRAME,
    TURN_PLAN_OUTPUT,
)
from sts_ai.teacher import (
    AGGREGATED_ROOT_VISITS,
    PUBLIC_OBSERVATION_VERSION,
    TEACHER_PRIVILEGE,
    validate_search_teacher_manifest,
    validate_search_teacher_rows,
)
from sts_ai.teacher_action_eval import action_text_completion, turn_plan_completion
from sts_ai.teacher_metrics import state_visit_stats
from sts_ai.train.sft_format import (
    build_example,
    chat_template_probe_hash,
    loss_mask_token_accounting,
)

SOURCE_STATE_SELECTIONS = ("every", "first_per_turn")
OUTPUT_CONTRACTS = (ACTION_ONLY_OUTPUT, ACTION_TEXT_OUTPUT, TURN_PLAN_OUTPUT)
TARGET_SAMPLING_MODES = ("consensus", "visit_sampled")


@dataclass(frozen=True)
class _SourceExample:
    state_hash: str
    row: dict[str, Any]
    consensus_action_index: int


@dataclass(frozen=True)
class _ScheduleItem:
    source_index: int
    pass_index: int


class _TurnPlanUnavailable(ValueError):
    """Signal a valid state that has no consensus-anchored stored plan."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _legal_action_descriptions(row: dict[str, Any]) -> dict[int, str]:
    state_hash = str(row.get("public_state_hash", ""))
    legal_actions = row.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError(f"legal_actions invalid for {state_hash}")
    descriptions: dict[int, str] = {}
    for action in legal_actions:
        if not isinstance(action, dict):
            raise ValueError(f"legal action invalid for {state_hash}")
        index = action.get("index")
        description = action.get("description")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(description, str)
            or not description
            or index in descriptions
        ):
            raise ValueError(f"legal action invalid for {state_hash}")
        descriptions[index] = description
    return descriptions


def _turn_plan(
    row: dict[str, Any],
    consensus_action_index: int,
) -> tuple[list[str], str]:
    """Derive a consensus-anchored plan and its stored-query provenance.

    Candidate queries are non-abstaining queries whose non-empty this-turn
    ``best_sequence`` line opens with the consensus displayed action's exact
    description (a byte match).  Prefer candidates whose
    ``teacher_vote.action_index`` equals consensus (eligible); within that pool,
    choose maximum ``search.best_evaluation``, then lowest
    ``query.search_seed``, then lowest ``query.draw_order_seed`` with ``None``
    first.  If no eligible candidate exists, apply the same tie-break to any
    candidate and record ``plan_source="non_eligible_query"`` instead of
    ``"eligible_query"``.  With no candidate, the caller skips and counts the
    state; it never emits a partial plan or one with a different first action.

    Per COMP-004A, aggregated-root-visits is the empirically better online
    policy than the raw winning/best line.  Plans are therefore anchored to the
    consensus first action, and unanchorable states are skipped and counted
    rather than trained on a different first action.  The retained plan contains
    this-turn descriptions in order and ends with ``"end turn"``.
    """

    state_hash = str(row.get("public_state_hash", ""))
    descriptions = _legal_action_descriptions(row)
    if consensus_action_index not in descriptions:
        raise ValueError(
            f"consensus action {consensus_action_index} is not legal for {state_hash}"
        )
    turn = row.get("turn")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
        raise ValueError(f"turn invalid for {state_hash}")
    queries = row.get("teacher_queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"teacher_queries invalid for {state_hash}")

    candidates: list[
        tuple[bool, float, int, tuple[int, int], list[str]]
    ] = []
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError(f"teacher query invalid for {state_hash}")
        vote = query.get("teacher_vote")
        if not isinstance(vote, dict):
            raise ValueError(f"teacher vote invalid for {state_hash}")
        abstained = vote.get("abstained")
        if not isinstance(abstained, bool):
            raise ValueError(f"teacher abstention invalid for {state_hash}")
        if abstained:
            continue
        vote_action_index = vote.get("action_index")
        if (
            isinstance(vote_action_index, bool)
            or not isinstance(vote_action_index, int)
            or vote_action_index < 0
        ):
            raise ValueError(f"teacher vote action index invalid for {state_hash}")
        search = query.get("search")
        query_spec = query.get("query")
        if not isinstance(search, dict) or not isinstance(query_spec, dict):
            raise ValueError(f"teacher query search invalid for {state_hash}")
        evaluation = search.get("best_evaluation")
        if (
            isinstance(evaluation, bool)
            or not isinstance(evaluation, (int, float))
            or not math.isfinite(float(evaluation))
        ):
            raise ValueError(f"best_evaluation invalid for {state_hash}")
        search_seed = query_spec.get("search_seed")
        if isinstance(search_seed, bool) or not isinstance(search_seed, int):
            raise ValueError(f"search_seed invalid for {state_hash}")
        draw_seed = query_spec.get("draw_order_seed")
        if draw_seed is not None and (
            isinstance(draw_seed, bool) or not isinstance(draw_seed, int)
        ):
            raise ValueError(f"draw_order_seed invalid for {state_hash}")
        best_sequence = search.get("best_sequence")
        if not isinstance(best_sequence, list):
            raise ValueError(f"best_sequence invalid for {state_hash}")
        plan: list[str] = []
        for entry in best_sequence:
            if not isinstance(entry, dict):
                raise ValueError(f"best_sequence entry invalid for {state_hash}")
            entry_turn = entry.get("turn")
            description = entry.get("description")
            if (
                isinstance(entry_turn, bool)
                or not isinstance(entry_turn, int)
                or entry_turn < 0
                or not isinstance(description, str)
                or not description
            ):
                raise ValueError(f"best_sequence entry invalid for {state_hash}")
            if entry_turn == turn:
                plan.append(description)
        if not plan or plan[0] != descriptions[consensus_action_index]:
            continue
        draw_key = (0, 0) if draw_seed is None else (1, draw_seed)
        candidates.append(
            (
                vote_action_index == consensus_action_index,
                -float(evaluation),
                search_seed,
                draw_key,
                plan,
            )
        )
    if not candidates:
        raise _TurnPlanUnavailable("no_candidate_query")

    eligible = [candidate for candidate in candidates if candidate[0]]
    pool = eligible or candidates
    selected = min(pool, key=lambda item: item[1:4])
    plan = selected[4]
    if not plan or plan[-1] != "end turn":
        plan.append("end turn")
    plan_source = "eligible_query" if selected[0] else "non_eligible_query"
    return plan, plan_source


def _sample_visit_target(
    row: dict[str, Any],
    *,
    schedule_seed: int,
    pass_index: int,
) -> tuple[int, float]:
    """Sample visits with a stable RNG constructed from the schedule identity.

    The exact seed material is UTF-8 JSON for
    ``[schedule_seed, public_state_hash, pass_index]`` rendered with compact
    separators and ``ensure_ascii=False``.  Its SHA-256 digest is interpreted as
    one big-endian integer and passed to ``random.Random``; one ``random()`` draw
    is mapped through cumulative visit shares in ascending displayed-index order.
    """

    state_hash = str(row.get("public_state_hash", ""))
    material = json.dumps(
        [schedule_seed, state_hash, pass_index],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    draw = random.Random(seed).random()
    shares = state_visit_stats(row).visit_share
    cumulative = 0.0
    selected = max(shares)
    for index in sorted(shares):
        cumulative += shares[index]
        if draw < cumulative:
            selected = index
            break
    return selected, shares[selected]


def _source_schedule(
    sources: list[_SourceExample],
    *,
    passes: int,
    schedule_seed: int,
) -> list[_ScheduleItem]:
    """Mirror COMP-015's per-pass SHA-256-ranked identity-arm schedule.

    Deliberate compatibility asymmetry: ``passes == 1`` preserves the selected
    input order for legacy dataset byte identity, while multi-pass pass 0 (like
    every later pass) uses the seeded content-addressed rank.
    """

    if passes == 1:
        return [
            _ScheduleItem(source_index=index, pass_index=0)
            for index in range(len(sources))
        ]
    source_rows = [source.row for source in sources]
    source_dataset_sha256 = _canonical_sha256(source_rows)
    source_row_hashes = [_canonical_sha256(row) for row in source_rows]
    schedule: list[_ScheduleItem] = []
    for pass_index in range(passes):
        ranked: list[tuple[str, _ScheduleItem]] = []
        for source_index, source_row_sha256 in enumerate(source_row_hashes):
            pair_id = _canonical_sha256(
                {
                    "contract": "search_teacher_sft_pass_schedule",
                    "version": 1,
                    "source_dataset_sha256": source_dataset_sha256,
                    "source_row_index": source_index,
                    "source_row_sha256": source_row_sha256,
                    "repetition_index": pass_index,
                }
            )
            sort_key = _canonical_sha256(
                {
                    "pair_id": pair_id,
                    "pass_index": pass_index,
                    "schedule_seed": schedule_seed,
                    "source_dataset_sha256": source_dataset_sha256,
                }
            )
            ranked.append((sort_key, _ScheduleItem(source_index, pass_index)))
        schedule.extend(item for _, item in sorted(ranked))
    return schedule


def _target_for_source(
    source: _SourceExample,
    *,
    output_contract: str,
    target_sampling: str,
    schedule_seed: int,
    pass_index: int,
) -> tuple[int, str, dict[str, Any] | None, str | None]:
    row = source.row
    action_index = source.consensus_action_index
    target_source: dict[str, Any] | None = None
    if target_sampling == "visit_sampled":
        action_index, share = _sample_visit_target(
            row,
            schedule_seed=schedule_seed,
            pass_index=pass_index,
        )
        target_source = {"kind": "visit_sampled", "share": share}
    if output_contract == ACTION_ONLY_OUTPUT:
        target = json.dumps({"action_index": action_index}, separators=(",", ":"))
        return action_index, target, target_source, None

    descriptions = _legal_action_descriptions(row)
    if action_index not in descriptions:
        raise ValueError(
            f"consensus action {action_index} is not legal for {source.state_hash}"
        )
    if output_contract == ACTION_TEXT_OUTPUT:
        target = action_text_completion(descriptions[action_index])
        return action_index, target, target_source or {"kind": "consensus"}, None

    plan, plan_source = _turn_plan(row, source.consensus_action_index)
    return (
        action_index,
        turn_plan_completion(plan),
        {"kind": "consensus"},
        plan_source,
    )


def _format_source_example(
    source: _SourceExample,
    *,
    action_index: int,
    target: str,
    target_source: dict[str, Any] | None,
    plan_source: str | None,
    tokenizer: Any,
    enable_thinking: bool,
    output_contract: str,
    selection_rule: str,
) -> dict[str, Any]:
    row = source.row
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
    example = build_example(
        pseudo_record,
        NEUTRAL_FRAME,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        loss_mask_mode="action",
        output_contract=output_contract,
    )
    example.update(
        {
            "public_state_hash": source.state_hash,
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
    if target_source is not None:
        example["target_source"] = target_source
    if plan_source is not None:
        example["plan_source"] = plan_source
    return example


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
    output_contract: str = ACTION_ONLY_OUTPUT,
    passes: int = 1,
    schedule_seed: int = 0,
    target_sampling: str = "consensus",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if enable_thinking:
        raise ValueError(
            "search-teacher action-only targets require enable_thinking=False; "
            "a future native-thinking arm must supply real thought-channel context"
        )
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    if output_contract not in OUTPUT_CONTRACTS:
        raise ValueError(
            "output_contract must be one of "
            + ", ".join(repr(value) for value in OUTPUT_CONTRACTS)
        )
    _positive_integer(passes, name="passes")
    _nonnegative_integer(schedule_seed, name="schedule_seed")
    if target_sampling not in TARGET_SAMPLING_MODES:
        raise ValueError(
            "target_sampling must be one of "
            + ", ".join(repr(value) for value in TARGET_SAMPLING_MODES)
        )
    if target_sampling == "visit_sampled" and (
        output_contract != ACTION_TEXT_OUTPUT or passes <= 1
    ):
        raise ValueError(
            "visit_sampled target sampling requires output_contract='action_text' "
            "and passes > 1"
        )
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

    sources: list[_SourceExample] = []
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

        sources.append(
            _SourceExample(
                state_hash=state_hash,
                row=variants[0],
                consensus_action_index=action_index,
            )
        )

    valid_sources: list[_SourceExample] = []
    consensus_examples: dict[str, dict[str, Any]] = {}
    turn_plan_skip_reasons: Counter[str] = Counter()
    turn_plan_skipped_hashes: list[str] = []
    for source in sources:
        try:
            action_index, target, target_source, plan_source = _target_for_source(
                source,
                output_contract=output_contract,
                target_sampling="consensus",
                schedule_seed=schedule_seed,
                pass_index=0,
            )
        except _TurnPlanUnavailable as exc:
            turn_plan_skip_reasons[exc.reason] += 1
            turn_plan_skipped_hashes.append(source.state_hash)
            continue
        try:
            example = _format_source_example(
                source,
                action_index=action_index,
                target=target,
                target_source=target_source,
                plan_source=plan_source,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                output_contract=output_contract,
                selection_rule=selection_rule,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped[f"format_error:{exc.__class__.__name__}"] += 1
            continue
        valid_sources.append(source)
        consensus_examples[source.state_hash] = example

    sources = valid_sources
    n_examples_before_limit = len(sources)
    if max_examples is not None:
        sources = sources[:max_examples]

    examples: list[dict[str, Any]] = []
    schedule = _source_schedule(
        sources,
        passes=passes,
        schedule_seed=schedule_seed,
    )
    for schedule_step, item in enumerate(schedule):
        source = sources[item.source_index]
        if target_sampling == "consensus":
            example = deepcopy(consensus_examples[source.state_hash])
        else:
            action_index, target, target_source, plan_source = _target_for_source(
                source,
                output_contract=output_contract,
                target_sampling=target_sampling,
                schedule_seed=schedule_seed,
                pass_index=item.pass_index,
            )
            # Expanded schedules must remain contiguous. A pass-specific sampled
            # target that cannot be formatted is a build failure, not a skippable
            # row after schedule assignment.
            example = _format_source_example(
                source,
                action_index=action_index,
                target=target,
                target_source=target_source,
                plan_source=plan_source,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                output_contract=output_contract,
                selection_rule=selection_rule,
            )
        if passes > 1:
            example.update(
                {
                    "pass_index": item.pass_index,
                    "schedule_step": schedule_step,
                    "source_identity": source.state_hash,
                }
            )
        examples.append(example)

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
        "output_contract": output_contract,
        "passes": passes,
        "schedule_seed": schedule_seed,
        "target_sampling": target_sampling,
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
        "n_examples_omitted_by_limit": n_examples_before_limit - len(sources),
        "skipped_record_counts": dict(skipped),
        "teacher_action_counts": dict(action_counts),
        "token_accounting": loss_mask_token_accounting(examples),
    }
    if output_contract == TURN_PLAN_OUTPUT:
        manifest["turn_plan_skips"] = {
            "n_skipped": len(turn_plan_skipped_hashes),
            "reason_counts": dict(sorted(turn_plan_skip_reasons.items())),
            "skipped_public_state_hashes": sorted(turn_plan_skipped_hashes),
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
        "--output-contract",
        choices=OUTPUT_CONTRACTS,
        default=ACTION_ONLY_OUTPUT,
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of deterministic full passes over the selected source states.",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=0,
        help="Non-negative seed for pass ordering and sampled semantic targets.",
    )
    parser.add_argument(
        "--target-sampling",
        choices=TARGET_SAMPLING_MODES,
        default="consensus",
    )
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
    if args.passes <= 0:
        raise ValueError("--passes must be positive")
    if args.schedule_seed < 0:
        raise ValueError("--schedule-seed must be non-negative")
    if args.target_sampling == "visit_sampled" and (
        args.output_contract != ACTION_TEXT_OUTPUT or args.passes <= 1
    ):
        raise ValueError(
            "--target-sampling visit_sampled requires "
            "--output-contract action_text and --passes > 1"
        )
    output_paths = (args.out, args.out.with_suffix(".manifest.json"))
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise ValueError(
            "refusing to overwrite output files: " + ", ".join(existing_outputs)
        )
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
        output_contract=args.output_contract,
        passes=args.passes,
        schedule_seed=args.schedule_seed,
        target_sampling=args.target_sampling,
    )
    if args.output_contract == TURN_PLAN_OUTPUT:
        turn_plan_skips = manifest["turn_plan_skips"]
        print(
            "turn_plan skips: "
            f"n_skipped={turn_plan_skips['n_skipped']} "
            "reason_counts="
            + json.dumps(turn_plan_skips["reason_counts"], sort_keys=True)
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
