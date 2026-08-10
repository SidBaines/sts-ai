"""Fail-closed action-order equivariance diagnostics for teacher SFT rows.

The diagnostic keeps the frozen dataset immutable.  It synthesizes prompts whose
legal-action descriptions are cyclically rotated, remaps the teacher target by
the exact positional permutation, and maps model predictions back to the
original semantic action identity for comparison with an unpermuted score
report.

The pure core depends only on the candidate scorer protocol.  MLX remains behind
``teacher_action_eval.MlxCandidateScorer`` and is never imported by unit tests.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Sequence

from sts_ai.teacher import PUBLIC_OBSERVATION_VERSION, public_observation_hash
from sts_ai.teacher_action_eval import (
    CANDIDATE_FORMAT,
    CandidateScorer,
    action_completion,
    declared_action_indices,
    score_teacher_row,
    validate_teacher_row,
)


DIAGNOSTIC_KIND = "teacher_action_order_equivariance"
DIAGNOSTIC_VERSION = 1
REFERENCE_REPORT_KIND = "teacher_action_candidate_likelihood"
REFERENCE_REPORT_VERSION = 1
_LEGAL_ACTIONS_MARKER = "\nLEGAL ACTIONS\n"
_GAME_STATE_MARKER = "\n\nGAME STATE\n"
_ACTION_LINE_RE = re.compile(r"^([0-9]+): (.+)$")
PromptRenderer = Callable[[Sequence[dict[str, str]]], str]


@dataclass(frozen=True)
class ParsedTeacherMenu:
    """Exact policy menu and prompt components extracted from one frozen row."""

    prompt: str
    user_content: str
    embedded_user_content: str
    chat_template_strips_terminal_user_newline: bool
    user_prefix: str
    state_text: str
    action_descriptions: tuple[str, ...]
    teacher_action_index: int
    window_id: str
    source_public_state_hash: str


@dataclass(frozen=True)
class RotatedTeacherRow:
    """One synthetic cyclic permutation and its exact index mappings."""

    scoring_row: dict[str, Any]
    rotation: int
    menu_size: int
    original_teacher_action_index: int
    assigned_teacher_action_index: int
    original_to_assigned: tuple[int, ...]
    assigned_to_original: tuple[int, ...]
    original_action_descriptions: tuple[str, ...]
    assigned_action_descriptions: tuple[str, ...]
    rotated_public_state_hash: str
    transformation_sha256: str


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _messages(row: dict[str, Any], completion: str) -> tuple[str, list[dict[str, Any]]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("messages_not_exact_user_assistant_pair")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("message_not_object")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("messages_not_exact_user_assistant_pair")
    user_content = user.get("content")
    if not isinstance(user_content, str) or not user_content:
        raise ValueError("missing_user_content")
    if assistant.get("content") != completion:
        raise ValueError("assistant_message_completion_mismatch")
    return user_content, [dict(user), dict(assistant)]


def parse_teacher_menu(row: dict[str, Any]) -> ParsedTeacherMenu:
    """Parse and cross-check the terminal legal-action block of a v3 row.

    The stored chat prompt must contain the exact user message once.  This lets
    rotations replace only policy-visible user bytes while preserving the
    tokenizer-specific chat-template prefix and generation suffix byte-for-byte.
    """

    prompt, action_indices, teacher_index, window_id = validate_teacher_row(row)
    completion = action_completion(teacher_index)
    user_content, _ = _messages(row, completion)
    if prompt.count(user_content) == 1:
        embedded_user_content = user_content
        strips_terminal_newline = False
    elif (
        user_content.endswith("\n")
        and prompt.count(user_content[:-1]) == 1
        and prompt.count(user_content) == 0
    ):
        # Gemma 4's frozen chat template strips exactly the terminal newline
        # from the user message before appending its turn delimiter.  Accept
        # only this exact, auditable normalization.
        embedded_user_content = user_content[:-1]
        strips_terminal_newline = True
    else:
        raise ValueError("prompt_user_content_embedding_mismatch")
    if user_content.count(_LEGAL_ACTIONS_MARKER) != 1:
        raise ValueError("legal_actions_block_count")
    user_prefix, action_block = user_content.split(_LEGAL_ACTIONS_MARKER)
    if not action_block.endswith("\n"):
        raise ValueError("legal_actions_block_not_terminal_newline")
    lines = action_block[:-1].split("\n")
    if not lines or any(not line for line in lines):
        raise ValueError("legal_action_line_empty")
    parsed: list[tuple[int, str]] = []
    for line in lines:
        match = _ACTION_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError("legal_action_line_malformed")
        index_text, description = match.groups()
        index = int(index_text)
        if str(index) != index_text:
            raise ValueError("legal_action_index_malformed")
        if not description.strip() or description != description.strip():
            raise ValueError("legal_action_description_malformed")
        parsed.append((index, description))
    parsed_indices = [index for index, _ in parsed]
    if parsed_indices != action_indices:
        raise ValueError("legal_action_lines_disagree_with_declaration")
    descriptions = tuple(description for _, description in parsed)
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("legal_action_descriptions_not_unique")
    if declared_action_indices(user_content) != action_indices:
        raise ValueError("user_and_stored_prompt_declarations_disagree")
    if user_prefix.count(_GAME_STATE_MARKER) != 1:
        raise ValueError("game_state_block_count")
    _, state_with_terminal_newline = user_prefix.split(_GAME_STATE_MARKER)
    if not state_with_terminal_newline.endswith("\n"):
        raise ValueError("game_state_block_missing_terminal_newline")
    state_text = state_with_terminal_newline[:-1]
    if not state_text:
        raise ValueError("game_state_empty")
    source_public_state_hash = row.get("public_state_hash")
    if (
        not isinstance(source_public_state_hash, str)
        or len(source_public_state_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_public_state_hash
        )
    ):
        raise ValueError("source_public_state_hash_invalid")
    observation_version = row.get("observation_version")
    if observation_version != PUBLIC_OBSERVATION_VERSION:
        raise ValueError("observation_version_not_combat_public_v2")
    computed_public_state_hash = public_observation_hash(
        state_text,
        [{"description": description} for description in descriptions],
        observation_version=observation_version,
    )
    if computed_public_state_hash != source_public_state_hash:
        raise ValueError("source_public_state_hash_mismatch")
    return ParsedTeacherMenu(
        prompt=prompt,
        user_content=user_content,
        embedded_user_content=embedded_user_content,
        chat_template_strips_terminal_user_newline=strips_terminal_newline,
        user_prefix=user_prefix,
        state_text=state_text,
        action_descriptions=descriptions,
        teacher_action_index=teacher_index,
        window_id=window_id,
        source_public_state_hash=source_public_state_hash,
    )


def cyclic_index_maps(menu_size: int, rotation: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return original→assigned and assigned→original maps for a right rotation."""

    if isinstance(menu_size, bool) or not isinstance(menu_size, int) or menu_size <= 0:
        raise ValueError("menu_size_must_be_positive")
    if isinstance(rotation, bool) or not isinstance(rotation, int):
        raise ValueError("rotation_must_be_integer")
    if rotation < 0 or rotation >= menu_size:
        raise ValueError("rotation_out_of_range")
    original_to_assigned = tuple(
        (original_index + rotation) % menu_size
        for original_index in range(menu_size)
    )
    assigned_to_original = tuple(
        (assigned_index - rotation) % menu_size
        for assigned_index in range(menu_size)
    )
    return original_to_assigned, assigned_to_original


def validate_prompt_round_trip(
    row: dict[str, Any],
    prompt_renderer: PromptRenderer,
) -> None:
    """Require the runtime tokenizer to reproduce the frozen prompt exactly."""

    parsed = parse_teacher_menu(row)
    rendered = prompt_renderer([{"role": "user", "content": parsed.user_content}])
    if not isinstance(rendered, str) or rendered != parsed.prompt:
        raise ValueError("runtime_chat_template_prompt_mismatch")


def rotate_teacher_row(
    row: dict[str, Any],
    rotation: int,
    *,
    prompt_renderer: PromptRenderer | None = None,
) -> RotatedTeacherRow:
    """Build one exact cyclic prompt permutation without mutating ``row``."""

    parsed = parse_teacher_menu(row)
    menu_size = len(parsed.action_descriptions)
    original_to_assigned, assigned_to_original = cyclic_index_maps(
        menu_size,
        rotation,
    )
    assigned_descriptions = tuple(
        parsed.action_descriptions[original_index]
        for original_index in assigned_to_original
    )
    rotated_block = "\n".join(
        f"{assigned_index}: {description}"
        for assigned_index, description in enumerate(assigned_descriptions)
    )
    rotated_user_content = (
        parsed.user_prefix + _LEGAL_ACTIONS_MARKER + rotated_block + "\n"
    )
    if prompt_renderer is None:
        rotated_embedded_user_content = (
            rotated_user_content[:-1]
            if parsed.chat_template_strips_terminal_user_newline
            else rotated_user_content
        )
        rotated_prompt = parsed.prompt.replace(
            parsed.embedded_user_content,
            rotated_embedded_user_content,
            1,
        )
    else:
        rotated_prompt = prompt_renderer(
            [{"role": "user", "content": rotated_user_content}]
        )
        if not isinstance(rotated_prompt, str) or not rotated_prompt:
            raise ValueError("runtime_chat_template_rotated_prompt_invalid")
        expected_embedded_user_content = (
            rotated_user_content[:-1]
            if parsed.chat_template_strips_terminal_user_newline
            else rotated_user_content
        )
        if rotated_prompt.count(expected_embedded_user_content) != 1:
            raise ValueError(
                "runtime_rotated_prompt_user_content_embedding_mismatch"
            )
        if declared_action_indices(rotated_prompt) != list(range(menu_size)):
            raise ValueError("runtime_rotated_prompt_action_declaration_mismatch")
    assigned_teacher_index = original_to_assigned[parsed.teacher_action_index]
    rotated_completion = action_completion(assigned_teacher_index)
    rotated_public_state_hash = public_observation_hash(
        parsed.state_text,
        [{"description": description} for description in assigned_descriptions],
        observation_version=PUBLIC_OBSERVATION_VERSION,
    )
    scoring_row = {
        "loss_mask_mode": "action",
        "output_contract": row["output_contract"],
        "prompt": rotated_prompt,
        "completion": rotated_completion,
        "target_action_index": assigned_teacher_index,
        "teacher_action_index": assigned_teacher_index,
        "window_id": parsed.window_id,
        "world_seed": row.get("world_seed"),
        "decision_index": row.get("decision_index"),
    }
    transformation = {
        "source_public_state_hash": parsed.source_public_state_hash,
        "rotated_public_state_hash": rotated_public_state_hash,
        "rotation": rotation,
        "menu_size": menu_size,
        "original_teacher_action_index": parsed.teacher_action_index,
        "assigned_teacher_action_index": assigned_teacher_index,
        "original_to_assigned": original_to_assigned,
        "assigned_to_original": assigned_to_original,
        "original_action_descriptions": parsed.action_descriptions,
        "chat_template_strips_terminal_user_newline": (
            parsed.chat_template_strips_terminal_user_newline
        ),
        "rotated_prompt_sha256": hashlib.sha256(
            rotated_prompt.encode("utf-8")
        ).hexdigest(),
    }
    return RotatedTeacherRow(
        scoring_row=scoring_row,
        rotation=rotation,
        menu_size=menu_size,
        original_teacher_action_index=parsed.teacher_action_index,
        assigned_teacher_action_index=assigned_teacher_index,
        original_to_assigned=original_to_assigned,
        assigned_to_original=assigned_to_original,
        original_action_descriptions=parsed.action_descriptions,
        assigned_action_descriptions=assigned_descriptions,
        rotated_public_state_hash=rotated_public_state_hash,
        transformation_sha256=_canonical_sha256(transformation),
    )


def all_nonzero_rotations(
    row: dict[str, Any],
    *,
    prompt_renderer: PromptRenderer | None = None,
) -> list[RotatedTeacherRow]:
    """Generate every non-identity cyclic rotation in ascending order."""

    menu_size = len(parse_teacher_menu(row).action_descriptions)
    return [
        rotate_teacher_row(
            row,
            rotation,
            prompt_renderer=prompt_renderer,
        )
        for rotation in range(1, menu_size)
    ]


def _finite_number(value: Any, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(reason)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(reason)
    return result


def validate_reference_report(
    rows: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cross-check an unpermuted candidate report against exact source rows."""

    if not isinstance(report, dict):
        raise ValueError("reference_report_not_object")
    if (
        report.get("kind") != REFERENCE_REPORT_KIND
        or report.get("version") != REFERENCE_REPORT_VERSION
    ):
        raise ValueError("reference_report_kind_or_version")
    if report.get("candidate_format") != CANDIDATE_FORMAT:
        raise ValueError("reference_report_candidate_format")
    if (
        report.get("output_contract") != "action_only"
        or report.get("candidate_includes_assistant_turn_terminator") is not False
    ):
        raise ValueError("reference_report_output_contract")
    if report.get("prompt_source") != "exact_stored_prompt":
        raise ValueError("reference_report_prompt_source")
    if (
        report.get("n_input_rows") != len(rows)
        or report.get("n_scored_rows") != len(rows)
        or report.get("n_invalid_rows") != 0
        or report.get("n_skipped_rows") != 0
    ):
        raise ValueError("reference_report_row_counts")
    scored_rows = report.get("rows")
    if not isinstance(scored_rows, list) or len(scored_rows) != len(rows):
        raise ValueError("reference_report_rows")

    validated: list[dict[str, Any]] = []
    for source_index, (row, scored) in enumerate(zip(rows, scored_rows)):
        parsed = parse_teacher_menu(row)
        if not isinstance(scored, dict) or scored.get("row_index") != source_index:
            raise ValueError("reference_report_row_index")
        if scored.get("public_state_hash") != row.get("public_state_hash"):
            raise ValueError("reference_report_public_state_hash")
        for key in ("window_id", "world_seed", "decision_index"):
            if scored.get(key) != row.get(key):
                raise ValueError(f"reference_report_{key}")
        expected_prompt_sha = hashlib.sha256(parsed.prompt.encode("utf-8")).hexdigest()
        if scored.get("prompt_sha256") != expected_prompt_sha:
            raise ValueError("reference_report_prompt_sha256")
        menu_size = len(parsed.action_descriptions)
        if (
            scored.get("n_candidates") != menu_size
            or scored.get("teacher_action_index") != parsed.teacher_action_index
        ):
            raise ValueError("reference_report_action_metadata")
        top1_index = scored.get("top1_action_index")
        top_indices = scored.get("top_action_indices")
        if (
            isinstance(top1_index, bool)
            or not isinstance(top1_index, int)
            or top1_index not in range(menu_size)
            or not isinstance(top_indices, list)
            or not top_indices
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index not in range(menu_size)
                for index in top_indices
            )
            or top_indices != sorted(set(top_indices))
            or top1_index != min(top_indices)
        ):
            raise ValueError("reference_report_top_actions")
        if scored.get("top1_tied") is not (len(top_indices) > 1):
            raise ValueError("reference_report_top1_tied")
        if scored.get("top1_agreement") is not (
            top1_index == parsed.teacher_action_index
        ):
            raise ValueError("reference_report_top1_agreement")
        candidates = scored.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != menu_size:
            raise ValueError("reference_report_candidates")
        probabilities: list[float] = []
        sequence_scores: list[float] = []
        for action_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError("reference_report_candidate_not_object")
            if (
                candidate.get("action_index") != action_index
                or candidate.get("completion") != action_completion(action_index)
            ):
                raise ValueError("reference_report_candidate_identity")
            probability = _finite_number(
                candidate.get("normalized_probability"),
                "reference_report_candidate_probability",
            )
            if probability < 0.0 or probability > 1.0:
                raise ValueError("reference_report_candidate_probability")
            probabilities.append(probability)
            sequence_score = _finite_number(
                candidate.get("sequence_log_probability"),
                "reference_report_candidate_sequence_score",
            )
            sequence_scores.append(sequence_score)
            token_count = candidate.get("n_tokens")
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count <= 0
            ):
                raise ValueError("reference_report_candidate_token_count")
            mean_token_score = _finite_number(
                candidate.get("mean_token_log_probability"),
                "reference_report_candidate_mean_token_score",
            )
            if not math.isclose(
                mean_token_score,
                sequence_score / token_count,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "reference_report_candidate_mean_score_mismatch"
                )
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("reference_report_candidate_probability_sum")
        score_maximum = max(sequence_scores)
        log_normalizer = score_maximum + math.log(
            sum(math.exp(score - score_maximum) for score in sequence_scores)
        )
        expected_probabilities = [
            math.exp(score - log_normalizer) for score in sequence_scores
        ]
        if any(
            not math.isclose(
                stored,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for stored, expected in zip(probabilities, expected_probabilities)
        ):
            raise ValueError("reference_report_probabilities_disagree_with_scores")
        for candidate, expected_normalized_log_probability in zip(
            candidates,
            (score - log_normalizer for score in sequence_scores),
        ):
            stored_normalized_log_probability = _finite_number(
                candidate.get("normalized_log_probability"),
                "reference_report_candidate_normalized_log_probability",
            )
            if not math.isclose(
                stored_normalized_log_probability,
                expected_normalized_log_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "reference_report_candidate_normalized_log_probability"
                )
        maximum = score_maximum
        expected_top = [
            index for index, score in enumerate(sequence_scores) if score == maximum
        ]
        if top_indices != expected_top:
            raise ValueError("reference_report_top_actions_disagree_with_scores")
        teacher_probability = probabilities[parsed.teacher_action_index]
        reported_probability = _finite_number(
            scored.get("teacher_normalized_probability"),
            "reference_report_teacher_probability",
        )
        reported_nll = _finite_number(
            scored.get("teacher_candidate_nll"),
            "reference_report_teacher_nll",
        )
        reported_normalized_log_probability = _finite_number(
            scored.get("teacher_normalized_log_probability"),
            "reference_report_teacher_normalized_log_probability",
        )
        reported_sequence_score = _finite_number(
            scored.get("teacher_sequence_log_probability"),
            "reference_report_teacher_sequence_score",
        )
        if not math.isclose(
            teacher_probability,
            reported_probability,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("reference_report_teacher_probability_mismatch")
        if teacher_probability <= 0.0 or not math.isclose(
            -math.log(teacher_probability),
            reported_nll,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("reference_report_teacher_nll_mismatch")
        if not math.isclose(
            math.log(teacher_probability),
            reported_normalized_log_probability,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "reference_report_teacher_normalized_log_probability_mismatch"
            )
        if not math.isclose(
            sequence_scores[parsed.teacher_action_index],
            reported_sequence_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("reference_report_teacher_sequence_score_mismatch")
        validated.append(scored)
    _validate_reference_overall_summary(report.get("overall"), validated)
    expected_windows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scored in validated:
        expected_windows[str(scored["window_id"])].append(scored)
    per_window = report.get("per_window")
    if not isinstance(per_window, dict) or set(per_window) != set(expected_windows):
        raise ValueError("reference_report_per_window_summary")
    for window_id, window_rows in expected_windows.items():
        _validate_reference_overall_summary(
            per_window.get(window_id),
            window_rows,
        )
    return validated


def _validate_reference_overall_summary(
    summary: Any,
    rows: Sequence[dict[str, Any]],
) -> None:
    if not isinstance(summary, dict):
        raise ValueError("reference_report_overall_summary")
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
    if summary.get("n") != expected["n"]:
        raise ValueError("reference_report_overall_summary")
    for key, expected_value in expected.items():
        if key == "n":
            continue
        actual = _finite_number(
            summary.get(key),
            "reference_report_overall_summary",
        )
        if not math.isclose(
            actual,
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("reference_report_overall_summary")


def _candidate_probabilities_by_original_index(
    scored: dict[str, Any],
    assigned_to_original: Sequence[int],
) -> list[float]:
    values = [0.0] * len(assigned_to_original)
    for candidate in scored["candidates"]:
        assigned_index = int(candidate["action_index"])
        original_index = assigned_to_original[assigned_index]
        values[original_index] = float(candidate["normalized_probability"])
    return values


def _variant_record(
    *,
    source_row_index: int,
    source_row: dict[str, Any],
    rotated: RotatedTeacherRow,
    scored: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    mapped_top_indices = sorted(
        rotated.assigned_to_original[index]
        for index in scored["top_action_indices"]
    )
    mapped_top1 = rotated.assigned_to_original[scored["top1_action_index"]]
    reference_top_indices = list(reference["top_action_indices"])
    mapped_probabilities = _candidate_probabilities_by_original_index(
        scored,
        rotated.assigned_to_original,
    )
    reference_probabilities = [
        float(candidate["normalized_probability"])
        for candidate in reference["candidates"]
    ]
    total_variation = 0.5 * sum(
        abs(current - baseline)
        for current, baseline in zip(mapped_probabilities, reference_probabilities)
    )
    teacher_description = rotated.original_action_descriptions[
        rotated.original_teacher_action_index
    ]
    mapped_description = rotated.original_action_descriptions[mapped_top1]
    return {
        "source_row_index": source_row_index,
        "source_public_state_hash": source_row.get("public_state_hash"),
        "rotated_public_state_hash": rotated.rotated_public_state_hash,
        "window_id": source_row.get("window_id"),
        "world_seed": source_row.get("world_seed"),
        "decision_index": source_row.get("decision_index"),
        "rotation": rotated.rotation,
        "menu_size": rotated.menu_size,
        "original_teacher_action_index": rotated.original_teacher_action_index,
        "assigned_teacher_action_index": rotated.assigned_teacher_action_index,
        "teacher_action_description": teacher_description,
        "original_action_descriptions": list(
            rotated.original_action_descriptions
        ),
        "assigned_action_descriptions": list(
            rotated.assigned_action_descriptions
        ),
        "assigned_top1_action_index": scored["top1_action_index"],
        "mapped_top1_action_index": mapped_top1,
        "mapped_top_action_indices": mapped_top_indices,
        "mapped_top1_action_description": mapped_description,
        "unpermuted_top1_action_index": reference["top1_action_index"],
        "unpermuted_top_action_indices": reference_top_indices,
        "semantic_top1_invariant": mapped_top1
        == reference["top1_action_index"],
        "semantic_top_set_invariant": mapped_top_indices == reference_top_indices,
        "semantic_probability_total_variation": total_variation,
        "teacher_probability_delta_from_unpermuted": (
            float(scored["teacher_normalized_probability"])
            - float(reference["teacher_normalized_probability"])
        ),
        "top1_tied": scored["top1_tied"],
        "top1_agreement": scored["top1_agreement"],
        "teacher_normalized_probability": scored[
            "teacher_normalized_probability"
        ],
        "teacher_candidate_nll": scored["teacher_candidate_nll"],
        "teacher_sequence_log_probability": scored[
            "teacher_sequence_log_probability"
        ],
        "prompt_sha256": scored["prompt_sha256"],
        "transformation_sha256": rotated.transformation_sha256,
        "original_to_assigned": list(rotated.original_to_assigned),
        "assigned_to_original": list(rotated.assigned_to_original),
        "mapped_candidate_probabilities": mapped_probabilities,
        "candidates": scored["candidates"],
    }


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "top1_agreement": None,
            "mean_teacher_normalized_probability": None,
            "mean_teacher_candidate_nll": None,
            "mean_teacher_sequence_log_probability": None,
            "top1_tied_count": 0,
        }
    count = len(rows)
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
    rotated = [row for row in rows if int(row["rotation"]) != 0]
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


def _group_summary(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        value: _metric_summary(group)
        for value, group in sorted(grouped.items())
    }


def _source_macro_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average per-source-row metrics so large menus do not get extra weight."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_row_index"])].append(row)
    if not grouped:
        return {
            "n_source_rows": 0,
            "mean_source_top1_agreement": None,
            "mean_source_teacher_probability": None,
            "mean_source_teacher_nll": None,
            "all_rotations_top1_invariant_source_count": 0,
            "all_rotations_top1_invariant_source_fraction": None,
            "all_rotations_top_set_invariant_source_count": 0,
            "all_rotations_top_set_invariant_source_fraction": None,
        }
    source_summaries = [_metric_summary(group) for group in grouped.values()]
    count = len(source_summaries)
    top1_invariant = sum(
        all(bool(row["semantic_top1_invariant"]) for row in group)
        for group in grouped.values()
    )
    top_set_invariant = sum(
        all(bool(row["semantic_top_set_invariant"]) for row in group)
        for group in grouped.values()
    )
    return {
        "n_source_rows": count,
        "mean_source_top1_agreement": sum(
            float(summary["top1_agreement"]) for summary in source_summaries
        )
        / count,
        "mean_source_teacher_probability": sum(
            float(summary["mean_teacher_normalized_probability"])
            for summary in source_summaries
        )
        / count,
        "mean_source_teacher_nll": sum(
            float(summary["mean_teacher_candidate_nll"])
            for summary in source_summaries
        )
        / count,
        "all_rotations_top1_invariant_source_count": top1_invariant,
        "all_rotations_top1_invariant_source_fraction": top1_invariant / count,
        "all_rotations_top_set_invariant_source_count": top_set_invariant,
        "all_rotations_top_set_invariant_source_fraction": top_set_invariant / count,
    }


def _per_source_row_summary(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_row_index"])].append(row)
    values: list[dict[str, Any]] = []
    for source_row_index, group in sorted(grouped.items()):
        rotated = [row for row in group if int(row["rotation"]) != 0]
        values.append(
            {
                "source_row_index": source_row_index,
                "source_public_state_hash": group[0][
                    "source_public_state_hash"
                ],
                "window_id": group[0]["window_id"],
                "menu_size": group[0]["menu_size"],
                "all_rotations_top1_invariant": all(
                    bool(row["semantic_top1_invariant"]) for row in rotated
                ),
                "all_rotations_top_set_invariant": all(
                    bool(row["semantic_top_set_invariant"]) for row in rotated
                ),
                "unpermuted_top1_action_index": group[0][
                    "unpermuted_top1_action_index"
                ],
                "metrics": _metric_summary(group),
            }
        )
    return values


def _reference_variant(
    source_row_index: int,
    source_row: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    parsed = parse_teacher_menu(source_row)
    identity = rotate_teacher_row(source_row, 0)
    probabilities = [
        float(candidate["normalized_probability"])
        for candidate in reference["candidates"]
    ]
    top1_index = int(reference["top1_action_index"])
    return {
        "source_row_index": source_row_index,
        "source_public_state_hash": source_row.get("public_state_hash"),
        "rotated_public_state_hash": identity.rotated_public_state_hash,
        "window_id": source_row.get("window_id"),
        "world_seed": source_row.get("world_seed"),
        "decision_index": source_row.get("decision_index"),
        "rotation": 0,
        "menu_size": len(parsed.action_descriptions),
        "original_teacher_action_index": parsed.teacher_action_index,
        "assigned_teacher_action_index": parsed.teacher_action_index,
        "teacher_action_description": parsed.action_descriptions[
            parsed.teacher_action_index
        ],
        "original_action_descriptions": list(parsed.action_descriptions),
        "assigned_action_descriptions": list(parsed.action_descriptions),
        "assigned_top1_action_index": top1_index,
        "mapped_top1_action_index": top1_index,
        "mapped_top_action_indices": list(reference["top_action_indices"]),
        "mapped_top1_action_description": parsed.action_descriptions[top1_index],
        "unpermuted_top1_action_index": top1_index,
        "unpermuted_top_action_indices": list(reference["top_action_indices"]),
        "semantic_top1_invariant": True,
        "semantic_top_set_invariant": True,
        "semantic_probability_total_variation": 0.0,
        "teacher_probability_delta_from_unpermuted": 0.0,
        "top1_tied": reference["top1_tied"],
        "top1_agreement": reference["top1_agreement"],
        "teacher_normalized_probability": reference[
            "teacher_normalized_probability"
        ],
        "teacher_candidate_nll": reference["teacher_candidate_nll"],
        "teacher_sequence_log_probability": reference[
            "teacher_sequence_log_probability"
        ],
        "prompt_sha256": reference["prompt_sha256"],
        "transformation_sha256": identity.transformation_sha256,
        "original_to_assigned": list(identity.original_to_assigned),
        "assigned_to_original": list(identity.assigned_to_original),
        "mapped_candidate_probabilities": probabilities,
        "candidates": reference["candidates"],
    }


def build_checkpoint_diagnostic(
    rows: Sequence[dict[str, Any]],
    scorer: CandidateScorer,
    reference_report: dict[str, Any],
    *,
    checkpoint_label: str,
    prompt_renderer: PromptRenderer,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score all non-zero rotations for one checkpoint and aggregate results."""

    if not isinstance(checkpoint_label, str) or not checkpoint_label:
        raise ValueError("checkpoint_label_must_be_nonempty")
    for row in rows:
        validate_prompt_round_trip(row, prompt_renderer)
    references = validate_reference_report(rows, reference_report)
    variants: list[dict[str, Any]] = []
    for source_row_index, (source_row, reference) in enumerate(
        zip(rows, references)
    ):
        variants.append(
            _reference_variant(source_row_index, source_row, reference)
        )
        for rotated in all_nonzero_rotations(
            source_row,
            prompt_renderer=prompt_renderer,
        ):
            scored = score_teacher_row(
                rotated.scoring_row,
                scorer,
                row_index=source_row_index,
            )
            variants.append(
                _variant_record(
                    source_row_index=source_row_index,
                    source_row=source_row,
                    rotated=rotated,
                    scored=scored,
                    reference=reference,
                )
            )
    unpermuted = [row for row in variants if row["rotation"] == 0]
    rotated_only = [row for row in variants if row["rotation"] != 0]
    transformation_sha256 = _canonical_sha256(
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
    return {
        "checkpoint_label": checkpoint_label,
        "n_source_rows": len(rows),
        "n_unpermuted_rows": len(unpermuted),
        "n_synthetic_rotated_rows": len(rotated_only),
        "n_all_variants": len(variants),
        "rotation_definition": (
            "right cyclic rotation by k positions: original_to_assigned[i] = "
            "(i + k) mod n and assigned_to_original[j] = (j - k) mod n; "
            "k=0 comes from the validated "
            "unpermuted reference report and every k in [1, menu_size-1] is scored"
        ),
        "n_candidate_sequence_scores": sum(
            int(row["menu_size"]) for row in rotated_only
        ),
        "transformation_set_sha256": transformation_sha256,
        "unpermuted": _metric_summary(unpermuted),
        "rotated": _metric_summary(rotated_only),
        "rotated_source_macro": _source_macro_summary(rotated_only),
        "all_variants": _metric_summary(variants),
        "all_variants_source_macro": _source_macro_summary(variants),
        "by_rotation": _group_summary(variants, "rotation"),
        "by_assigned_target_position": _group_summary(
            variants,
            "assigned_teacher_action_index",
        ),
        "by_original_target_position": _group_summary(
            variants,
            "original_teacher_action_index",
        ),
        "by_window": _group_summary(variants, "window_id"),
        "by_menu_size": _group_summary(variants, "menu_size"),
        "rotated_by_assigned_target_position": _group_summary(
            rotated_only,
            "assigned_teacher_action_index",
        ),
        "rotated_by_original_target_position": _group_summary(
            rotated_only,
            "original_teacher_action_index",
        ),
        "rotated_by_window": _group_summary(rotated_only, "window_id"),
        "rotated_by_menu_size": _group_summary(rotated_only, "menu_size"),
        "per_source_row": _per_source_row_summary(variants),
        "rows": variants,
        "provenance": dict(provenance or {}),
    }


def build_diagnostic_report(
    checkpoint_reports: dict[str, dict[str, Any]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Wrap independently scored checkpoints in one content-addressed report."""

    if not checkpoint_reports:
        raise ValueError("no_checkpoint_reports")
    labels = sorted(checkpoint_reports)
    if any(
        not isinstance(label, str)
        or not label
        or report.get("checkpoint_label") != label
        for label, report in checkpoint_reports.items()
    ):
        raise ValueError("checkpoint_report_label_mismatch")
    source_counts = {
        int(report["n_source_rows"]) for report in checkpoint_reports.values()
    }
    transformation_hashes = {
        str(report["transformation_set_sha256"])
        for report in checkpoint_reports.values()
    }
    if len(source_counts) != 1 or len(transformation_hashes) != 1:
        raise ValueError("checkpoint_reports_do_not_share_transformations")
    return {
        "kind": DIAGNOSTIC_KIND,
        "version": DIAGNOSTIC_VERSION,
        "diagnostic_only": True,
        "mutates_source_artifacts": False,
        "checkpoint_labels": labels,
        "n_source_rows": source_counts.pop(),
        "transformation_set_sha256": transformation_hashes.pop(),
        "checkpoints": {
            label: checkpoint_reports[label] for label in labels
        },
        "provenance": dict(provenance),
    }
