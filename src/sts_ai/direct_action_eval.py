"""Direct policy-token evaluation for compact search-teacher targets.

Unlike :mod:`sts_ai.teacher_action_eval`, this module does not compare summed
log-probabilities for complete JSON strings.  It identifies the one token that
differs between the canonical legal completions, evaluates that next-token
choice once, and reports both full-vocabulary and legal-candidate
probabilities.  The pure report core is independent of MLX.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Protocol, Sequence

from sts_ai.action_order_diagnostic import (
    RotatedTeacherRow,
    all_nonzero_rotations,
)
from sts_ai.teacher_action_eval import validate_teacher_row


DIRECT_ACTION_REPORT_KIND = "teacher_direct_action_token_likelihood"
DIRECT_ACTION_REPORT_VERSION = 1
POLICY_OUTPUT_PREFIX = '{"action_index":'


@dataclass(frozen=True)
class DirectActionTokenScore:
    """One legal action token's next-token score."""

    action_index: int
    token_id: int
    token_text: str
    logit: float
    full_vocabulary_log_probability: float


@dataclass(frozen=True)
class DirectActionScoreBatch:
    """All legal action-token scores from one shared model forward pass."""

    candidates: tuple[DirectActionTokenScore, ...]
    prompt_n_tokens: int
    context_n_tokens: int
    common_completion_prefix_token_ids: tuple[int, ...]
    common_completion_suffix_token_ids: tuple[int, ...]
    model_logits_dtype: str
    scoring_dtype: str
    transformer_hidden_dtype: str
    output_projection_weight_dtype: str
    output_projection_mode: str


@dataclass(frozen=True)
class GreedyCompletion:
    """Deterministic completion generated from the exact stored prompt."""

    text: str
    n_tokens: int
    max_tokens: int
    finish_reason: str


class DirectActionScorer(Protocol):
    """Boundary between the pure report builder and a model backend."""

    def score_action_tokens(
        self,
        prompt: str,
        action_indices: Sequence[int],
    ) -> DirectActionScoreBatch:
        """Score the single policy-bearing token for every legal action."""

    def generate_greedy(self, prompt: str, *, max_tokens: int) -> GreedyCompletion:
        """Generate one deterministic completion from the exact prompt."""


PromptRenderer = Callable[[Sequence[dict[str, str]]], str]


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _normalized_log_probabilities(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("direct_action_logits_not_finite")
    maximum = max(values)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return [value - log_normalizer for value in values]


def _finite_number(value: Any, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(reason)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(reason)
    return result


def _valid_token_ids(values: Any) -> bool:
    return isinstance(values, tuple) and all(
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
        for value in values
    )


def _greedy_result(
    completion: GreedyCompletion,
    action_indices: Sequence[int],
) -> dict[str, Any]:
    if (
        not isinstance(completion, GreedyCompletion)
        or isinstance(completion.n_tokens, bool)
        or completion.n_tokens < 0
        or isinstance(completion.max_tokens, bool)
        or completion.max_tokens <= 0
        or completion.n_tokens > completion.max_tokens
        or completion.finish_reason not in ("stop", "length")
    ):
        raise ValueError("greedy_completion_invalid")
    text = completion.text
    if not isinstance(text, str):
        raise ValueError("greedy_completion_text_not_string")
    stripped = text.strip()
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError) as exc:
        parse_error = type(exc).__name__
    json_object = isinstance(parsed, dict)
    schema_valid = json_object and set(parsed) == {"action_index"}
    value = parsed.get("action_index") if json_object else None
    action_is_integer = isinstance(value, int) and not isinstance(value, bool)
    action_is_legal = action_is_integer and value in action_indices
    canonical = (
        action_is_legal
        and schema_valid
        and text == json.dumps({"action_index": value}, separators=(",", ":"))
    )
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_tokens": completion.n_tokens,
        "max_tokens": completion.max_tokens,
        "finish_reason": completion.finish_reason,
        "max_tokens_reached": completion.finish_reason == "length",
        "json_parse_error": parse_error,
        "json_object": json_object,
        "schema_valid": schema_valid,
        "action_index": value if action_is_integer else None,
        "action_is_legal": action_is_legal,
        "canonical_action_json": canonical,
    }


def score_direct_action_row(
    row: dict[str, Any],
    scorer: DirectActionScorer,
    *,
    row_index: int,
    greedy_max_tokens: int | None,
) -> dict[str, Any]:
    """Score one strict teacher row at its policy-bearing action token."""

    prompt, action_indices, teacher_index, window_id = validate_teacher_row(row)
    batch = scorer.score_action_tokens(prompt, action_indices)
    if not isinstance(batch, DirectActionScoreBatch):
        raise ValueError("direct_action_score_batch_invalid")
    candidates = list(batch.candidates)
    if any(
        not isinstance(candidate, DirectActionTokenScore)
        for candidate in candidates
    ):
        raise ValueError("direct_action_candidate_score_invalid")
    if any(
        isinstance(candidate.action_index, bool)
        or not isinstance(candidate.action_index, int)
        for candidate in candidates
    ):
        raise ValueError("direct_action_candidate_indices_mismatch")
    if [candidate.action_index for candidate in candidates] != action_indices:
        raise ValueError("direct_action_candidate_indices_mismatch")
    if len({candidate.token_id for candidate in candidates}) != len(candidates):
        raise ValueError("direct_action_candidate_token_ids_not_unique")
    for candidate in candidates:
        if (
            isinstance(candidate.token_id, bool)
            or not isinstance(candidate.token_id, int)
            or candidate.token_id < 0
            or not isinstance(candidate.token_text, str)
            or not candidate.token_text
        ):
            raise ValueError("direct_action_candidate_score_invalid")
        _finite_number(
            candidate.logit,
            "direct_action_candidate_score_invalid",
        )
        full_log_probability = _finite_number(
            candidate.full_vocabulary_log_probability,
            "direct_action_candidate_score_invalid",
        )
        if full_log_probability > 1e-6:
            raise ValueError("direct_action_candidate_score_invalid")
    if (
        isinstance(batch.prompt_n_tokens, bool)
        or not isinstance(batch.prompt_n_tokens, int)
        or batch.prompt_n_tokens <= 0
        or isinstance(batch.context_n_tokens, bool)
        or not isinstance(batch.context_n_tokens, int)
        or batch.context_n_tokens < batch.prompt_n_tokens
        or not _valid_token_ids(batch.common_completion_prefix_token_ids)
        or not _valid_token_ids(batch.common_completion_suffix_token_ids)
        or batch.context_n_tokens
        != (
            batch.prompt_n_tokens
            + len(batch.common_completion_prefix_token_ids)
        )
        or not batch.model_logits_dtype
        or not batch.scoring_dtype
        or not batch.transformer_hidden_dtype
        or not batch.output_projection_weight_dtype
        or not batch.output_projection_mode
    ):
        raise ValueError("direct_action_score_metadata_invalid")

    logits = [float(candidate.logit) for candidate in candidates]
    normalized_log_probabilities = _normalized_log_probabilities(logits)
    candidate_probabilities = [
        math.exp(value) for value in normalized_log_probabilities
    ]
    maximum = max(logits)
    top_action_indices = [
        candidate.action_index
        for candidate in candidates
        if float(candidate.logit) == maximum
    ]
    top1_action_index = min(top_action_indices)
    teacher_position = action_indices.index(teacher_index)

    rendered_candidates = []
    for candidate, normalized, probability in zip(
        candidates,
        normalized_log_probabilities,
        candidate_probabilities,
    ):
        rendered_candidates.append(
            {
                "action_index": candidate.action_index,
                "policy_token_id": candidate.token_id,
                "policy_token_text": candidate.token_text,
                "direct_logit": float(candidate.logit),
                "full_vocabulary_log_probability": float(
                    candidate.full_vocabulary_log_probability
                ),
                "full_vocabulary_probability": math.exp(
                    float(candidate.full_vocabulary_log_probability)
                ),
                "candidate_normalized_log_probability": normalized,
                "candidate_normalized_probability": probability,
            }
        )

    teacher = candidates[teacher_position]
    result: dict[str, Any] = {
        "row_index": row_index,
        "window_id": window_id,
        "world_seed": row.get("world_seed"),
        "decision_index": row.get("decision_index"),
        "public_state_hash": row.get("public_state_hash"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "n_candidates": len(candidates),
        "teacher_action_index": teacher_index,
        "top1_action_index": top1_action_index,
        "top_action_indices": top_action_indices,
        "top1_tied": len(top_action_indices) > 1,
        "top_set_size": len(top_action_indices),
        "top1_agreement": top1_action_index == teacher_index,
        "teacher_in_top_set": teacher_index in top_action_indices,
        "teacher_candidate_normalized_probability": candidate_probabilities[
            teacher_position
        ],
        "teacher_candidate_nll": -normalized_log_probabilities[teacher_position],
        "teacher_full_vocabulary_probability": math.exp(
            float(teacher.full_vocabulary_log_probability)
        ),
        "teacher_full_vocabulary_nll": -float(
            teacher.full_vocabulary_log_probability
        ),
        "prompt_n_tokens": batch.prompt_n_tokens,
        "policy_context_n_tokens": batch.context_n_tokens,
        "common_completion_prefix_token_ids": list(
            batch.common_completion_prefix_token_ids
        ),
        "common_completion_suffix_token_ids": list(
            batch.common_completion_suffix_token_ids
        ),
        "model_logits_dtype": batch.model_logits_dtype,
        "scoring_dtype": batch.scoring_dtype,
        "transformer_hidden_dtype": batch.transformer_hidden_dtype,
        "output_projection_weight_dtype": batch.output_projection_weight_dtype,
        "output_projection_mode": batch.output_projection_mode,
        "candidates": rendered_candidates,
    }
    if greedy_max_tokens is not None:
        if (
            isinstance(greedy_max_tokens, bool)
            or not isinstance(greedy_max_tokens, int)
            or greedy_max_tokens <= 0
        ):
            raise ValueError("greedy_max_tokens_must_be_positive")
        greedy = _greedy_result(
            scorer.generate_greedy(prompt, max_tokens=greedy_max_tokens),
            action_indices,
        )
        greedy["teacher_agreement"] = (
            greedy["action_is_legal"]
            and greedy["action_index"] == teacher_index
        )
        result["greedy"] = greedy
    return result


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("direct_action_summary_empty")
    count = len(rows)
    result = {
        "n": count,
        "top1_agreement": sum(bool(row["top1_agreement"]) for row in rows)
        / count,
        "teacher_in_top_set_rate": sum(
            bool(row["teacher_in_top_set"]) for row in rows
        )
        / count,
        "top1_tied_rate": sum(bool(row["top1_tied"]) for row in rows) / count,
        "mean_top_set_size": sum(int(row["top_set_size"]) for row in rows)
        / count,
        "mean_n_candidates": sum(int(row["n_candidates"]) for row in rows)
        / count,
        "mean_teacher_candidate_nll": sum(
            float(row["teacher_candidate_nll"]) for row in rows
        )
        / count,
        "mean_teacher_candidate_normalized_probability": sum(
            float(row["teacher_candidate_normalized_probability"]) for row in rows
        )
        / count,
        "mean_teacher_full_vocabulary_nll": sum(
            float(row["teacher_full_vocabulary_nll"]) for row in rows
        )
        / count,
        "mean_teacher_full_vocabulary_probability": sum(
            float(row["teacher_full_vocabulary_probability"]) for row in rows
        )
        / count,
    }
    if all("greedy" in row for row in rows):
        result["greedy"] = {
            "n": count,
            "mean_n_tokens": sum(
                int(row["greedy"]["n_tokens"]) for row in rows
            )
            / count,
            "length_terminated_rate": sum(
                row["greedy"]["finish_reason"] == "length" for row in rows
            )
            / count,
            "json_object_rate": sum(
                bool(row["greedy"]["json_object"]) for row in rows
            )
            / count,
            "schema_valid_rate": sum(
                bool(row["greedy"]["schema_valid"]) for row in rows
            )
            / count,
            "legal_action_rate": sum(
                bool(row["greedy"]["action_is_legal"]) for row in rows
            )
            / count,
            "canonical_action_json_rate": sum(
                bool(row["greedy"]["canonical_action_json"]) for row in rows
            )
            / count,
            "teacher_agreement": sum(
                bool(row["greedy"]["teacher_agreement"]) for row in rows
            )
            / count,
        }
    elif any("greedy" in row for row in rows):
        raise ValueError("direct_action_greedy_rows_partial")
    return result


def _rotated_result(
    rotated: RotatedTeacherRow,
    scored: dict[str, Any],
    reference: dict[str, Any],
    *,
    source_row_index: int,
) -> dict[str, Any]:
    candidates = scored["candidates"]
    mapped_candidates = sorted(
        (
            {
                **candidate,
                "assigned_action_index": candidate["action_index"],
                "original_action_index": rotated.assigned_to_original[
                    candidate["action_index"]
                ],
            }
            for candidate in candidates
        ),
        key=lambda candidate: candidate["original_action_index"],
    )
    mapped_top_action_indices = sorted(
        rotated.assigned_to_original[index]
        for index in scored["top_action_indices"]
    )
    mapped_top1_action_index = rotated.assigned_to_original[
        scored["top1_action_index"]
    ]
    reference_probabilities = [
        float(candidate["candidate_normalized_probability"])
        for candidate in reference["candidates"]
    ]
    mapped_probabilities = [
        float(candidate["candidate_normalized_probability"])
        for candidate in mapped_candidates
    ]
    probability_tv = 0.5 * sum(
        abs(left - right)
        for left, right in zip(reference_probabilities, mapped_probabilities)
    )
    return {
        "source_row_index": source_row_index,
        "window_id": reference["window_id"],
        "source_public_state_hash": reference["public_state_hash"],
        "rotated_public_state_hash": rotated.rotated_public_state_hash,
        "rotation": rotated.rotation,
        "menu_size": rotated.menu_size,
        "transformation_sha256": rotated.transformation_sha256,
        "original_to_assigned": list(rotated.original_to_assigned),
        "assigned_to_original": list(rotated.assigned_to_original),
        "original_action_descriptions": list(
            rotated.original_action_descriptions
        ),
        "assigned_action_descriptions": list(
            rotated.assigned_action_descriptions
        ),
        "original_teacher_action_index": rotated.original_teacher_action_index,
        "assigned_teacher_action_index": rotated.assigned_teacher_action_index,
        "assigned_top1_action_index": scored["top1_action_index"],
        "assigned_top_action_indices": scored["top_action_indices"],
        "mapped_top1_action_index": mapped_top1_action_index,
        "mapped_top_action_indices": mapped_top_action_indices,
        "top1_tied": scored["top1_tied"],
        "top_set_size": scored["top_set_size"],
        "teacher_top1_agreement": (
            mapped_top1_action_index == rotated.original_teacher_action_index
        ),
        "teacher_in_top_set": (
            rotated.original_teacher_action_index in mapped_top_action_indices
        ),
        "unpermuted_top1_action_index": reference["top1_action_index"],
        "unpermuted_top_action_indices": reference["top_action_indices"],
        "semantic_top1_invariant": (
            mapped_top1_action_index == reference["top1_action_index"]
        ),
        "semantic_top_set_invariant": (
            mapped_top_action_indices == reference["top_action_indices"]
        ),
        "candidate_probability_tv_from_unpermuted": probability_tv,
        "teacher_candidate_normalized_probability": scored[
            "teacher_candidate_normalized_probability"
        ],
        "teacher_candidate_nll": scored["teacher_candidate_nll"],
        "teacher_full_vocabulary_probability": scored[
            "teacher_full_vocabulary_probability"
        ],
        "teacher_full_vocabulary_nll": scored[
            "teacher_full_vocabulary_nll"
        ],
        "prompt_sha256": scored["prompt_sha256"],
        "model_logits_dtype": scored["model_logits_dtype"],
        "scoring_dtype": scored["scoring_dtype"],
        "transformer_hidden_dtype": scored["transformer_hidden_dtype"],
        "output_projection_weight_dtype": scored[
            "output_projection_weight_dtype"
        ],
        "output_projection_mode": scored["output_projection_mode"],
        "candidates_mapped_to_original": mapped_candidates,
    }


def _rotation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("direct_action_rotation_summary_empty")
    count = len(rows)
    return {
        "n": count,
        "teacher_top1_agreement": sum(
            bool(row["teacher_top1_agreement"]) for row in rows
        )
        / count,
        "teacher_in_top_set_rate": sum(
            bool(row["teacher_in_top_set"]) for row in rows
        )
        / count,
        "top1_tied_rate": sum(bool(row["top1_tied"]) for row in rows) / count,
        "mean_top_set_size": sum(int(row["top_set_size"]) for row in rows)
        / count,
        "semantic_top1_invariance": sum(
            bool(row["semantic_top1_invariant"]) for row in rows
        )
        / count,
        "semantic_top_set_invariance": sum(
            bool(row["semantic_top_set_invariant"]) for row in rows
        )
        / count,
        "mean_candidate_probability_tv_from_unpermuted": sum(
            float(row["candidate_probability_tv_from_unpermuted"])
            for row in rows
        )
        / count,
        "mean_teacher_candidate_nll": sum(
            float(row["teacher_candidate_nll"]) for row in rows
        )
        / count,
        "mean_teacher_full_vocabulary_nll": sum(
            float(row["teacher_full_vocabulary_nll"]) for row in rows
        )
        / count,
    }


def _group_summary(
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    summary: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        value: summary(group)
        for value, group in sorted(grouped.items())
    }


def _rotation_source_macro_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Average within source rows so large legal menus are not overweighted."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_row_index"])].append(row)
    if not grouped:
        raise ValueError("direct_action_rotation_source_macro_empty")
    summaries = [_rotation_summary(group) for group in grouped.values()]
    count = len(summaries)
    all_top1_invariant = sum(
        all(bool(row["semantic_top1_invariant"]) for row in group)
        for group in grouped.values()
    )
    all_top_set_invariant = sum(
        all(bool(row["semantic_top_set_invariant"]) for row in group)
        for group in grouped.values()
    )
    return {
        "n_source_rows": count,
        "mean_source_teacher_top1_agreement": sum(
            float(summary["teacher_top1_agreement"]) for summary in summaries
        )
        / count,
        "mean_source_teacher_in_top_set_rate": sum(
            float(summary["teacher_in_top_set_rate"]) for summary in summaries
        )
        / count,
        "mean_source_semantic_top1_invariance": sum(
            float(summary["semantic_top1_invariance"]) for summary in summaries
        )
        / count,
        "mean_source_semantic_top_set_invariance": sum(
            float(summary["semantic_top_set_invariance"]) for summary in summaries
        )
        / count,
        "mean_source_candidate_probability_tv_from_unpermuted": sum(
            float(
                summary[
                    "mean_candidate_probability_tv_from_unpermuted"
                ]
            )
            for summary in summaries
        )
        / count,
        "mean_source_teacher_candidate_nll": sum(
            float(summary["mean_teacher_candidate_nll"])
            for summary in summaries
        )
        / count,
        "mean_source_teacher_full_vocabulary_nll": sum(
            float(summary["mean_teacher_full_vocabulary_nll"])
            for summary in summaries
        )
        / count,
        "all_rotations_top1_invariant_source_count": all_top1_invariant,
        "all_rotations_top1_invariant_source_fraction": (
            all_top1_invariant / count
        ),
        "all_rotations_top_set_invariant_source_count": all_top_set_invariant,
        "all_rotations_top_set_invariant_source_fraction": (
            all_top_set_invariant / count
        ),
    }


def build_direct_action_report(
    rows: Sequence[dict[str, Any]],
    scorer: DirectActionScorer,
    *,
    provenance: dict[str, Any] | None = None,
    greedy_max_tokens: int | None = 32,
    include_cyclic_rotations: bool = False,
    prompt_renderer: PromptRenderer | None = None,
) -> dict[str, Any]:
    """Build an exact, fail-closed direct action-token report.

    When cyclic rotations are requested, they use the same transformation
    implementation as ``diagnose_action_order.py`` and map every probability
    and prediction back to the original semantic action identity.
    """

    if not rows:
        raise ValueError("teacher_dataset_empty")
    scored_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            scored_rows.append(
                score_direct_action_row(
                    row,
                    scorer,
                    row_index=row_index,
                    greedy_max_tokens=greedy_max_tokens,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"teacher row {row_index}: {exc}") from exc

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_window[str(row["window_id"])].append(row)

    cyclic: dict[str, Any] | None = None
    if include_cyclic_rotations:
        if prompt_renderer is None:
            raise ValueError(
                "cyclic rotations require the runtime tokenizer prompt renderer"
            )
        rotation_rows: list[dict[str, Any]] = []
        transformations: list[dict[str, Any]] = []
        for source_row_index, (source, reference) in enumerate(
            zip(rows, scored_rows)
        ):
            try:
                rotations = all_nonzero_rotations(
                    source,
                    prompt_renderer=prompt_renderer,
                )
                for rotated in rotations:
                    scored = score_direct_action_row(
                        rotated.scoring_row,
                        scorer,
                        row_index=source_row_index,
                        greedy_max_tokens=None,
                    )
                    rotation_rows.append(
                        _rotated_result(
                            rotated,
                            scored,
                            reference,
                            source_row_index=source_row_index,
                        )
                    )
                    transformations.append(
                        {
                            "source_row_index": source_row_index,
                            "rotation": rotated.rotation,
                            "rotated_prompt_sha256": scored["prompt_sha256"],
                            "transformation_sha256": (
                                rotated.transformation_sha256
                            ),
                        }
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"teacher row {source_row_index} cyclic rotation: {exc}"
                ) from exc
        expected_rotations = sum(int(row["n_candidates"]) - 1 for row in scored_rows)
        if len(rotation_rows) != expected_rotations:
            raise ValueError("cyclic_rotation_count_mismatch")
        cyclic = {
            "transformation": "all_nonzero_cyclic_right_rotations",
            "semantic_mapping": "assigned_indices_mapped_back_to_original_identity",
            "n_source_rows": len(scored_rows),
            "n_rotated_rows": len(rotation_rows),
            "transformation_set_sha256": _canonical_sha256(transformations),
            "overall": _rotation_summary(rotation_rows),
            "source_macro": _rotation_source_macro_summary(rotation_rows),
            "by_rotation": _group_summary(
                rotation_rows,
                "rotation",
                summary=_rotation_summary,
            ),
            "by_assigned_teacher_action_index": _group_summary(
                rotation_rows,
                "assigned_teacher_action_index",
                summary=_rotation_summary,
            ),
            "by_original_teacher_action_index": _group_summary(
                rotation_rows,
                "original_teacher_action_index",
                summary=_rotation_summary,
            ),
            "by_assigned_top1_action_index": _group_summary(
                rotation_rows,
                "assigned_top1_action_index",
                summary=_rotation_summary,
            ),
            "by_window": _group_summary(
                rotation_rows,
                "window_id",
                summary=_rotation_summary,
            ),
            "by_menu_size": _group_summary(
                rotation_rows,
                "menu_size",
                summary=_rotation_summary,
            ),
            "rows": rotation_rows,
        }

    return {
        "kind": DIRECT_ACTION_REPORT_KIND,
        "version": DIRECT_ACTION_REPORT_VERSION,
        "output_contract": "action_only",
        "prompt_source": "exact_stored_prompt",
        "policy_token_definition": (
            "the unique single token that differs between tokenized exact "
            "canonical legal completions"
        ),
        "model_scoring_point": (
            "next token after the exact stored prompt and the maximal common "
            "canonical-completion token prefix"
        ),
        "tokenization_boundary_contract": (
            "for every candidate, separately tokenized prompt plus completion "
            "must exactly equal tokenization of their concatenated text; the "
            "varying token must decode to the standalone action digit"
        ),
        "full_vocabulary_probability_definition": (
            "softmax over the complete next-token vocabulary produced by the "
            "reported output-projection mode and scoring dtype"
        ),
        "precision_definition": (
            "the MLX backend casts the transformer hidden state and tied output "
            "embedding weights before the full-vocabulary matrix multiplication; "
            "source dtypes and projection mode are reported per row"
        ),
        "candidate_probability_definition": (
            "softmax over only the direct policy-token logits for declared legal "
            "actions; no JSON suffix or other format-token likelihood is included"
        ),
        "tie_definition": "exact equality of reported direct action-token logits",
        "candidate_normalization_dtype": (
            "Python float64 applied to values exported from the reported MLX "
            "scoring dtype"
        ),
        "greedy_scope": (
            "unpermuted_rows_only" if greedy_max_tokens is not None else "disabled"
        ),
        "n_input_rows": len(rows),
        "n_scored_rows": len(scored_rows),
        "model_logits_dtypes": sorted(
            {str(row["model_logits_dtype"]) for row in scored_rows}
        ),
        "scoring_dtypes": sorted(
            {str(row["scoring_dtype"]) for row in scored_rows}
        ),
        "transformer_hidden_dtypes": sorted(
            {str(row["transformer_hidden_dtype"]) for row in scored_rows}
        ),
        "output_projection_weight_dtypes": sorted(
            {
                str(row["output_projection_weight_dtype"])
                for row in scored_rows
            }
        ),
        "output_projection_modes": sorted(
            {str(row["output_projection_mode"]) for row in scored_rows}
        ),
        "overall": _summary(scored_rows),
        "per_window": {
            window_id: _summary(window_rows)
            for window_id, window_rows in sorted(by_window.items())
        },
        "by_teacher_action_index": _group_summary(
            scored_rows,
            "teacher_action_index",
            summary=_summary,
        ),
        "by_top1_action_index": _group_summary(
            scored_rows,
            "top1_action_index",
            summary=_summary,
        ),
        "by_menu_size": _group_summary(
            scored_rows,
            "n_candidates",
            summary=_summary,
        ),
        "rows": scored_rows,
        "cyclic_rotations": cyclic,
        "provenance": dict(provenance or {}),
    }


def _common_prefix_length(values: Sequence[Sequence[int]]) -> int:
    if not values:
        raise ValueError("canonical_completion_tokenizations_empty")
    length = min(len(value) for value in values)
    for index in range(length):
        if len({value[index] for value in values}) != 1:
            return index
    return length


def _common_suffix_length(
    values: Sequence[Sequence[int]],
    *,
    prefix_length: int,
) -> int:
    maximum = min(len(value) - prefix_length for value in values)
    for offset in range(1, maximum + 1):
        if len({value[-offset] for value in values}) != 1:
            return offset - 1
    return maximum


def isolate_policy_tokens(
    completion_token_ids: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Identify one candidate-varying token between common prefix and suffix."""

    if len(completion_token_ids) < 2:
        raise ValueError("direct_action_scoring_requires_at_least_two_candidates")
    if any(
        not value
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in value
        )
        for value in completion_token_ids
    ):
        raise ValueError("canonical_completion_tokenization_invalid")
    values = [tuple(value) for value in completion_token_ids]
    prefix_length = _common_prefix_length(values)
    suffix_length = _common_suffix_length(values, prefix_length=prefix_length)
    middle = [
        value[
            prefix_length : len(value) - suffix_length
            if suffix_length
            else len(value)
        ]
        for value in values
    ]
    if any(len(value) != 1 for value in middle):
        raise ValueError("canonical_completions_do_not_differ_by_one_token")
    policy_token_ids = tuple(value[0] for value in middle)
    if len(set(policy_token_ids)) != len(policy_token_ids):
        raise ValueError("canonical_action_policy_tokens_not_unique")
    return (
        values[0][:prefix_length],
        policy_token_ids,
        values[0][len(values[0]) - suffix_length :] if suffix_length else (),
    )


class MlxDirectActionScorer:
    """Lazy MLX scorer using a genuine float32 tied output projection.

    Gemma's transformer hidden state and tied embedding weights remain in their
    trained source dtypes.  This scorer casts both to ``mx.float32`` *before*
    the complete output-head matrix multiplication and applies any final logit
    soft-cap in float32.  Casting already-computed bfloat16 logits would not
    recover the precision lost in that projection and would not answer the
    quantization diagnostic.
    """

    def __init__(self, model_id: str, *, adapter_path: str | None = None):
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._output_weight_f32: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            from mlx_lm import load

            kwargs = {"tokenizer_config": {"trust_remote_code": True}}
            if self.adapter_path is not None:
                kwargs["adapter_path"] = self.adapter_path
            self._model, self._tokenizer = load(self.model_id, **kwargs)
        return self._model, self._tokenizer

    @staticmethod
    def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
        try:
            values = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            values = tokenizer.encode(text)
        return [int(value) for value in values]

    @staticmethod
    def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
        try:
            value = tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
            )
        except TypeError:
            value = tokenizer.decode(list(token_ids))
        if not isinstance(value, str):
            raise ValueError("tokenizer returned non-string decoded text")
        return value

    @classmethod
    def _decode_token(cls, tokenizer: Any, token_id: int) -> str:
        return cls._decode_ids(tokenizer, [token_id])

    def _float32_full_vocabulary_logits(
        self,
        model: Any,
        inputs: Any,
    ) -> tuple[Any, str, str, str]:
        """Run the supported tied output head in float32, failing closed."""

        import mlx.core as mx

        language_model = getattr(model, "language_model", model)
        text_model = getattr(language_model, "model", None)
        if text_model is None:
            raise ValueError("model has no supported text transformer")
        if getattr(language_model, "tie_word_embeddings", None) is not True:
            raise ValueError("direct float32 scorer requires tied word embeddings")
        embedding = getattr(text_model, "embed_tokens", None)
        weight = getattr(embedding, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            raise ValueError("model has no supported tied output embedding weight")

        hidden = text_model(inputs)
        if (
            getattr(hidden, "ndim", None) != 3
            or hidden.shape[0] != 1
            or hidden.shape[-1] != weight.shape[-1]
        ):
            raise ValueError("model returned an invalid transformer hidden state")
        hidden_dtype = str(hidden.dtype)
        weight_dtype = str(weight.dtype)
        if self._output_weight_f32 is None:
            self._output_weight_f32 = weight.astype(mx.float32)
            mx.eval(self._output_weight_f32)
        elif self._output_weight_f32.shape != weight.shape:
            raise ValueError("cached output projection shape changed")

        final_hidden = hidden[0, -1, :].astype(mx.float32)
        logits = final_hidden @ self._output_weight_f32.T
        softcap = getattr(language_model, "final_logit_softcapping", None)
        if softcap is not None:
            if (
                isinstance(softcap, bool)
                or not isinstance(softcap, (int, float))
                or not math.isfinite(float(softcap))
                or float(softcap) <= 0.0
            ):
                raise ValueError("model final logit softcap is invalid")
            softcap_f32 = mx.array(float(softcap), dtype=mx.float32)
            logits = mx.tanh(logits / softcap_f32) * softcap_f32
            projection_mode = (
                "tied_embedding_full_vocabulary_float32_projection_and_softcap"
            )
        else:
            projection_mode = "tied_embedding_full_vocabulary_float32_projection"
        mx.eval(logits)
        del hidden, final_hidden
        return logits, hidden_dtype, weight_dtype, projection_mode

    def score_action_tokens(
        self,
        prompt: str,
        action_indices: Sequence[int],
    ) -> DirectActionScoreBatch:
        import mlx.core as mx

        from sts_ai.teacher_action_eval import action_completion

        if list(action_indices) != list(range(len(action_indices))):
            raise ValueError("direct action indices must be zero-based contiguous")
        model, tokenizer = self._load()
        prompt_ids = self._encode(tokenizer, prompt, add_special_tokens=True)
        if not prompt_ids:
            raise ValueError("tokenized prompt is empty")
        completion_ids = [
            self._encode(
                tokenizer,
                action_completion(action_index),
                add_special_tokens=False,
            )
            for action_index in action_indices
        ]
        for action_index, candidate_ids in zip(action_indices, completion_ids):
            combined_ids = self._encode(
                tokenizer,
                prompt + action_completion(action_index),
                add_special_tokens=True,
            )
            if combined_ids != prompt_ids + candidate_ids:
                raise ValueError(
                    "prompt/completion tokenization seam disagrees with the "
                    "training contract"
                )
        common_prefix, policy_token_ids, common_suffix = isolate_policy_tokens(
            completion_ids
        )
        if (
            self._decode_ids(tokenizer, common_prefix) != POLICY_OUTPUT_PREFIX
            or self._decode_ids(tokenizer, common_suffix) != "}"
        ):
            raise ValueError(
                "canonical action token boundary is not the expected JSON digit"
            )
        policy_token_texts = tuple(
            self._decode_token(tokenizer, token_id)
            for token_id in policy_token_ids
        )
        if policy_token_texts != tuple(str(index) for index in action_indices):
            raise ValueError(
                "canonical action values are not standalone digit tokens"
            )
        context_ids = prompt_ids + list(common_prefix)
        if not context_ids:
            raise ValueError("direct action policy context is empty")

        inputs = mx.array([context_ids])
        (
            next_logits,
            hidden_dtype,
            projection_weight_dtype,
            projection_mode,
        ) = self._float32_full_vocabulary_logits(model, inputs)
        model_logits_dtype = str(next_logits.dtype)
        token_ids_array = mx.array(list(policy_token_ids))
        selected_logits = mx.take(next_logits, token_ids_array, axis=0)
        log_normalizer = mx.logsumexp(next_logits, axis=-1)
        selected_log_probabilities = selected_logits - log_normalizer
        mx.eval(selected_logits, selected_log_probabilities)
        scoring_dtype = str(next_logits.dtype)
        raw_logits = [float(value) for value in selected_logits.tolist()]
        raw_log_probabilities = [
            float(value) for value in selected_log_probabilities.tolist()
        ]
        candidates = tuple(
            DirectActionTokenScore(
                action_index=action_index,
                token_id=token_id,
                token_text=token_text,
                logit=logit,
                full_vocabulary_log_probability=log_probability,
            )
            for action_index, token_id, token_text, logit, log_probability in zip(
                action_indices,
                policy_token_ids,
                policy_token_texts,
                raw_logits,
                raw_log_probabilities,
            )
        )
        del (
            inputs,
            next_logits,
            token_ids_array,
            selected_logits,
            log_normalizer,
            selected_log_probabilities,
        )
        mx.clear_cache()
        return DirectActionScoreBatch(
            candidates=candidates,
            prompt_n_tokens=len(prompt_ids),
            context_n_tokens=len(context_ids),
            common_completion_prefix_token_ids=common_prefix,
            common_completion_suffix_token_ids=common_suffix,
            model_logits_dtype=model_logits_dtype,
            scoring_dtype=scoring_dtype,
            transformer_hidden_dtype=hidden_dtype,
            output_projection_weight_dtype=projection_weight_dtype,
            output_projection_mode=projection_mode,
        )

    def generate_greedy(self, prompt: str, *, max_tokens: int) -> GreedyCompletion:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        from mlx_lm import stream_generate

        model, tokenizer = self._load()
        prompt_ids = self._encode(tokenizer, prompt, add_special_tokens=True)
        text_parts: list[str] = []
        final_response: Any | None = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=max_tokens,
        ):
            if not isinstance(response.text, str):
                raise ValueError("MLX greedy generation returned non-string text")
            text_parts.append(response.text)
            final_response = response
        if final_response is None:
            raise ValueError("MLX greedy generation returned no response")
        generation_tokens = getattr(final_response, "generation_tokens", None)
        finish_reason = getattr(final_response, "finish_reason", None)
        if (
            isinstance(generation_tokens, bool)
            or not isinstance(generation_tokens, int)
            or generation_tokens < 0
            or generation_tokens > max_tokens
            or finish_reason not in ("stop", "length")
        ):
            raise ValueError("MLX greedy generation metadata is invalid")
        return GreedyCompletion(
            text="".join(text_parts),
            n_tokens=generation_tokens,
            max_tokens=max_tokens,
            finish_reason=finish_reason,
        )
