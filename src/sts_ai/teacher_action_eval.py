"""Candidate-set likelihood evaluation for action-only search-teacher data.

The pure evaluation core depends only on a small scorer protocol.  The MLX
implementation is lazy-imported so ordinary unit tests and data validation do
not require MLX or allocate a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol, Sequence


ACTION_ONLY_OUTPUT_CONTRACT = "action_only"
CANDIDATE_FORMAT = "compact_action_json_v1"
_VALID_ACTIONS_RE = re.compile(
    r"^Valid action_index values are:\s*"
    r"([0-9]+(?:,\s*[0-9]+)*)\.\s*(?:Use only\b.*)?$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class CandidateSequenceScore:
    """Unnormalized causal sequence score for one exact completion."""

    log_probability: float
    n_tokens: int


class CandidateScorer(Protocol):
    """Minimal boundary used by the pure report builder."""

    def score_candidates(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> Sequence[CandidateSequenceScore]:
        """Score each completion conditional on ``prompt``, in input order."""


def action_completion(action_index: int) -> str:
    """Canonical action-only completion used for both training and scoring."""

    if isinstance(action_index, bool) or not isinstance(action_index, int):
        raise TypeError("action_index must be an integer")
    if action_index < 0:
        raise ValueError("action_index must be non-negative")
    return json.dumps({"action_index": action_index}, separators=(",", ":"))


def declared_action_indices(prompt: str) -> list[int]:
    """Parse the prompt's declared policy action set, failing closed.

    Current policy prompts explicitly print one zero-based, contiguous list.
    We intentionally do not infer candidates from ``LEGAL ACTIONS`` lines: a
    malformed declaration is prompt-contract breakage and should be visible in
    the report rather than repaired by the evaluator.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("missing_prompt")
    matches = _VALID_ACTIONS_RE.findall(prompt)
    if len(matches) != 1:
        raise ValueError("valid_action_declaration_count")
    parts = [part.strip() for part in matches[0].split(",")]
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("valid_action_indices_malformed") from exc
    if any(str(value) != part for value, part in zip(values, parts)):
        raise ValueError("valid_action_indices_malformed")
    if values != list(range(len(values))):
        raise ValueError("valid_action_indices_not_zero_based_contiguous")
    if not values:
        raise ValueError("valid_action_indices_empty")
    return values


def _integer_field(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key}_not_integer")
    return value


def _teacher_action_index(row: dict[str, Any]) -> int:
    values = [
        value
        for key in ("target_action_index", "teacher_action_index")
        if (value := _integer_field(row, key)) is not None
    ]
    if not values:
        raise ValueError("missing_teacher_action_index")
    if len(set(values)) != 1:
        raise ValueError("teacher_action_index_mismatch")
    return values[0]


def validate_teacher_row(
    row: dict[str, Any],
) -> tuple[str, list[int], int, str]:
    """Validate and extract the exact scoring inputs from one SFT row."""

    if not isinstance(row, dict):
        raise ValueError("row_not_object")
    if row.get("loss_mask_mode") != "action":
        raise ValueError("loss_mask_mode_not_action")
    if row.get("output_contract") != ACTION_ONLY_OUTPUT_CONTRACT:
        raise ValueError("output_contract_not_action_only")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("missing_prompt")
    action_indices = declared_action_indices(prompt)
    teacher_index = _teacher_action_index(row)
    if teacher_index not in action_indices:
        raise ValueError("teacher_action_out_of_range")
    expected_completion = action_completion(teacher_index)
    if row.get("completion") != expected_completion:
        raise ValueError("noncanonical_action_only_completion")
    window_id = row.get("window_id")
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("missing_window_id")
    return prompt, action_indices, teacher_index, window_id


def _normalized_log_probabilities(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("candidate_log_probabilities_not_finite")
    maximum = max(values)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return [value - log_normalizer for value in values]


def score_teacher_row(
    row: dict[str, Any],
    scorer: CandidateScorer,
    *,
    row_index: int,
) -> dict[str, Any]:
    """Score every declared action candidate for one validated teacher row."""

    prompt, action_indices, teacher_index, window_id = validate_teacher_row(row)
    completions = [action_completion(index) for index in action_indices]
    scores = list(scorer.score_candidates(prompt, completions))
    if len(scores) != len(completions):
        raise ValueError("candidate_score_count_mismatch")
    if any(
        not isinstance(score, CandidateSequenceScore)
        or isinstance(score.n_tokens, bool)
        or score.n_tokens <= 0
        for score in scores
    ):
        raise ValueError("candidate_score_invalid")
    raw_log_probs = [float(score.log_probability) for score in scores]
    normalized_log_probs = _normalized_log_probabilities(raw_log_probs)
    probabilities = [math.exp(value) for value in normalized_log_probs]
    maximum = max(raw_log_probs)
    top_indices = [
        action_index
        for action_index, raw in zip(action_indices, raw_log_probs)
        if raw == maximum
    ]
    top1_index = min(top_indices)
    teacher_position = action_indices.index(teacher_index)

    candidates = []
    for action_index, completion, score, normalized, probability in zip(
        action_indices,
        completions,
        scores,
        normalized_log_probs,
        probabilities,
    ):
        candidates.append(
            {
                "action_index": action_index,
                "completion": completion,
                "n_tokens": score.n_tokens,
                "sequence_log_probability": float(score.log_probability),
                "mean_token_log_probability": (
                    float(score.log_probability) / score.n_tokens
                ),
                "normalized_log_probability": normalized,
                "normalized_probability": probability,
            }
        )

    teacher_probability = probabilities[teacher_position]
    return {
        "row_index": row_index,
        "window_id": window_id,
        "world_seed": row.get("world_seed"),
        "decision_index": row.get("decision_index"),
        "public_state_hash": row.get("public_state_hash"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "n_candidates": len(candidates),
        "teacher_action_index": teacher_index,
        "top1_action_index": top1_index,
        "top_action_indices": top_indices,
        "top1_tied": len(top_indices) > 1,
        "top1_agreement": top1_index == teacher_index,
        "teacher_normalized_probability": teacher_probability,
        "teacher_normalized_log_probability": normalized_log_probs[teacher_position],
        "teacher_candidate_nll": -normalized_log_probs[teacher_position],
        "teacher_sequence_log_probability": raw_log_probs[teacher_position],
        "candidates": candidates,
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "top1_agreement": None,
            "mean_n_candidates": None,
            "mean_teacher_candidate_nll": None,
            "mean_teacher_normalized_probability": None,
            "mean_teacher_sequence_log_probability": None,
        }
    count = len(rows)
    return {
        "n": count,
        "top1_agreement": sum(bool(row["top1_agreement"]) for row in rows) / count,
        "mean_n_candidates": sum(int(row["n_candidates"]) for row in rows) / count,
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


def build_teacher_action_report(
    rows: Sequence[dict[str, Any]],
    scorer: CandidateScorer,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable candidate-probability report.

    Dataset-contract errors are explicit skips.  Operational scorer exceptions
    propagate so an OOM/model failure cannot masquerade as a dataset skip.
    """

    scored: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skipped_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            validate_teacher_row(row)
        except ValueError as exc:
            reason = str(exc)
            skipped[reason] += 1
            skipped_rows.append({"row_index": row_index, "reason": reason})
            continue
        scored.append(score_teacher_row(row, scorer, row_index=row_index))

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_window[str(row["window_id"])].append(row)
    per_window = {
        window_id: _summary(window_rows)
        for window_id, window_rows in sorted(by_window.items())
    }
    return {
        "kind": "teacher_action_candidate_likelihood",
        "version": 1,
        "output_contract": ACTION_ONLY_OUTPUT_CONTRACT,
        "candidate_format": CANDIDATE_FORMAT,
        "candidate_includes_assistant_turn_terminator": False,
        "probability_definition": (
            "softmax over summed causal token log-probabilities of the exact "
            "canonical candidate strings; probability mass outside this candidate "
            "set is excluded"
        ),
        "prompt_source": "exact_stored_prompt",
        "n_input_rows": len(rows),
        "n_scored_rows": len(scored),
        "n_invalid_rows": sum(skipped.values()),
        "n_skipped_rows": sum(skipped.values()),
        "skipped_record_counts": dict(sorted(skipped.items())),
        "skipped_rows": skipped_rows,
        "overall": _summary(scored),
        "per_window": per_window,
        "rows": scored,
        "provenance": dict(provenance or {}),
    }


class MlxCandidateScorer:
    """Lazy MLX implementation of exact conditional sequence scoring."""

    def __init__(self, model_id: str, *, adapter_path: str | None = None):
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._model: Any | None = None
        self._tokenizer: Any | None = None

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

    def score_candidates(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> Sequence[CandidateSequenceScore]:
        import mlx.core as mx

        model, tokenizer = self._load()
        prompt_ids = self._encode(tokenizer, prompt, add_special_tokens=True)
        if not prompt_ids:
            raise ValueError("tokenized prompt is empty")
        results: list[CandidateSequenceScore] = []
        for completion in completions:
            completion_ids = self._encode(
                tokenizer,
                completion,
                add_special_tokens=False,
            )
            if not completion_ids:
                raise ValueError("tokenized candidate completion is empty")
            token_ids = prompt_ids + completion_ids
            inputs = mx.array([token_ids[:-1]])
            logits = model(inputs)
            candidate_logits = logits[
                0,
                len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(completion_ids),
                :,
            ]
            log_probs = candidate_logits - mx.logsumexp(
                candidate_logits,
                axis=-1,
                keepdims=True,
            )
            selected = mx.take_along_axis(
                log_probs,
                mx.array(completion_ids)[:, None],
                axis=-1,
            ).squeeze(-1)
            total = mx.sum(selected)
            mx.eval(total)
            results.append(
                CandidateSequenceScore(
                    log_probability=float(total.item()),
                    n_tokens=len(completion_ids),
                )
            )
            del inputs, logits, candidate_logits, log_probs, selected, total
            mx.clear_cache()
        return results
