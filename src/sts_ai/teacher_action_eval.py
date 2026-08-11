"""Contract-aware evaluation for compact search-teacher targets.

The pure evaluation core depends on small scorer and generator protocols. MLX
is imported lazily so ordinary validation and unit tests do not allocate a
model or require the optional runtime.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol, Sequence

from sts_ai.prompting import (
    ACTION_ONLY_OUTPUT,
    ACTION_TEXT_INSTRUCTION,
    ACTION_TEXT_OUTPUT,
    TURN_PLAN_INSTRUCTION,
    TURN_PLAN_OUTPUT,
)
from sts_ai.train.sft_format import TURN_PLAN_END_ACTION


ACTION_ONLY_OUTPUT_CONTRACT = ACTION_ONLY_OUTPUT
ACTION_TEXT_OUTPUT_CONTRACT = ACTION_TEXT_OUTPUT
TURN_PLAN_OUTPUT_CONTRACT = TURN_PLAN_OUTPUT
EVALUATED_OUTPUT_CONTRACTS = (
    ACTION_ONLY_OUTPUT_CONTRACT,
    ACTION_TEXT_OUTPUT_CONTRACT,
    TURN_PLAN_OUTPUT_CONTRACT,
)
CANDIDATE_FORMAT = "compact_action_json_v1"
ACTION_TEXT_CANDIDATE_FORMAT = "compact_action_text_json_v1"
DEFAULT_GENERATION_MAX_TOKENS = 128
_VALID_ACTIONS_RE = re.compile(
    r"^Valid action_index values are:\s*"
    r"([0-9]+(?:,\s*[0-9]+)*)\.\s*(?:Use only\b.*)?$",
    flags=re.MULTILINE,
)
_LEGAL_ACTIONS_HEADER_RE = re.compile(r"^LEGAL ACTIONS\s*$", flags=re.MULTILINE)
_LEGAL_ACTION_RE = re.compile(r"^([0-9]+): (.+)$")


class _SemanticContractInvariantError(ValueError):
    """A semantic dataset invariant that must abort rather than skip a row."""


@dataclass(frozen=True)
class CandidateSequenceScore:
    """Unnormalized causal sequence score for one exact completion."""

    log_probability: float
    n_tokens: int


class CandidateScorer(Protocol):
    """Minimal boundary used by candidate-likelihood reports."""

    def score_candidates(
        self,
        prompt: str,
        completions: Sequence[str],
    ) -> Sequence[CandidateSequenceScore]:
        """Score each completion conditional on ``prompt``, in input order."""


class GreedyGenerator(Protocol):
    """Minimal boundary used by generative contract reports."""

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        """Greedily generate one completion conditional on ``prompt``."""


@dataclass(frozen=True)
class _ValidatedTeacherRow:
    prompt: str
    action_indices: list[int]
    action_descriptions: list[str] | None
    teacher_index: int
    teacher_description: str | None
    window_id: str


def action_completion(action_index: int) -> str:
    """Canonical legacy action-only completion."""

    if isinstance(action_index, bool) or not isinstance(action_index, int):
        raise TypeError("action_index must be an integer")
    if action_index < 0:
        raise ValueError("action_index must be non-negative")
    return json.dumps({"action_index": action_index}, separators=(",", ":"))


def action_text_completion(description: str) -> str:
    """Canonical semantic single-action completion."""

    if not isinstance(description, str) or not description:
        raise ValueError("action description must be a non-empty string")
    return json.dumps({"action": description}, separators=(",", ":"))


def turn_plan_completion(plan: Sequence[str]) -> str:
    """Canonical ordered turn-plan completion."""

    if isinstance(plan, (str, bytes)):
        raise TypeError("turn plan must be a sequence of action descriptions")
    values = list(plan)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("turn plan must contain non-empty action descriptions")
    return json.dumps(
        {"plan": values, "action": values[0]},
        separators=(",", ":"),
    )


def declared_action_indices(prompt: str) -> list[int]:
    """Parse the legacy prompt's explicit action-index declaration."""

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


def declared_action_descriptions(
    prompt: str,
    *,
    assistant_turn_terminator: str,
) -> list[str]:
    """Infer semantic candidates from the bounded LEGAL ACTIONS menu.

    Description candidates are intentionally inferred from ``LEGAL ACTIONS`` for
    the semantic output contracts: those contracts choose exact displayed action
    text rather than an integer index. Cutting at the row's stored assistant-turn
    terminator prevents chat-template bytes from becoming part of the final menu
    entry. The caller additionally cross-checks the inferred teacher description
    against the stored target so prompt/schema drift fails loudly.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("missing_prompt")
    if (
        not isinstance(assistant_turn_terminator, str)
        or not assistant_turn_terminator
    ):
        raise _SemanticContractInvariantError(
            "missing_assistant_turn_terminator"
        )
    if prompt.count(assistant_turn_terminator) != 1:
        raise _SemanticContractInvariantError(
            "assistant_turn_terminator_count"
        )
    bounded_prompt = prompt.partition(assistant_turn_terminator)[0]
    headers = list(_LEGAL_ACTIONS_HEADER_RE.finditer(bounded_prompt))
    if len(headers) != 1:
        raise ValueError("legal_actions_declaration_count")
    lines = bounded_prompt[headers[0].end() :].splitlines()
    menu: list[tuple[int, str]] = []
    started = False
    for line in lines:
        match = _LEGAL_ACTION_RE.fullmatch(line)
        if match is not None:
            started = True
            menu.append((int(match.group(1)), match.group(2)))
        elif started:
            break
        elif line:
            raise ValueError("legal_actions_menu_malformed")
    if not menu:
        raise ValueError("legal_actions_empty")
    indices = [index for index, _description in menu]
    if indices != list(range(len(indices))):
        raise ValueError("legal_action_indices_not_zero_based_contiguous")
    return [description for _index, description in menu]


def rerender_turn_plan_prompt_for_action_text(prompt: str) -> str:
    """Swap only the frozen semantic instruction block used for scoring."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("missing_prompt")
    if prompt.count(TURN_PLAN_INSTRUCTION) != 1:
        raise ValueError("turn_plan_instruction_count")
    rendered = prompt.replace(
        TURN_PLAN_INSTRUCTION,
        ACTION_TEXT_INSTRUCTION,
        1,
    )
    source_body = prompt.partition("GAME STATE\n")[2]
    rendered_body = rendered.partition("GAME STATE\n")[2]
    if not source_body or source_body != rendered_body:
        raise ValueError("turn_plan_prompt_body_changed")
    return rendered


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


def _parse_exact_object(completion: Any, *, error: str) -> dict[str, Any]:
    if not isinstance(completion, str):
        raise ValueError(error)
    try:
        value = json.loads(completion)
    except json.JSONDecodeError as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def _validate_teacher_row(
    row: dict[str, Any],
    *,
    output_contract: str,
) -> _ValidatedTeacherRow:
    if output_contract not in EVALUATED_OUTPUT_CONTRACTS:
        raise ValueError("unknown_output_contract")
    if not isinstance(row, dict):
        raise ValueError("row_not_object")
    if row.get("loss_mask_mode") != "action":
        raise ValueError("loss_mask_mode_not_action")
    if row.get("output_contract") != output_contract:
        if output_contract == ACTION_ONLY_OUTPUT_CONTRACT:
            raise ValueError("output_contract_not_action_only")
        raise ValueError("output_contract_mismatch")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("missing_prompt")
    window_id = row.get("window_id")
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("missing_window_id")
    teacher_index = _teacher_action_index(row)

    if output_contract == ACTION_ONLY_OUTPUT_CONTRACT:
        action_indices = declared_action_indices(prompt)
        if teacher_index not in action_indices:
            raise ValueError("teacher_action_out_of_range")
        expected_completion = action_completion(teacher_index)
        if row.get("completion") != expected_completion:
            raise ValueError("noncanonical_action_only_completion")
        return _ValidatedTeacherRow(
            prompt=prompt,
            action_indices=action_indices,
            action_descriptions=None,
            teacher_index=teacher_index,
            teacher_description=None,
            window_id=window_id,
        )

    descriptions = declared_action_descriptions(
        prompt,
        assistant_turn_terminator=row.get("assistant_turn_terminator"),
    )
    action_indices = list(range(len(descriptions)))
    if teacher_index not in action_indices:
        raise ValueError("teacher_action_out_of_range")
    teacher_description = descriptions[teacher_index]
    target_description = row.get("target_action_description")
    if (
        not isinstance(target_description, str)
        or teacher_description.encode("utf-8")
        != target_description.encode("utf-8")
    ):
        raise _SemanticContractInvariantError(
            "target_action_description_mismatch"
        )
    completion = row.get("completion")
    if output_contract == ACTION_TEXT_OUTPUT_CONTRACT:
        parsed = _parse_exact_object(
            completion,
            error="noncanonical_action_text_completion",
        )
        expected = action_text_completion(teacher_description)
        if set(parsed) != {"action"} or completion != expected:
            raise ValueError("noncanonical_action_text_completion")
    else:
        parsed = _parse_exact_object(
            completion,
            error="noncanonical_turn_plan_completion",
        )
        plan = parsed.get("plan")
        action = parsed.get("action")
        if (
            set(parsed) != {"plan", "action"}
            or not isinstance(plan, list)
            or not plan
            or any(not isinstance(item, str) or not item for item in plan)
            or action != plan[0]
            or plan[-1] != TURN_PLAN_END_ACTION
            or action != teacher_description
        ):
            raise ValueError("noncanonical_turn_plan_completion")
        expected = turn_plan_completion(plan)
        if completion != expected:
            raise ValueError("noncanonical_turn_plan_completion")
    return _ValidatedTeacherRow(
        prompt=prompt,
        action_indices=action_indices,
        action_descriptions=descriptions,
        teacher_index=teacher_index,
        teacher_description=teacher_description,
        window_id=window_id,
    )


def validate_teacher_row(
    row: dict[str, Any],
    *,
    output_contract: str = ACTION_ONLY_OUTPUT_CONTRACT,
) -> tuple[str, list[int], int, str]:
    """Validate and extract the historical scoring tuple from one SFT row."""

    validated = _validate_teacher_row(row, output_contract=output_contract)
    return (
        validated.prompt,
        validated.action_indices,
        validated.teacher_index,
        validated.window_id,
    )


def _normalized_log_probabilities(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("candidate_log_probabilities_not_finite")
    maximum = max(values)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return [value - log_normalizer for value in values]


def _lowest_argmax(
    indices: Sequence[int],
    values: Sequence[float],
) -> tuple[int, list[int]]:
    maximum = max(values)
    tied = [index for index, value in zip(indices, values) if value == maximum]
    return min(tied), tied


def score_teacher_row(
    row: dict[str, Any],
    scorer: CandidateScorer,
    *,
    row_index: int,
    output_contract: str = ACTION_ONLY_OUTPUT_CONTRACT,
) -> dict[str, Any]:
    """Score every declared action candidate for one validated teacher row."""

    validated = _validate_teacher_row(row, output_contract=output_contract)
    scoring_prompt = validated.prompt
    if output_contract == ACTION_ONLY_OUTPUT_CONTRACT:
        completions = [action_completion(index) for index in validated.action_indices]
    else:
        if validated.action_descriptions is None:
            raise ValueError("semantic_action_descriptions_missing")
        completions = [
            action_text_completion(description)
            for description in validated.action_descriptions
        ]
        if output_contract == TURN_PLAN_OUTPUT_CONTRACT:
            scoring_prompt = rerender_turn_plan_prompt_for_action_text(scoring_prompt)
    scores = list(scorer.score_candidates(scoring_prompt, completions))
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
    top1_index, top_indices = _lowest_argmax(
        validated.action_indices,
        raw_log_probs,
    )
    teacher_position = validated.action_indices.index(validated.teacher_index)

    candidates = []
    for action_index, completion, score, normalized, probability in zip(
        validated.action_indices,
        completions,
        scores,
        normalized_log_probs,
        probabilities,
    ):
        candidate = {
            "action_index": action_index,
            "completion": completion,
            "n_tokens": score.n_tokens,
            "sequence_log_probability": float(score.log_probability),
            "mean_token_log_probability": float(score.log_probability) / score.n_tokens,
            "normalized_log_probability": normalized,
            "normalized_probability": probability,
        }
        if validated.action_descriptions is not None:
            candidate["action_description"] = validated.action_descriptions[
                action_index
            ]
        candidates.append(candidate)

    teacher_probability = probabilities[teacher_position]
    result = {
        "row_index": row_index,
        "window_id": validated.window_id,
        "world_seed": row.get("world_seed"),
        "decision_index": row.get("decision_index"),
        "public_state_hash": row.get("public_state_hash"),
        "prompt_sha256": hashlib.sha256(scoring_prompt.encode("utf-8")).hexdigest(),
        "n_candidates": len(candidates),
        "teacher_action_index": validated.teacher_index,
        "top1_action_index": top1_index,
        "top_action_indices": top_indices,
        "top1_tied": len(top_indices) > 1,
        "top1_agreement": top1_index == validated.teacher_index,
        "teacher_normalized_probability": teacher_probability,
        "teacher_normalized_log_probability": normalized_log_probs[teacher_position],
        "teacher_candidate_nll": -normalized_log_probs[teacher_position],
        "teacher_sequence_log_probability": raw_log_probs[teacher_position],
        "candidates": candidates,
    }
    if output_contract != ACTION_ONLY_OUTPUT_CONTRACT:
        means = [float(score.log_probability) / score.n_tokens for score in scores]
        mean_top1, _mean_ties = _lowest_argmax(validated.action_indices, means)
        result["top1_by_mean_token_index"] = mean_top1
        result["teacher_action_description"] = validated.teacher_description
    if output_contract == TURN_PLAN_OUTPUT_CONTRACT:
        result["scoring_contract"] = ACTION_TEXT_OUTPUT_CONTRACT
        result["source_prompt_sha256"] = hashlib.sha256(
            validated.prompt.encode("utf-8")
        ).hexdigest()
    return result


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
        ) / count,
        "mean_teacher_normalized_probability": sum(
            float(row["teacher_normalized_probability"]) for row in rows
        ) / count,
        "mean_teacher_sequence_log_probability": sum(
            float(row["teacher_sequence_log_probability"]) for row in rows
        ) / count,
    }


def build_teacher_action_report(
    rows: Sequence[dict[str, Any]],
    scorer: CandidateScorer,
    *,
    provenance: dict[str, Any] | None = None,
    output_contract: str = ACTION_ONLY_OUTPUT_CONTRACT,
) -> dict[str, Any]:
    """Build an auditable candidate-probability report."""

    if output_contract not in EVALUATED_OUTPUT_CONTRACTS:
        raise ValueError("unknown_output_contract")
    scored: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skipped_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            _validate_teacher_row(row, output_contract=output_contract)
        except _SemanticContractInvariantError:
            raise
        except ValueError as exc:
            reason = str(exc)
            skipped[reason] += 1
            skipped_rows.append({"row_index": row_index, "reason": reason})
            continue
        scored.append(
            score_teacher_row(
                row,
                scorer,
                row_index=row_index,
                output_contract=output_contract,
            )
        )

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
        "output_contract": output_contract,
        "candidate_format": (
            CANDIDATE_FORMAT
            if output_contract == ACTION_ONLY_OUTPUT_CONTRACT
            else ACTION_TEXT_CANDIDATE_FORMAT
        ),
        "candidate_includes_assistant_turn_terminator": False,
        "probability_definition": (
            "softmax over summed causal token log-probabilities of the exact "
            "canonical candidate strings; probability mass outside this candidate "
            "set is excluded"
        ),
        "prompt_source": (
            "action_text_instruction_rerender_of_stored_prompt"
            if output_contract == TURN_PLAN_OUTPUT_CONTRACT
            else "exact_stored_prompt"
        ),
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


def evaluate_generated_action(
    generated_text: str,
    *,
    output_contract: str,
    action_indices: Sequence[int],
    action_descriptions: Sequence[str] | None,
) -> dict[str, Any]:
    """Strictly parse one generated completion and map it to a legal index."""

    if not isinstance(generated_text, str):
        raise ValueError("generated_text_not_string")
    try:
        value = json.loads(generated_text)
    except json.JSONDecodeError:
        value = None
    valid_json = isinstance(value, dict)
    chosen_index: int | None = None
    if valid_json:
        assert isinstance(value, dict)
        if output_contract == ACTION_ONLY_OUTPUT_CONTRACT:
            action = value.get("action_index")
            schema_valid = set(value) == {"action_index"}
            if (
                schema_valid
                and isinstance(action, int)
                and not isinstance(action, bool)
                and action in action_indices
            ):
                chosen_index = action
        else:
            descriptions = list(action_descriptions or ())
            action = value.get("action")
            if output_contract == ACTION_TEXT_OUTPUT_CONTRACT:
                schema_valid = set(value) == {"action"}
            elif output_contract == TURN_PLAN_OUTPUT_CONTRACT:
                plan = value.get("plan")
                schema_valid = (
                    set(value) == {"plan", "action"}
                    and isinstance(plan, list)
                    and bool(plan)
                    and all(isinstance(item, str) for item in plan)
                    and action == plan[0]
                    and plan[-1] == TURN_PLAN_END_ACTION
                )
            else:
                raise ValueError("unknown_output_contract")
            if schema_valid and isinstance(action, str) and action in descriptions:
                chosen_index = descriptions.index(action)
    return {
        "generated_text": generated_text,
        "valid_json": valid_json,
        "matched": chosen_index is not None,
        "chosen_index": chosen_index,
    }


def _generation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "valid_json_rate": None,
            "matched_rate": None,
            "top1_rate": None,
        }
    count = len(rows)
    return {
        "n": count,
        "valid_json_rate": sum(bool(row["valid_json"]) for row in rows) / count,
        "matched_rate": sum(bool(row["matched"]) for row in rows) / count,
        "top1_rate": sum(bool(row["top1_agreement"]) for row in rows) / count,
    }


def build_teacher_generation_report(
    rows: Sequence[dict[str, Any]],
    generator: GreedyGenerator,
    *,
    output_contract: str = ACTION_ONLY_OUTPUT_CONTRACT,
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a separate greedy-generation contract-compliance report."""

    if output_contract not in EVALUATED_OUTPUT_CONTRACTS:
        raise ValueError("unknown_output_contract")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens_must_be_positive")
    generated_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skipped_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            validated = _validate_teacher_row(row, output_contract=output_contract)
        except _SemanticContractInvariantError:
            raise
        except ValueError as exc:
            reason = str(exc)
            skipped[reason] += 1
            skipped_rows.append({"row_index": row_index, "reason": reason})
            continue
        result = evaluate_generated_action(
            generator.generate(validated.prompt, max_tokens=max_tokens),
            output_contract=output_contract,
            action_indices=validated.action_indices,
            action_descriptions=validated.action_descriptions,
        )
        result.update(
            {
                "row_index": row_index,
                "window_id": validated.window_id,
                "world_seed": row.get("world_seed"),
                "decision_index": row.get("decision_index"),
                "public_state_hash": row.get("public_state_hash"),
                "teacher_action_index": validated.teacher_index,
                "top1_agreement": result["chosen_index"] == validated.teacher_index,
            }
        )
        generated_rows.append(result)
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in generated_rows:
        by_window[str(result["window_id"])].append(result)
    return {
        "kind": "teacher_action_greedy_generation",
        "version": 1,
        "output_contract": output_contract,
        "generation": {"temperature": 0, "max_tokens": max_tokens},
        "n_input_rows": len(rows),
        "n_scored_rows": len(generated_rows),
        "n_invalid_rows": sum(skipped.values()),
        "n_skipped_rows": sum(skipped.values()),
        "skipped_record_counts": dict(sorted(skipped.items())),
        "skipped_rows": skipped_rows,
        "overall": _generation_summary(generated_rows),
        "per_window": {
            window_id: _generation_summary(window_rows)
            for window_id, window_rows in sorted(by_window.items())
        },
        "rows": generated_rows,
        "provenance": dict(provenance or {}),
    }


class _LazyMlxModel:
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


class MlxCandidateScorer(_LazyMlxModel):
    """Lazy MLX implementation of exact conditional sequence scoring."""

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


class MlxGreedyGenerator(_LazyMlxModel):
    """Lazy MLX implementation of deterministic greedy generation."""

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._load()
        value = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        if not isinstance(value, str):
            raise ValueError("mlx_generation_returned_non_string")
        return value
