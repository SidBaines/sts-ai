"""Static no-native-thinking comparison for search-teacher combat states.

This module compares two *output contracts* while holding the policy-visible
state, legal-action menu, neutral framing, model, and generation settings fixed:

``action_only``
    ``{"action_index": 0}``

``reasoning_action``
    ``{"reasoning": "brief private reasoning", "action_index": 0}``

The reasoning field is visible generated text in the JSON answer.  It is not a
native/private model thinking channel: runtime prompt rendering is required to
use ``enable_thinking=False``.

The report-building core is backend-independent.  MLX is imported lazily by
``MlxStaticGenerator`` so unit tests do not load a model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
import time
from typing import Any, Protocol, Sequence

from sts_ai.action_order_diagnostic import ParsedTeacherMenu, parse_teacher_menu
from sts_ai.agents import parse_json_action
from sts_ai.prompting import (
    ACTION_ONLY_OUTPUT,
    NEUTRAL_FRAME,
    REASONING_ACTION_OUTPUT,
    render_action_prompt,
)
from sts_ai.schemas import LegalAction


REPORT_KIND = "no_native_thinking_output_contract_comparison"
REPORT_VERSION = 1
OUTPUT_CONTRACTS = (ACTION_ONLY_OUTPUT, REASONING_ACTION_OUTPUT)
ACTION_ONLY_SCHEMA = '{"action_index": 0}'
REASONING_ACTION_SCHEMA = (
    '{"reasoning": "brief private reasoning", "action_index": 0}'
)


@dataclass(frozen=True)
class ContractPrompts:
    """Exact user prompts reconstructed from one frozen teacher row."""

    parsed: ParsedTeacherMenu
    legal_actions: tuple[LegalAction, ...]
    user_prompts: dict[str, str]


@dataclass(frozen=True)
class GeneratedText:
    """One model completion and its measured generation metadata."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class StaticGenerator(Protocol):
    """Minimal backend boundary used by the pure paired evaluator."""

    def render_chat_prompt(self, user_prompt: str) -> str:
        """Apply the runtime chat template with native thinking disabled."""

    def generate(
        self,
        chat_prompt: str,
        *,
        max_tokens: int,
        seed: int,
    ) -> GeneratedText:
        """Generate one completion under the generator's frozen settings."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_text(rendered)


def build_contract_prompts(row: dict[str, Any]) -> ContractPrompts:
    """Rebuild both policy prompts and prove the intervention is one line.

    The compact/action-only reconstruction must equal the frozen user message
    byte-for-byte.  The reasoning-bearing reconstruction must equal that same
    message with exactly one schema-line replacement.  This prevents a framing,
    game-state, action-order, or instruction change from being mislabeled as a
    reasoning comparison.
    """

    parsed = parse_teacher_menu(row)
    legal_actions = tuple(
        LegalAction(index=index, bits=0, description=description)
        for index, description in enumerate(parsed.action_descriptions)
    )
    action_only = render_action_prompt(
        parsed.state_text,
        list(legal_actions),
        framing=NEUTRAL_FRAME,
        induce_reasoning=False,
        output_contract=ACTION_ONLY_OUTPUT,
    )
    if action_only != parsed.user_content:
        raise ValueError("action_only_user_prompt_rebuild_mismatch")
    if action_only.count(ACTION_ONLY_SCHEMA) != 1:
        raise ValueError("action_only_schema_line_count")

    reasoning_action = render_action_prompt(
        parsed.state_text,
        list(legal_actions),
        framing=NEUTRAL_FRAME,
        induce_reasoning=False,
        output_contract=REASONING_ACTION_OUTPUT,
    )
    expected_reasoning = action_only.replace(
        ACTION_ONLY_SCHEMA,
        REASONING_ACTION_SCHEMA,
        1,
    )
    if reasoning_action != expected_reasoning:
        raise ValueError("reasoning_prompt_changed_beyond_schema_line")
    if reasoning_action.count(REASONING_ACTION_SCHEMA) != 1:
        raise ValueError("reasoning_action_schema_line_count")
    return ContractPrompts(
        parsed=parsed,
        legal_actions=legal_actions,
        user_prompts={
            ACTION_ONLY_OUTPUT: action_only,
            REASONING_ACTION_OUTPUT: reasoning_action,
        },
    )


def _strict_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, type(exc).__name__
    if not isinstance(value, dict):
        return None, "top_level_not_object"
    return value, None


def evaluate_generated_text(
    generated: GeneratedText,
    *,
    legal_actions: Sequence[LegalAction],
    output_contract: str,
    teacher_action_index: int,
) -> dict[str, Any]:
    """Parse one output using both harness and exact-contract definitions."""

    if output_contract not in OUTPUT_CONTRACTS:
        raise ValueError("unknown_output_contract")
    if (
        not isinstance(generated, GeneratedText)
        or not isinstance(generated.text, str)
        or isinstance(generated.prompt_tokens, bool)
        or generated.prompt_tokens <= 0
        or isinstance(generated.completion_tokens, bool)
        or generated.completion_tokens < 0
        or isinstance(generated.latency_s, bool)
        or not isinstance(generated.latency_s, (int, float))
        or not math.isfinite(float(generated.latency_s))
        or generated.latency_s < 0
    ):
        raise ValueError("generated_text_metadata_invalid")
    actions = list(legal_actions)
    decision = parse_json_action(
        generated.text,
        actions,
        completion_tokens=generated.completion_tokens,
    )
    strict_object, strict_error = _strict_json_object(generated.text)
    if output_contract == ACTION_ONLY_OUTPUT:
        expected_keys = {"action_index"}
        reasoning_value: Any = None
        reasoning_field_valid = True
    else:
        expected_keys = {"reasoning", "action_index"}
        reasoning_value = (
            strict_object.get("reasoning")
            if strict_object is not None
            else None
        )
        reasoning_field_valid = (
            isinstance(reasoning_value, str) and bool(reasoning_value.strip())
        )
    exact_schema = (
        strict_object is not None
        and set(strict_object) == expected_keys
        and reasoning_field_valid
    )
    action_value = (
        strict_object.get("action_index")
        if strict_object is not None
        else None
    )
    strict_action_legal = (
        isinstance(action_value, int)
        and not isinstance(action_value, bool)
        and 0 <= action_value < len(actions)
    )
    exact_contract_valid = exact_schema and strict_action_legal
    parsed_action_index = decision.action_index if decision.valid else None
    native_thinking_evidence = bool(decision.thinking)
    return {
        "raw_response": generated.text,
        "raw_response_sha256": _sha256_text(generated.text),
        "prompt_tokens": generated.prompt_tokens,
        "completion_tokens": generated.completion_tokens,
        "latency_s": float(generated.latency_s),
        "harness_action_valid": decision.valid,
        "harness_parse_error": decision.metadata.get("parse_error"),
        "parsed_action_index": parsed_action_index,
        "teacher_action_index": teacher_action_index,
        "teacher_agreement": (
            decision.valid and decision.action_index == teacher_action_index
        ),
        "strict_json_parse_error": strict_error,
        "exact_schema_valid": exact_schema,
        "strict_action_legal": strict_action_legal,
        "exact_contract_valid": exact_contract_valid,
        "visible_reasoning": (
            reasoning_value if isinstance(reasoning_value, str) else None
        ),
        "visible_reasoning_n_chars": (
            len(reasoning_value) if isinstance(reasoning_value, str) else 0
        ),
        "native_thinking_evidence": native_thinking_evidence,
        "parser_metadata": decision.metadata,
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "harness_action_valid_rate": None,
            "exact_contract_valid_rate": None,
            "teacher_agreement_rate": None,
            "teacher_agreement_given_harness_valid": None,
            "native_thinking_evidence_rate": None,
            "mean_latency_s": None,
            "median_latency_s": None,
            "p95_latency_s": None,
            "mean_prompt_tokens": None,
            "mean_completion_tokens": None,
            "mean_visible_reasoning_chars": None,
        }
    count = len(rows)
    valid = [row for row in rows if row["harness_action_valid"]]
    latencies = [float(row["latency_s"]) for row in rows]
    return {
        "n": count,
        "harness_action_valid_rate": sum(
            bool(row["harness_action_valid"]) for row in rows
        )
        / count,
        "exact_contract_valid_rate": sum(
            bool(row["exact_contract_valid"]) for row in rows
        )
        / count,
        "teacher_agreement_rate": sum(
            bool(row["teacher_agreement"]) for row in rows
        )
        / count,
        "teacher_agreement_given_harness_valid": (
            sum(bool(row["teacher_agreement"]) for row in valid) / len(valid)
            if valid
            else None
        ),
        "native_thinking_evidence_rate": sum(
            bool(row["native_thinking_evidence"]) for row in rows
        )
        / count,
        "mean_latency_s": statistics.fmean(latencies),
        "median_latency_s": statistics.median(latencies),
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_prompt_tokens": statistics.fmean(
            int(row["prompt_tokens"]) for row in rows
        ),
        "mean_completion_tokens": statistics.fmean(
            int(row["completion_tokens"]) for row in rows
        ),
        "mean_visible_reasoning_chars": statistics.fmean(
            int(row["visible_reasoning_n_chars"]) for row in rows
        ),
    }


def _validate_generation_settings(
    *,
    max_tokens: int,
    seed: int,
    samples_per_row: int,
) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens_must_be_positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed_must_be_nonnegative")
    if (
        isinstance(samples_per_row, bool)
        or not isinstance(samples_per_row, int)
        or samples_per_row <= 0
    ):
        raise ValueError("samples_per_row_must_be_positive")


def build_model_comparison(
    rows: Sequence[dict[str, Any]],
    generator: StaticGenerator,
    *,
    model_label: str,
    max_tokens: int,
    seed: int,
    samples_per_row: int = 1,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate paired contract outputs for one already-loaded model label."""

    _validate_generation_settings(
        max_tokens=max_tokens,
        seed=seed,
        samples_per_row=samples_per_row,
    )
    if not isinstance(model_label, str) or not model_label:
        raise ValueError("model_label_missing")
    if not rows:
        raise ValueError("teacher_rows_empty")

    results: list[dict[str, Any]] = []
    seen_public_hashes: set[str] = set()
    for row_index, row in enumerate(rows):
        prompts = build_contract_prompts(row)
        public_hash = prompts.parsed.source_public_state_hash
        if public_hash in seen_public_hashes:
            raise ValueError("teacher_public_state_hash_not_unique")
        seen_public_hashes.add(public_hash)

        chat_prompts: dict[str, str] = {}
        for contract in OUTPUT_CONTRACTS:
            chat_prompt = generator.render_chat_prompt(
                prompts.user_prompts[contract]
            )
            if not isinstance(chat_prompt, str) or not chat_prompt:
                raise ValueError("runtime_chat_prompt_invalid")
            chat_prompts[contract] = chat_prompt
        if chat_prompts[ACTION_ONLY_OUTPUT] != prompts.parsed.prompt:
            raise ValueError("runtime_action_only_chat_prompt_mismatch")

        for sample_index in range(samples_per_row):
            # The same seed is deliberately paired across contracts.  Greedy
            # decoding ignores it; sampled decoding is reproducible and starts
            # both interventions from the same RNG state.
            sample_seed = seed + row_index * samples_per_row + sample_index
            for contract in OUTPUT_CONTRACTS:
                generated = generator.generate(
                    chat_prompts[contract],
                    max_tokens=max_tokens,
                    seed=sample_seed,
                )
                parsed_output = evaluate_generated_text(
                    generated,
                    legal_actions=prompts.legal_actions,
                    output_contract=contract,
                    teacher_action_index=prompts.parsed.teacher_action_index,
                )
                results.append(
                    {
                        "row_index": row_index,
                        "sample_index": sample_index,
                        "sample_seed": sample_seed,
                        "output_contract": contract,
                        "window_id": prompts.parsed.window_id,
                        "world_seed": row.get("world_seed"),
                        "decision_index": row.get("decision_index"),
                        "public_state_hash": public_hash,
                        "teacher_action_description": (
                            prompts.parsed.action_descriptions[
                                prompts.parsed.teacher_action_index
                            ]
                        ),
                        "legal_action_descriptions": list(
                            prompts.parsed.action_descriptions
                        ),
                        "user_prompt": prompts.user_prompts[contract],
                        "user_prompt_sha256": _sha256_text(
                            prompts.user_prompts[contract]
                        ),
                        "chat_prompt": chat_prompts[contract],
                        "chat_prompt_sha256": _sha256_text(
                            chat_prompts[contract]
                        ),
                        **parsed_output,
                    }
                )

    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        contract = str(result["output_contract"])
        window_id = str(result["window_id"])
        by_contract[contract].append(result)
        by_window[window_id][contract].append(result)
    contract_summaries = {
        contract: _summary(by_contract[contract])
        for contract in OUTPUT_CONTRACTS
    }
    paired_deltas = {
        key: (
            contract_summaries[REASONING_ACTION_OUTPUT][key]
            - contract_summaries[ACTION_ONLY_OUTPUT][key]
        )
        for key in (
            "harness_action_valid_rate",
            "exact_contract_valid_rate",
            "teacher_agreement_rate",
            "mean_latency_s",
            "mean_completion_tokens",
        )
        if (
            contract_summaries[REASONING_ACTION_OUTPUT][key] is not None
            and contract_summaries[ACTION_ONLY_OUTPUT][key] is not None
        )
    }
    return {
        "model_label": model_label,
        "n_input_rows": len(rows),
        "samples_per_row": samples_per_row,
        "n_generations": len(results),
        "native_thinking_enabled": False,
        "framing": NEUTRAL_FRAME,
        "intervention": (
            "only the requested JSON schema line changes from action_only "
            "to reasoning_action; game state, legal actions, framing, runtime "
            "chat template, model, and paired sampling seed are held fixed"
        ),
        "contract_summaries": contract_summaries,
        "reasoning_minus_action_only": paired_deltas,
        "per_window": {
            window_id: {
                contract: _summary(contract_rows)
                for contract, contract_rows in sorted(contract_groups.items())
            }
            for window_id, contract_groups in sorted(by_window.items())
        },
        "rows": results,
        "provenance": dict(provenance or {}),
    }


def build_report(
    model_comparisons: Sequence[dict[str, Any]],
    *,
    dataset_provenance: dict[str, Any],
    generation_settings: dict[str, Any],
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble multiple separately-loaded model labels into one report."""

    if not model_comparisons:
        raise ValueError("model_comparisons_empty")
    labels = [comparison.get("model_label") for comparison in model_comparisons]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("model_comparison_label_invalid")
    if len(set(labels)) != len(labels):
        raise ValueError("model_comparison_label_duplicate")
    row_counts = {comparison.get("n_input_rows") for comparison in model_comparisons}
    if len(row_counts) != 1:
        raise ValueError("model_comparison_row_count_mismatch")
    return {
        "kind": REPORT_KIND,
        "version": REPORT_VERSION,
        "evaluation_scope": "static_nonfinal_search_teacher_states",
        "native_thinking_enabled": False,
        "output_contracts": list(OUTPUT_CONTRACTS),
        "teacher_agreement_definition": (
            "fraction of all generated outputs whose harness-parsed legal "
            "action_index equals the fixed search-teacher action; invalid "
            "outputs count as disagreement"
        ),
        "exact_contract_validity_definition": (
            "the stripped response is exactly one JSON object with no extra "
            "keys; action_only requires only action_index, reasoning_action "
            "requires a non-empty string reasoning plus action_index; the "
            "action_index must be a legal integer"
        ),
        "dataset": dict(dataset_provenance),
        "generation_settings": dict(generation_settings),
        "source_provenance": dict(source_provenance),
        "model_comparisons": list(model_comparisons),
        "report_content_sha256": _canonical_sha256(
            {
                "dataset": dataset_provenance,
                "generation_settings": generation_settings,
                "source_provenance": source_provenance,
                "model_comparisons": model_comparisons,
            }
        ),
        "limitations": [
            "This is a static teacher-agreement test, not a game rollout or win-rate test.",
            "The visible reasoning text has no expert reasoning target and is evaluated only through its subsequent action, validity, length, and latency.",
            "One search-teacher action is treated as the reference even when other legal actions may also be strong.",
            "Reasoning-action changes the generated token history before action_index, so any effect combines useful scratch work with extra opportunities for generation error.",
            "No result from this development diagnostic licenses inspection of the embargoed final cohort.",
        ],
    }


class MlxStaticGenerator:
    """Lazy local-MLX generator with native thinking forced off."""

    def __init__(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        if not model_path:
            raise ValueError("model_path_missing")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError("temperature_invalid")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(float(top_p))
            or top_p < 0
            or top_p > 1
        ):
            raise ValueError("top_p_invalid")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k_invalid")

        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        kwargs: dict[str, Any] = {
            "tokenizer_config": {"trust_remote_code": True},
        }
        if adapter_path is not None:
            kwargs["adapter_path"] = adapter_path
        self._model, self._tokenizer = load(model_path, **kwargs)
        self._generate = generate
        self._sampler = make_sampler(
            temp=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
        )
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.native_thinking_toggle = self._probe_native_thinking_toggle()

    @staticmethod
    def _encode(
        tokenizer: Any,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        try:
            values = tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        except TypeError:
            values = tokenizer.encode(text)
        return [int(value) for value in values]

    def _probe_native_thinking_toggle(self) -> str:
        messages = [{"role": "user", "content": "probe"}]
        try:
            enabled = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            disabled = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return "unsupported"
        return "supported_and_effective" if enabled != disabled else "accepted_no_effect"

    def render_chat_prompt(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        try:
            value = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            value = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(value, str) or not value:
            raise ValueError("runtime_chat_template_returned_invalid_prompt")
        return value

    def generate(
        self,
        chat_prompt: str,
        *,
        max_tokens: int,
        seed: int,
    ) -> GeneratedText:
        import mlx.core as mx

        mx.random.seed(seed)
        started = time.perf_counter()
        text = self._generate(
            self._model,
            self._tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        latency = time.perf_counter() - started
        if not isinstance(text, str):
            raise ValueError("mlx_generation_returned_non_string")
        return GeneratedText(
            text=text,
            prompt_tokens=len(
                self._encode(
                    self._tokenizer,
                    chat_prompt,
                    add_special_tokens=True,
                )
            ),
            completion_tokens=len(
                self._encode(
                    self._tokenizer,
                    text,
                    add_special_tokens=False,
                )
            ),
            latency_s=latency,
        )

    def tokenizer_provenance(self) -> dict[str, Any]:
        from sts_ai.train.sft_format import chat_template_probe_hash

        return {
            "chat_template_probe_hash_enable_thinking_false": (
                chat_template_probe_hash(
                    self._tokenizer,
                    enable_thinking=False,
                )
            ),
            "native_thinking_toggle": self.native_thinking_toggle,
        }

    def close(self) -> None:
        """Drop the model before the next label is loaded."""

        self._model = None
        self._tokenizer = None
        self._generate = None
        self._sampler = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass
