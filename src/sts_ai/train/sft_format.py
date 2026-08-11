"""Skew-free SFT prompt/completion reconstruction.

The role-based ``messages`` pair is the canonical training surface for SFT data.
The legacy ``{"prompt", "completion"}`` text pair remains available for
provenance/eval. ``tokenize_example`` is a reference helper that makes the
intended completion-only loss mask explicit and testable.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sts_ai.prompting import (
    ACTION_TEXT_OUTPUT,
    REASONING_ACTION_OUTPUT,
    TURN_PLAN_OUTPUT,
    render_action_prompt,
)
from sts_ai.schemas import LegalAction

__all__ = [
    "TURN_PLAN_END_ACTION",
    "chat_template_probe_hash",
    "user_content",
    "reconstruct_prompt",
    "completion_text",
    "assistant_turn_content",
    "assistant_turn_terminator",
    "build_example",
    "loss_mask_token_accounting",
    "resolve_loss_mask_mode",
    "tokenize_example",
]


TURN_PLAN_END_ACTION = "end turn"
LOSS_MASK_MODES = ("completion", "action")
_ACTION_KEY = "action_index"
_SEMANTIC_ACTION_KEY = "action"
_FORMAT_MARKER_RE = re.compile(
    r"(?:<think>|</think>|<\|channel>thought|<channel\|>|"
    r"<\|channel\|>(?:thought|final)|```(?:json)?)",
    flags=re.IGNORECASE,
)


def _legal_actions_from_record(record: dict) -> list[LegalAction]:
    return [
        LegalAction(
            index=action["index"],
            bits=action["bits"],
            description=action["description"],
        )
        for action in record["legal_actions"]
    ]


def chat_template_probe_hash(
    tokenizer,
    *,
    enable_thinking: bool,
    probe: str = "__sts_probe__",
) -> str:
    """Stable short hash of the model's chat template applied to a probe message.
    Used by the dataset builder (producer) and the trainers (consumer) so the
    skew guard compares like-for-like. MUST be the single definition of this hash.
    """
    messages = [{"role": "user", "content": probe}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


def user_content(
    record: dict,
    framing: str,
    *,
    induce_reasoning: bool = False,
    output_contract: str = REASONING_ACTION_OUTPUT,
) -> str:
    legal_actions = _legal_actions_from_record(record)
    return render_action_prompt(
        record["state_text"],
        legal_actions,
        framing,
        induce_reasoning=induce_reasoning,
        output_contract=output_contract,
    )


def reconstruct_prompt(
    record: dict,
    framing: str,
    *,
    tokenizer,
    enable_thinking: bool,
    induce_reasoning: bool = False,
    output_contract: str = REASONING_ACTION_OUTPUT,
) -> str:
    prompt = user_content(
        record,
        framing,
        induce_reasoning=induce_reasoning,
        output_contract=output_contract,
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def completion_text(record: dict) -> str:
    return record["agent"]["raw_response"]


def assistant_turn_content(record: dict, *, reasoning_format: str | None) -> str:
    """Return the assistant message content used as the SFT target.

    Gemma thought-channel generation preserves the native channel markers in
    ``raw_response``. Keep the passthrough centralized so future normalization,
    if needed, has one round-trip-gated home.
    """
    raw_response = record["agent"]["raw_response"]
    if reasoning_format == "gemma_thought":
        return raw_response
    return raw_response


def build_example(
    record: dict,
    framing: str,
    *,
    tokenizer,
    enable_thinking: bool,
    induce_reasoning: bool = False,
    loss_mask_mode: str = "completion",
    output_contract: str = REASONING_ACTION_OUTPUT,
) -> dict:
    _validate_loss_mask_mode(loss_mask_mode)
    user_message = user_content(
        record,
        framing,
        induce_reasoning=induce_reasoning,
        output_contract=output_contract,
    )
    metadata = record["agent"].get("metadata", {})
    reasoning_format = metadata.get("reasoning_format")
    completion = assistant_turn_content(record, reasoning_format=reasoning_format)
    example = {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": completion},
        ],
        "prompt": reconstruct_prompt(
            record,
            framing,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            induce_reasoning=induce_reasoning,
            output_contract=output_contract,
        ),
        "completion": completion,
        "world_seed": record.get("world_seed"),
        "decision_index": record.get("decision_index"),
        "phase": record.get("phase", "out_of_combat"),
    }
    if output_contract != REASONING_ACTION_OUTPUT:
        example["output_contract"] = output_contract
    if loss_mask_mode == "action":
        agent = record.get("agent") or {}
        expected_action_index = agent.get("action_index")
        if isinstance(expected_action_index, bool) or not isinstance(
            expected_action_index, int
        ):
            raise ValueError("action loss requires an integer agent.action_index")
        example["loss_mask_mode"] = "action"
        example["target_action_index"] = expected_action_index
        if output_contract in (ACTION_TEXT_OUTPUT, TURN_PLAN_OUTPUT):
            descriptions = {
                action.index: action.description
                for action in _legal_actions_from_record(record)
            }
            if expected_action_index not in descriptions:
                raise ValueError(
                    "action loss requires agent.action_index to identify a legal action"
                )
            example["target_action_description"] = descriptions[
                expected_action_index
            ]
        example["assistant_turn_terminator"] = assistant_turn_terminator(tokenizer)
        tokenized = tokenize_example(example, tokenizer, loss_mask_mode="action")
        example["token_counts"] = _token_counts(tokenized)
    return example


def _validate_loss_mask_mode(loss_mask_mode: str) -> None:
    if loss_mask_mode not in LOSS_MASK_MODES:
        raise ValueError(
            "loss_mask_mode must be one of "
            + ", ".join(repr(mode) for mode in LOSS_MASK_MODES)
        )


def resolve_loss_mask_mode(
    requested: str,
    *,
    manifest_path: Any | None,
) -> str:
    """Resolve a trainer mask mode against the dataset manifest.

    Old manifests predate loss-mask provenance and therefore mean the historical
    completion loss. ``auto`` preserves that compatibility. An explicit mode is
    rejected when it disagrees with a new manifest rather than silently changing
    the experiment that the manifest describes.
    """
    if requested not in ("auto", *LOSS_MASK_MODES):
        raise ValueError("loss_mask_mode must be 'auto', 'completion', or 'action'")

    manifest_mode: str | None = None
    if manifest_path is not None:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not read dataset manifest {manifest_path}: {exc}") from exc
        value = manifest.get("loss_mask_mode")
        if value is not None:
            manifest_mode = str(value)
            _validate_loss_mask_mode(manifest_mode)

    if requested == "auto":
        return manifest_mode or "completion"
    if manifest_mode is not None and manifest_mode != requested:
        raise ValueError(
            "requested loss_mask_mode disagrees with dataset manifest: "
            f"requested={requested!r} manifest={manifest_mode!r}"
        )
    return requested


def assistant_turn_terminator(tokenizer: Any) -> str:
    """Infer the exact assistant-turn suffix used by the chat template.

    Gemma 4 uses ``<turn|>\n`` rather than its generic EOS token. Deriving the
    suffix from a two-message template keeps the training target aligned with
    inference for other tokenizers as well. The EOS string is a conservative
    fallback for legacy templates.
    """
    probe = "__sts_assistant_content_probe__"
    messages = [{"role": "user", "content": "__sts_user_probe__"}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": probe}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": probe}],
                tokenize=False,
                add_generation_prompt=False,
            )
        except (AttributeError, TypeError):
            prompt = full = ""
    except AttributeError:
        prompt = full = ""

    if full.startswith(prompt):
        remainder = full[len(prompt) :]
        probe_at = remainder.find(probe)
        if probe_at != -1:
            suffix = remainder[probe_at + len(probe) :]
            if suffix:
                return suffix

    eos_token = getattr(tokenizer, "eos_token", None)
    return str(eos_token) if eos_token else ""


def _balanced_json_candidates(text: str) -> list[tuple[str, tuple[int, int]]]:
    candidates: list[tuple[str, tuple[int, int]]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append((text[start : index + 1], (start, index + 1)))
                start = None
    return candidates


def _action_object_span(
    text: str,
    *,
    action_key: str,
    expected_action: int | str | None,
    output_contract: str,
) -> tuple[int, int]:
    stripped = text.strip()
    offset = len(text) - len(text.lstrip())
    candidates = [(stripped, (offset, offset + len(stripped)))]
    candidates.extend(reversed(_balanced_json_candidates(text)))
    seen: set[tuple[int, int]] = set()
    for candidate, span in candidates:
        if span in seen:
            continue
        seen.add(span)
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        action = parsed.get(action_key)
        if action_key == _ACTION_KEY:
            if isinstance(action, bool) or not isinstance(action, int):
                raise ValueError("final JSON object has no integer action_index")
        elif not isinstance(action, str):
            raise ValueError("final JSON object has no string action")
        if expected_action is not None and action != expected_action:
            raise ValueError(
                f"final JSON {action_key} disagrees with recorded action: "
                f"json={action!r} recorded={expected_action!r}"
            )
        if output_contract == TURN_PLAN_OUTPUT:
            plan = parsed.get("plan")
            if (
                not isinstance(plan, list)
                or not plan
                or any(not isinstance(item, str) for item in plan)
                or action != plan[0]
                or plan[-1] != TURN_PLAN_END_ACTION
            ):
                raise ValueError(
                    "final JSON turn plan must be a non-empty string list whose "
                    "first entry matches action and whose final entry is end turn"
                )
        return span
    raise ValueError("completion has no parseable JSON action object")


def _json_members(
    text: str,
    object_span: tuple[int, int],
) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    """Return top-level ``(key, key_span, value_span)`` members."""
    decoder = json.JSONDecoder()
    object_start, object_end = object_span
    index = object_start + 1
    members: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    while index < object_end - 1:
        while index < object_end and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= object_end - 1:
            break
        key_start = index
        key, key_end = decoder.raw_decode(text, index)
        if not isinstance(key, str):
            raise ValueError("final JSON action object has a non-string key")
        index = key_end
        while index < object_end and text[index].isspace():
            index += 1
        if index >= object_end or text[index] != ":":
            raise ValueError("malformed final JSON action object")
        index += 1
        while index < object_end and text[index].isspace():
            index += 1
        value_start = index
        _value, value_end = decoder.raw_decode(text, index)
        members.append((key, (key_start, key_end), (value_start, value_end)))
        index = value_end
    return members


def _character_categories(
    completion: str,
    *,
    expected_action_index: int | None,
    expected_action_description: str | None,
    output_contract: str,
    turn_terminator: str,
) -> tuple[str, list[str]]:
    target = completion
    if turn_terminator and not target.endswith(turn_terminator):
        target += turn_terminator

    # Everything outside the final structured action is private thought by
    # default. Explicit syntax/channel markers and the turn suffix are promoted
    # to format tokens below.
    categories = ["thought"] * len(target)
    semantic_output = output_contract in (ACTION_TEXT_OUTPUT, TURN_PLAN_OUTPUT)
    action_key = _SEMANTIC_ACTION_KEY if semantic_output else _ACTION_KEY
    expected_action: int | str | None = (
        expected_action_description if semantic_output else expected_action_index
    )
    object_span = _action_object_span(
        completion,
        action_key=action_key,
        expected_action=expected_action,
        output_contract=output_contract,
    )
    object_start, object_end = object_span
    if semantic_output:
        # Semantic contracts contain no optional rationale fields. Their entire
        # JSON object is required format, except for the policy-bearing string
        # value overridden to ``action`` below. In particular, turn_plan's plan
        # array is format context rather than another policy target.
        for index in range(object_start, object_end):
            categories[index] = "format"
    else:
        categories[object_start] = "format"
        categories[object_end - 1] = "format"

    found_action = False
    for key, key_span, value_span in _json_members(completion, object_span):
        if key == action_key:
            found_action = True
            # Supervise exactly the syntax needed for the policy-bearing field,
            # not keys/punctuation belonging to optional rationale/metadata.
            for index in range(key_span[0], value_span[0]):
                categories[index] = "format"
            for index in range(*value_span):
                categories[index] = "action"
    if not found_action:
        raise ValueError(f"final JSON object has no {action_key} member")

    for marker in _FORMAT_MARKER_RE.finditer(target):
        for index in range(marker.start(), marker.end()):
            categories[index] = "format"
    if turn_terminator and target.endswith(turn_terminator):
        for index in range(len(target) - len(turn_terminator), len(target)):
            categories[index] = "format"
    return target, categories


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[Any]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


def _completion_ids_and_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[Any], list[tuple[int, int]]]:
    # mlx-lm wraps HF tokenizers in a non-callable TokenizerWrapper while
    # exposing the offset-capable tokenizer as ``_tokenizer``.
    candidates = (tokenizer, getattr(tokenizer, "_tokenizer", None))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            encoded = candidate(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            input_ids = list(encoded["input_ids"])
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            if len(input_ids) == len(offsets):
                return input_ids, offsets
        except (TypeError, AttributeError, KeyError, NotImplementedError):
            continue

    input_ids = _encode(tokenizer, text, add_special_tokens=False)
    if len(input_ids) == len(text):
        return input_ids, [(index, index + 1) for index in range(len(text))]

    words = list(re.finditer(r"\S+", text))
    if len(words) == len(input_ids):
        return input_ids, [(match.start(), match.end()) for match in words]
    raise ValueError(
        "action loss requires a tokenizer that supplies offset_mapping (or a "
        "simple character/whitespace tokenizer in tests)"
    )


def _token_counts(tokenized: dict[str, Any]) -> dict[str, int]:
    keys = (
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
    return {key: int(tokenized.get(key, 0)) for key in keys}


def loss_mask_token_accounting(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-example auditable action-mask token counts."""
    totals: dict[str, int] = {}
    counted = 0
    for example in examples:
        counts = example.get("token_counts")
        if not isinstance(counts, dict):
            continue
        counted += 1
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + int(value)
    return {
        "n_examples": len(examples),
        "n_examples_counted": counted,
        "totals": totals,
    }


def tokenize_example(
    example: dict,
    tokenizer,
    *,
    loss_mask_mode: str = "completion",
) -> dict[str, Any]:
    """Tokenize one policy example with completion- or action-only labels.

    Action mode keeps the full sampled reasoning in the causal context, but only
    supervises required formatting/channel tokens, the top-level
    ``action_index`` value, and the assistant-turn terminator. This is a token
    mask, not a text rewrite, so the action is scored under the context that
    actually preceded it at generation time.
    """
    _validate_loss_mask_mode(loss_mask_mode)
    prompt_ids = _encode(tokenizer, example["prompt"], add_special_tokens=True)

    categories: list[str]
    if loss_mask_mode == "action":
        terminator = str(
            example.get("assistant_turn_terminator")
            or assistant_turn_terminator(tokenizer)
        )
        completion, char_categories = _character_categories(
            str(example["completion"]),
            expected_action_index=example.get("target_action_index"),
            expected_action_description=example.get("target_action_description"),
            output_contract=str(
                example.get("output_contract", REASONING_ACTION_OUTPUT)
            ),
            turn_terminator=terminator,
        )
        completion_ids, offsets = _completion_ids_and_offsets(tokenizer, completion)
        categories = []
        # Strict action-only invariant: a subword that contains any private
        # thought is masked, even when it also straddles adjacent JSON syntax.
        # Action payload overlap remains highest priority so a tokenizer merge
        # like ``: 12}`` still trains the selected action.
        priority = {"format": 0, "thought": 1, "action": 2}
        for start, end in offsets:
            if end <= start:
                categories.append("format")
                continue
            overlapping = char_categories[start:end]
            categories.append(max(overlapping, key=priority.__getitem__))
        completion_labels = [
            token_id if category in ("format", "action") else -100
            for token_id, category in zip(completion_ids, categories)
        ]
    else:
        completion_ids = _encode(
            tokenizer,
            example["completion"],
            add_special_tokens=False,
        )
        categories = ["thought"] * len(completion_ids)
        completion_labels = list(completion_ids)

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_labels
    n_format_tokens = categories.count("format")
    n_thought_tokens = categories.count("thought")
    n_action_tokens = categories.count("action")
    supervised_categories = [
        category
        for category, label in zip(categories, completion_labels)
        if label != -100
    ]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "action_mask": [False] * len(prompt_ids)
        + [category == "action" for category in categories],
        "format_mask": [False] * len(prompt_ids)
        + [category == "format" for category in categories],
        "thought_mask": [False] * len(prompt_ids)
        + [category == "thought" for category in categories],
        "n_prompt_tokens": len(prompt_ids),
        "n_completion_tokens": len(completion_ids),
        "n_format_tokens": n_format_tokens,
        "n_thought_tokens": n_thought_tokens,
        "n_action_tokens": n_action_tokens,
        "n_supervised_format_tokens": supervised_categories.count("format"),
        "n_supervised_thought_tokens": supervised_categories.count("thought"),
        "n_supervised_action_tokens": supervised_categories.count("action"),
        "n_supervised_tokens": len(supervised_categories),
    }
