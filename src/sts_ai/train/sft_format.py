"""Skew-free SFT prompt/completion reconstruction.

The role-based ``messages`` pair is the canonical training surface for SFT data.
The legacy ``{"prompt", "completion"}`` text pair remains available for
provenance/eval. ``tokenize_example`` is a reference helper that makes the
intended completion-only loss mask explicit and testable.
"""
from __future__ import annotations

import hashlib
from typing import Any

from sts_ai.prompting import render_action_prompt
from sts_ai.schemas import LegalAction

__all__ = [
    "chat_template_probe_hash",
    "user_content",
    "reconstruct_prompt",
    "completion_text",
    "build_example",
    "tokenize_example",
]


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
) -> str:
    legal_actions = _legal_actions_from_record(record)
    return render_action_prompt(
        record["state_text"],
        legal_actions,
        framing,
        induce_reasoning=induce_reasoning,
    )


def reconstruct_prompt(
    record: dict,
    framing: str,
    *,
    tokenizer,
    enable_thinking: bool,
    induce_reasoning: bool = False,
) -> str:
    prompt = user_content(
        record,
        framing,
        induce_reasoning=induce_reasoning,
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


def build_example(
    record: dict,
    framing: str,
    *,
    tokenizer,
    enable_thinking: bool,
    induce_reasoning: bool = False,
) -> dict:
    user_message = user_content(
        record,
        framing,
        induce_reasoning=induce_reasoning,
    )
    completion = completion_text(record)
    return {
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
        ),
        "completion": completion,
        "world_seed": record.get("world_seed"),
        "decision_index": record.get("decision_index"),
        "phase": record.get("phase", "out_of_combat"),
    }


def tokenize_example(example: dict, tokenizer) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(example["prompt"])
    try:
        completion_ids = tokenizer.encode(
            example["completion"],
            add_special_tokens=False,
        )
    except TypeError:
        completion_ids = tokenizer.encode(example["completion"])

    prompt_ids = list(prompt_ids)
    completion_ids = list(completion_ids)
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "n_prompt_tokens": len(prompt_ids),
        "n_completion_tokens": len(completion_ids),
    }
