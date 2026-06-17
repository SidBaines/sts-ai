"""Skew-free SFT prompt/completion reconstruction.

The ``{"prompt", "completion"}`` text pair is the source of truth for SFT data.
Production trainers should re-tokenize that pair with the real training
tokenizer. ``tokenize_example`` is a reference helper that makes the intended
completion-only loss mask explicit and testable.
"""
from __future__ import annotations

from typing import Any

from sts_ai.prompting import render_action_prompt
from sts_ai.schemas import LegalAction

__all__ = [
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


def reconstruct_prompt(
    record: dict,
    framing: str,
    *,
    tokenizer,
    enable_thinking: bool,
    induce_reasoning: bool = False,
) -> str:
    legal_actions = _legal_actions_from_record(record)
    prompt = render_action_prompt(
        record["state_text"],
        legal_actions,
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
    return {
        "prompt": reconstruct_prompt(
            record,
            framing,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            induce_reasoning=induce_reasoning,
        ),
        "completion": completion_text(record),
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
