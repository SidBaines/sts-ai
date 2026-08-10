from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sts_ai.local_tasks import base, get_task
from sts_ai.train.sft_format import (
    build_example,
    chat_template_probe_hash,
    loss_mask_token_accounting,
    resolve_loss_mask_mode,
)


def _include_window(window: dict[str, Any], label_mode: str) -> bool:
    label = str(window["label"])
    if label_mode == "won":
        return label in {"won", "convincing"}
    if label_mode == "convincing":
        return label == "convincing"
    if label_mode == "all":
        return True
    raise ValueError("label_mode must be 'won', 'convincing', or 'all'")


def _multiplicity(window: dict[str, Any], weighting_mode: str) -> int:
    if weighting_mode == "filter":
        return 1
    if weighting_mode == "rwr":
        return int(window.get("rwr_multiplicity", 0) or 0)
    raise ValueError("weighting_mode must be 'filter' or 'rwr'")


def _one_value(values: list[Any], name: str) -> Any:
    unique = set(values)
    if len(unique) > 1:
        rendered = ", ".join(repr(v) for v in sorted(unique, key=repr))
        raise ValueError(f"Refusing to mix {name} values: {rendered}")
    return values[0] if values else None


def build_local_sft_dataset(
    manifest: dict[str, Any],
    *,
    framing: str,
    tokenizer: Any,
    tokenizer_id: str,
    split: str = "train",
    label_mode: str = "won",
    weighting_mode: str = "rwr",
    require_no_thinking: bool = True,
    require_framing_match: bool = True,
    loss_mask_mode: str = "completion",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loss_mask_mode = resolve_loss_mask_mode(loss_mask_mode, manifest_path=None)
    task = get_task(str(manifest["task_id"]))
    windows = [
        window
        for window in manifest["windows"]
        if window.get("split") == split and _include_window(window, label_mode)
    ]

    metas = [base.source_meta_for_window(manifest, window) for window in windows]
    reasoning_mode = _one_value(
        [base.reasoning_mode_from_meta(meta) for meta in metas],
        "reasoning_mode",
    )
    if reasoning_mode is None:
        reasoning_mode = "none"
    if require_no_thinking and reasoning_mode not in (None, "none"):
        raise ValueError(
            "Refusing to build no-thinking local-task SFT data from "
            f"reasoning_mode={reasoning_mode!r}"
        )

    generation_framings = [meta.get("framing") for meta in metas]
    generation_framing = _one_value(generation_framings, "generation_framing")
    if require_framing_match and generation_framing is not None and generation_framing != framing:
        raise ValueError(
            "Refusing to reconstruct framing that differs from source rollout "
            f"framing: requested={framing!r}, found={generation_framing!r}"
        )

    enable_thinking = reasoning_mode == "native"
    induce_reasoning = reasoning_mode == "prompted"
    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    multiplicity_histogram: Counter[int] = Counter()
    n_unique_examples = 0
    included_window_ids: list[str] = []

    for window in windows:
        multiplicity = _multiplicity(window, weighting_mode)
        multiplicity_histogram[multiplicity] += 1
        if multiplicity <= 0:
            continue
        included_window_ids.append(str(window["window_id"]))
        window_examples: list[dict[str, Any]] = []
        for record in base.source_records_for_window(manifest, window):
            skip = base.skip_reason(record)
            if skip is not None:
                skipped[skip] += 1
                continue
            try:
                example = build_example(
                    record,
                    framing,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                    induce_reasoning=induce_reasoning,
                    loss_mask_mode=loss_mask_mode,
                )
            except ValueError:
                if loss_mask_mode != "action":
                    raise
                skipped["action_mask_unavailable"] += 1
                continue
            example.update(
                {
                    "local_task": str(manifest["task_id"]),
                    "task_window_id": window["window_id"],
                    "source_stem": window["source_stem"],
                    "task_label": window["label"],
                    "task_reward": float(window["reward"]),
                    "multiplicity": multiplicity,
                    "weighting_mode": weighting_mode,
                    "stem": window["source_stem"],
                }
            )
            window_examples.append(example)
        n_unique_examples += len(window_examples)
        for example in window_examples:
            for _ in range(multiplicity):
                examples.append(dict(example))

    split_windows = [window for window in manifest["windows"] if window.get("split") == split]
    manifest_out: dict[str, Any] = {
        "task_id": task.task_id,
        "source_manifest_task_id": manifest["task_id"],
        "source_rollout_dir": manifest["source_rollout_dir"],
        "tokenizer_id": tokenizer_id,
        "chat_template_hash": chat_template_probe_hash(
            tokenizer,
            enable_thinking=enable_thinking,
        ),
        "framing": framing,
        "generation_framing": generation_framing,
        "reasoning_mode": reasoning_mode,
        "enable_thinking": enable_thinking,
        "induce_reasoning": induce_reasoning,
        "loss_mask_mode": loss_mask_mode,
        "split": split,
        "label_mode": label_mode,
        "weighting_mode": weighting_mode,
        "n_split_windows": len(split_windows),
        "n_candidate_windows": len(windows),
        "n_included_windows": len(included_window_ids),
        "included_window_ids": included_window_ids,
        "n_unique_examples": n_unique_examples,
        "n_examples": len(examples),
        "label_counts": base.label_counts(windows),
        "multiplicity_histogram": dict(multiplicity_histogram),
        "skipped_record_counts": dict(skipped),
    }
    if loss_mask_mode == "action":
        manifest_out["token_accounting"] = loss_mask_token_accounting(examples)
    return examples, manifest_out
