"""Policy-gradient dataset construction from recorded rollout traces."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sts_ai.train.advantage import offline_advantages, group_relative_advantages
from sts_ai.train.dataset_builder import (
    _count_missing_meta,
    _load_json,
    _load_jsonl,
    _one_value,
    _reasoning_mode,
    _skip_reason,
    discover_rollouts,
)
from sts_ai.train.reward import label_trajectories
from sts_ai.train.sft_format import build_example, chat_template_probe_hash

__all__ = ["build_pg_dataset"]


def build_pg_dataset(
    rollout_dir: Path,
    *,
    framing: str,
    tokenizer: Any,
    tokenizer_id: str,
    mode: str = "offline",
    baseline: str = "median",
    std_norm: bool = True,
    eps: float = 1e-6,
    min_act: int = 1,
    require_no_thinking: bool = True,
    require_framing_match: bool = True,
    drop_phases: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollout_dir = Path(rollout_dir)

    pairs = discover_rollouts(rollout_dir)
    metas = [_load_json(meta_path) for _jsonl_path, meta_path in pairs]

    reasoning_mode = _one_value(
        [_reasoning_mode(meta) for meta in metas],
        name="reasoning_mode",
    )
    if reasoning_mode is None:
        reasoning_mode = "none"
    if require_no_thinking and reasoning_mode not in (None, "none"):
        raise ValueError(
            "Refusing to build no-thinking PG data from "
            f"reasoning_mode={reasoning_mode!r}"
        )

    generation_framings = [meta.get("framing") for meta in metas]
    generation_framing = generation_framings[0] if generation_framings else None
    if len(set(generation_framings)) > 1:
        generation_framing = "mixed"
    if require_framing_match:
        mismatched = [
            meta.get("framing")
            for meta in metas
            if meta.get("framing") != framing
        ]
        if mismatched:
            raise ValueError(
                "Refusing to reconstruct framing that differs from rollout "
                f"generation framing: requested={framing!r}, "
                f"found={mismatched[0]!r}"
            )

    enable_thinking = reasoning_mode == "native"
    induce_reasoning = reasoning_mode == "prompted"

    labels, label_report = label_trajectories(metas, min_act=min_act)

    if mode == "offline":
        advantage_by_stem, adv_report = offline_advantages(labels, baseline=baseline)
    elif mode == "group":
        advantage_by_stem, adv_report = group_relative_advantages(
            labels,
            std_norm=std_norm,
            eps=eps,
        )
    else:
        raise ValueError("mode must be 'offline' or 'group'")

    jsonl_by_stem = {jsonl_path.stem: jsonl_path for jsonl_path, _meta_path in pairs}
    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for label in labels:
        if label.stem not in advantage_by_stem:
            continue
        jsonl_path = jsonl_by_stem[label.stem]
        advantage = float(advantage_by_stem[label.stem])
        for record in _load_jsonl(jsonl_path):
            skip_reason = _skip_reason(record, drop_phases)
            if skip_reason is not None:
                skipped[skip_reason] += 1
                continue
            example = build_example(
                record,
                framing,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                induce_reasoning=induce_reasoning,
            )
            example["advantage"] = advantage
            example["stem"] = label.stem
            examples.append(example)

    manifest: dict[str, Any] = {
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
        "mode": mode,
        "min_act": min_act,
        "n_rollouts_discovered": len(pairs),
        "n_missing_meta": _count_missing_meta(rollout_dir),
        "n_trajectories_with_advantage": len(advantage_by_stem),
        "n_examples": len(examples),
        "advantage_report": adv_report,
        "skipped_record_counts": dict(skipped),
        "label_report": label_report,
    }
    if mode == "offline":
        manifest["baseline"] = baseline
    else:
        manifest["std_norm"] = std_norm
        manifest["eps"] = eps

    return examples, manifest
