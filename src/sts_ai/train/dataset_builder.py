"""Offline SFT dataset construction from recorded rollout traces."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sts_ai.train.reward import label_trajectories, rwr_multiplicities
from sts_ai.train.sft_format import build_example, chat_template_probe_hash

__all__ = ["discover_rollouts", "build_dataset"]


def discover_rollouts(rollout_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted JSONL/meta pairs directly under ``rollout_dir``."""
    rollout_dir = Path(rollout_dir)
    pairs: list[tuple[Path, Path]] = []
    for jsonl_path in sorted(rollout_dir.glob("seed_*_r*.jsonl")):
        meta_path = jsonl_path.with_suffix(".meta.json")
        if meta_path.exists():
            pairs.append((jsonl_path, meta_path))
    return pairs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _count_missing_meta(rollout_dir: Path) -> int:
    return sum(
        1
        for jsonl_path in rollout_dir.glob("seed_*_r*.jsonl")
        if not jsonl_path.with_suffix(".meta.json").exists()
    )


def _reasoning_mode(meta: dict[str, Any]) -> str:
    extra = meta.get("extra") or {}
    agent_config = extra.get("agent_config") or {}
    mode = agent_config.get("reasoning_mode")
    return "none" if mode is None else str(mode)


def _one_value(values: list[Any], *, name: str) -> Any:
    unique = set(values)
    if len(unique) > 1:
        rendered = ", ".join(repr(value) for value in sorted(unique, key=repr))
        raise ValueError(f"Refusing to mix {name} values: {rendered}")
    if not values:
        return None
    return values[0]


def _chat_template_hash(tokenizer: Any, *, enable_thinking: bool) -> str:
    return chat_template_probe_hash(tokenizer, enable_thinking=enable_thinking)


def _skip_reason(record: dict[str, Any], drop_phases: tuple[str, ...]) -> str | None:
    agent = record.get("agent") or {}
    if record.get("action_executed", True) is False:
        return "action_not_executed"
    if agent.get("valid", True) is False:
        return "agent_invalid"
    if int(agent.get("retries") or 0) > 0:
        return "agent_retried"
    if record.get("phase", "out_of_combat") in drop_phases:
        return "dropped_phase"
    return None


def build_dataset(
    rollout_dir: Path,
    *,
    framing: str,
    tokenizer: Any,
    tokenizer_id: str,
    min_act: int = 1,
    fallback_floor_quantile: float = 0.8,
    min_positives: int = 20,
    weighting_mode: str = "filter",
    rwr_beta: float = 5.0,
    rwr_baseline: str = "median",
    rwr_max_multiplier: int = 8,
    require_no_thinking: bool = True,
    require_framing_match: bool = True,
    drop_phases: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollout_dir = Path(rollout_dir)
    if weighting_mode not in {"filter", "rwr"}:
        raise ValueError("weighting_mode must be 'filter' or 'rwr'")

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
            "Refusing to build no-thinking SFT data from "
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

    labels, report = label_trajectories(
        metas,
        min_act=min_act,
        fallback_floor_quantile=fallback_floor_quantile,
        min_positives=min_positives,
    )

    rwr_report = None
    multiplicity_by_stem: dict[str, int] = {}
    if weighting_mode == "rwr":
        multiplicity_by_stem, rwr_report = rwr_multiplicities(
            labels,
            beta=rwr_beta,
            baseline=rwr_baseline,
            max_multiplier=rwr_max_multiplier,
        )

    jsonl_by_stem = {jsonl_path.stem: jsonl_path for jsonl_path, _meta_path in pairs}
    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    n_unique_examples = 0
    if weighting_mode == "filter":
        for label in labels:
            if not label.kept:
                continue
            jsonl_path = jsonl_by_stem[label.stem]
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
                example["stem"] = label.stem
                example["keep_reason"] = label.keep_reason
                examples.append(example)
        n_unique_examples = len(examples)
        n_trajectories_with_multiplicity = sum(1 for label in labels if label.kept)
    else:
        for label in labels:
            multiplicity = multiplicity_by_stem.get(label.stem, 0)
            if multiplicity == 0:
                continue
            jsonl_path = jsonl_by_stem[label.stem]
            trajectory_examples: list[dict[str, Any]] = []
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
                example["stem"] = label.stem
                example["keep_reason"] = label.keep_reason
                example["multiplicity"] = multiplicity
                example["weighting_mode"] = weighting_mode
                trajectory_examples.append(example)
            n_unique_examples += len(trajectory_examples)
            for example in trajectory_examples:
                for _copy_index in range(multiplicity):
                    # Safe shallow copy: prompt/completion strings and messages are serialized/read only.
                    examples.append(dict(example))
        n_trajectories_with_multiplicity = sum(
            1 for multiplicity in multiplicity_by_stem.values() if multiplicity >= 1
        )

    manifest = {
        "tokenizer_id": tokenizer_id,
        "chat_template_hash": _chat_template_hash(
            tokenizer,
            enable_thinking=enable_thinking,
        ),
        "framing": framing,
        "generation_framing": generation_framing,
        "reasoning_mode": reasoning_mode,
        "enable_thinking": enable_thinking,
        "induce_reasoning": induce_reasoning,
        "min_act": min_act,
        "n_rollouts_discovered": len(pairs),
        "n_missing_meta": _count_missing_meta(rollout_dir),
        "n_kept_trajectories": n_trajectories_with_multiplicity,
        "n_examples": len(examples),
        "weighting_mode": weighting_mode,
        "rwr_report": rwr_report,
        "n_unique_examples": n_unique_examples,
        "n_trajectories_with_multiplicity": n_trajectories_with_multiplicity,
        "skipped_record_counts": dict(skipped),
        "filter_report": report,
    }
    return examples, manifest
