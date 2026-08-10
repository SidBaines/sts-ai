from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics
from typing import Any

from sts_ai.local_tasks import base
from sts_ai.train.sft_format import (
    build_example,
    chat_template_probe_hash,
    loss_mask_token_accounting,
    resolve_loss_mask_mode,
)


def _discover_pairs(rollout_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for jsonl_path in sorted(Path(rollout_dir).glob("seed_*_r*.jsonl")):
        meta_path = jsonl_path.with_suffix(".meta.json")
        if meta_path.exists():
            pairs.append((jsonl_path, meta_path))
    return pairs


def _task_extra(meta: dict[str, Any]) -> dict[str, Any]:
    extra = meta.get("extra") or {}
    task = extra.get("local_task") or {}
    if not isinstance(task, dict):
        return {}
    return task


def _std(values: list[float], eps: float) -> float:
    if len(values) <= 1:
        return 1.0
    value = statistics.pstdev(values)
    return value if value > eps else 1.0


def _advantages(
    metas: list[dict[str, Any]],
    *,
    mode: str,
    std_norm: bool,
    eps: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    rewards_by_stem: dict[str, float] = {}
    groups: dict[str, list[tuple[str, float]]] = {}
    for meta in metas:
        task = _task_extra(meta)
        if not task:
            continue
        if str(meta.get("stopped_reason")) == "simulator_error":
            continue
        stem = base.rollout_stem(int(meta["world_seed"]), int(meta.get("rollout_index", 0)))
        reward = float(task.get("reward", 0.0))
        rewards_by_stem[stem] = reward
        groups.setdefault(str(task.get("window_id", stem)), []).append((stem, reward))

    advantages: dict[str, float] = {}
    if mode == "offline":
        rewards = sorted(rewards_by_stem.values())
        baseline_value = statistics.median(rewards) if rewards else 0.0
        for stem, reward in rewards_by_stem.items():
            advantages[stem] = reward - baseline_value
        report = {"mode": mode, "baseline_value": baseline_value}
    elif mode == "group":
        for rows in groups.values():
            rewards = [reward for _stem, reward in rows]
            mean = statistics.mean(rewards) if rewards else 0.0
            denom = _std(rewards, eps) if std_norm else 1.0
            for stem, reward in rows:
                advantages[stem] = (reward - mean) / denom
        report = {"mode": mode, "n_groups": len(groups), "std_norm": std_norm, "eps": eps}
    else:
        raise ValueError("mode must be 'offline' or 'group'")

    reward_values = list(rewards_by_stem.values())
    report.update(
        {
            "n_episodes": len(reward_values),
            "reward_mean": statistics.mean(reward_values) if reward_values else 0.0,
            "reward_min": min(reward_values) if reward_values else 0.0,
            "reward_max": max(reward_values) if reward_values else 0.0,
            "advantage_mean": statistics.mean(advantages.values()) if advantages else 0.0,
            "advantage_min": min(advantages.values()) if advantages else 0.0,
            "advantage_max": max(advantages.values()) if advantages else 0.0,
        }
    )
    return advantages, report


def _one_value(values: list[Any], name: str) -> Any:
    unique = set(values)
    if len(unique) > 1:
        rendered = ", ".join(repr(v) for v in sorted(unique, key=repr))
        raise ValueError(f"Refusing to mix {name} values: {rendered}")
    return values[0] if values else None


def build_local_pg_dataset(
    rollout_dir: Path,
    *,
    framing: str,
    tokenizer: Any,
    tokenizer_id: str,
    mode: str = "group",
    std_norm: bool = True,
    eps: float = 1e-6,
    require_no_thinking: bool = True,
    require_framing_match: bool = True,
    loss_mask_mode: str = "completion",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loss_mask_mode = resolve_loss_mask_mode(loss_mask_mode, manifest_path=None)
    pairs = _discover_pairs(Path(rollout_dir))
    metas = [base.load_json(meta_path) for _jsonl_path, meta_path in pairs]
    reasoning_mode = _one_value(
        [base.reasoning_mode_from_meta(meta) for meta in metas],
        "reasoning_mode",
    )
    if reasoning_mode is None:
        reasoning_mode = "none"
    if require_no_thinking and reasoning_mode not in (None, "none"):
        raise ValueError(
            "Refusing to build no-thinking local-task PG data from "
            f"reasoning_mode={reasoning_mode!r}"
        )

    generation_framing = _one_value([meta.get("framing") for meta in metas], "framing")
    if require_framing_match and generation_framing is not None and generation_framing != framing:
        raise ValueError(
            "Refusing to reconstruct framing that differs from source rollout "
            f"framing: requested={framing!r}, found={generation_framing!r}"
        )

    advantages, advantage_report = _advantages(
        metas,
        mode=mode,
        std_norm=std_norm,
        eps=eps,
    )
    enable_thinking = reasoning_mode == "native"
    induce_reasoning = reasoning_mode == "prompted"
    jsonl_by_stem = {jsonl_path.stem: jsonl_path for jsonl_path, _meta_path in pairs}

    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for stem, advantage in advantages.items():
        records = base.load_jsonl(jsonl_by_stem[stem])
        meta = next(meta for meta in metas if base.rollout_stem(int(meta["world_seed"]), int(meta.get("rollout_index", 0))) == stem)
        task = _task_extra(meta)
        for record in records:
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
                    "advantage": float(advantage),
                    "stem": stem,
                    "local_task": task.get("task_id"),
                    "task_window_id": task.get("window_id"),
                    "task_reward": float(task.get("reward", 0.0)),
                }
            )
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
        "loss_mask_mode": loss_mask_mode,
        "mode": mode,
        "std_norm": std_norm,
        "eps": eps,
        "n_rollouts_discovered": len(pairs),
        "n_episodes_with_advantage": len(advantages),
        "n_examples": len(examples),
        "advantage_report": advantage_report,
        "skipped_record_counts": dict(skipped),
    }
    if loss_mask_mode == "action":
        manifest["token_accounting"] = loss_mask_token_accounting(examples)
    return examples, manifest
