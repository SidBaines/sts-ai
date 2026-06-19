#!/usr/bin/env python
"""Run predetermined full-game rollout specs for one model.

In fresh mode, this script launches exactly ``--target`` world seeds, each with
``--rollouts-per-seed`` rollout starts, keeping up to N live (``--concurrency``
for vLLM streaming or ``--batch-size`` for MLX batching), then lets every
launched rollout finish. Split mode draws exact frozen world seeds from
``--seeds-config``/``--split``. The stopping rule is on starts, not completions:
stopping when M completions are observed would bias the sample toward fast
rollouts because failures tend to finish faster than wins, thereby oversampling
deaths. We accept some idle GPU on the tail, when the last long rollouts drain
after no new specs remain, in exchange for an unbiased sample of predetermined
seeds. ``run_streaming_rollouts`` already drains every finite spec it is given,
so passing it a finite list of specs gives exactly this behavior.

Example:
    PYTHONPATH=src .venv/bin/python scripts/run_until.py \
        --model Qwen/Qwen3-4B --backend vllm --target 500 --concurrency 48 \
        --exclude-seeds-config configs/frozen_seeds.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import gc
import json
from pathlib import Path
from typing import Any, Iterable

from sts_ai.seeding import expand_specs, rollout_stem

Spec = tuple[int, int]


def parse_seed_list(value: str | None) -> set[int]:
    """Parse a comma-separated seed list."""
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _collect_ints(value: Any) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, list):
        seeds: set[int] = set()
        for item in value:
            seeds.update(_collect_ints(item))
        return seeds
    if isinstance(value, dict):
        seeds: set[int] = set()
        for item in value.values():
            seeds.update(_collect_ints(item))
        return seeds
    return set()


def load_excluded_seed_config(path: Path) -> set[int]:
    """Read excluded/errored seeds from a tolerant JSON config shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()

    seeds: set[int] = set()
    for key in (
        "excluded",
        "exclude",
        "excluded_seeds",
        "errored",
        "errored_seeds",
        "errored_any_agent",
    ):
        if key in payload:
            seeds.update(_collect_ints(payload[key]))
    return seeds


def generate_specs(
    target: int | None,
    seed_start: int,
    excluded: set[int] | None = None,
    already_done_specs: set[Spec] | None = None,
    *,
    rollouts_per_seed: int = 1,
    seeds: Iterable[int] | None = None,
    overwrite: bool = False,
) -> list[Spec]:
    """Generate fresh ``(world_seed, rollout_index)`` specs.

    In fresh mode, ``target`` counts world seeds, not rollout pairs. In split
    mode, ``seeds`` fixes the exact candidate world seeds and ``target`` is
    ignored. Excluded world seeds are always skipped. Existing rollout pairs are
    skipped unless ``overwrite`` is true.
    """
    if rollouts_per_seed < 1:
        raise ValueError("rollouts_per_seed must be >= 1")

    excluded = set(excluded or set())
    blocked = set() if overwrite else set(already_done_specs or set())

    if seeds is not None:
        world_seeds = [seed for seed in sorted(set(seeds)) if seed not in excluded]
        return [
            spec
            for spec in expand_specs(world_seeds, rollouts_per_seed)
            if spec not in blocked
        ]

    if target is None or target < 1:
        raise ValueError("target must be >= 1")

    specs: list[Spec] = []
    seed = seed_start
    counted_world_seeds = 0
    while counted_world_seeds < target:
        if seed not in excluded:
            seed_specs = expand_specs([seed], rollouts_per_seed)
            fresh_specs = [spec for spec in seed_specs if spec not in blocked]
            if fresh_specs:
                specs.extend(fresh_specs)
                counted_world_seeds += 1
        seed += 1
    return specs


def _parse_rollout_meta_name(name: str) -> Spec | None:
    if not name.startswith("seed_") or not name.endswith(".meta.json"):
        return None
    stem = name[len("seed_") : -len(".meta.json")]
    try:
        raw_seed, raw_rollout = stem.rsplit("_r", 1)
        return int(raw_seed), int(raw_rollout)
    except ValueError:
        return None


def existing_rollout_specs(out_dir: Path) -> set[Spec]:
    """Return rollout specs with an existing meta sidecar."""
    specs: set[Spec] = set()
    for path in out_dir.glob("seed_*_r*.meta.json"):
        spec = _parse_rollout_meta_name(path.name)
        if spec is not None:
            specs.add(spec)
    return specs


def load_split_seeds(config_path: Path, split: str) -> list[int]:
    """Read a named frozen seed split from a JSON config."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object")

    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"{config_path} must contain a 'splits' object")
    if split not in splits:
        raise KeyError(f"split {split!r} not found in {config_path}")

    split_seeds = splits[split]
    if not isinstance(split_seeds, list):
        raise ValueError(f"split {split!r} in {config_path} must be a list of ints")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in split_seeds):
        raise ValueError(f"split {split!r} in {config_path} must be a list of ints")
    return sorted(split_seeds)


def _skip_counts(
    candidate_world_seeds: Iterable[int],
    excluded: set[int],
    already_done_specs: set[Spec],
    *,
    rollouts_per_seed: int,
) -> tuple[int, int]:
    excluded_count = 0
    existing_count = 0
    for seed in sorted(set(candidate_world_seeds)):
        if seed in excluded:
            excluded_count += 1
            continue
        for rollout_index in range(rollouts_per_seed):
            if (seed, rollout_index) in already_done_specs:
                existing_count += 1
    return excluded_count, existing_count


def _remove_existing_outputs(specs: list[Spec], out_dir: Path) -> None:
    for world_seed, rollout_index in specs:
        output_path = out_dir / f"{rollout_stem(world_seed, rollout_index)}.jsonl"
        for path in (
            output_path,
            output_path.with_suffix(".meta.json"),
            output_path.with_suffix(".error.json"),
        ):
            if path.exists():
                path.unlink()


def _outcome_counts(results: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        outcome = str((result.terminal_state or {}).get("outcome", ""))
        if "VICTORY" in outcome:
            counts["victories"] += 1
        elif "PLAYER_LOSS" in outcome or "LOSS" in outcome or "DEATH" in outcome:
            counts["deaths"] += 1
        elif result.stopped_reason == "max_decisions":
            counts["budget_truncated"] += 1
        elif result.stopped_reason == "agent_invalid":
            counts["agent_invalid"] += 1
        elif result.stopped_reason == "simulator_error":
            counts["simulator_error"] += 1
        elif "UNDECIDED" in outcome:
            counts["undecided"] += 1
        else:
            counts["other"] += 1
    return counts


def print_summary(specs: list[Spec], results: list[Any]) -> None:
    stopped = Counter(result.stopped_reason for result in results)
    outcomes = _outcome_counts(results)
    budget_truncated = sum(1 for result in results if result.stopped_reason == "max_decisions")

    print(f"[summary] launched: {len(specs)}")
    print(f"[summary] completed: {len(results)}")
    print(
        "[summary] stopped_reason: "
        + ", ".join(f"{key}: {stopped[key]}" for key in sorted(stopped))
    )
    outcome_keys = [
        "victories",
        "deaths",
        "budget_truncated",
        "agent_invalid",
        "simulator_error",
        "undecided",
        "other",
    ]
    print(
        "[summary] outcomes: "
        + ", ".join(f"{key}: {outcomes[key]}" for key in outcome_keys)
    )
    if budget_truncated:
        print(
            f"WARNING: {budget_truncated} rollouts hit --max-decisions; "
            "their outcomes are budget-truncated, not final."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run predetermined full-game rollout specs for one model: either "
            "--target fresh ascending world seeds, or a frozen --split, each "
            "expanded to --rollouts-per-seed rollouts."
        )
    )
    parser.add_argument("--model", required=True, help="HF/mlx model id.")
    parser.add_argument("--backend", choices=["vllm", "mlx"], default="vllm")
    parser.add_argument(
        "--target",
        type=int,
        required=False,
        default=None,
        help="Total fresh world seeds to launch. Required unless --split is set.",
    )
    parser.add_argument("--concurrency", type=int, default=48, help="vLLM in-flight rollout cap.")
    parser.add_argument("--batch-size", type=int, default=8, help="MLX lockstep batch size.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--rollouts-per-seed",
        type=int,
        default=1,
        help="Number of rollout indices to launch for each selected world seed.",
    )
    parser.add_argument("--exclude-seeds", default=None, help="Comma-separated world seeds to skip.")
    parser.add_argument(
        "--exclude-seeds-config",
        type=Path,
        default=None,
        help="JSON config containing excluded/errored seed lists.",
    )
    parser.add_argument(
        "--seeds-config",
        type=Path,
        default=None,
        help="JSON config containing frozen seed splits and optional exclusions.",
    )
    parser.add_argument("--split", default=None, help="Named split in --seeds-config to run.")
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--combat-control", choices=["search", "llm"], default="llm")
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--preserve-special-tokens",
        choices=["auto", "on", "off"],
        default="auto",
        help="vLLM-only: preserve native special tokens in completions (auto = native thinking only).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="vLLM top-p (nucleus) sampling (1.0 = disabled). E.g. Gemma uses 0.95.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="vLLM top-k sampling (-1 = disabled). E.g. Gemma uses 64.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Generation cap. Keep high for reasoning models so JSON is not truncated.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Invalid-response retries before stopping a rollout with agent_invalid.",
    )
    parser.add_argument(
        "--hints",
        choices=["off", "on"],
        default="off",
        help="vLLM streaming only: enable tactical hint re-decide/launder stages.",
    )
    parser.add_argument(
        "--hint-block-hp-fraction",
        type=float,
        default=1.0,
        help="Hint when full block is possible and incoming damage reaches this HP fraction.",
    )
    parser.add_argument(
        "--hint-on-launder-fail",
        choices=["action_only", "drop"],
        default="action_only",
        help="Hint behavior when the laundering response does not select the target action.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM prefix caching. vLLM-only and numerically transparent.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="LoRA adapter dir to load on top of --model for eval.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=16,
        help="vLLM max LoRA rank; must be >= the trained adapter's rank.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "rollouts" / "run_until")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.hints == "on" and args.backend == "mlx":
        parser.error("--hints on requires the vllm backend (hinting runs on the streaming path only)")
    if args.split is not None and args.seeds_config is None:
        parser.error("--split requires --seeds-config")
    if args.split is None:
        if args.target is None:
            parser.error("--target is required unless --split is provided")
        if args.target < 1:
            parser.error("--target must be >= 1")
    if args.rollouts_per_seed < 1:
        parser.error("--rollouts-per-seed must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    from sts_ai.agent_factory import agent_label

    excluded = parse_seed_list(args.exclude_seeds)
    if args.exclude_seeds_config is not None:
        excluded.update(load_excluded_seed_config(args.exclude_seeds_config))
    elif args.seeds_config is not None:
        excluded.update(load_excluded_seed_config(args.seeds_config))

    split_seeds: list[int] | None = None
    if args.split is not None:
        split_seeds = load_split_seeds(args.seeds_config, args.split)

    label = agent_label(
        args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )
    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    already_done_specs = existing_rollout_specs(out_dir)

    if split_seeds is not None:
        specs = generate_specs(
            None,
            args.seed_start,
            excluded,
            already_done_specs,
            rollouts_per_seed=args.rollouts_per_seed,
            seeds=split_seeds,
            overwrite=args.overwrite,
        )
        candidate_world_seeds: Iterable[int] = split_seeds
        selection = f"split={args.split}"
    else:
        specs = generate_specs(
            args.target,
            args.seed_start,
            excluded,
            already_done_specs,
            rollouts_per_seed=args.rollouts_per_seed,
            overwrite=args.overwrite,
        )
        candidate_world_seeds = range(args.seed_start, specs[-1][0] + 1)
        selection = f"target={args.target}"

    if not specs:
        print(
            f"[{label}] no rollouts to launch "
            f"({selection}, rollouts_per_seed={args.rollouts_per_seed}; "
            "all selected specs already exist or are excluded)"
        )
        return

    visible_done_specs = set() if args.overwrite else already_done_specs
    skipped_excluded, skipped_existing = _skip_counts(
        candidate_world_seeds,
        excluded,
        visible_done_specs,
        rollouts_per_seed=args.rollouts_per_seed,
    )
    effective_concurrency = args.concurrency if args.backend == "vllm" else args.batch_size
    print(
        f"[{label}] launching {len(specs)} rollouts "
        f"({selection}, rollouts_per_seed={args.rollouts_per_seed}, "
        f"effective_concurrency={effective_concurrency}, "
        f"first={specs[0]}, last={specs[-1]}, skipped_excluded={skipped_excluded}, "
        f"skipped_existing_pairs={skipped_existing})"
    )
    if args.overwrite:
        _remove_existing_outputs(specs, out_dir)

    from sts_ai.agent_factory import build_agent
    from sts_ai.hinting import HintConfig
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.parallel_rollout import run_parallel_rollouts
    from sts_ai.rollout import current_git_sha
    from sts_ai.streaming_rollout import run_streaming_rollouts

    git_sha = current_git_sha()

    def make_env(seed: int) -> LightspeedHybridEnv:
        return LightspeedHybridEnv(
            world_seed=seed,
            combat_control=args.combat_control,
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )

    preserve_special_tokens = {"auto": None, "on": True, "off": False}[args.preserve_special_tokens]
    hint_cfg = HintConfig(
        enabled=args.hints == "on",
        full_block_hp_fraction=args.hint_block_hp_fraction,
        on_launder_fail=args.hint_on_launder_fail,
    )
    agent = build_agent(
        args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_retries=args.max_retries,
        thinking=args.thinking,
        preserve_special_tokens=preserve_special_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        adapter_path=args.adapter_path,
        max_lora_rank=args.max_lora_rank,
    )
    try:
        extra: dict[str, Any] = {
            "orchestrator": "streaming" if args.backend == "vllm" else "parallel",
            "concurrency": effective_concurrency,
            "target": None if args.split is not None else args.target,
            "seed_start": args.seed_start,
            "rollouts_per_seed": args.rollouts_per_seed,
            "excluded_seed_count": len(excluded),
            "skipped_existing_pairs": skipped_existing,
            "hints": args.hints,
            "hint_block_hp_fraction": args.hint_block_hp_fraction,
            "hint_on_launder_fail": args.hint_on_launder_fail,
        }
        if args.split is not None:
            extra["split"] = args.split

        run_meta = {
            "git_sha": git_sha,
            "battle_simulations": args.battle_simulations,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "extra": extra,
        }
        if args.backend == "vllm":
            results = run_streaming_rollouts(
                specs,
                make_env,
                agent,
                output_for=lambda ws, ri, d=out_dir: d / f"{rollout_stem(ws, ri)}.jsonl",
                concurrency=args.concurrency,
                max_decisions=args.max_decisions,
                run_meta=run_meta,
                hint_cfg=hint_cfg,
            )
        else:
            results = run_parallel_rollouts(
                specs,
                make_env,
                agent,
                output_for=lambda ws, ri, d=out_dir: d / f"{rollout_stem(ws, ri)}.jsonl",
                batch_size=args.batch_size,
                max_decisions=args.max_decisions,
                max_retries=args.max_retries,
                run_meta=run_meta,
            )
        print_summary(specs, results)
    finally:
        if args.backend == "vllm":
            del agent
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
