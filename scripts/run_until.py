#!/usr/bin/env python
"""Run exactly M predetermined full-game rollouts for one model.

This script launches exactly ``--target`` rollout STARTS, keeping up to N live
(``--concurrency`` for vLLM streaming or ``--batch-size`` for MLX batching), then
lets every launched rollout finish. The stopping rule is on starts, not
completions: stopping when M completions are observed would bias the sample
toward fast rollouts because failures tend to finish faster than wins, thereby
oversampling deaths. We accept some idle GPU on the tail, when the last long
rollouts drain after no new specs remain, in exchange for an unbiased sample of
exactly M predetermined seeds. ``run_streaming_rollouts`` already drains every
finite spec it is given, so passing it a finite list of M specs gives exactly
this behavior.

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
from typing import Any

from sts_ai.seeding import rollout_stem

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
    target: int,
    seed_start: int,
    excluded: set[int] | None = None,
    already_done: set[int] | None = None,
    *,
    overwrite: bool = False,
) -> list[Spec]:
    """Generate exactly ``target`` fresh ``(world_seed, rollout_index)`` specs.

    Seeds are distinct, ascending, and use rollout index 0. Excluded seeds are
    always skipped. Existing outputs are skipped unless ``overwrite`` is true.
    """
    if target < 1:
        raise ValueError("target must be >= 1")

    blocked = set(excluded or set())
    if not overwrite:
        blocked.update(already_done or set())

    specs: list[Spec] = []
    seed = seed_start
    while len(specs) < target:
        if seed not in blocked:
            specs.append((seed, 0))
        seed += 1
    return specs


def existing_rollout_seeds(out_dir: Path) -> set[int]:
    """Return world seeds with an existing rollout-index-0 meta sidecar."""
    seeds: set[int] = set()
    for path in out_dir.glob("seed_*_r0.meta.json"):
        name = path.name
        if not name.startswith("seed_") or not name.endswith("_r0.meta.json"):
            continue
        raw_seed = name[len("seed_") : -len("_r0.meta.json")]
        try:
            seeds.add(int(raw_seed))
        except ValueError:
            continue
    return seeds


def _skip_counts(
    seed_start: int,
    last_seed: int,
    excluded: set[int],
    already_done: set[int],
) -> tuple[int, int]:
    excluded_count = 0
    existing_count = 0
    for seed in range(seed_start, last_seed + 1):
        if seed in excluded:
            excluded_count += 1
        elif seed in already_done:
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
        description="Run exactly M fresh full-game rollouts for one model."
    )
    parser.add_argument("--model", required=True, help="HF/mlx model id.")
    parser.add_argument("--backend", choices=["vllm", "mlx"], default="vllm")
    parser.add_argument("--target", type=int, required=True, help="Total fresh rollouts to launch.")
    parser.add_argument("--concurrency", type=int, default=48, help="vLLM in-flight rollout cap.")
    parser.add_argument("--batch-size", type=int, default=8, help="MLX lockstep batch size.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--exclude-seeds", default=None, help="Comma-separated world seeds to skip.")
    parser.add_argument(
        "--exclude-seeds-config",
        type=Path,
        default=None,
        help="JSON config containing excluded/errored seed lists.",
    )
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

    if args.target < 1:
        parser.error("--target must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    from sts_ai.agent_factory import agent_label, build_agent
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.parallel_rollout import run_parallel_rollouts
    from sts_ai.rollout import current_git_sha
    from sts_ai.streaming_rollout import run_streaming_rollouts

    excluded = parse_seed_list(args.exclude_seeds)
    if args.exclude_seeds_config is not None:
        excluded.update(load_excluded_seed_config(args.exclude_seeds_config))

    label = agent_label(
        args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )
    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    already_done = existing_rollout_seeds(out_dir)

    specs = generate_specs(
        args.target,
        args.seed_start,
        excluded,
        already_done,
        overwrite=args.overwrite,
    )
    last_seed = specs[-1][0]
    skipped_excluded, skipped_existing = _skip_counts(
        args.seed_start,
        last_seed,
        excluded,
        set() if args.overwrite else already_done,
    )
    effective_concurrency = args.concurrency if args.backend == "vllm" else args.batch_size
    print(
        f"[{label}] launching {len(specs)} rollouts "
        f"(target={args.target}, effective_concurrency={effective_concurrency}, "
        f"first={specs[0]}, last={specs[-1]}, skipped_excluded={skipped_excluded}, "
        f"skipped_existing={skipped_existing})"
    )
    if args.overwrite:
        _remove_existing_outputs(specs, out_dir)

    git_sha = current_git_sha()

    def make_env(seed: int) -> LightspeedHybridEnv:
        return LightspeedHybridEnv(
            world_seed=seed,
            combat_control=args.combat_control,
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )

    preserve_special_tokens = {"auto": None, "on": True, "off": False}[args.preserve_special_tokens]
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
        run_meta = {
            "git_sha": git_sha,
            "battle_simulations": args.battle_simulations,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "extra": {
                "orchestrator": "streaming" if args.backend == "vllm" else "parallel",
                "concurrency": effective_concurrency,
                "target": args.target,
                "seed_start": args.seed_start,
                "excluded_seed_count": len(excluded),
                "skipped_existing": skipped_existing,
            },
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
