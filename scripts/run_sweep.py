#!/usr/bin/env python
"""Batched multi-model rollout sweep.

Runs models x {thinking on/off} x seeds through the cross-rollout batched
orchestrator (sts_ai.parallel_rollout), writing per-rollout JSONL + meta sidecars
under <output-dir>/<agent_label>/. One model load is reused across both thinking
modes (thinking is a per-call chat-template toggle).

Example:
    PYTHONPATH=src .venv/bin/python scripts/run_sweep.py \
        --models mlx-community/Qwen3-1.7B-4bit,mlx-community/Qwen3-4B-4bit \
        --thinking both --seeds 3,4,5 --batch-size 8 --max-decisions 200
"""
from __future__ import annotations

import argparse
import datetime
import gc
from pathlib import Path

from sts_ai.agent_factory import agent_label, build_agent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.parallel_rollout import run_parallel_rollouts
from sts_ai.rollout import current_git_sha
from sts_ai.seeding import expand_specs, rollout_stem

_THINKING_MODES = {"off": [False], "on": [True], "both": [False, True]}


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched multi-model rollout sweep.")
    parser.add_argument("--models", required=True, help="Comma-separated HF/mlx model ids.")
    parser.add_argument("--backend", choices=["mlx", "vllm"], default="mlx")
    parser.add_argument("--thinking", choices=list(_THINKING_MODES), default="off")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--rollouts-per-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument("--combat-control", choices=["search", "llm"], default="llm")
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--max-act", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Generation cap. Must be large for reasoning/thinking models — a small "
                        "cap (e.g. 256) truncates mid-thought so no JSON is emitted and the agent "
                        "falls back to action 0. Harmless for no-thinking (stops at EOS first).")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "rollouts" / "sweep")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args)
    try:
        specs = expand_specs(seeds, args.rollouts_per_seed)
    except ValueError as exc:
        raise SystemExit(f"--{str(exc).replace('_', '-')}") from exc
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.backend == "vllm" and len(models) > 1:
        print("WARNING: each vLLM model loads fresh in-process and vLLM does not free GPU memory between loads; the robust path is one model per process, e.g. invoke run_sweep once per model.")
    modes = _THINKING_MODES[args.thinking]
    git_sha = current_git_sha()

    def make_env(seed: int) -> LightspeedHybridEnv:
        return LightspeedHybridEnv(
            world_seed=seed,
            combat_control=args.combat_control,
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )

    for model in models:
        # One load per model id; both thinking modes reuse the same weights.
        try:
            agent = build_agent(args.backend, model=model, max_tokens=args.max_tokens,
                                temperature=args.temperature, thinking=modes[0])
        except Exception as exc:  # noqa: BLE001 - skip a model that won't load, keep the sweep going
            print(f"[{model}] FAILED to load ({exc.__class__.__name__}: {exc}); skipping")
            continue
        for thinking in modes:
            agent.enable_thinking = thinking
            label = agent_label(args.backend, model=model, max_tokens=args.max_tokens, thinking=thinking)
            out_dir = args.output_dir / label
            out_dir.mkdir(parents=True, exist_ok=True)
            to_run = [
                (world_seed, rollout_index)
                for world_seed, rollout_index in specs
                if args.overwrite
                or not (out_dir / f"{rollout_stem(world_seed, rollout_index)}.meta.json").exists()
            ]
            print(
                f"[{label}] specs={len(to_run)} rollouts (e.g. {to_run[:4]}...) "
                f"(skipped {len(specs) - len(to_run)} existing)"
            )
            if not to_run:
                continue
            run_meta = {
                "git_sha": git_sha,
                "battle_simulations": args.battle_simulations,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            results = run_parallel_rollouts(
                to_run, make_env, agent,
                output_for=lambda ws, ri, d=out_dir: d / f"{rollout_stem(ws, ri)}.jsonl",
                batch_size=args.batch_size,
                max_decisions=args.max_decisions,
                run_meta=run_meta,
            )
            wins = sum(1 for r in results if "VICTORY" in str(r.terminal_state.get("outcome", "")))
            print(f"[{label}] done: {len(results)} rollouts, {wins} wins")
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
