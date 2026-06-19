#!/usr/bin/env python
"""Run the in-process GRPO outer loop on CUDA/vLLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run in-process streaming GRPO.")
    parser.add_argument("--base-model", required=True, help="HF model id for vLLM and TRL.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer id.")
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--train-seeds-config", type=Path, required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seeds-per-iter", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--combat-control", choices=("search", "llm"), default="llm")
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--std-norm", dest="std_norm", action="store_true", default=True)
    parser.add_argument("--no-std-norm", dest="std_norm", action="store_false")

    args = parser.parse_args(argv)
    if args.num_iterations < 1:
        parser.error("--num-iterations must be >= 1")
    if args.group_size < 1:
        parser.error("--group-size must be >= 1")
    if args.seeds_per_iter < 1:
        parser.error("--seeds-per-iter must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.max_decisions < 1:
        parser.error("--max-decisions must be >= 1")
    return args


def _resolve_framing(value: str) -> str:
    if value == "neutral":
        from sts_ai.prompting import NEUTRAL_FRAME

        return NEUTRAL_FRAME
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from scripts.run_until import load_split_seeds
    from sts_ai.agents import VllmJsonAgent
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.train import grpo_loop
    from transformers import AutoTokenizer

    train_seeds = load_split_seeds(args.train_seeds_config, args.train_split)
    framing = _resolve_framing(args.framing)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    agent = VllmJsonAgent(
        model_id=args.base_model,
        framing=framing,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_lora=True,
        enable_sleep_mode=True,
    )

    def make_env(seed: int) -> LightspeedHybridEnv:
        return LightspeedHybridEnv(
            world_seed=seed,
            combat_control=args.combat_control,
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )

    summary = grpo_loop.run_grpo(
        agent=agent,
        make_env=make_env,
        base_model=args.base_model,
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer,
        framing=framing,
        train_seeds=train_seeds,
        out_dir=args.out_dir,
        num_iterations=args.num_iterations,
        group_size=args.group_size,
        seeds_per_iter=args.seeds_per_iter,
        concurrency=args.concurrency,
        max_decisions=args.max_decisions,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        learning_rate=args.learning_rate,
        std_norm=args.std_norm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
