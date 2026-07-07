#!/usr/bin/env python
"""Run the in-process GRPO outer loop on CUDA/vLLM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Make the repo root importable so ``from scripts.run_until import ...`` resolves
# regardless of how this entrypoint is launched. Running ``python scripts/run_grpo.py``
# only puts ``scripts/`` (not the repo root) on sys.path, and ``PYTHONPATH=src``
# (per docs/grpo_dryrun_checklist.md and runpod/run_grpo.sh) does not add the repo
# root either — so the ``scripts`` namespace package would otherwise be unfindable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run in-process streaming GRPO.")
    parser.add_argument("--base-model", required=True, help="HF model id for vLLM and TRL.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer id.")
    parser.add_argument("--backend", choices=("cuda", "mlx"), default="cuda")
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable native thinking for the MLX agent; used by --backend mlx",
    )
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
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Max sequence length for MLX (--backend mlx) training. Lower for large models on limited unified memory "
        "— e.g. 1024-2048 for gemma-4-e4b on a 48GB Mac, which OOMs at 4096. Ignored by --backend cuda.",
    )
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--combat-control", choices=("search", "llm"), default="llm")
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU fraction. <0.9 leaves headroom so wake() after a co-resident "
        "training step does not OOM re-acquiring memory.",
    )
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=4,
        help="Training micro-batch size. >1 fills the idle GPU during the (dominant) "
        "training phase; effective batch = this * --grad-accum.",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
        help="Recompute activations in backward (numerically identical) to fit a larger "
        "training batch at long seq-len. On by default.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--std-norm", dest="std_norm", action="store_true", default=True)
    parser.add_argument("--no-std-norm", dest="std_norm", action="store_false")
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Resume: iteration index to start from (loop runs [start, num_iterations)). "
        "Pair with --resume-adapter to continue an interrupted run.",
    )
    parser.add_argument(
        "--resume-adapter",
        default=None,
        help="Resume: local path to the latest completed adapter to seed training from "
        "(e.g. downloaded from HF grpo/iter_<N>/adapter). Use with --start-iteration N+1.",
    )
    parser.add_argument("--wandb-project", default=None, help="wandb project for the per-iteration dashboard.")
    parser.add_argument("--run-name", default=None, help="wandb run name.")
    parser.add_argument("--hf-repo", default=None, help="HF model repo id to push adapters/datasets to (e.g. user/sts-grpo).")
    parser.add_argument("--hf-private", dest="hf_private", action="store_true", default=True)
    parser.add_argument("--no-hf-private", dest="hf_private", action="store_false")

    args = parser.parse_args(argv)
    if args.thinking and args.backend == "cuda":
        parser.error("--thinking applies only to --backend mlx")
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
    if args.per_device_batch_size < 1:
        parser.error("--per-device-batch-size must be >= 1")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.start_iteration < 0:
        parser.error("--start-iteration must be >= 0")
    if args.start_iteration >= args.num_iterations:
        parser.error("--start-iteration must be < --num-iterations")
    if args.start_iteration > 0 and args.resume_adapter is None:
        parser.error("--start-iteration > 0 requires --resume-adapter (the latest adapter to continue from)")
    return args


def _resolve_framing(value: str) -> str:
    if value == "neutral":
        from sts_ai.prompting import NEUTRAL_FRAME

        return NEUTRAL_FRAME
    return value


def _select_agent_and_overrides(
    args: argparse.Namespace,
    framing: str,
) -> tuple[object, dict[str, object]]:
    if args.backend == "mlx":
        from sts_ai.train.mlx_grpo import build_mlx_backend

        backend = build_mlx_backend(
            base_model=args.base_model,
            framing=framing,
            thinking=args.thinking,
            temperature=args.temperature,
            max_seq_len=args.max_seq_len,
            resume_adapter=args.resume_adapter,
        )
        return backend.agent, {
            "run_streaming_fn": backend.run_fn,
            "train_fn": backend.train_fn,
            "build_dataset_fn": backend.build_dataset_fn,
        }

    from sts_ai.agents import VllmJsonAgent

    agent = VllmJsonAgent(
        model_id=args.base_model,
        framing=framing,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_lora=True,
        enable_sleep_mode=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return agent, {}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from scripts.run_until import load_split_seeds
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.train import grpo_loop
    from transformers import AutoTokenizer

    train_seeds = load_split_seeds(args.train_seeds_config, args.train_split)
    framing = _resolve_framing(args.framing)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    agent, extra_kwargs = _select_agent_and_overrides(args, framing)

    def make_env(seed: int) -> LightspeedHybridEnv:
        return LightspeedHybridEnv(
            world_seed=seed,
            combat_control=args.combat_control,
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )

    wandb_config = {
        "backend": args.backend,
        "base_model": args.base_model,
        "thinking": args.thinking,
        "train_split": args.train_split,
        "num_iterations": args.num_iterations,
        "group_size": args.group_size,
        "seeds_per_iter": args.seeds_per_iter,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_seq_len": args.max_seq_len,
        "max_act": args.max_act,
        "max_decisions": args.max_decisions,
        "battle_simulations": args.battle_simulations,
        "combat_control": args.combat_control,
        "clip_eps": args.clip_eps,
        "kl_beta": args.kl_beta,
        "learning_rate": args.learning_rate,
        "std_norm": args.std_norm,
        "framing": args.framing,
        "n_train_seeds": len(train_seeds),
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum": args.grad_accum,
        "gradient_checkpointing": args.gradient_checkpointing,
        "start_iteration": args.start_iteration,
        "resumed": args.resume_adapter is not None,
    }

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
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        start_iteration=args.start_iteration,
        init_adapter_path=args.resume_adapter,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        wandb_config=wandb_config,
        hf_repo=args.hf_repo,
        hf_private=args.hf_private,
        **extra_kwargs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
