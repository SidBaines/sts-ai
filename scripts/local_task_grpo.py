#!/usr/bin/env python
"""Run MLX GRPO on a detachable local-curriculum task."""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-task MLX GRPO.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", choices=("mlx",), default="mlx")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seeds-per-iter", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.num_iterations < 1:
        parser.error("--num-iterations must be >= 1")
    if args.group_size < 1:
        parser.error("--group-size must be >= 1")
    if args.seeds_per_iter < 1:
        parser.error("--seeds-per-iter must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.start_iteration < 0:
        parser.error("--start-iteration must be >= 0")
    if args.start_iteration >= args.num_iterations:
        parser.error("--start-iteration must be < --num-iterations")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from transformers import AutoTokenizer

    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.local_tasks import base, get_task
    from sts_ai.local_tasks.pg_dataset import build_local_pg_dataset
    from sts_ai.local_tasks.runner import (
        one_window_per_seed,
        replay_task_start,
        run_local_task_episodes,
        windows_for_split,
    )
    from sts_ai.prompting import NEUTRAL_FRAME
    from sts_ai.train import grpo_loop
    from sts_ai.train.mlx_grpo import build_mlx_backend

    manifest = base.load_manifest(args.manifest)
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(f"manifest task_id={manifest['task_id']!r} != --task {args.task!r}")

    windows = windows_for_split(manifest, args.train_split)
    windows_by_seed = one_window_per_seed(windows)
    train_seeds = sorted(windows_by_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    framing = NEUTRAL_FRAME
    backend = build_mlx_backend(
        base_model=args.base_model,
        framing=framing,
        thinking=args.thinking,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_seq_len=args.max_seq_len,
        resume_adapter=args.init_adapter,
    )

    def make_env(seed: int) -> LightspeedHybridEnv:
        env = LightspeedHybridEnv(
            world_seed=seed,
            combat_control="llm",
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )
        replay_task_start(env, windows_by_seed[int(seed)])
        return env

    run_fn = functools.partial(
        run_local_task_episodes,
        task=task,
        windows_by_seed=windows_by_seed,
    )
    build_dataset_fn = functools.partial(
        build_local_pg_dataset,
        require_no_thinking=False,
    )
    summary = grpo_loop.run_grpo(
        agent=backend.agent,
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
        start_iteration=args.start_iteration,
        init_adapter_path=args.init_adapter,
        concurrency=args.concurrency,
        max_decisions=args.max_decisions,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        gradient_checkpointing=False,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        wandb_config={
            "local_task": task.task_id,
            "source_manifest": str(args.manifest),
            "backend": args.backend,
            "train_split": args.train_split,
            "num_train_windows": len(train_seeds),
            "init_adapter": args.init_adapter,
        },
        build_dataset_fn=build_dataset_fn,
        train_fn=backend.train_fn,
        run_streaming_fn=run_fn,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
