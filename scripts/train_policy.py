#!/usr/bin/env python
"""Dispatch LoRA policy training to an optional backend."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _default_manifest(dataset: Path) -> Path | None:
    manifest = dataset.with_suffix(".manifest.json")
    return manifest if manifest.exists() else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA policy adapter.")
    parser.add_argument("--backend", choices=("mlx", "trl"), required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)

    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps-per-eval", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="TRL backend: cap optimizer steps (>0 overrides --epochs). -1 = full epochs.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    parser.add_argument("--eval-steps", type=int, default=50)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = _default_manifest(args.dataset)
    return args


def dispatch(args: argparse.Namespace) -> Path:
    if args.backend == "mlx":
        from sts_ai.train import train_mlx

        return train_mlx.train(
            args.dataset,
            args.base_model,
            args.out,
            num_layers=args.num_layers,
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            manifest_path=args.manifest,
            wandb_project=args.wandb_project,
            steps_per_eval=args.steps_per_eval,
            steps_per_report=args.steps_per_report,
            save_every=args.save_every,
            val_batches=args.val_batches,
        )

    from sts_ai.train import train_trl

    return train_trl.train(
        args.dataset,
        args.base_model,
        args.out,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        manifest_path=args.manifest,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        eval_fraction=args.eval_fraction,
        eval_steps=args.eval_steps,
    )


def main(argv: Sequence[str] | None = None) -> None:
    adapter_dir = dispatch(parse_args(argv))
    print(adapter_dir)


if __name__ == "__main__":
    main()
