#!/usr/bin/env python
"""Dispatch LoRA policy training to an optional backend."""
from __future__ import annotations

import argparse
import math
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
    parser.add_argument(
        "--loss-mask",
        choices=("auto", "action", "completion"),
        default="auto",
        help="Default 'auto' reads loss_mask_mode from the manifest and treats "
        "legacy manifests as historical completion loss.",
    )

    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps-per-eval", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument(
        "--expected-example-count",
        type=int,
        default=None,
        help=(
            "MLX strict search-teacher gate: require this many manifest rows and "
            "this many examples retained after tokenization."
        ),
    )
    parser.add_argument("--mlx-seed", type=int, default=None)
    parser.add_argument("--mlx-lora-rank", type=int, default=None)
    parser.add_argument("--mlx-lora-scale", type=float, default=None)
    parser.add_argument("--mlx-lora-dropout", type=float, default=None)
    parser.add_argument("--mlx-grad-accum", type=int, default=None)
    parser.add_argument(
        "--mlx-model-revision",
        default=None,
        help=(
            "Strict search-teacher MLX only: require and load this exact "
            "40-character cached Hugging Face model revision."
        ),
    )
    parser.add_argument(
        "--mlx-action-token-only",
        action="store_true",
        help=(
            "Strict search-teacher MLX only: supervise only tokens categorized "
            "as the action_index value, retaining format tokens as causal context."
        ),
    )
    parser.add_argument(
        "--mlx-action-token-weight",
        type=float,
        default=None,
        help=(
            "Strict search-teacher MLX only: relative loss weight for tokens "
            "categorized as the action_index value (default: 1.0). Format "
            "tokens retain weight 1.0."
        ),
    )
    parser.add_argument(
        "--mlx-preserve-row-order",
        action="store_true",
        help=(
            "Strict search-teacher MLX only, batch size 1: keep the dataset's "
            "pre-shuffled row order through preprocessing and every train pass "
            "(no length sort or iterator shuffle)."
        ),
    )

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
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Fraction reserved for validation (default: 0.1 for MLX, 0.0 for TRL).",
    )
    parser.add_argument("--eval-steps", type=int, default=50)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args(argv)
    if args.eval_fraction is None:
        args.eval_fraction = 0.1 if args.backend == "mlx" else 0.0
    if not 0.0 <= args.eval_fraction < 1.0:
        parser.error("--eval-fraction must be in [0.0, 1.0)")
    if args.expected_example_count is not None:
        if args.backend != "mlx":
            parser.error("--expected-example-count is supported only by --backend mlx")
        if args.expected_example_count <= 0:
            parser.error("--expected-example-count must be positive")
    mlx_knobs = {
        "--mlx-seed": args.mlx_seed,
        "--mlx-lora-rank": args.mlx_lora_rank,
        "--mlx-lora-scale": args.mlx_lora_scale,
        "--mlx-lora-dropout": args.mlx_lora_dropout,
        "--mlx-grad-accum": args.mlx_grad_accum,
        "--mlx-model-revision": args.mlx_model_revision,
        "--mlx-action-token-weight": args.mlx_action_token_weight,
    }
    if args.backend != "mlx" and any(value is not None for value in mlx_knobs.values()):
        parser.error("--mlx-* options are supported only by --backend mlx")
    if args.backend != "mlx" and args.mlx_action_token_only:
        parser.error("--mlx-action-token-only is supported only by --backend mlx")
    if args.backend != "mlx" and args.mlx_preserve_row_order:
        parser.error("--mlx-preserve-row-order is supported only by --backend mlx")
    if args.backend == "mlx":
        args.mlx_seed = 0 if args.mlx_seed is None else args.mlx_seed
        args.mlx_lora_rank = 8 if args.mlx_lora_rank is None else args.mlx_lora_rank
        args.mlx_lora_scale = (
            20.0 if args.mlx_lora_scale is None else args.mlx_lora_scale
        )
        args.mlx_lora_dropout = (
            0.0 if args.mlx_lora_dropout is None else args.mlx_lora_dropout
        )
        args.mlx_grad_accum = (
            1 if args.mlx_grad_accum is None else args.mlx_grad_accum
        )
        args.mlx_action_token_weight = (
            1.0
            if args.mlx_action_token_weight is None
            else args.mlx_action_token_weight
        )
        if args.mlx_seed < 0:
            parser.error("--mlx-seed must be non-negative")
        if args.mlx_lora_rank <= 0:
            parser.error("--mlx-lora-rank must be positive")
        if not args.mlx_lora_scale > 0:
            parser.error("--mlx-lora-scale must be positive")
        if not 0.0 <= args.mlx_lora_dropout < 1.0:
            parser.error("--mlx-lora-dropout must be in [0.0, 1.0)")
        if args.mlx_grad_accum <= 0:
            parser.error("--mlx-grad-accum must be positive")
        if (
            not math.isfinite(args.mlx_action_token_weight)
            or args.mlx_action_token_weight <= 0
        ):
            parser.error("--mlx-action-token-weight must be finite and positive")
        if args.mlx_action_token_only and args.mlx_action_token_weight != 1.0:
            parser.error(
                "--mlx-action-token-weight must be 1 with "
                "--mlx-action-token-only"
            )
        if args.mlx_model_revision is not None and (
            len(args.mlx_model_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in args.mlx_model_revision
            )
        ):
            parser.error(
                "--mlx-model-revision must be a lowercase 40-character commit"
            )
        if args.mlx_preserve_row_order and args.batch_size != 1:
            parser.error("--mlx-preserve-row-order requires --batch-size 1")
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
            max_seq_length=args.max_seq_len,
            loss_mask_mode=args.loss_mask,
            valid_fraction=args.eval_fraction,
            expected_example_count=args.expected_example_count,
            seed=args.mlx_seed,
            lora_rank=args.mlx_lora_rank,
            lora_scale=args.mlx_lora_scale,
            lora_dropout=args.mlx_lora_dropout,
            grad_accumulation_steps=args.mlx_grad_accum,
            action_token_only=args.mlx_action_token_only,
            action_token_weight=args.mlx_action_token_weight,
            preserve_row_order=args.mlx_preserve_row_order,
            model_revision=args.mlx_model_revision,
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
        loss_mask_mode=args.loss_mask,
    )


def main(argv: Sequence[str] | None = None) -> None:
    adapter_dir = dispatch(parse_args(argv))
    print(adapter_dir)


if __name__ == "__main__":
    main()
