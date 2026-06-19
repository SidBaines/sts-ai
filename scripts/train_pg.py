#!/usr/bin/env python
"""Train a LoRA policy adapter with the clipped PG objective."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _default_manifest(dataset: Path) -> Path | None:
    manifest = dataset.with_suffix(".manifest.json")
    return manifest if manifest.exists() else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PG LoRA policy adapter.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = _default_manifest(args.dataset)
    return args


def dispatch(args: argparse.Namespace) -> Path:
    from sts_ai.train import train_pg_trl

    return train_pg_trl.train(
        args.dataset,
        args.base_model,
        args.out,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        manifest_path=args.manifest,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


def main(argv: Sequence[str] | None = None) -> None:
    adapter_dir = dispatch(parse_args(argv))
    print(adapter_dir)


if __name__ == "__main__":
    main()
