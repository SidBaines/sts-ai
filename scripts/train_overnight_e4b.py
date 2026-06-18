#!/usr/bin/env python
"""Overnight MLX LoRA launcher: filtered-BC on Gemma-4 E4B rollouts.

Builds an SFT dataset from a rollout directory (filtered to act-boss-clear
positives), holds out a fraction of the *seeds* as a quarantined validation set
(so train/eval games are disjoint — no leakage), writes the mlx-lm chat format,
and runs `mlx_lm lora` with assistant-only loss (`--mask-prompt`) while logging
train + validation loss to Weights & Biases.

"Eval" here is held-out **validation loss** on the quarantined seeds (logged at
`--steps-per-eval` intervals) — the robust signal for an unattended run. A
game-performance eval (running rollouts with the trained adapter) is a separate,
slower step better run afterwards on a GPU box.

Reuses the tested `build_dataset` (reward join + filter + skew guards) and
`build_lora_cmd`. Defaults target the 240-rollout E4B thinking run; pass
`--allow-thinking` is implicit here (the launcher sets require_no_thinking=False)
since that data is thinking-mode.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from sts_ai.prompting import NEUTRAL_FRAME
from sts_ai.train.dataset_builder import build_dataset
from sts_ai.train.train_mlx import build_lora_cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rollout-dir",
        default="data/rollouts/e4b_think_perf/vllm_gemma_4_E4B_it_thinking_8192",
    )
    p.add_argument("--model", default="mlx-community/gemma-4-e4b-it-bf16")
    p.add_argument("--out-dir", type=Path, default=Path("data/train_run1"))
    p.add_argument("--holdout-frac", type=float, default=0.1, help="Fraction of kept seeds quarantined for eval.")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-positives", type=int, default=20)
    p.add_argument("--fallback-floor-quantile", type=float, default=0.8)
    # Allow thinking-mode data (the 240 E4B run is native-thinking).
    p.add_argument("--require-no-thinking", action="store_true", default=False)
    # mlx_lm lora hyperparameters
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--steps-per-eval", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--wandb-project", default="sts-e4b-offline-bc")
    p.add_argument("--run-name", default="iter1-thinking-mlx")
    return p.parse_args()


def _write_chat_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer (same one used at generation; the skew-guard hash checks this).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 2. Build the SFT dataset (reward join + act-boss-clear filter + skew guards).
    examples, manifest = build_dataset(
        Path(args.rollout_dir),
        framing=NEUTRAL_FRAME,
        tokenizer=tokenizer,
        tokenizer_id=args.model,
        min_act=1,
        fallback_floor_quantile=args.fallback_floor_quantile,
        min_positives=args.min_positives,
        require_no_thinking=args.require_no_thinking,
        require_framing_match=True,
    )
    if not examples:
        print("No examples after filtering — aborting.", file=sys.stderr)
        sys.exit(2)

    # 3. Seed-quarantined split: hold out a fraction of the KEPT seeds for eval so
    #    train/eval games never overlap (no memorization leakage).
    seeds = sorted({int(ex["world_seed"]) for ex in examples})
    rng = random.Random(args.split_seed)
    n_hold = max(1, round(len(seeds) * args.holdout_frac))
    n_hold = min(n_hold, len(seeds) - 1)  # always leave >=1 train seed
    holdout = set(rng.sample(seeds, n_hold))
    train_ex = [ex for ex in examples if int(ex["world_seed"]) not in holdout]
    valid_ex = [ex for ex in examples if int(ex["world_seed"]) in holdout]
    if not train_ex or not valid_ex:
        print("Seed split produced an empty side — adjust --holdout-frac.", file=sys.stderr)
        sys.exit(2)

    data_dir = args.out_dir / "mlx_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_chat_jsonl(data_dir / "train.jsonl", train_ex)
    _write_chat_jsonl(data_dir / "valid.jsonl", valid_ex)

    # 4. Record exactly what we did (morning review).
    summary = {
        "model": args.model,
        "rollout_dir": args.rollout_dir,
        "n_kept_examples": len(examples),
        "n_train_examples": len(train_ex),
        "n_valid_examples": len(valid_ex),
        "n_kept_seeds": len(seeds),
        "n_holdout_seeds": len(holdout),
        "holdout_seeds": sorted(holdout),
        "filter_report": manifest["filter_report"],
        "reasoning_mode": manifest["reasoning_mode"],
        "hparams": {
            "iters": args.iters,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "steps_per_eval": args.steps_per_eval,
            "val_batches": args.val_batches,
            "save_every": args.save_every,
        },
    }
    (args.out_dir / "overnight_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== run summary ===")
    print(json.dumps({k: summary[k] for k in (
        "n_kept_examples", "n_train_examples", "n_valid_examples",
        "n_kept_seeds", "n_holdout_seeds", "holdout_seeds", "reasoning_mode",
    )}, indent=2))
    print("filter_report:", json.dumps(manifest["filter_report"], indent=2))

    # 5. Train: reuse build_lora_cmd (core flags + --mask-prompt + wandb), then add
    #    the long-sequence / memory flags it doesn't model.
    adapter_dir = args.out_dir / "adapter_iter1_thinking"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_lora_cmd(
        python_exe=sys.executable,
        base_model=args.model,
        data_dir=data_dir,
        out_adapter_dir=adapter_dir,
        num_layers=args.num_layers,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wandb_project=args.wandb_project,
        steps_per_eval=args.steps_per_eval,
        save_every=args.save_every,
        val_batches=args.val_batches,
        mask_prompt=True,
    )
    cmd += [
        "--max-seq-length", str(args.max_seq_length),
        "--grad-checkpoint",
        "--grad-accumulation-steps", str(args.grad_accum),
        "--seed", str(args.train_seed),
    ]
    env = dict(os.environ)
    env.setdefault("WANDB_NAME", args.run_name)  # names the wandb run
    print("=== launching ===\n" + " ".join(cmd))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    subprocess.run(cmd, check=True, env=env)
    print(f"DONE. adapter: {adapter_dir} | started {started}")


if __name__ == "__main__":
    main()
