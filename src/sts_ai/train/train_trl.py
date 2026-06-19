"""Thin CUDA/TRL LoRA trainer wrapper."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sts_ai.train.sft_format import chat_template_probe_hash

__all__ = ["train"]


def _check_manifest(manifest_path: Path, *, tokenizer: Any, base_model: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_hash = manifest.get("chat_template_hash")
    if expected_hash:
        enable_thinking = bool(manifest.get("enable_thinking", False))
        actual_hash = chat_template_probe_hash(
            tokenizer,
            enable_thinking=enable_thinking,
        )
        if actual_hash != expected_hash:
            raise ValueError(
                "dataset chat_template_hash does not match base model tokenizer: "
                f"manifest={expected_hash!r} actual={actual_hash!r}"
            )

    tokenizer_id = manifest.get("tokenizer_id")
    if tokenizer_id and str(tokenizer_id) != base_model:
        print(
            "WARNING: dataset manifest tokenizer_id "
            f"{tokenizer_id!r} differs from base_model {base_model!r}.",
            file=sys.stderr,
        )


def train(
    dataset_path: Path,
    base_model: str,
    out_adapter_dir: Path,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 1,
    max_steps: int = -1,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_len: int = 4096,
    manifest_path: Path | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
    eval_fraction: float = 0.0,
    eval_steps: int = 50,
) -> Path:
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install .[train-cuda]") from exc

    dataset_path = Path(dataset_path)
    out_adapter_dir = Path(out_adapter_dir)
    out_adapter_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    # Use the skew-free prompt/completion pair (not messages): Gemma-4 E4B is a
    # VLM arch and TRL blocks assistant_only_loss for VLMs, but completion_only_loss
    # over prompt/completion is VLM-safe and gives the same prompt-masked loss.
    # The prompt is already chat-templated by sft_format.reconstruct_prompt and the
    # completion is the verbatim raw_response, so TRL must NOT re-template (dropping
    # the messages column keeps the format unambiguously prompt-completion). RWR
    # weighting is preserved because rows are physically replicated by multiplicity.
    required_columns = {"prompt", "completion"}
    missing = sorted(required_columns.difference(ds.column_names))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    drop_columns = [name for name in ds.column_names if name not in required_columns]
    if drop_columns:
        ds = ds.remove_columns(drop_columns)

    train_dataset = ds
    eval_dataset = None
    eval_strategy = "no"
    if eval_fraction > 0:
        split = ds.train_test_split(test_size=eval_fraction, seed=0)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        eval_strategy = "steps"

    if manifest_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        _check_manifest(Path(manifest_path), tokenizer=tokenizer, base_model=base_model)

    if wandb_project is not None:
        import os

        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    # Cannot be run/verified in this sandbox (CUDA-only); eval_strategy,
    # run_name, and report_to are TRL/transformers names to confirm on the pod.
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(out_adapter_dir),
        "num_train_epochs": epochs,
        # -1 = use num_train_epochs; >0 caps optimizer steps (transformers default
        # semantics). Lets us bound a multi-epoch-equivalent dataset to a sane budget.
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        # TRL renamed max_seq_length -> max_length (verified on the CUDA pod with
        # trl 1.6.0 + transformers 5.12.1; the old name raises TypeError there).
        "max_length": max_seq_len,
        # Prompt-masked loss over the prompt/completion pair. completion_only_loss
        # (not assistant_only_loss) because Gemma-4 E4B is a VLM arch and TRL blocks
        # assistant_only_loss for VLMs (verified on the CUDA pod, trl 1.6.0).
        "completion_only_loss": True,
        "eval_strategy": eval_strategy,
        "run_name": run_name,
        "report_to": report_to,
    }
    if eval_dataset is not None:
        sft_kwargs["eval_steps"] = eval_steps
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": base_model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_adapter_dir))
    return out_adapter_dir
