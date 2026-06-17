"""Thin CUDA/TRL LoRA trainer wrapper."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["train"]


def _chat_template_hash(tokenizer: Any) -> str:
    messages = [{"role": "user", "content": "__sts_probe__"}]
    try:
        probe = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        probe = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    return hashlib.sha256(probe.encode()).hexdigest()[:16]


def _check_manifest(manifest_path: Path, *, tokenizer: Any, base_model: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_hash = manifest.get("chat_template_hash")
    if expected_hash:
        actual_hash = _chat_template_hash(tokenizer)
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
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_len: int = 4096,
    manifest_path: Path | None = None,
) -> Path:
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install .[train-cuda]") from exc

    _ = AutoModelForCausalLM
    dataset_path = Path(dataset_path)
    out_adapter_dir = Path(out_adapter_dir)
    out_adapter_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    required_columns = {"prompt", "completion"}
    missing = sorted(required_columns.difference(ds.column_names))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    drop_columns = [name for name in ds.column_names if name not in required_columns]
    if drop_columns:
        ds = ds.remove_columns(drop_columns)

    if manifest_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        _check_manifest(Path(manifest_path), tokenizer=tokenizer, base_model=base_model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    # Cannot be run/verified in this sandbox (CUDA-only); flag/SFTConfig arg
    # names are verified against TRL 0.12, confirm on the pod before relying on it.
    sft_config = SFTConfig(
        output_dir=str(out_adapter_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_len,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=base_model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(out_adapter_dir))
    return out_adapter_dir
