"""CUDA/TRL LoRA trainer for signed-advantage policy-gradient updates.

This implements the mu=1 path used by TRL's default GRPO-style objective:
``logp_old = logp_new.detach()``, so the clipped surrogate is value-inert while
preserving the same loss path. mu>1 batch-reuse requires a frozen old-logp
snapshot and is a documented GPU-validated follow-up, not implemented here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sts_ai.train.sft_format import chat_template_probe_hash, tokenize_example

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


def _ensure_pad_token(tokenizer: Any) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        return int(pad_token_id)

    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define a pad token or eos token")
    tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.eos_token_id)


def train(
    dataset_path: Path,
    base_model: str,
    out_adapter_dir: Path,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_len: int = 4096,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
    manifest_path: Path | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
) -> Path:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("install .[train-cuda]") from exc

    from sts_ai.train.pg_loss import grpo_loss, selective_logps

    dataset_path = Path(dataset_path)
    out_adapter_dir = Path(out_adapter_dir)
    out_adapter_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    required_columns = {"prompt", "completion", "advantage"}
    missing = sorted(required_columns.difference(ds.column_names))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_token_id = _ensure_pad_token(tokenizer)

    if manifest_path is not None:
        _check_manifest(Path(manifest_path), tokenizer=tokenizer, base_model=base_model)

    def encode_row(example: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenize_example(example, tokenizer)
        if len(tokenized["input_ids"]) > max_seq_len:
            tokenized["input_ids"] = tokenized["input_ids"][:max_seq_len]
            tokenized["labels"] = tokenized["labels"][:max_seq_len]
            tokenized["n_completion_tokens"] = sum(
                1 for label in tokenized["labels"] if label != -100
            )
        tokenized["advantage"] = float(example["advantage"])
        return tokenized

    train_dataset = ds.map(encode_row, remove_columns=ds.column_names)

    def data_collator(features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        advantages = torch.empty(batch_size, dtype=torch.float)

        for index, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            row_labels = torch.tensor(feature["labels"], dtype=torch.long)
            row_len = ids.numel()
            input_ids[index, :row_len] = ids
            labels[index, :row_len] = row_labels
            attention_mask[index, :row_len] = 1
            advantages[index] = float(feature["advantage"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "completion_mask": labels.ne(-100),
            "advantages": advantages,
        }

    if wandb_project is not None:
        import os

        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    base = AutoModelForCausalLM.from_pretrained(base_model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)

    class PGTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            advantages = inputs["advantages"]
            completion_mask = inputs["completion_mask"][:, 1:]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logp_new = selective_logps(outputs.logits, input_ids)
            logp_old = logp_new.detach()

            if kl_beta > 0:
                # Assumes an unwrapped single-GPU PeftModel; a DDP/accelerate
                # wrapper would need .module unwrapping before disable_adapter()
                # (multi-GPU is a follow-up, same as the mu>1 batch-reuse path).
                with torch.no_grad(), model.disable_adapter():
                    ref_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                logp_ref = selective_logps(ref_outputs.logits, input_ids)
            else:
                logp_ref = None

            loss, metrics = grpo_loss(
                logp_new,
                logp_old,
                logp_ref,
                advantages,
                completion_mask,
                clip_eps=clip_eps,
                kl_beta=kl_beta,
            )
            self.log(metrics)
            if return_outputs:
                return loss, outputs
            return loss

    args = TrainingArguments(
        output_dir=str(out_adapter_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
    )
    trainer = PGTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(out_adapter_dir))
    return out_adapter_dir
