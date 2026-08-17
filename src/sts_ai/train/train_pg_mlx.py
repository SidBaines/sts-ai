"""In-process MLX LoRA trainer for signed-advantage PG updates."""
from __future__ import annotations

from contextlib import contextmanager
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Iterator

from sts_ai.train.sft_format import (
    chat_template_probe_hash,
    resolve_loss_mask_mode,
    tokenize_example,
)

__all__ = ["train"]

_REQUIRED_COLUMNS = {"prompt", "completion", "advantage"}
_GRADIENT_CHECKPOINTING_WARNED = False


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _tokenize_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_len: int,
    loss_mask_mode: str = "completion",
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        missing = sorted(_REQUIRED_COLUMNS.difference(record))
        if missing:
            raise ValueError(
                f"dataset record {index} is missing required columns: {missing}"
            )

        tokenized = tokenize_example(
            record,
            tokenizer,
            loss_mask_mode=loss_mask_mode,
        )
        input_ids = list(tokenized["input_ids"])
        labels = list(tokenized["labels"])
        action_mask = list(tokenized["action_mask"])
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
            action_mask = action_mask[:max_seq_len]

        n_completion_tokens = sum(1 for label in labels if label != -100)
        if n_completion_tokens == 0 or (
            loss_mask_mode == "action" and not any(action_mask)
        ):
            continue

        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "completion_mask": [label != -100 for label in labels],
                "n_completion_tokens": n_completion_tokens,
                "n_action_tokens": sum(bool(value) for value in action_mask),
                "n_format_tokens": sum(
                    bool(value)
                    for value in tokenized["format_mask"][: len(input_ids)]
                ),
                "advantage": float(record["advantage"]),
            }
        )

    if not examples:
        raise ValueError("dataset has no completion tokens after truncation")
    return examples


@contextmanager
def _lora_disabled(model: Any, lora_types: tuple[type[Any], ...]) -> Iterator[None]:
    saved_scales: list[tuple[Any, float]] = []
    for _name, module in model.named_modules():
        if isinstance(module, lora_types):
            saved_scales.append((module, module.scale))
            module.scale = 0.0
    try:
        yield
    finally:
        for module, scale in saved_scales:
            module.scale = scale


def _load_and_attach_lora(
    *,
    mlx_lm: Any,
    linear_to_lora_layers: Callable[..., Any],
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    init_adapter_path: str | None,
    manifest_path: Path | None,
    num_layers: int | None,
) -> tuple[Any, Any, int, dict[str, float | int]]:
    model, tokenizer = mlx_lm.load(base_model)

    if manifest_path is not None:
        _check_manifest(Path(manifest_path), tokenizer=tokenizer, base_model=base_model)

    n_layers = int(num_layers or len(model.layers))
    lora_parameters: dict[str, float | int] = {
        "rank": lora_r,
        "scale": lora_alpha / lora_r,
        "dropout": lora_dropout,
    }
    model.freeze()
    linear_to_lora_layers(model, n_layers, lora_parameters)
    if init_adapter_path is not None:
        model.load_weights(f"{init_adapter_path}/adapters.safetensors", strict=False)
    return model, tokenizer, n_layers, lora_parameters


def _loss_fn(
    model: Any,
    input_ids: Any,
    comp_mask: Any,
    advantages: Any,
    logp_ref: Any,
    *,
    mx: Any,
    pg_loss_mlx: Any,
    clip_eps: float,
    kl_beta: float,
) -> tuple[Any, dict[str, float]]:
    logits = model(input_ids)
    logp_new = pg_loss_mlx.selective_logps(logits, input_ids)
    logp_old = mx.stop_gradient(logp_new)
    comp_mask_shifted = comp_mask[:, 1:]
    return pg_loss_mlx.grpo_loss(
        logp_new,
        logp_old,
        logp_ref,
        advantages,
        comp_mask_shifted,
        clip_eps=clip_eps,
        kl_beta=kl_beta,
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
    learning_rate: float = 1e-5,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_len: int = 4096,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
    gradient_checkpointing: bool = False,
    init_adapter_path: str | None = None,
    manifest_path: Path | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
    num_layers: int | None = None,
    loss_mask_mode: str = "auto",
) -> Path:
    """Train an MLX LoRA adapter with the GRPO-style PG loss.

    The signature is compatible with ``train_pg_trl.train`` so the GRPO outer
    loop can inject this trainer. MLX v1 keeps the implementation eager and
    processes one example at a time; ``per_device_batch_size`` is accepted for
    compatibility only.
    ``wandb_project`` and ``run_name`` are accepted for signature parity but
    intentionally unused; MLX telemetry flows through ``trainer_log.json`` into
    the GRPO loop's single wandb run.
    ``gradient_checkpointing`` is accepted and ignored with a one-time warning
    because MLX gradient checkpointing is not implemented in v1.
    The default ``max_seq_len`` matches the torch trainer; call sites can lower
    it if a large model hits the unified-memory ceiling.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import mlx_lm
        from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx.utils import tree_flatten, tree_map
    except ImportError as exc:
        raise RuntimeError("install .[train-mlx]") from exc

    from sts_ai.train import pg_loss_mlx

    del per_device_batch_size, wandb_project, run_name

    if gradient_checkpointing:
        global _GRADIENT_CHECKPOINTING_WARNED
        if not _GRADIENT_CHECKPOINTING_WARNED:
            warnings.warn(
                "gradient_checkpointing=True was requested, but MLX gradient "
                "checkpointing is not implemented in v1; ignoring it.",
                stacklevel=2,
            )
            _GRADIENT_CHECKPOINTING_WARNED = True
    if lora_r < 1:
        raise ValueError("lora_r must be >= 1")
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")

    dataset_path = Path(dataset_path)
    out_adapter_dir = Path(out_adapter_dir)
    out_adapter_dir.mkdir(parents=True, exist_ok=True)
    resolved_loss_mask_mode = resolve_loss_mask_mode(
        loss_mask_mode,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
    )

    model, tokenizer, n_layers, lora_parameters = _load_and_attach_lora(
        mlx_lm=mlx_lm,
        linear_to_lora_layers=linear_to_lora_layers,
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_adapter_path=init_adapter_path,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        num_layers=num_layers,
    )
    examples = _tokenize_dataset(
        _load_jsonl(dataset_path),
        tokenizer,
        max_seq_len=max_seq_len,
        loss_mask_mode=resolved_loss_mask_mode,
    )

    lora_types = (LoRALinear, LoRAEmbedding, LoRASwitchLinear)
    optimizer = optim.AdamW(learning_rate=learning_rate)
    vg = nn.value_and_grad(
        model,
        lambda *args: _loss_fn(
            *args,
            mx=mx,
            pg_loss_mlx=pg_loss_mlx,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        ),
    )

    model.train()
    log_history: list[dict[str, float | int]] = []
    accum_grad = None
    accum_count = 0
    step = 0

    def apply_accumulated_grad() -> None:
        nonlocal accum_count, accum_grad
        if accum_grad is None or accum_count == 0:
            return
        grad = tree_map(lambda value: value / accum_count, accum_grad)
        optimizer.update(model, grad)
        accum_grad = None
        accum_count = 0
        mx.eval(model.state, optimizer.state)
        mx.clear_cache()

    examples = list(examples)
    for _epoch in range(epochs):
        # Rollout order groups trajectories (and phases) together; unshuffled
        # accumulation windows would average highly correlated gradients.
        random.Random(17 + _epoch).shuffle(examples)
        for example in examples:
            input_ids = mx.array([example["input_ids"]], dtype=mx.int32)
            comp_mask = mx.array([example["completion_mask"]], dtype=mx.bool_)
            advantages = mx.array([example["advantage"]], dtype=mx.float32)

            logp_ref = None
            if kl_beta > 0:
                with _lora_disabled(model, lora_types):
                    ref_logits = model(input_ids)
                    logp_ref = pg_loss_mlx.selective_logps(ref_logits, input_ids)
                    mx.eval(logp_ref)

            ((loss, metrics), grad) = vg(
                model,
                input_ids,
                comp_mask,
                advantages,
                logp_ref,
            )
            step += 1
            accum_grad = (
                grad
                if accum_grad is None
                else tree_map(lambda left, right: left + right, accum_grad, grad)
            )
            accum_count += 1
            mx.eval(accum_grad, loss)

            entry: dict[str, float | int] = {
                "loss": float(loss.item()),
                "mean_kl": float(metrics["mean_kl"]),
                "mean_ratio": float(metrics["mean_ratio"]),
                "clip_fraction": float(metrics["clip_fraction"]),
                "mean_advantage": float(metrics["mean_advantage"]),
                "step": step,
                "action_token_count": int(example["n_action_tokens"]),
                "supervised_token_count": int(example["n_completion_tokens"]),
            }
            log_history.append(entry)

            if accum_count >= grad_accum:
                apply_accumulated_grad()

        apply_accumulated_grad()

    mx.save_safetensors(
        str(out_adapter_dir / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    (out_adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "num_layers": n_layers,
                "lora_parameters": lora_parameters,
                "fine_tune_type": "lora",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    try:
        (out_adapter_dir / "trainer_log.json").write_text(
            json.dumps(log_history, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Same release-order rule as MlxQwenJsonAgent.sleep(): refs must be gone
    # (del + gc) BEFORE the cache clear, or a model image strands in the cache
    # and the agent's post-training wake() reload swap-storms.
    del model, optimizer
    import gc

    gc.collect()
    mx.clear_cache()
    return out_adapter_dir
