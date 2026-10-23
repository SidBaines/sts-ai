"""Small synchronous LoRA GRPO trainer and CUDA optimizer benchmark.

Training samples from the current HF/PEFT model, freezes behavior log-probs
before any update, and rewards single decisions with the non-LLM teacher's
probability. It never trains on teacher reasoning or SFT target completions.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Sequence

TRAINING_FORMAT_VERSION = 3


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def normalized_advantages(rewards: Sequence[float], epsilon: float = 1e-6) -> tuple[list[float], float]:
    if not rewards or any(not math.isfinite(value) for value in rewards):
        raise ValueError("rewards must be a finite nonempty group")
    mean = statistics.fmean(rewards)
    variance = statistics.fmean((value - mean) ** 2 for value in rewards)
    std = math.sqrt(variance)
    return ([0.0] * len(rewards) if std <= epsilon else [(value - mean) / std for value in rewards], variance)


@dataclass
class Completion:
    prompt_id: str
    phase: str
    prompt_ids: list[int]
    output_ids: list[int]
    reward: float
    advantage: float = 0.0
    old_logps: list[float] = field(default_factory=list)
    ref_logps: list[float] = field(default_factory=list)
    raw_text: str = ""
    action_index: int | None = None
    valid: bool = False
    truncated: bool = False
    error: str | None = None


def clipped_response_loss(new_logps: Any, old_logps: Any, ref_logps: Any,
                          advantages: Any, mask: Any, *, clip_epsilon: float,
                          kl_beta: float, loss_normalization: str = "response_mean",
                          loss_token_budget: int = 2048) -> tuple[Any, dict[str, Any]]:
    """Completion-token surrogate with explicit sequence normalization.

    `old_logps` and `ref_logps` are immutable cohort measurements. This function
    intentionally never recomputes or detaches a new log-prob to create old.
    Both policy and KL use either each response's realized length or the same
    fixed positive token budget. Diagnostic probability ratios remain per-token.
    """
    import torch
    if loss_normalization not in {"response_mean", "fixed_token_budget"}:
        raise ValueError("unknown loss normalization")
    if not math.isfinite(loss_token_budget) or loss_token_budget <= 0:
        raise ValueError("loss token budget must be a finite positive constant")
    new = new_logps.float()
    old = old_logps.detach().float()
    ref = ref_logps.detach().float()
    mask = mask.bool()
    if new.shape != old.shape or new.shape != ref.shape or new.shape != mask.shape:
        raise ValueError("log-probs and mask must have identical shapes")
    lengths = mask.sum(-1)
    if torch.any(lengths == 0):
        raise ValueError("every response must contain at least one generated token")
    # Replace padding before exponentiation; arbitrary masked values cannot
    # overflow and contaminate the loss through inf*0.
    log_ratio = torch.where(mask, new - old, torch.zeros_like(new))
    ref_delta = torch.where(mask, ref - new, torch.zeros_like(new))
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
    # Nonnegative sampled k3 KL proxy, not an exact vocabulary KL.
    kl = torch.expm1(ref_delta) - ref_delta
    sums = lambda value: torch.where(mask, value, torch.zeros_like(value)).sum(-1)
    per_token_means = lambda value: (sums(value) / lengths.float()).mean()
    denom = lengths.float() if loss_normalization == "response_mean" else torch.full_like(
        lengths, float(loss_token_budget), dtype=torch.float32)
    objective_means = lambda value: (sums(value) / denom).mean()
    policy_loss = -objective_means(surrogate)
    kl_mean = objective_means(kl)
    loss = policy_loss + kl_beta * kl_mean
    metrics = {
        "policy_loss": policy_loss.detach(), "sampled_kl_proxy": kl_mean.detach(),
        "sampled_kl_per_token": per_token_means(kl).detach(),
        "mean_ratio": per_token_means(ratio).detach(),
        "clip_fraction": per_token_means((ratio != clipped).float()).detach(),
        "mean_abs_log_ratio": per_token_means(log_ratio.abs()).detach(),
        "max_abs_log_ratio": torch.where(mask, log_ratio.abs(), torch.zeros_like(new)).max().detach(),
        "mean_response_tokens": lengths.float().mean().detach(),
    }
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite policy/KL loss; no optimizer step was taken")
    return loss, metrics


def response_logps(model: Any, examples: Sequence[Completion], *, pad_id: int,
                   temperature: float, chunk_size: int = 256) -> list[Any]:
    """Score only generated-token predictions; no prompt/pad loss or cache."""
    import torch
    import torch.nn.functional as functional
    if not examples or temperature <= 0:
        raise ValueError("examples required and temperature must be positive")
    device = next(model.parameters()).device
    totals = [len(item.prompt_ids) + len(item.output_ids) for item in examples]
    prompts = [len(item.prompt_ids) for item in examples]
    if min(prompts) < 1 or min(len(item.output_ids) for item in examples) < 1:
        raise ValueError("empty prompt or completion")
    width = max(totals)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for index, item in enumerate(examples):
        tokens = item.prompt_ids + item.output_ids
        ids[index, :len(tokens)] = torch.tensor(tokens, device=device)
        attention[index, :len(tokens)] = 1
    start = min(prompts) - 1
    # K2 supports tensor logits_to_keep: omit the large prompt-vocabulary
    # projection, and omit the final token's unused next-token prediction.
    positions = torch.arange(start, width - 1, device=device)
    output = model(input_ids=ids, attention_mask=attention, use_cache=False,
                   logits_to_keep=positions)
    logits = output.logits
    if logits.shape[1] != len(positions):
        raise ValueError("model did not honor tensor logits_to_keep")
    targets = ids[:, start + 1:]
    # Chunk FP32 vocabulary normalization. Only BF16 logits persist for the
    # full suffix; this avoids multiple full FP32 [length,vocabulary] arrays.
    chunks = []
    for offset in range(0, logits.shape[1], chunk_size):
        chunk = logits[:, offset:offset + chunk_size].float() / temperature
        labels = targets[:, offset:offset + chunk_size]
        logps = -functional.cross_entropy(chunk.transpose(1, 2), labels, reduction="none")
        chunks.append(logps)
    all_logps = torch.cat(chunks, dim=1)
    return [all_logps[index, prompts[index] - 1 - start:totals[index] - 1 - start]
            for index in range(len(examples))]


def freeze_cohort_logps(model: Any, examples: list[Completion], *, microbatch: int,
                         pad_id: int, temperature: float) -> None:
    import torch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for offset in range(0, len(examples), microbatch):
                batch = examples[offset:offset + microbatch]
                old = response_logps(model, batch, pad_id=pad_id, temperature=temperature)
                for item, values in zip(batch, old):
                    item.old_logps = values.detach().cpu().tolist()
                del old
                # PEFT's base weights are frozen. Disabling adapters gives the
                # same fixed reference across every iteration and PPO epoch.
                with model.disable_adapter():
                    reference = response_logps(model, batch, pad_id=pad_id, temperature=temperature)
                for item, values in zip(batch, reference):
                    item.ref_logps = values.detach().cpu().tolist()
                del reference
    finally:
        model.train(was_training)


def padded_objective(model: Any, examples: list[Completion], args: Any, pad_id: int) -> tuple[Any, dict[str, Any]]:
    import torch
    scored = response_logps(model, examples, pad_id=pad_id, temperature=args.temperature)
    width = max(value.numel() for value in scored)
    # pad_sequence preserves gradients; assigning into a fresh buffer is less
    # explicit and can accidentally detach in later refactors.
    new = torch.nn.utils.rnn.pad_sequence(scored, batch_first=True)
    old = torch.zeros_like(new)
    ref = torch.zeros_like(new)
    mask = torch.zeros_like(new, dtype=torch.bool)
    for index, (item, values) in enumerate(zip(examples, scored)):
        count = values.numel()
        if len(item.old_logps) != count or len(item.ref_logps) != count:
            raise ValueError("cohort log-probs missing or incompatible with generated tokens")
        old[index, :count] = torch.tensor(item.old_logps, device=new.device)
        ref[index, :count] = torch.tensor(item.ref_logps, device=new.device)
        mask[index, :count] = True
    adv = torch.tensor([item.advantage for item in examples], dtype=torch.float32, device=new.device)
    return clipped_response_loss(new, old, ref, adv, mask,
                                 clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta,
                                 loss_normalization=args.loss_normalization,
                                 loss_token_budget=args.loss_token_budget)


def optimizer_step(model: Any, optimizer: Any, examples: list[Completion], args: Any,
                   pad_id: int) -> dict[str, float]:
    import torch
    optimizer.zero_grad(set_to_none=True)
    model.train()
    metrics: dict[str, float] = {}
    loss_total = 0.0
    for offset in range(0, len(examples), args.microbatch):
        batch = examples[offset:offset + args.microbatch]
        loss, observed = padded_objective(model, batch, args, pad_id)
        # Weight by response count, including an incomplete final microbatch.
        weight = len(batch) / len(examples)
        (loss * weight).backward()
        loss_total += float(loss.detach()) * weight
        for key, value in observed.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(f"nonfinite {key}; optimizer step withheld")
            if key == "max_abs_log_ratio":
                metrics[key] = max(metrics.get(key, 0.0), scalar)
            else:
                metrics[key] = metrics.get(key, 0.0) + scalar * weight
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    metrics.update(loss=loss_total, gradient_norm_before_clip=float(norm), responses=len(examples),
                   generated_tokens=sum(len(item.output_ids) for item in examples))
    return metrics


def synchronize(device: str) -> None:
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def trainable_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(args: Any) -> tuple[Any, Any, int]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, code_revision=args.revision,
        trust_remote_code=True, torch_dtype=dtype,
        attn_implementation=args.attention, device_map=args.device)
    if args.resume or args.adapter:
        model = PeftModel.from_pretrained(model, args.resume or args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=0.0, bias="none", target_modules=args.lora_targets.split(",")))
    model.config.use_cache = False
    # Exact old/new comparisons need identical stochastic-network semantics.
    # Checkpointing requires train mode, so disable dropout in both modes.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    stop_id = tokenizer.convert_tokens_to_ids("<|ifm|im_end|>")
    if not isinstance(stop_id, int) or tokenizer.convert_ids_to_tokens(stop_id) != "<|ifm|im_end|>":
        raise ValueError("missing IFM final-turn stop token")
    return model, tokenizer, stop_id


def load_rows(path: Path, tokenizer: Any, args: Any) -> list[dict[str, Any]]:
    rows = []
    benchmark_only = args.benchmark_updates or args.benchmark_generation
    source_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Validate the entire declared split surface before applying a limit. A
    # small smoke limit must not conceal an unsplit row later in the file.
    if not benchmark_only:
        for row in source_rows:
            if row.get("split") not in {"train", "validation", "test"}:
                raise ValueError("training requires every dataset row to declare split=train/validation/test")
    for row in source_rows:
        if not benchmark_only and row["split"] != "train":
            continue
        if "teacher" not in row and not benchmark_only:
            raise ValueError("training rows require teacher distribution metadata")
        if not row.get("legal_indices") or not row.get("messages"):
            raise ValueError("every row requires legal_indices and messages")
        row["id"] = str(row.get("id", len(rows)))
        row["phase"] = str(row.get("phase", "unknown"))
        row["prompt_ids"] = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                    add_generation_prompt=True, reasoning_effort=args.effort,
                                    return_dict=False)
        if hasattr(row["prompt_ids"], "tolist"):
            row["prompt_ids"] = row["prompt_ids"].tolist()
        if not isinstance(row["prompt_ids"], list) or not row["prompt_ids"] or any(
                isinstance(token, bool) or not isinstance(token, int) for token in row["prompt_ids"]):
            raise ValueError("chat template must return a nonempty flat integer-token list")
        if len(row["prompt_ids"]) + args.max_new_tokens > args.max_sequence_length:
            raise ValueError(f"prompt {row['id']} plus completion cap exceeds max sequence length; no silent truncation")
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("no selected input rows (training selects only split=train)")
    args.selected_split = "all_explicitly_provided_benchmark_only" if benchmark_only else "train"
    args.selected_row_count = len(rows)
    args.selected_selection_sha256 = hashlib.sha256(json.dumps([
        {"id": row["id"], "split": row.get("split"), "world_identity": row.get("world_identity"),
         "prompt_ids": row["prompt_ids"]} for row in rows],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return rows


def sample_cohort(model: Any, tokenizer: Any, stop_id: int, rows: list[dict[str, Any]],
                  args: Any, *, iteration: int) -> tuple[list[Completion], dict[str, Any]]:
    import torch
    from .generation import parse_action
    from .teacher import teacher_reward
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define a padding token")
    device = next(model.parameters()).device
    cohort = []
    variances = []
    model.eval()
    began = time.perf_counter()
    completed_path = args.output / f"generation_{iteration:04d}.jsonl"
    partial_path = args.output / f"generation_{iteration:04d}.partial.jsonl"
    if completed_path.exists():
        raise FileExistsError(f"completed generation cohort exists: {completed_path}")
    generated_tokens = 0
    # This is the raw generation ledger, before group advantages are assigned.
    # Each finished microbatch survives a later generation failure. The later
    # cohort_NNNN.jsonl additionally stores the frozen old/reference log-probs.
    with partial_path.open("x") as partial:
        for row in rows:
            group = []
            for offset in range(0, args.group_size, args.generation_microbatch):
                count = min(args.generation_microbatch, args.group_size - offset)
                prompt = torch.tensor([row["prompt_ids"]] * count, dtype=torch.long, device=device)
                torch.manual_seed(args.seed + iteration * 1000003 + len(cohort) + offset)
                with torch.no_grad():
                    outputs = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                        max_new_tokens=args.max_new_tokens, pad_token_id=pad_id,
                        eos_token_id=[stop_id, tokenizer.eos_token_id], use_cache=True,
                        repetition_penalty=1.0, renormalize_logits=False)
                for sequence in outputs:
                    tokens = sequence[prompt.shape[1]:].tolist()
                    endings = [i for i, token in enumerate(tokens) if token in {stop_id, tokenizer.eos_token_id}]
                    if endings:
                        tokens = tokens[:endings[0] + 1]
                    truncated = not endings and len(tokens) >= args.max_new_tokens
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    parsed = parse_action(text, row["legal_indices"], finish_reason="length" if truncated else "stop")
                    reward = teacher_reward(row, parsed.action_index if parsed.valid else None,
                                            invalid_reward=args.invalid_reward) if "teacher" in row else 0.0
                    item = Completion(row["id"], row["phase"], row["prompt_ids"], tokens,
                        reward, raw_text=text, action_index=parsed.action_index,
                        valid=parsed.valid, truncated=truncated, error=parsed.error)
                    group.append(item)
                    generated_tokens += len(tokens)
                    partial.write(json.dumps({"stage": "generation_before_advantage_normalization",
                                               **asdict(item)}) + "\n")
                partial.flush()
                os.fsync(partial.fileno())
                elapsed = time.perf_counter() - began
                print(json.dumps({"event": "generation_progress", "iteration": iteration,
                    "completed_responses": len(cohort) + len(group), "generated_tokens": generated_tokens,
                    "elapsed_seconds": elapsed, "generated_tokens_per_second": generated_tokens / elapsed,
                    "partial_path": str(partial_path)}), flush=True)
                del outputs, prompt
            if args.benchmark_updates:
                # Throughput-only synthetic advantages guarantee a real
                # backward/update even when every answer is truncated.
                for index, item in enumerate(group):
                    item.reward = float(index % 2)
            advantages, variance = normalized_advantages([item.reward for item in group])
            variances.append(variance)
            for item, advantage in zip(group, advantages):
                item.advantage = advantage
            cohort.extend(group)
    os.replace(partial_path, completed_path)
    return cohort, {"mean_group_reward_variance": statistics.fmean(variances),
                    "zero_variance_group_fraction": sum(v <= 1e-12 for v in variances) / len(variances),
                    "mean_reward": statistics.fmean(item.reward for item in cohort),
                    "valid_fraction": sum(item.valid for item in cohort) / len(cohort),
                    "truncation_fraction": sum(item.truncated for item in cohort) / len(cohort)}


def offline_benchmark_cohort(path: Path, rows: list[dict[str, Any]], tokenizer: Any,
                             args: Any) -> list[Completion]:
    by_id = {row["id"]: row for row in rows}
    cohort = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        prompt_id = str(record.get("prompt_id", record.get("id")))
        if prompt_id not in by_id:
            continue
        row = by_id[prompt_id]
        if record.get("effort") and record["effort"] != args.effort:
            raise ValueError("offline cohort effort differs from --effort; use the matching prompt template")
        tokens = record.get("output_token_ids")
        if not isinstance(tokens, list) or not tokens:
            continue
        if len(row["prompt_ids"]) + len(tokens) > args.max_sequence_length:
            raise ValueError("offline completion exceeds max sequence length; no silent truncation")
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        # This path is ONLY timing replayed forwards/backwards, never a claim
        # that vLLM samples came from the current adapter's exact policy.
        cohort.append(Completion(prompt_id, row["phase"], row["prompt_ids"], tokens,
                                 float(len(cohort) % 2), advantage=(-1.0 if len(cohort) % 2 == 0 else 1.0),
                                 raw_text=text))
    if len(cohort) < 2:
        raise ValueError("update benchmark needs >=2 records with actual server output_token_ids")
    return cohort


def benchmark_updates(model: Any, tokenizer: Any, cohort: list[Completion], args: Any,
                      output: Path) -> dict[str, Any]:
    import torch
    pad_id = tokenizer.pad_token_id
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.0)
    freeze_start = time.perf_counter()
    freeze_cohort_logps(model, cohort, microbatch=args.microbatch, pad_id=pad_id, temperature=args.temperature)
    synchronize(args.device)
    freeze_seconds = time.perf_counter() - freeze_start
    before = trainable_hash(model)
    timed_before = None
    effective_batch = args.microbatch * args.gradient_accumulation
    history = []
    # Each timed step performs real loss/backward/gradient clipping/AdamW.
    for step in range(args.warmup_steps + args.benchmark_steps):
        batch = [cohort[(step * effective_batch + i) % len(cohort)] for i in range(effective_batch)]
        if step == args.warmup_steps:
            timed_before = trainable_hash(model)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
        synchronize(args.device)
        began = time.perf_counter()
        metrics = optimizer_step(model, optimizer, batch, args, pad_id)
        synchronize(args.device)
        metrics.update(step=step, warmup=step < args.warmup_steps,
                       seconds=time.perf_counter() - began)
        history.append(metrics)
        atomic_json(output / "update_steps.json", history)
    after = trainable_hash(model)
    timed = [item for item in history if not item["warmup"]]
    seconds = sum(item["seconds"] for item in timed)
    summary = {
        "purpose": "update_throughput_only_synthetic_advantages",
        "cohort_source": "offline_vllm_not_on_policy" if args.cohort_jsonl else "synchronous_hf_generation",
        "on_policy_training_result": False, "original_checkpoint_modified": False,
        "adapter_saved": False, "trainable_weights_changed": before != after,
        "timed_trainable_weights_changed": timed_before != after,
        "trainable_hash_before": before, "trainable_hash_after": after,
        "trainable_hash_after_warmup": timed_before,
        "frozen_logprob_seconds": freeze_seconds,
        "warmup_steps_excluded": args.warmup_steps, "timed_optimizer_steps": len(timed),
        "loss_normalization": args.loss_normalization, "loss_token_budget": args.loss_token_budget,
        "timed_seconds": seconds, "optimizer_steps_per_second": len(timed) / seconds,
        "responses_per_second": sum(item["responses"] for item in timed) / seconds,
        "generated_training_tokens_per_second": sum(item["generated_tokens"] for item in timed) / seconds,
        "prompt_token_lengths": [len(item.prompt_ids) for item in cohort],
        "response_token_lengths": [len(item.output_ids) for item in cohort],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
        "steps": history,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def benchmark_generation(model: Any, tokenizer: Any, stop_id: int, rows: list[dict[str, Any]],
                         args: Any, output: Path) -> dict[str, Any]:
    """Time the actual HF rollout path used by training, without an update."""
    import torch
    selected = rows[:args.states_per_iteration]
    warmup_records = []
    warmup_args = copy.copy(args)
    warmup_args.group_size = args.generation_warmup_responses
    for warmup in range(args.warmup_steps):
        began = time.perf_counter()
        warmup_cohort, _ = sample_cohort(model, tokenizer, stop_id, selected[:1], warmup_args, iteration=10000 + warmup)
        synchronize(args.device)
        warmup_records.append({"responses": len(warmup_cohort),
            "generated_tokens": sum(len(item.output_ids) for item in warmup_cohort),
            "seconds": time.perf_counter() - began,
            "cohort_path": str(output / f"generation_{10000 + warmup:04d}.jsonl")})
    synchronize(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    cohort, metrics = sample_cohort(model, tokenizer, stop_id, selected, args, iteration=0)
    synchronize(args.device)
    seconds = time.perf_counter() - began
    generated = sum(len(item.output_ids) for item in cohort)
    valid = sum(item.valid for item in cohort)
    (output / "generated_cohort.jsonl").write_text("".join(json.dumps(asdict(item)) + "\n" for item in cohort))
    summary = {"purpose": "hf_training_generation_throughput_only", "optimizer_steps": 0,
        "original_checkpoint_modified": False, "adapter_saved": False,
        "effort": args.effort, "generation_microbatch": args.generation_microbatch,
        "group_size": args.group_size, "timed_states": len(selected), "timed_responses": len(cohort),
        "warmup_groups_excluded": args.warmup_steps,
        "warmup_responses_per_group": args.generation_warmup_responses,
        "warmup_records": warmup_records,
        "timed_seconds": seconds, "generated_tokens": generated,
        "generated_tokens_per_second": generated / seconds,
        "decisions_per_second": len(cohort) / seconds,
        "valid_decisions_per_second": valid / seconds,
        "prompt_token_lengths": [len(item.prompt_ids) for item in cohort],
        "response_token_lengths": [len(item.output_ids) for item in cohort],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
        "sampling_top_p": 1.0, "temperature": args.temperature, **metrics}
    atomic_json(output / "summary.json", summary)
    return summary


def training_loop(model: Any, tokenizer: Any, stop_id: int, rows: list[dict[str, Any]],
                  args: Any, output: Path) -> dict[str, Any]:
    import torch
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.0)
    history = []
    rng = random.Random(args.seed)
    start_iteration = 0
    last_checkpoint = str(args.resume or args.adapter) if args.resume or args.adapter else None
    invariant_names = ("model", "revision", "effort", "temperature", "max_new_tokens", "group_size",
        "generation_microbatch", "states_per_iteration", "microbatch", "gradient_accumulation",
        "update_epochs", "learning_rate", "clip_epsilon", "kl_beta", "invalid_reward", "seed",
        "max_grad_norm", "dtype", "attention", "gradient_checkpointing", "limit",
        "selected_split", "selected_row_count", "selected_selection_sha256",
        "loss_normalization", "loss_token_budget")
    settings = {name: getattr(args, name) for name in invariant_names}
    settings["training_format_version"] = TRAINING_FORMAT_VERSION
    settings["prompt_file_sha256"] = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    if args.resume:
        # This is our own explicitly selected training artifact, not arbitrary
        # downloaded model code. Python RNG tuples require weights_only=False.
        state = torch.load(Path(args.resume) / "training_state.pt", map_location=args.device, weights_only=False)
        if state["settings"] != settings:
            raise ValueError("exact-resume settings/data differ; use --warm-start-adapter for a declared new condition")
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        start_iteration, history = state["next_iteration"], state["history"]

    def stopped(iteration: int, updates: list[dict[str, Any]] | None = None,
                cohort_path: Path | None = None) -> dict[str, Any] | None:
        if not args.stop_file.exists():
            return None
        summary = {"purpose": "single_decision_teacher_probability_grpo", "status": "stopped",
            "last_completed_checkpoint": last_checkpoint, "incomplete_iteration": iteration,
            "completed_updates_in_incomplete_iteration": len(updates or []),
            "incomplete_cohort": str(cohort_path) if cohort_path else None,
            "resume_semantics": "Resume the last completed checkpoint; incomplete iteration is regenerated. Partial optimizer updates are not promoted.",
            "iterations": history}
        atomic_json(output / "stopped.json", summary)
        atomic_json(output / "summary.json", summary)
        return summary

    for iteration in range(start_iteration, args.iterations):
        stop_result = stopped(iteration)
        if stop_result is not None:
            return stop_result
        selected = rng.sample(rows, min(args.states_per_iteration, len(rows)))
        began = time.perf_counter()
        cohort, reward_metrics = sample_cohort(model, tokenizer, stop_id, selected, args, iteration=iteration)
        synchronize(args.device)
        generation_seconds = time.perf_counter() - began
        freeze_cohort_logps(model, cohort, microbatch=args.microbatch,
                             pad_id=tokenizer.pad_token_id, temperature=args.temperature)
        # Freeze and persist once. Later epochs must reuse these exact values.
        cohort_path = output / f"cohort_{iteration:04d}.jsonl"
        temporary = cohort_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(asdict(item)) + "\n" for item in cohort))
        os.replace(temporary, cohort_path)
        updates = []
        update_start = time.perf_counter()
        effective_batch = args.microbatch * args.gradient_accumulation
        for epoch in range(args.update_epochs):
            order = list(range(len(cohort)))
            rng.shuffle(order)
            for offset in range(0, len(order), effective_batch):
                stop_result = stopped(iteration, updates, cohort_path)
                if stop_result is not None:
                    return stop_result
                batch = [cohort[index] for index in order[offset:offset + effective_batch]]
                metrics = optimizer_step(model, optimizer, batch, args, tokenizer.pad_token_id)
                updates.append(dict(epoch=epoch, **metrics))
        synchronize(args.device)
        update_seconds = time.perf_counter() - update_start
        save_start = time.perf_counter()
        adapter_dir = output / f"adapter_{iteration:04d}"
        pending_dir = output / f"adapter_{iteration:04d}.pending"
        if adapter_dir.exists() or pending_dir.exists():
            raise FileExistsError("checkpoint output exists; refusing to replace completed or recoverable state")
        model.save_pretrained(pending_dir)
        tokenizer.save_pretrained(pending_dir)
        iteration_metrics = {"iteration": iteration, **reward_metrics,
            "generation_seconds": generation_seconds,
            "update_seconds": update_seconds,
            "save_seconds": time.perf_counter() - save_start,
            "generated_tokens": sum(len(item.output_ids) for item in cohort),
            "updates": updates, "adapter": str(adapter_dir), "cohort": str(cohort_path)}
        history.append(iteration_metrics)
        state = {"optimizer": optimizer.state_dict(), "python_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
            "next_iteration": iteration + 1, "history": history, "settings": settings}
        torch.save(state, pending_dir / "training_state.pt")
        os.replace(pending_dir, adapter_dir)
        last_checkpoint = str(adapter_dir)
        atomic_json(output / "progress.json", history)
        print(json.dumps({key: value for key, value in iteration_metrics.items() if key != "updates"}), flush=True)
    summary = {"purpose": "single_decision_teacher_probability_grpo", "status": "complete",
               "iterations": history, "last_completed_checkpoint": last_checkpoint,
               "generation": "synchronous_current_hf_peft_policy", "sampling_top_p": 1.0,
               "loss_tokens": "all_generated_reasoning_final_and_eos_tokens",
               "loss_normalization": args.loss_normalization,
               "loss_token_budget": args.loss_token_budget,
               "reference": "frozen_base_with_adapters_disabled", "sft_targets": False}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--revision", default="7c75f2f49e74ac97c1852e62b4c860091d6a00ea")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start-adapter", "--adapter", dest="adapter",
                        help="Weights-only warm start: resets optimizer/RNG; not an exact resume")
    parser.add_argument("--resume", type=Path, help="Completed adapter checkpoint containing training_state.pt; restores optimizer/RNG")
    parser.add_argument("--stop-file", type=Path, help="Checked between iterations and optimizer steps; defaults to OUTPUT/STOP")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-sequence-length", type=int, default=40960)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--generation-microbatch", type=int, default=2)
    parser.add_argument("--states-per-iteration", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=.2)
    parser.add_argument("--kl-beta", type=float, default=.02)
    parser.add_argument("--loss-normalization", choices=("response_mean", "fixed_token_budget"),
                        help="Required for training; benchmarks default to response_mean")
    parser.add_argument("--loss-token-budget", type=int, default=2048,
                        help="Fixed positive denominator for fixed_token_budget policy and KL loss")
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--invalid-reward", type=float, default=-1.)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true", help="One state/group, one iteration; respects requested token cap")
    parser.add_argument("--benchmark-updates", action="store_true")
    parser.add_argument("--benchmark-generation", action="store_true")
    parser.add_argument("--generation-warmup-responses", type=int, default=2,
                        help="Benchmark-generation only: responses per excluded warmup group")
    parser.add_argument("--cohort-jsonl", type=Path, help="Benchmark-only raw generated records with actual output_token_ids")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--benchmark-steps", type=int, default=3)
    args = parser.parse_args()
    if args.cohort_jsonl and not args.benchmark_updates:
        parser.error("offline cohorts are restricted to update-throughput benchmarking")
    if args.benchmark_updates and args.benchmark_generation:
        parser.error("choose one benchmark mode")
    if args.resume and (args.adapter or args.benchmark_updates or args.benchmark_generation):
        parser.error("exact resume is training-only and cannot be combined with a weights warm start")
    if args.loss_normalization is None:
        if args.benchmark_updates or args.benchmark_generation:
            args.loss_normalization = "response_mean"
        else:
            parser.error("actual training requires an explicit --loss-normalization choice")
    if args.dtype == "float16":
        parser.error("float16 requires loss scaling; use bfloat16 or float32")
    positive = (args.temperature, args.max_new_tokens, args.max_sequence_length, args.group_size,
                args.generation_microbatch, args.microbatch, args.gradient_accumulation,
                args.states_per_iteration, args.iterations, args.update_epochs, args.benchmark_steps,
                args.learning_rate, args.max_grad_norm, args.lora_rank, args.lora_alpha,
                args.generation_warmup_responses, args.loss_token_budget)
    if min(positive) <= 0 or args.group_size < 2 or args.warmup_steps < 0:
        parser.error("positive sizes/temperature and group-size >=2 required")
    if (not all(math.isfinite(value) for value in positive) or
            not math.isfinite(args.invalid_reward) or not 0 < args.clip_epsilon < 1 or args.kl_beta < 0 or
            not math.isfinite(args.kl_beta) or args.limit is not None and args.limit <= 0):
        parser.error("finite settings, positive limit, 0<clip-epsilon<1, and nonnegative kl-beta required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stop_file is None:
        args.stop_file = args.output / "STOP"
    if args.smoke:
        args.states_per_iteration = args.iterations = args.update_epochs = 1
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(sampling_top_p=1.0, reasoning_sft=False,
                  prompt_file_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                  source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  training_format_version=TRAINING_FORMAT_VERSION,
                  source_files_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in Path(__file__).parent.glob("*.py")},
                  warm_start_resets_optimizer=bool(args.adapter),
                  exact_resume=bool(args.resume))
    checkpoint_source = Path(args.resume or args.adapter) if args.resume or args.adapter else None
    if checkpoint_source:
        config["input_checkpoint_files_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in checkpoint_source.glob("*")
            if path.is_file() and (path.suffix in (".safetensors", ".pt") or path.name == "adapter_config.json")}
    atomic_json(args.output / "config.json", config)
    try:
        import torch
        torch.manual_seed(args.seed)
        model, tokenizer, stop_id = load_model(args)
        rows = load_rows(args.prompts, tokenizer, args)
        config.update(stop_token_ids=[stop_id], torch_version=torch.__version__,
                      selected_split=args.selected_split,
                      selected_row_count=args.selected_row_count,
                      selected_selection_sha256=args.selected_selection_sha256,
                      cuda_version=torch.version.cuda,
                      device_name=torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      teacher_checkpoint_sha256=sorted({row["teacher"]["checkpoint_sha256"] for row in rows
                        if "teacher" in row and row["teacher"].get("checkpoint_sha256")}))
        atomic_json(args.output / "config.json", config)
        if args.benchmark_generation:
            summary = benchmark_generation(model, tokenizer, stop_id, rows, args, args.output)
        elif args.benchmark_updates:
            if args.cohort_jsonl:
                cohort = offline_benchmark_cohort(args.cohort_jsonl, rows, tokenizer, args)
            else:
                cohort, _ = sample_cohort(model, tokenizer, stop_id, rows[:args.states_per_iteration], args, iteration=0)
            summary = benchmark_updates(model, tokenizer, cohort, args, args.output)
        else:
            summary = training_loop(model, tokenizer, stop_id, rows, args, args.output)
        print(json.dumps(summary), flush=True)
        return 0
    except Exception as exc:
        progress_path = args.output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
        atomic_json(args.output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
            "last_completed_checkpoint": progress[-1]["adapter"] if progress else str(args.resume or args.adapter or "base model"),
            "completed_iterations_in_output": len(progress),
            "cohort_files_retained": [str(path) for path in args.output.glob("cohort_*.jsonl")]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
