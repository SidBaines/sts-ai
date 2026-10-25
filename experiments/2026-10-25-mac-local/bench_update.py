"""Time LoRA policy-gradient updates on Apple silicon with MLX.

Mirrors `sts_llm.training --benchmark-updates --cohort-jsonl`: the same saved
completions, LoRA rank 16 / alpha 32 on every attention and MLP projection,
temperature-scaled generated-token log-probabilities, the clipped surrogate plus
the sampled k3 KL proxy, gradient-norm clipping at 1 and AdamW. Advantages are
synthetic (alternating +1/-1) so every step has a real backward pass; this
measures speed and memory, not learning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = "IFM/K2-Horizon-0.9B"
REVISION = "7c75f2f49e74ac97c1852e62b4c860091d6a00ea"
LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def load_cohort(prompts: Path, cohort: Path, tokenizer, effort: str):
    rows = {json.loads(line)["id"]: json.loads(line) for line in prompts.read_text().splitlines() if line.strip()}
    items = []
    for line in cohort.read_text().splitlines():
        record = json.loads(line)
        if record["prompt_id"] not in rows or not record.get("output_token_ids"):
            continue
        prompt = list(tokenizer.apply_chat_template(rows[record["prompt_id"]]["messages"], tokenize=True,
                                                    add_generation_prompt=True, return_dict=False,
                                                    reasoning_effort=effort))
        items.append({"prompt": prompt, "output": list(record["output_token_ids"]),
                      "advantage": -1.0 if len(items) % 2 == 0 else 1.0})
    return items


def batch_arrays(items):
    """Right-pad prompt+completion; mask marks generated-token predictions."""
    import mlx.core as mx
    start = min(len(item["prompt"]) for item in items) - 1
    width = max(len(item["prompt"]) + len(item["output"]) for item in items)
    ids, mask = [], []
    for item in items:
        tokens = item["prompt"] + item["output"]
        ids.append(tokens + [0] * (width - len(tokens)))
        row = [0.0] * (width - 1 - start)
        for position in range(len(item["prompt"]) - 1, len(tokens) - 1):
            row[position - start] = 1.0
        mask.append(row)
    return mx.array(ids), mx.array(mask), start


def token_logps(model, ids, start, temperature):
    import mlx.core as mx
    hidden = model.model(ids)[:, start:-1]
    logits = model.lm_head(hidden).astype(mx.float32) / temperature
    logps = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(logps, ids[:, start + 1:, None], axis=-1)[..., 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--benchmark-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--repeat-cohort", type=int, default=1,
                        help="Concatenate the cohort with itself (longer runs, same lengths)")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    cohort = load_cohort(args.prompts, args.cohort, tokenizer, args.effort) * args.repeat_cohort
    model, _ = load(str(args.mlx_model))
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {"rank": args.lora_rank, "dropout": 0.0,
                          "scale": args.lora_alpha / args.lora_rank, "keys": LORA_KEYS})
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    # Frozen behaviour and reference scores, once per cohort (as in training).
    mx.reset_peak_memory()
    began = time.perf_counter()
    for item in cohort:
        ids, mask, start = batch_arrays([item])
        scores = token_logps(model, ids, start, args.temperature)
        mx.eval(scores)
        item["old"] = item["ref"] = scores[0]
    freeze_seconds = time.perf_counter() - began

    def loss_fn(model, ids, mask, start, old, ref, advantages):
        new = token_logps(model, ids, start, args.temperature)
        ratio = mx.exp((new - old) * mask)
        surrogate = mx.minimum(ratio * advantages[:, None],
                               mx.clip(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages[:, None])
        delta = (ref - new) * mask
        kl = mx.exp(delta) - 1 - delta
        lengths = mask.sum(-1)
        per_response = ((-surrogate + args.kl_beta * kl) * mask).sum(-1) / lengths
        return per_response.mean()

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    effective = args.microbatch * args.gradient_accumulation
    history = []
    for step in range(args.warmup_steps + args.benchmark_steps):
        batch = [cohort[(step * effective + i) % len(cohort)] for i in range(effective)]
        if step == args.warmup_steps:
            mx.reset_peak_memory()
            before = {k: v for k, v in tree_flatten(model.trainable_parameters())}
        began = time.perf_counter()
        grads_total, loss_total = None, 0.0
        for offset in range(0, effective, args.microbatch):
            chunk = batch[offset:offset + args.microbatch]
            ids, mask, start = batch_arrays(chunk)
            width = mask.shape[1]
            old = mx.stack([pad_scores(item, start, width, "old") for item in chunk])
            ref = mx.stack([pad_scores(item, start, width, "ref") for item in chunk])
            advantages = mx.array([item["advantage"] for item in chunk])
            loss, grads = value_and_grad(model, ids, mask, start, old, ref, advantages)
            weight = len(chunk) / effective
            grads = tree_map(lambda g: g * weight, grads)
            grads_total = grads if grads_total is None else tree_map(mx.add, grads_total, grads)
            mx.eval(loss, grads_total)
            loss_total += loss.item() * weight
        grads_total, norm = optim.clip_grad_norm(grads_total, 1.0)
        optimizer.update(model, grads_total)
        mx.eval(model.trainable_parameters(), optimizer.state)
        seconds = time.perf_counter() - began
        history.append({"step": step, "warmup": step < args.warmup_steps, "seconds": seconds, "loss": loss_total,
                        "gradient_norm_before_clip": norm.item(), "responses": len(batch),
                        "generated_tokens": sum(len(item["output"]) for item in batch)})
        print(json.dumps(history[-1]), flush=True)
    after = {k: v for k, v in tree_flatten(model.trainable_parameters())}
    changed = any(not mx.array_equal(before[k], after[k]).item() for k in before)
    timed = [h for h in history if not h["warmup"]]
    total = sum(h["seconds"] for h in timed)
    summary = {"backend": "mlx", "microbatch": args.microbatch, "gradient_accumulation": args.gradient_accumulation,
               "grad_checkpoint": args.grad_checkpoint, "trainable_parameters": trainable,
               "frozen_logprob_seconds_per_response": freeze_seconds / len(cohort),
               "timed_steps": len(timed), "timed_seconds": total,
               "responses_per_second": sum(h["responses"] for h in timed) / total,
               "generated_training_tokens_per_second": sum(h["generated_tokens"] for h in timed) / total,
               "peak_memory_gb": mx.get_peak_memory() / 1e9, "timed_trainable_weights_changed": changed,
               "prompt_token_lengths": [len(i["prompt"]) for i in cohort],
               "response_token_lengths": [len(i["output"]) for i in cohort],
               "synthetic_advantages": True, "steps": history}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("steps", "prompt_token_lengths")}))
    return 0


def pad_scores(item, start, width, key):
    """Place one response's frozen scores at its generated-token positions."""
    import mlx.core as mx
    first = len(item["prompt"]) - 1 - start
    values = item[key]
    return mx.concatenate([mx.zeros((first,)), values, mx.zeros((width - first - values.shape[0],))])


if __name__ == "__main__":
    raise SystemExit(main())
