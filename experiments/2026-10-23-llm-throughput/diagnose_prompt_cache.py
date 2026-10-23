#!/usr/bin/env python3
"""Bounded K2 cache-math versus dtype/batch diagnostic. No optimizer steps.

Run separately for --dtype bfloat16 and float32, using an external shell
deadline and a fresh output directory. The model loads once per invocation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from sts_llm.training import (atomic_json, expanded_prompt_cache, load_model,
                              load_rows, prefill_prompt_cache, synchronize)
from sts_llm.generation import parse_action


def tensor_summary(value):
    if value is None:
        return None
    if not hasattr(value, "shape"):
        return str(value)
    flat = value.detach().reshape(-1)
    return {"shape": list(value.shape), "first": flat[:8].cpu().tolist(),
            "last": flat[-8:].cpu().tolist()}


def distribution_comparison(reference, candidate, top_k):
    import torch
    reference, candidate = reference.float(), candidate.float()
    if candidate.shape[0] == 1:
        candidate = candidate.expand_as(reference)
    log_p = torch.log_softmax(reference, dim=-1)
    log_q = torch.log_softmax(candidate, dim=-1)
    p, q = log_p.exp(), log_q.exp()
    difference = reference - candidate
    top_ids = torch.topk(reference[0], top_k).indices
    candidate_top = torch.topk(candidate[0], top_k).indices
    return {
        "max_abs_logit_difference": float(difference.abs().max()),
        "mean_abs_logit_difference": float(difference.abs().mean()),
        "rms_logit_difference": float(difference.square().mean().sqrt()),
        "kl_reference_to_candidate_by_row": (p * (log_p - log_q)).sum(-1).cpu().tolist(),
        "kl_candidate_to_reference_by_row": (q * (log_q - log_p)).sum(-1).cpu().tolist(),
        "total_variation_by_row": (.5 * (p - q).abs().sum(-1)).cpu().tolist(),
        "argmax_reference": reference.argmax(-1).cpu().tolist(),
        "argmax_candidate": candidate.argmax(-1).cpu().tolist(),
        "reference_top_ids": top_ids.cpu().tolist(),
        "reference_top_logits": reference[0, top_ids].cpu().tolist(),
        "candidate_logits_at_reference_top": candidate[0, top_ids].cpu().tolist(),
        "candidate_top_ids": candidate_top.cpu().tolist(),
        "candidate_top_logits": candidate[0, candidate_top].cpu().tolist(),
        "candidate_within_batch_max_abs_spread": float((candidate - candidate[:1]).abs().max()),
    }


def greedy(model, tokenizer, row, full, prefix, stop_id, args):
    """Trace the actual first forwards inside HF generate with supplied past."""
    import torch
    trace = []
    base = model.get_base_model()
    def hook(_module, positional, kwargs):
        if len(trace) >= 3:
            return
        cache = kwargs.get("past_key_values")
        trace.append({"input_ids": tensor_summary(kwargs.get("input_ids", positional[0] if positional else None)),
            "attention_mask": tensor_summary(kwargs.get("attention_mask")),
            "position_ids": tensor_summary(kwargs.get("position_ids")),
            "cache_position": tensor_summary(kwargs.get("cache_position")),
            "past_length_before_forward": cache.get_seq_length() if cache is not None else None,
            "use_cache": kwargs.get("use_cache")})
    handle = base.register_forward_pre_hook(hook, with_kwargs=True)
    past = None
    if prefix is not None:
        source_batch = next(layer.keys.shape[0] for layer in prefix.layers if layer.keys is not None)
        if source_batch not in {1, full.shape[0]}:
            raise ValueError("prefix batch does not match the diagnostic generation batch")
        past = expanded_prompt_cache(prefix, full.shape[0] // source_batch)
    try:
        with torch.no_grad():
            output = model.generate(input_ids=full, attention_mask=torch.ones_like(full),
                past_key_values=past, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=[stop_id, tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id,
                use_cache=True, repetition_penalty=1.0)
    finally:
        handle.remove()
    del past
    records = []
    for sequence in output:
        ids = sequence[full.shape[1]:].tolist()
        ends = [i for i, token in enumerate(ids) if token in {stop_id, tokenizer.eos_token_id}]
        if ends:
            ids = ids[:ends[0] + 1]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        parsed = parse_action(text, row["legal_indices"], finish_reason="stop" if ends else "length")
        records.append({"output_token_ids": ids, "raw_text": text, "valid": parsed.valid,
            "truncated": not bool(ends), "action_index": parsed.action_index, "error": parsed.error})
    return {"forward_trace": trace, "records": records}


def diagnose_state(model, tokenizer, stop_id, row, args, output):
    import torch
    device = next(model.parameters()).device
    one = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    full = one.repeat(args.batch_size, 1)
    model.eval()
    logits = {}
    with torch.no_grad():
        logits["full_b4_no_cache"] = model(input_ids=full, attention_mask=torch.ones_like(full),
            use_cache=False, logits_to_keep=1).logits[:, -1].detach()
        result = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=True, logits_to_keep=1)
        logits["full_b4_with_cache"] = result.logits[:, -1].detach()
        del result
        logits["full_b1_no_cache"] = model(input_ids=one, attention_mask=torch.ones_like(one),
            use_cache=False, logits_to_keep=1).logits[:, -1].detach()
        prefix_one = prefill_prompt_cache(model, row["prompt_ids"])
        cache = expanded_prompt_cache(prefix_one, args.batch_size)
        result = model(input_ids=full[:, -1:], attention_mask=torch.ones_like(full),
            past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits["split_prefix_b1_expand_b4"] = result.logits[:, -1].detach()
        del result, cache
        result = model(input_ids=full[:, :-1], attention_mask=torch.ones_like(full[:, :-1]),
            use_cache=True, logits_to_keep=1)
        prefix_batch = result.past_key_values
        del result
        cache = expanded_prompt_cache(prefix_batch, 1)
        result = model(input_ids=full[:, -1:], attention_mask=torch.ones_like(full),
            past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits["split_prefix_b4"] = result.logits[:, -1].detach()
        del result, cache
        cache = expanded_prompt_cache(prefix_one, 1)
        result = model(input_ids=one[:, -1:], attention_mask=torch.ones_like(one),
            past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits["split_prefix_b1_decode_b1"] = result.logits[:, -1].detach()
        del result, cache
    reference = logits["full_b4_with_cache"]
    comparisons = {name: distribution_comparison(reference, value, args.top_k)
                   for name, value in logits.items()}
    state_dir = output / row["id"]
    state_dir.mkdir()
    torch.save({name: value.cpu() for name, value in logits.items()}, state_dir / "next_token_logits.pt")
    result = {"prompt_id": row["id"], "phase": row["phase"], "prompt_length": len(row["prompt_ids"]),
        "batch_size": args.batch_size, "reference": "full_b4_with_cache", "comparisons": comparisons,
        "logits_file": str(state_dir / "next_token_logits.pt")}
    atomic_json(state_dir / "logits.json", result)
    print(json.dumps({"event":"logits_done", "prompt_id":row["id"],
        "max_differences":{name:value["max_abs_logit_difference"] for name, value in comparisons.items()}}), flush=True)
    if not args.skip_greedy:
        for name, prefix in (("uncached", None), ("prefix_b1_expanded", prefix_one), ("prefix_b4", prefix_batch)):
            began = time.perf_counter()
            result[name] = greedy(model, tokenizer, row, full, prefix, stop_id, args)
            synchronize(args.device)
            result[name]["seconds"] = time.perf_counter() - began
            atomic_json(state_dir / "comparison.json", result)
            print(json.dumps({"event":"greedy_done", "prompt_id":row["id"], "variant":name,
                              "seconds":result[name]["seconds"]}), flush=True)
        baseline = result["uncached"]["records"]
        for variant in ("prefix_b1_expanded", "prefix_b4"):
            other = result[variant]["records"]
            result[variant]["token_ids_match_baseline"] = [a["output_token_ids"] == b["output_token_ids"]
                                                            for a, b in zip(baseline, other)]
            result[variant]["completed_action_match_baseline"] = [
                a["action_index"] == b["action_index"] if a["valid"] and b["valid"] else None
                for a, b in zip(baseline, other)]
        result["prefix_one_length_unchanged"] = prefix_one.get_seq_length() == len(row["prompt_ids"]) - 1
        result["prefix_batch_length_unchanged"] = prefix_batch.get_seq_length() == len(row["prompt_ids"]) - 1
    atomic_json(state_dir / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--revision", default="7c75f2f49e74ac97c1852e62b4c860091d6a00ea")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=40960)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if min(args.states, args.batch_size, args.max_new_tokens, args.top_k) <= 0:
        parser.error("counts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a fresh output directory; failed probes are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    args.resume = None
    args.lora_rank, args.lora_alpha = 16, 32
    args.lora_targets = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    args.gradient_checkpointing = False
    args.benchmark_updates, args.benchmark_generation = False, True
    args.static_cache_compile = False
    args.limit = args.states
    config = {key:str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(prompt_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                  diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  deadline="external shell timeout required", optimizer_steps=0)
    atomic_json(args.output / "config.json", config)
    import torch
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, stop_id = load_model(args)
    rows = load_rows(args.prompts, tokenizer, args)
    config.update(torch_version=torch.__version__, actual_parameter_dtype=str(next(model.parameters()).dtype),
                  tf32=False, device_name=torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device)
    atomic_json(args.output / "config.json", config)
    results = []
    for row in rows:
        results.append(diagnose_state(model, tokenizer, stop_id, row, args, args.output))
        atomic_json(args.output / "summary.json", {"dtype":args.dtype, "results":results,
            "interpretation":"Compare FP32 split/full with BF16 and batch-size effects; no tolerance was changed or gate relaxed."})
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
