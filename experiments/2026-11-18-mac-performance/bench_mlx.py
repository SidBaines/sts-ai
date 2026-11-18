"""Compare greedy job ordering and token scoring on the saved K2 LoRA adapter."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import MethodType, SimpleNamespace


def gather_first(policy, prompt, output):
    mx = policy.mx
    ids = mx.array([prompt + output])
    start = len(prompt) - 1
    hidden = policy.model.model(ids)[:, start:-1]
    logits = policy.model.lm_head(hidden).astype(mx.float32) / policy.args.temperature
    target = mx.take_along_axis(logits, ids[:, start + 1 :, None], axis=-1)
    return (target - mx.logsumexp(logits, axis=-1, keepdims=True))[0, :, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worktree", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    opts = p.parse_args()
    import sys

    sys.path.insert(0, str(opts.worktree / "src"))
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from sts_llm.mlx_training import MLXPolicy, labelled_prompt

    runs = opts.worktree / "data/llm_runs"
    source = runs / "learning_trials_20261027/lr_small_step"
    args = SimpleNamespace(
        **json.loads((source / "config.json").read_text())["settings"]
    )
    args.mlx_model = opts.worktree / "data/models/k2-horizon-0.9b-mlx"
    args.adapter = (
        runs / "stage1_screen_ps_20261026/lr_3e-5/adapters/iter_0048.safetensors"
    )
    args.reference_adapter = args.adapter
    policy = MLXPolicy(args)
    summary = {"generation": [], "scoring": [], "gradients": []}

    def save():
        opts.out.write_text(json.dumps(summary, indent=2) + "\n")

    def read(path):
        return list(map(json.loads, path.open()))

    states = {
        r["id"]: r for r in read(runs / "broader_states_selected_20261026/probe.jsonl")
    }
    jobs = [
        r
        for r in read(
            runs / "learning_trials_20261027/lr_full_parent/evaluations/iter_0000.jsonl"
        )
        if r["id"] in states
    ][::3]
    prompts = [
        policy.prompt_ids(
            labelled_prompt(states[r["id"]]["observation"], r["order"]).messages
        )
        for r in jobs
    ]
    summary["input_lengths"] = list(map(len, prompts))
    # Fixed budget for this diagnostic. Real training/evaluation caps do not change.
    args.max_new_tokens = 96
    width = 8
    results = {}

    def generate(arm):
        order = list(range(len(jobs)))
        if arm == "sorted":
            order.sort(key=lambda i: len(prompts[i]))
        out = policy.generate(
            [prompts[i] for i in order],
            temperature=0.0,
            seed=args.seed,
            concurrency=width,
        )
        return dict(zip(order, out))

    policy.generate([prompts[0]], temperature=0.0, seed=args.seed, concurrency=1)
    for arm in ("original", "sorted", "sorted", "original"):
        mx.synchronize()
        mx.reset_peak_memory()
        start = time.perf_counter()
        values = generate(arm)
        mx.synchronize()
        rec = {
            "arm": arm,
            "seconds": time.perf_counter() - start,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "generated_tokens": sum(len(y[0]) for y in values.values()),
            "capped": sum(y[1] == "length" for y in values.values()),
            "hash": hashlib.sha256(
                json.dumps(sorted(values.items())).encode()
            ).hexdigest(),
        }
        if "original" in results:
            rec["identical_sequences"] = sum(
                v == results["original"][i] for i, v in values.items()
            )
        results[arm] = values
        summary["generation"].append(rec)
        save()
        print(rec, flush=True)
    examples = [(pr, r["output_token_ids"]) for pr, r in zip(prompts, jobs)][:6]
    summary["scoring_lengths"] = [(len(a), len(b)) for a, b in examples]
    original = policy.token_logps
    ref_out = None
    for arm in ("original", "gather_first", "gather_first", "original"):
        method = original if arm == "original" else MethodType(gather_first, policy)
        mx.synchronize()
        mx.reset_peak_memory()
        start = time.perf_counter()
        values = []
        for a, b in examples:
            arr = method(a, b)
            mx.eval(arr)
            values.append(np.asarray(arr))
        if ref_out is None:
            ref_out = values
        rec = {
            "arm": arm,
            "seconds": time.perf_counter() - start,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "max_abs_logp_diff": max(
                float(np.max(np.abs(v - r))) for v, r in zip(values, ref_out)
            ),
        }
        summary["scoring"].append(rec)
        save()
        print(rec, flush=True)
    a, b = examples[0]
    policy.model.train()
    reference = None
    for arm in ("original", "gather_first", "gather_first", "original"):
        method = original if arm == "original" else MethodType(gather_first, policy)
        mx.synchronize()
        mx.reset_peak_memory()
        start = time.perf_counter()
        loss, grads = nn.value_and_grad(
            policy.model, lambda: -method(a, b).sum() / 2048
        )()
        mx.eval(loss, grads)
        seconds = time.perf_counter() - start
        flat = dict(tree_flatten(grads))
        if reference is None:
            reference = flat
        diff = max(
            float(
                mx.max(
                    mx.abs(v.astype(mx.float32) - reference[k].astype(mx.float32))
                ).item()
            )
            for k, v in flat.items()
        )
        rec = {
            "arm": arm,
            "seconds": seconds,
            "loss": loss.item(),
            "max_abs_gradient_diff": diff,
            "parameters": len(flat),
        }
        summary["gradients"].append(rec)
        save()
        print(rec, flush=True)


if __name__ == "__main__":
    main()
