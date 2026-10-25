"""Check that the MLX conversion scores saved completions like the original model.

Teacher-forces saved K2 completions through the original Hugging Face model
(FP32 on CPU as the reference; BF16 on MPS as a precision control) and through
the MLX Llama conversion (FP32 and BF16), then compares generated-token
log-probabilities and greedy next-token agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = "IFM/K2-Horizon-0.9B"
REVISION = "7c75f2f49e74ac97c1852e62b4c860091d6a00ea"


def load_examples(prompts: Path, cohort: Path, tokenizer, effort: str, count: int):
    rows = {json.loads(line)["id"]: json.loads(line) for line in prompts.read_text().splitlines() if line.strip()}
    examples = []
    for line in cohort.read_text().splitlines():
        record = json.loads(line)
        if record["prompt_id"] not in rows or not record.get("output_token_ids"):
            continue
        prompt_ids = tokenizer.apply_chat_template(rows[record["prompt_id"]]["messages"], tokenize=True,
                                                   add_generation_prompt=True, return_dict=False,
                                                   reasoning_effort=effort)
        examples.append((record["prompt_id"], list(prompt_ids), list(record["output_token_ids"])))
        if len(examples) >= count:
            break
    return examples


def torch_scores(model, prompt, output, device):
    import torch
    ids = torch.tensor([prompt + output], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1].float()
    logps = torch.log_softmax(logits, -1)
    return logps.gather(-1, torch.tensor(output, device=device)[:, None])[:, 0].cpu().numpy(), logits.argmax(-1).cpu().numpy()


def mlx_scores(model, prompt, output):
    import mlx.core as mx
    logits = model(mx.array([prompt + output]))[0, len(prompt) - 1:-1].astype(mx.float32)
    logps = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logps, mx.array(output)[:, None], axis=-1)[:, 0]
    return np.array(picked), np.array(mx.argmax(logits, axis=-1))


def compare(reference, candidate):
    (ref_lp, ref_arg), (lp, arg) = reference, candidate
    diff = np.abs(ref_lp - lp)
    return {"mean_abs_logprob_diff": float(diff.mean()), "max_abs_logprob_diff": float(diff.max()),
            "sequence_logprob_diff": float(lp.sum() - ref_lp.sum()),
            "greedy_argmax_agreement": float((ref_arg == arg).mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    import mlx.core as mx
    import torch
    from mlx.utils import tree_map
    from mlx_lm import load
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    examples = load_examples(args.prompts, args.cohort, tokenizer, args.effort, args.count)
    reference = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, code_revision=REVISION,
        trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    ref = {pid: torch_scores(reference, p, o, "cpu") for pid, p, o in examples}
    del reference
    conditions = {}
    hf_bf16 = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, code_revision=REVISION,
        trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()
    conditions["hf_bf16_mps"] = {pid: compare(ref[pid], torch_scores(hf_bf16, p, o, "mps")) for pid, p, o in examples}
    del hf_bf16
    model, _ = load(str(args.mlx_model))
    conditions["mlx_bf16"] = {pid: compare(ref[pid], mlx_scores(model, p, o)) for pid, p, o in examples}
    model.update(tree_map(lambda value: value.astype(mx.float32), model.parameters()))
    conditions["mlx_fp32"] = {pid: compare(ref[pid], mlx_scores(model, p, o)) for pid, p, o in examples}
    result = {"reference": "hf_fp32_cpu", "examples": [{"prompt_id": pid, "prompt_tokens": len(p),
              "output_tokens": len(o)} for pid, p, o in examples], "conditions": conditions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in conditions.items():
        print(name, {key: round(float(np.mean([v[key] for v in values.values()])), 5)
                     for key in next(iter(values.values()))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
