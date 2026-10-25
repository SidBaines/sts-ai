"""Time batched K2 generation with MLX on Apple silicon, matching the RTX serving panel.

Each case renders the saved public-state prompts with the requested reasoning
effort, generates with mlx-lm's continuous BatchGenerator (at most `concurrency`
sequences decoding at once), and scores every response with the same strict
parser and teacher table as the RTX study. Raw output token IDs are retained.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO = "IFM/K2-Horizon-0.9B"
REVISION = "7c75f2f49e74ac97c1852e62b4c860091d6a00ea"


def teacher_probability(row, action):
    if action is None:
        return 0.0
    table = dict(zip(row["teacher"]["legal_indices"], row["teacher"]["probabilities"]))
    return float(table.get(action, 0.0))


def run_case(model, tokenizer, rows, teacher, *, effort, concurrency, samples, temperature,
             top_p, max_tokens, seed, output: Path, prefill_batch_size: int = 8, grouped: bool = False):
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from sts_llm.generation import K2_ASSISTANT_END_TOKEN_ID, parse_action

    stops = [[K2_ASSISTANT_END_TOKEN_ID], [tokenizer.eos_token_id]]
    prompts, meta = [], []
    for row in rows:
        ids = list(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                                 return_dict=False, reasoning_effort=effort))
        for repeat in range(samples):
            prompts.append(ids)
            meta.append((row, repeat))
    mx.random.seed(seed)
    # Grouped mode generates one state's samples per batch: identical prompts
    # need no padding mask, unlike a batch mixing prompts of different lengths.
    batches = ([list(range(i, i + samples)) for i in range(0, len(prompts), samples)]
               if grouped else [list(range(len(prompts)))])
    mx.reset_peak_memory()
    began = time.perf_counter()
    outputs = {}
    for batch in batches:
        generator = BatchGenerator(model, max_tokens=max_tokens, stop_tokens=stops,
                                   sampler=make_sampler(temp=temperature, top_p=top_p),
                                   completion_batch_size=concurrency,
                                   prefill_batch_size=min(prefill_batch_size, concurrency))
        uids = generator.insert([prompts[i] for i in batch], [max_tokens] * len(batch))
        position = dict(zip(uids, batch))
        tokens = {uid: [] for uid in uids}
        while responses := generator.next_generated():
            for response in responses:
                tokens[response.uid].append(int(response.token))
                if response.finish_reason is not None:
                    outputs[position[response.uid]] = (tokens[response.uid], response.finish_reason,
                                                       time.perf_counter() - began)
        generator.close()
    seconds = time.perf_counter() - began
    records = []
    for index, (row, repeat) in enumerate(meta):
        output_ids, reason, latency = outputs[index]
        text = tokenizer.decode(output_ids, skip_special_tokens=False)
        parsed = parse_action(text, row["legal_indices"], finish_reason=reason)
        action = parsed.action_index if parsed.valid else None
        records.append({"prompt_id": row["id"], "phase": row["phase"], "repeat": repeat,
                        "legal_indices": row["legal_indices"], "output_token_ids": output_ids,
                        "raw_text": text, "finish_reason": reason, "completion_seconds": latency,
                        "valid": parsed.valid, "error": parsed.error, "action_index": action,
                        "first_option": action == row["legal_indices"][0] if action is not None else None,
                        "teacher_probability": teacher_probability(teacher[row["id"]], action)})
    valid = sum(item["valid"] for item in records)
    generated = sum(len(item["output_token_ids"]) for item in records)
    lengths = sorted(len(item["output_token_ids"]) for item in records)
    summary = {
        "effort": effort, "concurrency": concurrency, "states": len(rows), "samples_per_state": samples,
        "responses": len(records), "valid": valid, "capped": sum(r["finish_reason"] == "length" for r in records),
        "seconds": seconds, "valid_decisions_per_second": valid / seconds,
        "generated_tokens": generated, "generated_tokens_per_second": generated / seconds,
        "median_output_tokens": statistics.median(lengths), "max_output_tokens": lengths[-1],
        "mean_teacher_probability": statistics.fmean(r["teacher_probability"] for r in records),
        "first_option_rate_of_valid": (sum(bool(r["first_option"]) for r in records) / valid) if valid else None,
        "peak_memory_gb": mx.get_peak_memory() / 1e9, "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens, "seed": seed, "backend": "mlx-lm BatchGenerator, BF16 weights",
        "prefill_batch_size": min(prefill_batch_size, concurrency), "one_group_per_batch": grouped,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Run directory; one subdirectory per case")
    parser.add_argument("--case", action="append", required=True,
                        help="name:effort:concurrency:samples:top_p:max_tokens[:limit]")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--warmup-prompts", type=Path)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--one-group-at-a-time", action="store_true",
                        help="Generate each state's samples as their own batch (no mixed prompt lengths)")
    args = parser.parse_args()

    from mlx_lm import load
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model, _ = load(str(args.mlx_model))
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    teacher = {json.loads(line)["id"]: json.loads(line) for line in args.teacher.read_text().splitlines() if line.strip()}
    if args.warmup_prompts:
        warm = [json.loads(line) for line in args.warmup_prompts.read_text().splitlines() if line.strip()][:2]
        for row in warm:
            teacher.setdefault(row["id"], {"teacher": {"legal_indices": [], "probabilities": []}})
        run_case(model, tokenizer, warm, teacher, effort="medium", concurrency=2, samples=1,
                 temperature=args.temperature, top_p=0.95, max_tokens=256, seed=args.seed,
                 output=args.output / "warmup")
    for spec in args.case:
        parts = spec.split(":")
        name, effort, concurrency, samples, top_p, max_tokens = parts[:6]
        selected = rows[:int(parts[6])] if len(parts) > 6 else rows
        summary = run_case(model, tokenizer, selected, teacher, effort=effort, concurrency=int(concurrency),
                           samples=int(samples), temperature=args.temperature, top_p=float(top_p),
                           max_tokens=int(max_tokens), seed=args.seed, output=args.output / name,
                           prefill_batch_size=args.prefill_batch_size, grouped=args.one_group_at_a_time)
        print(json.dumps({"case": name, **{k: summary[k] for k in (
            "responses", "valid", "capped", "seconds", "valid_decisions_per_second",
            "generated_tokens_per_second", "mean_teacher_probability", "first_option_rate_of_valid",
            "peak_memory_gb")}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
