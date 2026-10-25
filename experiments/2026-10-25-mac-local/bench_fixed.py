"""Equal-work MLX generation probe, matching the RTX fixed-token probe.

Sends `requests` prompts of exactly `input_tokens` tokens (real panel prompt
tokens, repeated to length) and decodes exactly `output_tokens` each with no
stop tokens, at most `concurrency` at a time. Output tokens per second then
measures execution speed without natural length differences. A separate
single-prompt pass reports prefill and decode rates on their own.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = "IFM/K2-Horizon-0.9B"
REVISION = "7c75f2f49e74ac97c1852e62b4c860091d6a00ea"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, action="append", required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import BatchGenerator
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model, _ = load(str(args.mlx_model))
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    prompts = []
    for index in range(args.requests):
        ids = list(tokenizer.apply_chat_template(rows[index % len(rows)]["messages"], tokenize=True,
                   add_generation_prompt=True, return_dict=False, reasoning_effort="medium"))
        prompts.append((ids * (args.input_tokens // len(ids) + 1))[:args.input_tokens])
    results = {}
    # Single stream: prefill and decode rates separately (mlx-lm's own counters).
    for response in stream_generate(model, _, prompts[0], max_tokens=args.output_tokens):
        last = response
    results["single_stream"] = {"prompt_tokens_per_second": last.prompt_tps,
                                "decode_tokens_per_second": last.generation_tps,
                                "peak_memory_gb": last.peak_memory}
    for concurrency in args.concurrency:
        generator = BatchGenerator(model, max_tokens=args.output_tokens, stop_tokens=None,
                                   completion_batch_size=concurrency, prefill_batch_size=min(8, concurrency))
        mx.reset_peak_memory()
        began = time.perf_counter()
        generator.insert(prompts, [args.output_tokens] * len(prompts))
        produced = 0
        with generator.stats() as stats:
            while responses := generator.next_generated():
                produced += len(responses)
        seconds = time.perf_counter() - began
        generator.close()
        results[f"concurrency_{concurrency}"] = {
            "requests": len(prompts), "output_tokens": produced, "seconds": seconds,
            "output_tokens_per_second": produced / seconds,
            "prompt_tokens_per_second_during_batch": stats.prompt_tps,
            "decode_tokens_per_second_during_batch": stats.generation_tps,
            "peak_memory_gb": mx.get_peak_memory() / 1e9}
        print(json.dumps({f"concurrency_{concurrency}": results[f"concurrency_{concurrency}"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                                       **results}, indent=2) + "\n")
    print(json.dumps(results["single_stream"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
