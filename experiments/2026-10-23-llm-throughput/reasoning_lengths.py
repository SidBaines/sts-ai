#!/usr/bin/env python3
"""Count reasoning tokens in saved serving responses (produces reasoning_lengths.csv).

Reasoning tokens are the server-returned output token IDs before the first K2
reasoning closer, excluding the tokenizer's added control tokens; the final JSON
and end tokens after the closer are not counted. A response with no closer is
counted as unclosed and kept out of the distribution, because its length is only
a lower bound. Requires Transformers and the pinned tokenizer revision.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from sts_llm.generation import K2_REASONING_CLOSERS

DEFAULT_CASES = ("fi_graphs_b8192_low_c32", "fi_graphs_b8192_medium_c32", "fi_graphs_b8192_high_c32")


def percentile(values, q):
    """Linear interpolation between closest ranks (NumPy's default method)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def reasoning_count(token_ids, closer_ids, control_ids):
    """Return (reasoning tokens, closed) for one response's output token IDs."""
    end = next((i for i, token in enumerate(token_ids) if token in closer_ids), None)
    prefix = token_ids if end is None else token_ids[:end]
    return sum(token not in control_ids for token in prefix), end is not None


def summarize(path, closer_ids, control_ids):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts, unclosed, effort = [], 0, None
    for row in rows:
        effort = effort or row.get("effort")
        token_ids = row["output_token_ids"]
        if not isinstance(token_ids, list):
            raise ValueError(f"{path}: output_token_ids missing for {row.get('prompt_id')}")
        count, closed = reasoning_count(token_ids, closer_ids, control_ids)
        if closed:
            counts.append(count)
        else:
            unclosed += 1
    return {"effort": effort, "n": len(rows),
            "mean": statistics.mean(counts) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "p95": percentile(counts, .95) if counts else None,
            "maximum": max(counts) if counts else None,
            "empty_reasoning": sum(c == 0 for c in counts), "unclosed_reasoning": unclosed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Directory holding <case>/requests.jsonl")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relative-to", type=Path, default=Path("data/llm_runs"),
                        help="Write source paths relative to this directory")
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--revision", default="7c75f2f49e74ac97c1852e62b4c860091d6a00ea")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    closer_ids = {tokenizer.convert_tokens_to_ids(tag) for tag in K2_REASONING_CLOSERS}
    if tokenizer.unk_token_id in closer_ids or None in closer_ids:
        raise ValueError("a K2 reasoning closer is not a single token in this tokenizer")
    control_ids = set(tokenizer.get_added_vocab().values())
    rows = []
    for name in args.cases:
        path = args.results / name / "requests.jsonl"
        try:
            source = path.resolve().relative_to(args.relative_to.resolve())
        except ValueError:
            source = path
        rows.append({"case": name, **summarize(path, closer_ids, control_ids), "source": str(source)})
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
