"""Collect the Mac run's summaries into the CSVs behind the research page."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

GENERATION = ("case", "effort", "concurrency", "states", "samples_per_state", "one_group_per_batch",
              "prefill_batch_size", "temperature", "top_p", "max_tokens", "responses", "valid", "capped",
              "seconds", "valid_decisions_per_second", "generated_tokens", "generated_tokens_per_second",
              "median_output_tokens", "max_output_tokens", "mean_teacher_probability",
              "first_option_rate_of_valid", "peak_memory_gb")
UPDATES = ("case", "backend", "microbatch", "gradient_accumulation", "grad_checkpoint", "timed_steps",
           "timed_seconds", "responses_per_second", "generated_training_tokens_per_second",
           "frozen_logprob_seconds_per_response", "peak_memory_gb", "timed_trainable_weights_changed")


def write(path: Path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generation = []
    for summary in sorted(args.run.glob("*/*/summary.json")):
        if summary.parent.name == "warmup" or summary.parts[-3] in ("updates", "hf_generation"):
            continue
        row = json.loads(summary.read_text())
        row.setdefault("one_group_per_batch", False)
        row.setdefault("prefill_batch_size", min(8, row["concurrency"]))
        generation.append({"case": summary.parent.name, **row})
    for summary in sorted((args.run / "hf_generation").glob("*/summary.json")):
        row = json.loads(summary.read_text())
        generation.append({"case": summary.parent.name, "effort": row["effort"],
                           "concurrency": row["generation_microbatch"], "states": row["timed_states"],
                           "samples_per_state": row["group_size"], "one_group_per_batch": True,
                           "temperature": row["temperature"], "top_p": row["sampling_top_p"],
                           "responses": row["timed_responses"], "seconds": row["timed_seconds"],
                           "valid_decisions_per_second": row["valid_decisions_per_second"],
                           "generated_tokens": row["generated_tokens"],
                           "generated_tokens_per_second": row["generated_tokens_per_second"]})
    write(args.output / "generation.csv", GENERATION, generation)
    updates = []
    for summary in sorted((args.run / "updates").glob("*/summary.json")):
        row = json.loads(summary.read_text())
        if row.get("backend") != "mlx":  # the shared PyTorch trainer's own summary format
            row = {"backend": "pytorch_mps", "microbatch": 1, "gradient_accumulation": 4, "grad_checkpoint": True,
                   "timed_steps": row["timed_optimizer_steps"], "timed_seconds": row["timed_seconds"],
                   "responses_per_second": row["responses_per_second"],
                   "generated_training_tokens_per_second": row["generated_training_tokens_per_second"],
                   "frozen_logprob_seconds_per_response": row["frozen_logprob_seconds"] / (2 * len(row["response_token_lengths"])),
                   "timed_trainable_weights_changed": row["timed_trainable_weights_changed"]}
        updates.append({"case": summary.parent.name, **row})
    write(args.output / "updates.csv", UPDATES, updates)
    fixed = json.loads((args.run / "fixed_work.json").read_text())
    parity = json.loads((args.run / "parity.json").read_text())
    with (args.output / "fixed_work_and_parity.json").open("w") as stream:
        json.dump({"fixed_work": fixed, "parity": parity}, stream, indent=2)
        stream.write("\n")
    print(f"{len(generation)} generation rows, {len(updates)} update rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
