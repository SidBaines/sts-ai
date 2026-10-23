#!/usr/bin/env python3
"""Describe every generation case and compare paired saved-decision teacher scores."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from sts_llm.generation import parse_action
from sts_llm.statistics import paired_cluster_bootstrap
from sts_llm.teacher import teacher_reward, validate_teacher


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def case(path, teachers):
    summary = json.loads(path.read_text())
    config_path = path.with_name("config.json")
    meta = {**(json.loads(config_path.read_text()) if config_path.exists() else {}), **summary}
    server_path = path.with_name("server_config.json")
    if server_path.exists():
        meta.update(json.loads(server_path.read_text()))
    observations, scheduled = {}, set()
    records = jsonl(path.with_name("requests.jsonl"))
    counts = {"complete_responses": 0, "invalid_or_truncated": 0, "valid_responses": 0}
    for record in records:
        prompt_id = str(record["prompt_id"])
        if prompt_id not in teachers:
            raise ValueError(f"unknown teacher prompt {prompt_id} in {path}")
        scheduled.add(prompt_id)
        if not record.get("http_ok") or not record.get("response_complete"):
            continue  # Infrastructure-incomplete responses remain missing, not zero reward.
        row = teachers[prompt_id]
        parsed = parse_action(record.get("decoded_output_token_ids", record.get("raw_text", "")),
            row["legal_indices"], finish_reason=record.get("finish_reason"),
            reasoning_content=record.get("server_reasoning_content"))
        teacher = row["teacher"]
        top = teacher["legal_indices"][max(range(len(teacher["probabilities"])), key=teacher["probabilities"].__getitem__)]
        score = teacher_reward(row, parsed.action_index if parsed.valid else None)
        observations.setdefault(prompt_id, []).append((score, float(parsed.valid and parsed.action_index == top)))
        counts["complete_responses"] += 1
        counts["valid_responses"] += int(parsed.valid)
        counts["invalid_or_truncated"] += int(not parsed.valid)
    decisions = [{"prompt_id": p, "world_id": teachers[p]["world_identity"], "repeats": len(values),
        "teacher_probability": math.fsum(v[0] for v in values)/len(values),
        "teacher_top_agreement": math.fsum(v[1] for v in values)/len(values)}
        for p, values in sorted(observations.items())]
    metrics = summary.get("metrics", {})
    fields = ("model", "effort", "concurrency", "max_tokens", "endpoint", "backend", "execution", "prefix", "max_num_batched_tokens", "max_num_seqs", "temperature", "top_p")
    result = {"case_id": str(path.parent.resolve()), "case_name": meta.get("case_name", path.parent.name),
        **{key: meta.get(key) for key in fields}, "revision": meta.get("revision", meta.get("tokenizer_revision")),
        "end_reason": summary.get("end_reason"), **counts, "scheduled_decisions": len(scheduled),
        "scored_decisions": len(decisions), "missing_scheduled_decisions": len(scheduled-observations.keys()),
        "missing_scheduled_prompt_ids": sorted(scheduled-observations.keys()),
        "incomplete_requests": len(records)-counts["complete_responses"], "decisions": decisions,
        **{key: metrics.get(key) for key in ("valid_decisions_per_second", "generated_tokens_per_second",
            "median_latency_seconds", "p95_latency_seconds", "median_output_tokens", "truncation_rate_of_http_ok")}}
    for metric in ("teacher_probability", "teacher_top_agreement"):
        result["mean_"+metric] = math.fsum(d[metric] for d in decisions)/len(decisions) if decisions else None
    return result


def analyze(teacher_path, results, *, seed=1729, replicates=10000):
    rows = jsonl(teacher_path)
    teachers = {str(row["id"]): row for row in rows}
    if len(teachers) != len(rows):
        raise ValueError("duplicate teacher prompt IDs")
    for row in rows:
        validate_teacher(row)
        if not isinstance(row.get("world_identity"), str) or not row["world_identity"]:
            raise ValueError("every teacher decision needs an explicit world_identity")
    paths = sorted({path for root in results for path in root.rglob("summary.json")})
    if not paths:
        raise ValueError("no summary.json files found under the requested result directories")
    cases = [case(path, teachers) for path in paths]
    comparisons = []
    match = ("model", "revision", "concurrency", "max_tokens", "endpoint", "backend", "execution", "prefix", "max_num_batched_tokens", "max_num_seqs", "temperature", "top_p")
    effort_order = {"low": 0, "medium": 1, "high": 2}
    for first, second in itertools.combinations(cases, 2):
        if any(first[key] is None or second[key] is None for key in ("model", "concurrency", "max_tokens")):
            continue
        if any(first[key] != second[key] for key in match) or first["effort"] == second["effort"]:
            continue
        if first["effort"] not in effort_order or second["effort"] not in effort_order:
            continue
        left, right = sorted((first, second), key=lambda c: effort_order[c["effort"]])
        for metric in ("teacher_probability", "teacher_top_agreement"):
            records = lambda c: [{"prompt_id": d["prompt_id"], "world_id": d["world_id"], "value": d[metric]}
                                 for d in c["decisions"]]
            comparisons.append({"left_case": left["case_id"], "right_case": right["case_id"],
                "left_effort": left["effort"], "right_effort": right["effort"], "metric": metric,
                "matched_configuration": {key: left[key] for key in match},
                **paired_cluster_bootstrap(records(left), records(right), seed=seed, replicates=replicates)})
    return {"interpretation": "Teacher policy preferences, not game-win probability or reasoning faithfulness.",
        "decision_weighting": "Average repeats per decision, then weight every observed decision equally.",
        "missingness": "Completed invalid/capped outputs score zero; incomplete HTTP responses are missing.",
        "missingness_caveat": "Complete-pair estimates can be biased if slower or harder responses are disproportionately missing.",
        "timings": "Descriptive single-cohort measurements; no timing confidence intervals.",
        "configuration": "Effort comparisons match recorded fields; null backend/execution fields mean unrecorded, not verified equal. Server sidecars preserve controls for early pilot cases.",
        "seed": seed, "bootstrap_replicates": replicates, "cases": cases, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    report = analyze(args.teacher, args.results, seed=args.seed, replicates=args.replicates)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    flat = [{k: v for k, v in c.items() if k not in ("decisions", "missing_scheduled_prompt_ids")} for c in report["cases"]]
    with (args.output/"cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else ["case_id"])
        writer.writeheader()
        writer.writerows(flat)
    paired = []
    for comparison in report["comparisons"]:
        row = {k: v for k, v in comparison.items() if k not in
               ("pairs", "matched_configuration", "missing_left_prompt_ids", "missing_right_prompt_ids", "ci95")}
        row.update(ci95_low=comparison["ci95"][0] if comparison["ci95"] else None,
                   ci95_high=comparison["ci95"][1] if comparison["ci95"] else None)
        paired.append(row)
    with (args.output/"comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]) if paired else ["left_case", "right_case"])
        writer.writeheader()
        writer.writerows(paired)


if __name__ == "__main__":
    main()
