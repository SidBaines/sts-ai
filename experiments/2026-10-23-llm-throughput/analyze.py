#!/usr/bin/env python3
"""Describe every generation case, compare paired saved-decision teacher scores,
and compare each arm with simple baselines on the same states.

Writes analysis.json plus teacher_cases.csv, teacher_comparisons.csv and (with
--panel) teacher_baselines.csv to --output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from sts_llm.generation import parse_action
from sts_llm.statistics import DEFAULT_MIN_CLUSTERS, paired_cluster_bootstrap
from sts_llm.teacher import teacher_reward, validate_teacher

PHASES = ("all", "combat", "out_of_combat")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values) if values else None


def teacher_top(row):
    teacher = row["teacher"]
    return teacher["legal_indices"][max(range(len(teacher["probabilities"])), key=teacher["probabilities"].__getitem__)]


def case(path, teachers):
    summary = json.loads(path.read_text())
    config_path = path.with_name("config.json")
    meta = {**(json.loads(config_path.read_text()) if config_path.exists() else {}), **summary}
    server_path = path.with_name("server_config.json")
    if server_path.exists():
        meta.update(json.loads(server_path.read_text()))
    observations, scheduled = {}, set()
    records = jsonl(path.with_name("requests.jsonl"))
    counts = {"complete_responses": 0, "invalid_or_truncated": 0, "valid_responses": 0,
              "valid_first_option": 0, "recorded_validity_disagreements": 0}
    fingerprint = hashlib.sha256()
    for record in sorted(records, key=lambda r: (str(r["prompt_id"]), r.get("repeat", 0))):
        fingerprint.update(json.dumps([str(record["prompt_id"]), record.get("repeat", 0),
                                       record.get("output_token_ids")]).encode())
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
        recorded_valid = bool(record.get("legal_action") and record.get("thinking_closed") and not record.get("truncated"))
        counts["recorded_validity_disagreements"] += int(recorded_valid != parsed.valid)
        score = teacher_reward(row, parsed.action_index if parsed.valid else None)
        first = parsed.valid and parsed.action_index == row["legal_indices"][0]
        observations.setdefault(prompt_id, []).append(
            (score, float(parsed.valid and parsed.action_index == teacher_top(row)), parsed.valid, first))
        counts["complete_responses"] += 1
        counts["valid_responses"] += int(parsed.valid)
        counts["valid_first_option"] += int(first)
        counts["invalid_or_truncated"] += int(not parsed.valid)
    decisions = [{"prompt_id": p, "world_id": teachers[p]["world_identity"], "phase": teachers[p].get("phase"),
        "repeats": len(values), "valid_repeats": sum(v[2] for v in values),
        "valid_first_option": sum(v[3] for v in values),
        "teacher_probability": mean(v[0] for v in values),
        "teacher_top_agreement": mean(v[1] for v in values),
        # Sensitivity only: mean over valid answers, None when no answer was valid.
        "valid_teacher_probability": mean(v[0] for v in values if v[2])}
        for p, values in sorted(observations.items())]
    metrics = summary.get("metrics", {})
    fields = ("model", "effort", "concurrency", "max_tokens", "endpoint", "backend", "execution", "prefix", "max_num_batched_tokens", "max_num_seqs", "temperature", "top_p")
    result = {"case_id": str(path.parent.resolve()), "case_name": meta.get("case_name", path.parent.name),
        **{key: meta.get(key) for key in fields}, "revision": meta.get("revision", meta.get("tokenizer_revision")),
        "end_reason": summary.get("end_reason"), **counts,
        "valid_first_option_rate": counts["valid_first_option"]/counts["valid_responses"] if counts["valid_responses"] else None,
        "scheduled_decisions": len(scheduled),
        "scored_decisions": len(decisions), "missing_scheduled_decisions": len(scheduled-observations.keys()),
        "missing_scheduled_prompt_ids": sorted(scheduled-observations.keys()),
        "incomplete_requests": len(records)-counts["complete_responses"],
        "output_fingerprint": fingerprint.hexdigest()[:16], "identical_outputs_to": None,
        "scheduled_prompt_ids": sorted(scheduled), "decisions": decisions,
        **{key: metrics.get(key) for key in ("valid_decisions_per_second", "generated_tokens_per_second",
            "median_latency_seconds", "p95_latency_seconds", "median_output_tokens", "truncation_rate_of_http_ok")}}
    for metric in ("teacher_probability", "teacher_top_agreement"):
        result["mean_"+metric] = mean(d[metric] for d in decisions)
    return result


def as_records(decisions, metric):
    return [{"prompt_id": d["prompt_id"], "world_id": d["world_id"], "value": d[metric]}
            for d in decisions if d[metric] is not None]


def baselines(teachers, panel, cases, *, seed, replicates, min_clusters):
    """Random legal action, always the first menu option and the teacher's own top
    choice on the panel states; then each arm on exactly those states against always-first."""
    rows = [teachers[p] for p in panel]
    fixed = {
        "random_legal_action": lambda r: mean(teacher_reward(r, i) for i in r["legal_indices"]),
        "always_first_option": lambda r: teacher_reward(r, r["legal_indices"][0]),
        "teacher_top_choice": lambda r: teacher_reward(r, teacher_top(r)),
    }
    def in_phase(phase, value):
        return phase == "all" or value == phase
    output = []
    for phase in PHASES:
        selected = [r for r in rows if in_phase(phase, r.get("phase"))]
        for name, score in fixed.items():
            output.append({"kind": "baseline", "name": name, "phase": phase, "states": len(selected),
                "worlds": len({r["world_identity"] for r in selected}),
                "median_menu_size": statistics.median(len(r["legal_indices"]) for r in selected) if selected else None,
                "mean_teacher_probability": mean(score(r) for r in selected)})
    always_first = [{"prompt_id": str(r["id"]), "world_id": r["world_identity"], "phase": r.get("phase"),
                     "value": fixed["always_first_option"](r)} for r in rows]
    for arm in cases:
        if set(arm["scheduled_prompt_ids"]) != set(panel):
            continue  # Warmup, calibration and other panels are not compared with these baselines.
        for phase in PHASES:
            decisions = [d for d in arm["decisions"] if in_phase(phase, d["phase"])]
            left = [{k: x[k] for k in ("prompt_id", "world_id", "value")} for x in always_first if in_phase(phase, x["phase"])]
            report = paired_cluster_bootstrap(left, as_records(decisions, "teacher_probability"),
                                              seed=seed, replicates=replicates, min_clusters=min_clusters)
            # Sensitivity: states with at least one valid answer, scored over valid answers only.
            valid_only = paired_cluster_bootstrap(left, as_records(decisions, "valid_teacher_probability"),
                                                  seed=seed, replicates=replicates, min_clusters=min_clusters)
            valid = sum(d["valid_repeats"] for d in decisions)
            output.append({"kind": "arm", "name": arm["case_name"], "phase": phase, "effort": arm["effort"],
                "identical_outputs_to": arm["identical_outputs_to"],
                "states": len(decisions), "worlds": len({d["world_id"] for d in decisions}),
                "responses": sum(d["repeats"] for d in decisions), "valid_responses": valid,
                "mean_teacher_probability": mean(d["teacher_probability"] for d in decisions),
                "valid_first_option_rate": sum(d["valid_first_option"] for d in decisions)/valid if valid else None,
                "difference_vs_always_first": report["difference"],
                "ci95_low": report["ci95"][0] if report["ci95"] else None,
                "ci95_high": report["ci95"][1] if report["ci95"] else None,
                "n_pairs": report["n_pairs"], "n_worlds": report["n_worlds"],
                "interval_unavailable_reason": report["interval_unavailable_reason"],
                "valid_only_difference_vs_always_first": valid_only["difference"],
                "valid_only_ci95_low": valid_only["ci95"][0] if valid_only["ci95"] else None,
                "valid_only_ci95_high": valid_only["ci95"][1] if valid_only["ci95"] else None,
                "valid_only_n_pairs": valid_only["n_pairs"]})
    return output


def analyze(teacher_path, results, *, panel_path=None, seed=1729, replicates=10000, min_clusters=DEFAULT_MIN_CLUSTERS):
    rows = jsonl(teacher_path)
    teachers = {str(row["id"]): row for row in rows}
    if len(teachers) != len(rows):
        raise ValueError("duplicate teacher prompt IDs")
    for row in rows:
        validate_teacher(row)
        if not isinstance(row.get("world_identity"), str) or not row["world_identity"]:
            raise ValueError("every teacher decision needs an explicit world_identity")
    # Only generation cases with request records carry saved decisions.
    paths = sorted({path for root in results for path in root.rglob("summary.json")
                    if path.with_name("requests.jsonl").exists()})
    if not paths:
        raise ValueError("no summary.json with requests.jsonl found under the requested result directories")
    cases = [case(path, teachers) for path in paths]
    seen = {}
    for item in cases:  # Sorted by path, so the earlier-named case is the reference copy.
        if item["output_fingerprint"] in seen:
            item["identical_outputs_to"] = seen[item["output_fingerprint"]]
        else:
            seen[item["output_fingerprint"]] = item["case_name"]
    comparisons, previous = [], {}
    match = ("model", "revision", "concurrency", "max_tokens", "endpoint", "backend", "execution", "prefix", "max_num_batched_tokens", "max_num_seqs", "temperature", "top_p")
    effort_order = {"low": 0, "medium": 1, "high": 2}
    for first, second in itertools.combinations(cases, 2):
        if any(first[key] is None or second[key] is None for key in ("model", "concurrency", "max_tokens")):
            continue
        if any(first[key] != second[key] for key in match) or first["effort"] == second["effort"]:
            continue
        if first["effort"] not in effort_order or second["effort"] not in effort_order:
            continue
        if not set(first["scheduled_prompt_ids"]) & set(second["scheduled_prompt_ids"]):
            continue  # Disjoint warmup panels share settings but no states.
        left, right = sorted((first, second), key=lambda c: effort_order[c["effort"]])
        for metric, values in (("teacher_probability", "teacher_probability"),
                               ("teacher_top_agreement", "teacher_top_agreement"),
                               ("teacher_probability_valid_pairs_only", "valid_teacher_probability")):
            key = (left["output_fingerprint"], right["output_fingerprint"], metric)
            comparisons.append({"left_case": left["case_name"], "right_case": right["case_name"],
                "left_case_id": left["case_id"], "right_case_id": right["case_id"],
                "left_effort": left["effort"], "right_effort": right["effort"], "metric": metric,
                "duplicate_of": previous.get(key),
                "matched_configuration": {k: left[k] for k in match},
                **paired_cluster_bootstrap(as_records(left["decisions"], values), as_records(right["decisions"], values),
                                           seed=seed, replicates=replicates, min_clusters=min_clusters)})
            previous.setdefault(key, f"{left['case_name']} / {right['case_name']}")
    report = {"interpretation": "Teacher policy preferences, not game-win probability or reasoning faithfulness.",
        "decision_weighting": "Average repeats per decision, then weight every observed decision equally.",
        "missingness": "Completed invalid/capped outputs score zero; incomplete HTTP responses are missing.",
        "valid_pairs_sensitivity": "teacher_probability_valid_pairs_only averages valid answers only and keeps a state only when both arms have one; it is a sensitivity check, not the pre-declared score.",
        "missingness_caveat": "Complete-pair estimates can be biased if slower or harder responses are disproportionately missing.",
        "duplicates": "identical_outputs_to / duplicate_of mark cohorts that reproduced another cohort's token sequences; they are not replications.",
        "timings": "Descriptive single-cohort measurements; no timing confidence intervals.",
        "configuration": "Effort comparisons match recorded fields; null backend/execution fields mean unrecorded, not verified equal. Server sidecars preserve controls for early pilot cases.",
        "seed": seed, "bootstrap_replicates": replicates, "min_clusters": min_clusters,
        "cases": cases, "comparisons": comparisons}
    if panel_path is not None:
        panel = [str(row["id"]) for row in jsonl(panel_path)]
        if not panel or len(set(panel)) != len(panel) or any(p not in teachers for p in panel):
            raise ValueError("panel IDs must be nonempty, unique and present in the teacher file")
        report["baselines"] = baselines(teachers, panel, cases, seed=seed, replicates=replicates, min_clusters=min_clusters)
    return report


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--panel", type=Path, help="JSONL of the shared state panel (rows with id) for baselines")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--min-clusters", type=int, default=DEFAULT_MIN_CLUSTERS)
    args = parser.parse_args()
    report = analyze(args.teacher, args.results, panel_path=args.panel, seed=args.seed,
                     replicates=args.replicates, min_clusters=args.min_clusters)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    # CSVs name cases rather than local paths; analysis.json keeps the full case IDs.
    write_csv(args.output/"teacher_cases.csv", [{k: v for k, v in c.items() if k not in
        ("case_id", "decisions", "missing_scheduled_prompt_ids", "scheduled_prompt_ids")} for c in report["cases"]])
    paired = []
    for comparison in report["comparisons"]:
        row = {k: v for k, v in comparison.items() if k not in ("pairs", "matched_configuration", "left_case_id",
               "right_case_id", "missing_left_prompt_ids", "missing_right_prompt_ids", "ci95")}
        row.update(ci95_low=comparison["ci95"][0] if comparison["ci95"] else None,
                   ci95_high=comparison["ci95"][1] if comparison["ci95"] else None)
        paired.append(row)
    write_csv(args.output/"teacher_comparisons.csv", paired)
    if "baselines" in report:
        write_csv(args.output/"teacher_baselines.csv", report["baselines"])


if __name__ == "__main__":
    main()
