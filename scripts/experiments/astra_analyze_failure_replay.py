#!/usr/bin/env python
"""Analyze complete arms of the historical-failure replay, independently of 133-state probes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.astra_analyze_policy import (
    MODEL_BUNDLE_CAVEAT, indexed_rows, read_rows, validate_arm,
)
from sts_ai.agents import parse_json_action
from sts_ai.schemas import LegalAction

MODELS = ("base", "rl1", "rl5", "gemma26")
EXPECTED_ARMS = tuple(f"{model}_temp{temp}" for model in MODELS for temp in ("0.0", "0.7"))
LIMITATIONS = [
    "Outcome-selected cohort: all 56 historical fatal/unexecuted iteration-6 decisions, not a representative state sample.",
    "53 states have one legal action; validity there measures action communication, not strategic choice.",
    "Replay uses each recorded state's standard first-attempt prompt; the stored historical failure was the final retry (retries=2).",
    "Historical policy was iteration-5 at temperature 0.7, cap4096; replay cap256 and random stream differ. This is not an exact historical sampling reproduction.",
    "One generated response per state per arm, including sampled arms; no estimate of repeated-sampling failure probability.",
    "No rollout outcomes or teacher-quality values are measured. Action changes on the three branching states do not establish improvement.",
    "States share original training-world seeds; paired counts are descriptive and not independent-state inferential estimates.",
    MODEL_BUNDLE_CAVEAT,
]


def failure_category(row: dict, decision: dict) -> str:
    """Mutually exclusive categories of raw output, with live parser as authority."""
    actions = [LegalAction(**a) for a in row["legal_actions"]]
    parsed = parse_json_action(decision["raw_response"], actions, output_contract="action_text")
    if parsed.valid:
        return "valid_live_action"
    obj = parsed.metadata.get("parsed")
    if not isinstance(obj, dict):
        return "no_parseable_json_object"
    value = obj.get("action")
    if not isinstance(value, str):
        return "missing_or_nonstring_action"
    value = re.sub(r"^\d+:\s+", "", value)
    descriptions = [a.description for a in actions]
    if any(value.casefold() == text.casefold() for text in descriptions):
        return "capitalization_only"
    if sum(text.startswith(value) for text in descriptions) > 1 and value:
        return "ambiguous_legal_description_prefix"
    if re.fullmatch(r"\d+", value):
        return "numeric_string_in_action_field"
    if re.fullmatch(r"x\s*=\s*\d+", value):
        return "map_coordinate_in_action_field"
    if re.fullmatch(r"\d+:\s*\d*", value):
        return "malformed_numeric_selector"
    if value and value in row["state_text"]:
        return "observation_text_in_action_field"
    if value.lower().startswith("choice map node"):
        return "map_verb_typo"
    if "map node" in value.lower():
        return "nonlegal_map_node_description"
    return "other_nonlegal_action_text"


def resolution_path(row: dict, decision: dict) -> str:
    parsed = parse_json_action(decision["raw_response"], [LegalAction(**a) for a in row["legal_actions"]],
                               output_contract="action_text")
    if not parsed.valid:
        return "invalid"
    return parsed.metadata.get("semantic_fallback") or parsed.metadata.get("semantic_match") or "exact_action_text"


def stats(rows: list[dict], labels: dict, strict_ids: set[int] | None = None) -> dict:
    n = len(rows)
    valid = sum(bool(r["decision"]["valid"]) for r in rows)
    tokens = [r["decision"].get("completion_tokens", 0) for r in rows]
    return {"n": n, "valid": valid, "invalid": n - valid,
            "valid_rate": valid / n if n else None,
            "strict_action_contract_valid": None if strict_ids is None else sum(r["probe_id"] in strict_ids for r in rows),
            "completion_tokens_mean": statistics.mean(tokens) if tokens else None,
            "completion_tokens_max": max(tokens) if tokens else None,
            "valid_resolution_paths": dict(Counter(resolution_path(labels[r["probe_id"]], r["decision"]) for r in rows if r["decision"]["valid"])),
            "raw_output_categories": dict(Counter(failure_category(labels[r["probe_id"]], r["decision"]) for r in rows))}


def summarize_arm(rows: list[dict], labels: dict, strict: list[dict] | None = None) -> dict:
    strict_ids = None if strict is None else {r["probe_id"] for r in strict if r["valid_json"] and r["matched"]}
    return {"overall": stats(rows, labels, strict_ids),
            "by_menu_size": {name: stats([r for r in rows if (len(labels[r["probe_id"]]["legal_actions"]) == 1) == singleton], labels, strict_ids)
                             for name, singleton in (("singleton", True), ("branching", False))},
            "by_screen": {screen: stats([r for r in rows if r["probe_screen"] == screen], labels, strict_ids)
                          for screen in sorted({r["probe_screen"] for r in rows})},
            "invalid_probe_ids": [r["probe_id"] for r in rows if not r["decision"]["valid"]]}


def paired_changes(first: list[dict], second: list[dict], labels: dict) -> dict:
    a, b = indexed_rows(first, name="first"), indexed_rows(second, name="second")
    if list(a) != list(b) or set(a) != set(labels):
        raise ValueError("paired predictions do not have the same complete ordered cohort")
    def counts(ids):
        transitions = Counter()
        switches = []
        for key in ids:
            av, bv = bool(a[key]["decision"]["valid"]), bool(b[key]["decision"]["valid"])
            transitions[f"{'valid' if av else 'invalid'}_to_{'valid' if bv else 'invalid'}"] += 1
            if av and bv and a[key]["decision"]["action_index"] != b[key]["decision"]["action_index"]:
                switches.append(key)
        return {"n": len(ids), "transitions": dict(transitions),
                "valid_count_delta": transitions["invalid_to_valid"] - transitions["valid_to_invalid"],
                "action_switches_both_valid": len(switches), "action_switch_probe_ids": switches}
    return {"overall": counts(list(a)),
            "by_menu_size": {name: counts([i for i in a if (len(labels[i]["legal_actions"]) == 1) == singleton])
                             for name, singleton in (("singleton", True), ("branching", False))}}


def validate_pair(first_name: str, second_name: str, a: dict, b: dict) -> dict:
    # Exact named pairs only: same-temperature model/adapter effects, or temperature within model.
    af, at = first_name.split("_temp")
    bf, bt = second_name.split("_temp")
    if first_name not in EXPECTED_ARMS or second_name not in EXPECTED_ARMS or (af != bf and at != bt):
        raise ValueError("undeclared failure-replay comparison")
    allowed = {"out", "started", "adapter", "adapter_identity"}
    if af == bf:
        allowed.add("temperature")
    if af != bf and "gemma26" in (af, bf):
        allowed.add("model")
    required = {"source", "source_sha256", "model", "temperature", "batch_size", "max_tokens", "max_retries", "script_sha256", "contract", "constrained"}
    if not required.issubset(a) or not required.issubset(b):
        raise ValueError("comparison missing generation provenance")
    bad = [key for key in a.keys() | b.keys() if key not in allowed and a.get(key) != b.get(key)]
    if bad:
        raise ValueError(f"undeclared failure-replay config changes: {bad}")
    return {"observed_interventions": {key: [a.get(key), b.get(key)] for key in sorted(allowed) if a.get(key) != b.get(key)}}


def analyze(directory: Path) -> dict:
    source = directory / "cohort.jsonl"
    manifest = json.loads((directory / "cohort.manifest.json").read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    labels = indexed_rows(read_rows(source), name="frozen failure cohort")
    if len(labels) != 56 or manifest["n"] != 56 or manifest["sha256"] != source_hash:
        raise ValueError("failure cohort does not match frozen 56-state manifest")
    historical = [{**{k: row[k] for k in ("probe_id", "probe_screen", "world_seed", "decision_index")}, "decision": row["agent"]} for row in labels.values()]
    arms, predictions, configs, pending = {}, {}, {}, []
    for name in EXPECTED_ARMS:
        folder = directory / name
        if not all((folder / f).exists() for f in ("report.json", "config.json", "predictions.jsonl")):
            pending.append(name)
            continue
        config = json.loads((folder / "config.json").read_text())
        if config["source_sha256"] != source_hash or config["contract"] != "action_text" or config["constrained"]:
            raise ValueError(f"{name}: expected exact-source, free action-text replay")
        if config["temperature"] != float(name.split("_temp")[1]):
            raise ValueError(f"{name}: temperature disagrees with arm name")
        rows, sources, strict = validate_arm(config, read_rows(folder / "predictions.jsonl"), labels)
        if any(row["decision"].get("retries") != 0 for row in rows):
            raise ValueError(f"{name}: replay contains retries, expected one first attempt")
        if list(sources.values()) != list(labels.values()):
            raise ValueError(f"{name}: source records differ from frozen failure cohort")
        report = json.loads((folder / "report.json").read_text())
        if report["overall"]["n"] != 56 or report["overall"]["valid"] != sum(r["decision"]["valid"] for r in rows):
            raise ValueError(f"{name}: report disagrees with complete predictions")
        arms[name] = summarize_arm(rows, labels, strict)
        arms[name]["provenance"] = {key: config.get(key) for key in ("model", "adapter", "adapter_identity", "temperature", "max_tokens", "batch_size", "source_sha256", "script_sha256")}
        predictions[name], configs[name] = rows, config
    pairs = [(f"{a}_temp{t}", f"{b}_temp{t}") for t in ("0.0", "0.7") for a, b in (("base", "rl1"), ("base", "rl5"), ("rl1", "rl5"), ("base", "gemma26"))]
    pairs += [(f"{model}_temp0.0", f"{model}_temp0.7") for model in MODELS]
    comparisons = {}
    for first, second in pairs:
        if first not in arms or second not in arms:
            continue
        provenance = validate_pair(first, second, configs[first], configs[second])
        comparisons[f"{first} -> {second}"] = {**paired_changes(predictions[first], predictions[second], labels), **provenance}
    return {"kind": "astra_outcome_selected_failure_replay_v1", "source_sha256": source_hash,
            "limitations": LIMITATIONS, "completed_arms": list(arms), "pending_arms": pending,
            "historical_final_retry_descriptive_only": summarize_arm(historical, labels),
            "arms": arms, "paired_changes": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/experiments/astra_5sep/failure_replay"))
    parser.add_argument("--out", type=Path, default=Path("docs/astra_5sep_failure_replay_results.json"))
    args = parser.parse_args()
    report = analyze(args.directory)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"completed": report["completed_arms"], "pending": report["pending_arms"]}))


if __name__ == "__main__":
    main()
