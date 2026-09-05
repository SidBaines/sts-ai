"""Join fixed-state diagnostic predictions to continuation-dependent values."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from sts_ai.eval_stats import bootstrap_ci
from sts_ai.agents import parse_json_action
from sts_ai.provenance import adapter_provenance
from sts_ai.schemas import LegalAction
from sts_ai.teacher_action_eval import evaluate_generated_action


DECLARED_MODEL_COMPARISONS = frozenset({
    ("base_text_free", "gemma26_text_free"),
    ("base_text_constrained", "gemma26_text_constrained"),
    ("base_text_enriched_constrained", "gemma26_text_enriched_constrained"),
    ("base_text_reversed_constrained", "gemma26_text_reversed_constrained"),
    ("base_text_sampled", "gemma26_text_sampled"),
})

MODEL_BUNDLE_CAVEAT = (
    "Cross-model comparisons change the complete model bundle: model weights and architecture, "
    "tokenizer/chat-template, quantization and any model-local custom loader. "
    "The staged gemma26_ollama bundle uses Ollama NVFP4 weights and a custom MLX loader; "
    "the alternative Hugging Face mlx-community bundle uses 4-bit affine weights with "
    "8-bit router projections. The E4B control is BF16. Read each arm model path and "
    "bundle provenance to identify the actual implementation. Gemma26 and E4B chat "
    "templates may also differ in empty-thought-channel prefilling. "
    "Differences cannot be attributed to parameter count alone."
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return statistics.mean(values) if values else None


def indexed_rows(rows, *, name):
    result = {}
    for row in rows:
        key = row.get("probe_id")
        if isinstance(key, bool) or not isinstance(key, int) or key in result:
            raise ValueError(f"invalid/duplicate probe_id in {name}: {key!r}")
        result[key] = row
    return result


def original_index_map(source, label):
    """Validate a displayed-menu permutation before mapping to teacher indices."""
    displayed = source["legal_actions"]
    original = label["legal_actions"]
    if any(type(action.get("index")) is not int for action in displayed + original):
        raise ValueError("menu indices must be integers, not booleans")
    if [a["index"] for a in displayed] != list(range(len(displayed))):
        raise ValueError("source display indices must be contiguous list positions")
    original_by_index = {a["index"]: a for a in original}
    if len(original_by_index) != len(original) or len(displayed) != len(original):
        raise ValueError("source and teacher menu sizes/identities differ")
    recorded = source.get("probe_original_indices")
    if recorded is None:
        mapping = {a["index"]: a["index"] for a in displayed}
    elif isinstance(recorded, list):
        mapping = dict(enumerate(recorded))
    elif isinstance(recorded, dict):
        mapping = {}
        for key, value in recorded.items():
            if isinstance(key, bool) or str(key) not in {str(a["index"]) for a in displayed}:
                raise ValueError("invalid displayed index in probe_original_indices")
            parsed = int(key)
            if parsed in mapping:
                raise ValueError("duplicate displayed index in probe_original_indices")
            mapping[parsed] = value
    else:
        raise ValueError("probe_original_indices must be a list or dictionary")
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in mapping.values())
            or set(mapping) != {a["index"] for a in displayed}
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) != set(original_by_index)):
        raise ValueError("probe_original_indices is not a bijection onto teacher actions")
    for action in displayed:
        target = original_by_index[mapping[action["index"]]]
        if any(action[key] != target[key] for key in ("bits", "description")):
            raise ValueError("remapped action bits/description differ from teacher action")
    return mapping


def validate_identity(row, label, *, name):
    for key in ("probe_id", "world_seed", "decision_index", "probe_screen"):
        if row.get(key) != label.get(key):
            raise ValueError(f"{name} disagrees with teacher {key}")


def validate_arm(config, predictions, labels):
    if config.get("mode") != "probe" or config.get("contract") not in {"action_text", "action_only"}:
        raise ValueError("analysis requires an action_text/action_only single-attempt probe")
    source_path = Path(config["source"])
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if not config.get("source_sha256") or config["source_sha256"] != source_hash:
        raise ValueError("source_sha256 disagrees with actual source cohort")
    if config.get("adapter_identity") != adapter_provenance(config.get("adapter")):
        raise ValueError("recorded adapter identity disagrees with current adapter files")
    sources = indexed_rows(read_rows(source_path), name=str(source_path))
    prediction_by_id = indexed_rows(predictions, name="predictions")
    if set(sources) != set(labels) or set(prediction_by_id) != set(labels):
        raise ValueError("incomplete/foreign source or prediction cohort")
    if list(sources) != list(prediction_by_id):
        raise ValueError("prediction ordering differs from source cohort")
    remapped, strict = [], []
    for prediction in predictions:
        label = labels[prediction["probe_id"]]
        source = sources[prediction["probe_id"]]
        validate_identity(source, label, name="source")
        validate_identity(prediction, label, name="prediction")
        if source.get("source_trace") != label.get("source_trace"):
            raise ValueError("source trace identity differs from teacher")
        mapping = original_index_map(source, label)
        decision = prediction["decision"]
        raw = decision["raw_response"]
        parsed = parse_json_action(raw, [LegalAction(**a) for a in source["legal_actions"]],
                                   output_contract=config["contract"])
        if parsed.valid != decision["valid"] or (parsed.valid and parsed.action_index != decision["action_index"]):
            raise ValueError("recorded decision disagrees with parsing its actual displayed menu")
        result = evaluate_generated_action(raw, output_contract=config["contract"],
            action_indices=[a["index"] for a in source["legal_actions"]],
            action_descriptions=[a["description"] for a in source["legal_actions"]])
        result["probe_id"] = prediction["probe_id"]
        result["original_chosen_index"] = mapping[result["chosen_index"]] if result["chosen_index"] is not None else None
        strict.append(result)
        if config["constrained"]:
            key = "action" if config["contract"] == "action_text" else "action_index"
            field = "description" if config["contract"] == "action_text" else "index"
            candidates = [json.dumps({key: a[field]}, separators=(",", ":")) for a in source["legal_actions"]]
            if not decision["valid"] or raw not in candidates:
                raise ValueError("constrained arm emitted a noncanonical/nonlegal candidate")
        remapped_decision = dict(decision)
        remapped_decision["displayed_action_index"] = decision["action_index"]
        if decision["valid"]:
            remapped_decision["action_index"] = mapping[decision["action_index"]]
        remapped.append({**prediction, "decision": remapped_decision})
    return remapped, sources, strict


def validate_comparison(first_config, second_config, first_sources, second_sources, *, comparison_pair=None):
    # Metadata and the four intended interventions may differ. All other CLI
    # and generator-code provenance must match, including unknown future fields.
    allowed = {"source", "source_sha256", "out", "started", "adapter", "adapter_identity", "contract", "constrained"}
    if comparison_pair in DECLARED_MODEL_COMPARISONS:
        allowed.add("model")
    required = {"model", "temperature", "batch_size", "max_tokens", "max_retries", "script_sha256"}
    for config in (first_config, second_config):
        if not required.issubset(config):
            raise ValueError("comparison missing required generation provenance")
    differences = [key for key in first_config.keys() | second_config.keys()
                   if key not in allowed and first_config.get(key) != second_config.get(key)]
    if differences:
        raise ValueError(f"undeclared generation/config intervention: {sorted(differences)}")
    if list(first_sources) != list(second_sources):
        raise ValueError("paired source cohort membership/order differs")
    allowed_row_fields = {"state_text", "legal_actions", "probe_original_indices"}
    for key in first_sources:
        a = {k: v for k, v in first_sources[key].items() if k not in allowed_row_fields}
        b = {k: v for k, v in second_sources[key].items() if k not in allowed_row_fields}
        if a != b:
            raise ValueError(f"undeclared source-row intervention at probe {key}")


def action_consistency(first, second):
    """Compare executed intent in ORIGINAL teacher-index coordinates."""
    a = indexed_rows(first, name="first arm")
    b = indexed_rows(second, name="second arm")
    if set(a) != set(b):
        raise ValueError("action-consistency cohorts differ")
    worlds, screens = defaultdict(list), defaultdict(list)
    for probe in a:
        if a[probe]["decision"]["valid"] and b[probe]["decision"]["valid"]:
            same = a[probe]["decision"]["action_index"] == b[probe]["decision"]["action_index"]
            worlds[a[probe]["world_seed"]].append(same)
            screens[a[probe]["probe_screen"]].append(same)
    values = [value for group in worlds.values() for value in group]
    world_means = [statistics.mean(group) for group in worlds.values()]
    return {"n_states": len(a), "n_common_valid": len(values),
            "same_original_action_rate_given_both_valid": mean(values),
            "n_worlds": len(worlds), "mean_world_consistency": mean(world_means),
            "world_bootstrap_ci": list(bootstrap_ci(world_means, seed=905)) if world_means else None,
            "by_screen": {screen: {"n_common_valid": len(group), "same_original_action_rate_given_both_valid": mean(group)}
                          for screen, group in screens.items()}}


def values_for(label, kind):
    if kind == "heuristic":
        entry = label["values"][kind]
        return entry["action_final_floors"] if entry["complete_terminal_action_set"] else None
    return label.get("random3_mean", {}).get("action_final_floors")


def action_value_consistency(first, second, labels, kind):
    """Separate changed action identity from changed continuation-policy value."""
    a = indexed_rows(first, name="first arm")
    b = indexed_rows(second, name="second arm")
    if set(a) != set(b):
        raise ValueError("action-value consistency cohorts differ")
    rows = []
    for probe in a:
        values = values_for(labels[probe], kind)
        if values is None or not (a[probe]["decision"]["valid"] and b[probe]["decision"]["valid"]):
            continue
        ia, ib = (str(row[probe]["decision"]["action_index"]) for row in (a, b))
        if ia not in values or ib not in values:
            raise ValueError("valid action absent from complete teacher values")
        equal = math.isclose(values[ia], values[ib], rel_tol=0, abs_tol=1e-9)
        rows.append({"probe_id": probe, "same_action": ia == ib, "equal_value": equal,
                     "informative": max(values.values()) > min(values.values()),
                     "second_value_minus_first": values[ib] - values[ia]})

    def summarize_subset(items):
        return {"n": len(items), "n_same_action": sum(row["same_action"] for row in items),
                "n_different_action_equal_value": sum(not row["same_action"] and row["equal_value"] for row in items),
                "n_different_action_unequal_value": sum(not row["same_action"] and not row["equal_value"] for row in items),
                "same_action_rate": mean([row["same_action"] for row in items]),
                "equivalent_value_rate": mean([row["equal_value"] for row in items]),
                "n_second_higher_value": sum(not row["equal_value"] and row["second_value_minus_first"] > 0 for row in items),
                "n_second_lower_value": sum(not row["equal_value"] and row["second_value_minus_first"] < 0 for row in items)}

    return {"complete_teacher_and_both_valid": summarize_subset(rows),
            "informative_complete_teacher_and_both_valid": summarize_subset([row for row in rows if row["informative"]]),
            "interpretation": "Value ties are under this finite privileged continuation-policy label, not proof of strategically interchangeable actions."}


def paired_regret_on_subset(first, second, probe_ids):
    a = {row["probe_id"]: row for row in first if row["probe_id"] in probe_ids}
    b = {row["probe_id"]: row for row in second if row["probe_id"] in probe_ids}
    worlds = defaultdict(list)
    for probe in sorted(set(a) & set(b)):
        if a[probe]["valid"] and b[probe]["valid"]:
            worlds[a[probe]["world_seed"]].append(b[probe]["regret"] - a[probe]["regret"])
    deltas = [statistics.mean(group) for group in worlds.values()]
    return {"first": summary(list(a.values())), "second": summary(list(b.values())),
            "n_common_valid_states": sum(map(len, worlds.values())), "n_worlds": len(worlds),
            "mean_world_regret_delta_second_minus_first": mean(deltas),
            "bootstrap_ci": list(bootstrap_ci(deltas, seed=905)) if deltas else None}


def evaluate(prediction, label, values):
    value_list = list(values.values())
    best, worst = max(value_list), min(value_list)
    decision = prediction["decision"]
    chosen = str(decision["action_index"])
    valid = decision["valid"] and chosen in values
    regret = best - values[chosen] if valid else None
    return {"probe_id": label["probe_id"], "world_seed": label["world_seed"],
            "screen": label["probe_screen"], "valid": valid,
            "informative": best > worst, "regret": regret,
            "normalized_regret": regret/(best-worst) if valid and best > worst else None,
            "top_set_match": regret == 0 if valid else None}


def summary(rows):
    valid = [r for r in rows if r["valid"]]
    informative = [r for r in valid if r["informative"]]
    return {"n": len(rows), "n_valid": len(valid),
            "n_valid_informative": len(informative),
            "mean_regret_given_valid": mean([r["regret"] for r in valid]),
            "mean_normalized_regret_given_valid_informative": mean([r["normalized_regret"] for r in informative]),
            "top_set_rate_given_valid_informative": mean([r["top_set_match"] for r in informative])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("data/experiments/astra_5sep/policy"))
    parser.add_argument("--teacher", type=Path, default=Path("data/experiments/astra_5sep/teacher/labels.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/astra_5sep/policy/analysis.json"))
    args = parser.parse_args()
    labels = indexed_rows(read_rows(args.teacher), name="teacher labels")
    arms, per_state, arm_configs, arm_sources, arm_predictions = {}, {}, {}, {}, {}
    for path in sorted(args.policy.glob("*/predictions.jsonl")):
        if path.parent.name.startswith("smoke") or not (path.parent/"report.json").exists():
            continue
        predictions = read_rows(path)
        config = json.loads((path.parent/"config.json").read_text())
        predictions, sources, strict = validate_arm(config, predictions, labels)
        arm_configs[path.parent.name] = config
        arm_sources[path.parent.name] = sources
        arm_predictions[path.parent.name] = predictions
        arm = {"n": len(predictions), "pipeline_valid": sum(p["decision"]["valid"] for p in predictions),
               "strict_generation": strict, "values": {},
               "validity_semantics": "single generation attempt; no retry loop",
               "validated_source_sha256": config["source_sha256"],
               "menu_remapped": any("probe_original_indices" in row for row in sources.values())}
        per_state[path.parent.name] = {}
        for kind in ("heuristic", "random3_mean"):
            evaluated = []
            for p in predictions:
                label = labels[p["probe_id"]]
                values = values_for(label, kind)
                if values is not None:
                    evaluated.append(evaluate(p, label, values))
            groups = defaultdict(list)
            for r in evaluated:
                groups[r["screen"]].append(r)
            arm["values"][kind] = {"overall": summary(evaluated), "by_screen": {k: summary(v) for k,v in groups.items()}}
            per_state[path.parent.name][kind] = evaluated
        arms[path.parent.name] = arm

    comparisons, consistencies, value_consistencies, changed_text_comparisons = {}, {}, {}, {}
    comparison_provenance = {}
    within_model_comparisons = [
        ("base_text_free", "base_text_constrained"),
        ("base_index_free", "base_index_constrained"),
        ("base_text_constrained", "base_index_constrained"),
        ("base_text_constrained", "rl1_text_constrained"),
        ("base_text_constrained", "rl5_text_constrained"),
        ("base_text_constrained", "base_text_enriched_constrained"),
        ("base_index_constrained", "base_index_enriched_constrained"),
        ("rl1_text_constrained", "rl1_text_enriched_constrained"),
        ("base_text_constrained", "base_text_reversed_constrained"),
        ("base_index_constrained", "base_index_reversed_constrained"),
        ("rl1_text_constrained", "rl1_text_reversed_constrained"),
        ("base_text_sampled", "rl1_text_sampled"),
        ("base_text_sampled", "rl5_text_sampled"),
        ("gemma26_text_free", "gemma26_text_constrained"),
        ("gemma26_text_constrained", "gemma26_text_enriched_constrained"),
        ("gemma26_text_constrained", "gemma26_text_reversed_constrained"),
    ]
    for first, second in within_model_comparisons + sorted(DECLARED_MODEL_COMPARISONS):
        if first not in per_state or second not in per_state:
            continue
        validate_comparison(arm_configs[first], arm_configs[second], arm_sources[first], arm_sources[second],
                            comparison_pair=(first, second))
        model_changed = arm_configs[first]["model"] != arm_configs[second]["model"]
        comparison_provenance[f"{second}_minus_{first}"] = {
            "model_intervention_declared": (first, second) in DECLARED_MODEL_COMPARISONS,
            "model_intervention_observed": model_changed,
            "first_model": arm_configs[first]["model"], "second_model": arm_configs[second]["model"],
            "caveat": MODEL_BUNDLE_CAVEAT if model_changed else None}
        consistencies[f"{second}_versus_{first}"] = action_consistency(arm_predictions[first], arm_predictions[second])
        value_consistencies[f"{second}_versus_{first}"] = {
            kind: action_value_consistency(arm_predictions[first], arm_predictions[second], labels, kind)
            for kind in ("heuristic", "random3_mean")}
        changed_ids = {probe for probe in arm_sources[first]
                       if arm_sources[first][probe]["state_text"] != arm_sources[second][probe]["state_text"]}
        if changed_ids:
            changed_text_comparisons[f"{second}_minus_{first}"] = {
                "subset_definition": "state_text actually differs between the two validated source cohorts",
                "n_changed_source_states": len(changed_ids), "probe_ids": sorted(changed_ids),
                "action_consistency": action_consistency(
                    [row for row in arm_predictions[first] if row["probe_id"] in changed_ids],
                    [row for row in arm_predictions[second] if row["probe_id"] in changed_ids]),
                "values": {kind: paired_regret_on_subset(per_state[first][kind], per_state[second][kind], changed_ids)
                           for kind in ("heuristic", "random3_mean")}}
        comparison = {}
        for kind in ("heuristic", "random3_mean"):
            a = {r["probe_id"]: r for r in per_state[first][kind]}
            b = {r["probe_id"]: r for r in per_state[second][kind]}
            worlds = defaultdict(list)
            for probe in sorted(set(a) & set(b)):
                if a[probe]["valid"] and b[probe]["valid"]:
                    worlds[a[probe]["world_seed"]].append(b[probe]["regret"] - a[probe]["regret"])
            deltas = [statistics.mean(v) for v in worlds.values()]
            comparison[kind] = {"n_common_valid_states": sum(map(len, worlds.values())), "n_worlds": len(worlds),
                "mean_world_regret_delta_second_minus_first": mean(deltas),
                "bootstrap_ci": list(bootstrap_ci(deltas, seed=905)) if deltas else None}
        comparisons[f"{second}_minus_{first}"] = comparison
    report = {"arms": arms, "paired_conditional_regret": comparisons, "paired_action_consistency": consistencies,
        "paired_action_value_consistency": value_consistencies,
        "secondary_changed_state_text_comparisons": changed_text_comparisons,
        "comparison_provenance": comparison_provenance,
        "limitations": ["Only complete matched-cohort arms enter comparisons; incomplete teacher action sets excluded equally.",
                        "Validity is single-attempt generation; max_retries config is not used in probe mode.",
                        "Regret is against privileged continuation-policy values, not optimal-action truth.",
                        "Regret given valid excludes malformed choices and is selection biased; read validity beside it.",
                        "Pairwise intervals use world means across states valid under both arms; descriptive exploratory comparisons.",
                        MODEL_BUNDLE_CAVEAT]}
    args.out.write_text(json.dumps(report, indent=2))
    args.out.with_name("analysis_per_state.json").write_text(json.dumps(per_state, indent=2))
    for name, arm in arms.items():
        print(name, arm["pipeline_valid"], arm["values"]["heuristic"]["overall"], arm["values"]["random3_mean"]["overall"])


if __name__ == "__main__":
    main()
