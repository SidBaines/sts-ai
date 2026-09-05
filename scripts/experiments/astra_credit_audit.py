"""CPU-only audit of saved OOC PG credit assignment and MLX trainer outliers.

Run from the repository root with PYTHONPATH=src. Loads a cached tokenizer,
never model weights. Counterfactual advantages are descriptive sensitivity
checks, not unbiased alternative policy-gradient estimators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import tempfile
from typing import Any
import unittest

from sts_ai.train.dataset_builder import _skip_reason, trajectory_is_contiguous
from sts_ai.train.sft_format import chat_template_probe_hash, tokenize_example


def prepare_failure_cohort(source: Path, out: Path, compare_cohort: Path, iteration: int = 6) -> None:
    """Freeze every fatal decision, including single-action menus excluded by the broad probe."""
    from transformers import AutoTokenizer
    from sts_ai.prompting import NEUTRAL_FRAME, render_action_prompt
    from sts_ai.schemas import LegalAction

    directory = source / f"iter_{iteration}" / "rollouts"
    selected, inputs = [], []
    for path in sorted(directory.glob("seed_*_r*.jsonl")):
        records = read_jsonl(path)
        fatal = [row for row in records if row.get("action_executed") is False]
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        if len(fatal) != int(meta["stopped_reason"] == "agent_invalid"):
            raise ValueError(f"fatal record count disagrees with episode outcome: {path}")
        if not fatal:
            continue
        row = dict(fatal[0])
        if row != records[-1] or row["agent"]["valid"] or row["phase"] != "out_of_combat":
            raise ValueError(f"unexpected fatal decision shape: {path}")
        row["probe_id"] = len(selected)
        screen = row["state"]["screen_state"].split(".")[-1]
        row["probe_screen"] = "NEOW" if screen == "EVENT_SCREEN" and row["state"]["floor"] == 0 else screen
        row["source_trace"] = str(path)
        selected.append(row)
        for input_path in (path, meta_path):
            inputs.append({"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()})
    out.mkdir(parents=True, exist_ok=True)
    cohort = out / "cohort.jsonl"
    with cohort.open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    manifest = {"kind": "astra_original_fatal_state_replay_v1", "n": len(selected),
                "source": str(directory), "iteration": iteration, "cohort": str(cohort),
                "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_inputs": inputs, "selected_by_screen": dict(Counter(row["probe_screen"] for row in selected)),
                "legal_action_count_histogram": dict(Counter(len(row["legal_actions"]) for row in selected)),
                "original_retries": dict(Counter(row["agent"].get("retries", 0) for row in selected)),
                "selection": "Every action_executed=false final record from a completed agent_invalid episode; no minimum-menu-size filter; original record fields retained verbatim.",
                "limitations": ["Conditioned on historical failures; not a representative policy-quality cohort.",
                                "Probe replays exact recorded state_text/legal_actions under its standard first-attempt prompt; historical raw responses here were final retry attempts.",
                                "Historical iteration6 policy used iter5/adapter; sampling temperature0.7, cap4096. Current probe cap/sampling seed may differ."]}
    (out / "cohort.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cohort": str(cohort), "n": len(selected), "actions": manifest["legal_action_count_histogram"]}), flush=True)

    config = json.loads((source / "run_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    telemetry = []
    for name, rows in (("historical_failures", selected), ("fresh133", read_jsonl(compare_cohort))):
        for row in rows:
            prompt = render_action_prompt(row["state_text"], [LegalAction(**a) for a in row["legal_actions"]], NEUTRAL_FRAME, output_contract="action_text")
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            telemetry.append({"cohort": name, "probe_id": row["probe_id"], "world_seed": row["world_seed"],
                              "probe_screen": row["probe_screen"], "floor": row["state"]["floor"],
                              "n_legal_actions": len(row["legal_actions"]), "state_text_chars": len(row["state_text"]),
                              "first_attempt_prompt_tokens": len(tokenizer.encode(rendered)),
                              "stored_original_prompt_tokens": row["agent"].get("prompt_tokens"),
                              "stored_original_completion_tokens": row["agent"].get("completion_tokens"),
                              "stored_original_raw_response": row["agent"].get("raw_response"),
                              "original_error": (row["agent"].get("metadata") or {}).get("error"),
                              "original_retries": row["agent"].get("retries", 0)})

    def distribution(values):
        values = [value for value in values if isinstance(value, (int, float))]
        return {"n": len(values), "min": min(values) if values else None, "max": max(values) if values else None,
                "mean": statistics.mean(values) if values else None, "median": statistics.median(values) if values else None}

    comparison = {"compare_cohort": str(compare_cohort), "compare_cohort_sha256": hashlib.sha256(compare_cohort.read_bytes()).hexdigest(),
                  "chat_template_probe_hash": chat_template_probe_hash(tokenizer, enable_thinking=False), "cohorts": {},
                  "interpretation": "Fresh cohort source traces were generated by a random agent, so their stored model-token telemetry is unavailable/zero; reconstructed first-attempt prompt lengths are comparable."}
    for name in ("historical_failures", "fresh133"):
        rows = [row for row in telemetry if row["cohort"] == name]
        comparison["cohorts"][name] = {"n": len(rows), "screens": dict(Counter(row["probe_screen"] for row in rows)),
            "n_single_action": sum(row["n_legal_actions"] == 1 for row in rows),
            **{key: distribution([row[key] for row in rows]) for key in
               ("n_legal_actions", "floor", "first_attempt_prompt_tokens", "state_text_chars", "stored_original_prompt_tokens", "stored_original_completion_tokens")}}
    (out / "cohort_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    write_csv(out / "cohort_telemetry.csv", telemetry)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_snapshot_pair(left: Path, right: Path, out: Path) -> None:
    """Compare exact text-model tensor bytes without loading any ML tensors."""
    def headers(directory: Path) -> dict[str, tuple[Path, int, dict[str, Any]]]:
        tensors = {}
        for path in directory.glob("*.safetensors"):
            with path.open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            for name, item in header.items():
                if name != "__metadata__":
                    tensors[name] = (path, size + 8, item)
        return tensors

    def digest(item: tuple[Path, int, dict[str, Any]]) -> str:
        path, offset, header = item
        start, stop = header["data_offsets"]
        remaining = stop - start
        result = hashlib.sha256()
        with path.open("rb") as handle:
            handle.seek(offset + start)
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"truncated safetensors file: {path}")
                result.update(block)
                remaining -= len(block)
        return result.hexdigest()

    a, b = headers(left), headers(right)
    rows = []
    for name in sorted(a.keys() & b.keys()):
        if not name.startswith("language_model."):
            continue
        ha, hb = a[name][2], b[name][2]
        da = digest(a[name])
        db = da if a[name][0].resolve() == b[name][0].resolve() and ha == hb else digest(b[name])
        rows.append({"name": name, "sha256_left": da, "sha256_right": db,
                     "identical": da == db and ha["dtype"] == hb["dtype"] and ha["shape"] == hb["shape"]})
    ca = json.loads((left / "config.json").read_text())
    cb = json.loads((right / "config.json").read_text())
    result = {"left": str(left), "right": str(right),
              "removed_keys": sorted(a.keys() - b.keys()), "added_keys": sorted(b.keys() - a.keys()),
              "config_differences": {key: {"left": ca.get(key), "right": cb.get(key)} for key in sorted(ca.keys() | cb.keys()) if ca.get(key) != cb.get(key)},
              "n_common_language_tensors": len(rows), "n_changed_common_language_tensors": sum(not row["identical"] for row in rows),
              "language_tensors": rows,
              "method": "SHA256 of every byte of every common language_model tensor, parsed by safetensors offsets; shapes and dtype also compared; 1MiB buffers; no ML weights loaded."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshot_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("n_common_language_tensors", "n_changed_common_language_tensors")}))


def normalized_advantages(floors: list[int], eps: float = 1e-6) -> list[float]:
    if not floors:
        return []
    mean = statistics.mean(floors)
    std = statistics.pstdev(floors)
    return [(floor - mean) / (std + eps) if std else 0.0 for floor in floors]


def action_class(record: dict[str, Any]) -> str:
    description = str((record.get("selected_action") or {}).get("description", ""))
    if not record.get("action_executed", True):
        return "unexecuted"
    if description.startswith("choose map node"):
        return "map_" + description.partition("room=")[2].lower()
    for prefix, name in (
        ("rest", "rest"), ("smith", "smith"), ("upgrade", "upgrade"),
        ("take card", "take_card"), ("take relic", "take_relic"),
        ("take potion", "take_potion"), ("take gold", "take_gold"),
        ("skip rewards", "skip_rewards"), ("event option", "event"),
        ("buy card", "buy_card"), ("buy relic", "buy_relic"),
        ("buy potion", "buy_potion"), ("remove", "remove"),
        ("select card", "select_card"), ("open chest", "open_chest"),
        ("skip chest", "skip_chest"), ("leave", "leave"),
    ):
        if description.startswith(prefix):
            return name
    return description.split(" ", 1)[0] or "none"


def coefficient_summary(rows: list[dict[str, Any]], field: str = "advantage") -> dict[str, Any]:
    """Mass of row coefficients, not gradient norms or realized parameter changes."""
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "n_rows": len(rows), "n_scored": len(values),
        "n_positive": sum(value > 1e-10 for value in values),
        "n_negative": sum(value < -1e-10 for value in values),
        "signed_mass": sum(values), "absolute_mass": sum(abs(value) for value in values),
        "positive_mass": sum(value for value in values if value > 0),
        "negative_mass": sum(value for value in values if value < 0),
    }


def grouped_coefficients(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [dict(zip(keys, key)) | coefficient_summary(value)
            for key, value in sorted(groups.items())]


def group_diagnostic(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in episodes if row["stopped_reason"] != "simulator_error"]
    clean = [row for row in valid if row["stopped_reason"] != "agent_invalid"]
    floors = [row["final_floor"] for row in valid]
    clean_floors = [row["final_floor"] for row in clean]
    for row, advantage in zip(valid, normalized_advantages(floors)):
        row["advantage"] = advantage
    for row in episodes:
        row.setdefault("advantage", None)
        row["counterfactual_advantage"] = None
    for row, advantage in zip(clean, normalized_advantages(clean_floors)):
        row["counterfactual_advantage"] = advantage
    variance = statistics.pvariance(floors) if floors else 0.0
    clean_variance = statistics.pvariance(clean_floors) if clean_floors else 0.0
    if not variance:
        classification = "zero_variance"
    elif len(clean) < 2:
        classification = "insufficient_nonformat_repeats"
    elif not clean_variance:
        classification = "format_only_observed_separation"
    else:
        classification = "nonformat_outcome_separation"
    overall_mean = statistics.mean(floors) if floors else 0.0
    between_ss = 0.0
    for format_dead in (False, True):
        subset = [row["final_floor"] for row in valid
                  if (row["stopped_reason"] == "agent_invalid") == format_dead]
        if subset:
            between_ss += len(subset) * (statistics.mean(subset) - overall_mean) ** 2
    return {
        "iteration": episodes[0]["iteration"], "world_seed": episodes[0]["world_seed"],
        "n_episodes": len(valid), "n_format_dead": len(valid) - len(clean),
        "n_nonformat": len(clean), "mean_floor": overall_mean,
        "floor_std": math.sqrt(variance), "nonformat_floor_std": math.sqrt(clean_variance),
        "classification": classification,
        "format_vs_nonformat_variance_share": between_ss / (len(valid) * variance) if variance else 0.0,
        "n_nonformat_sign_flips": sum(row["advantage"] * row["counterfactual_advantage"] < -1e-10 for row in clean),
        "n_nonformat_advantage_zeroed": sum(abs(row["advantage"]) > 1e-10 and abs(row["counterfactual_advantage"]) < 1e-10 for row in clean),
    }


def training_rows(examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct train_pg_mlx token filtering and epoch-zero shuffle exactly."""
    kept, dropped = [], []
    for example in examples:
        tokenized = tokenize_example(example, tokenizer, loss_mask_mode="action")
        labels = tokenized["labels"][:max_seq_len]
        action_mask = tokenized["action_mask"][:max_seq_len]
        row = dict(example)
        row["supervised_token_count"] = sum(label != -100 for label in labels)
        row["action_token_count"] = sum(bool(value) for value in action_mask)
        row["actual_total_tokens"] = min(len(tokenized["input_ids"]), max_seq_len)
        row["was_truncated"] = len(tokenized["input_ids"]) > max_seq_len
        if not row["supervised_token_count"] or not row["action_token_count"]:
            dropped.append(row)
        else:
            kept.append(row)
    random.Random(17).shuffle(kept)
    return kept, dropped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_summary(report: dict[str, Any], top_kl: list[dict[str, Any]]) -> str:
    lines = [
        "# OOC credit-assignment audit — astra-5sep", "",
        "CPU-only audit of iterations 0–6. Every saved training row was reconstructed "
        "with the cached tokenizer, exact action mask and truncation, and the MLX "
        "epoch-zero shuffle. All 6,995 trainer log rows matched their reconstructed "
        "action-token count, supervised-token count, and advantage. No model was loaded.", "",
        "`mass` below means the sum of per-example advantage coefficients, not a "
        "measured gradient norm. Counterfactuals remove format-dead episodes before "
        "recomputing each seed-group baseline and standard deviation, then rescore "
        "the identical retained nonformat rows. This is post-treatment selection "
        "and is only a sensitivity diagnostic.", "",
        "| Iter | Format-dead episodes | Trained rows from format deaths | "
        "Nonformat-row sign flips / rows | Format-dominated groups (variance share ≥75%) | "
        "Mean / median KL | Largest row's share of KL |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["iterations"]:
        episode = next((item for item in row["episode_by_stop"] if item["stop_class"] == "format_death"), {})
        trained = next((item for item in row["row_by_stage_and_stop"] if item["stage"] == "trained" and item["stop_class"] == "format_death"), {})
        cf = row["counterfactual_on_identical_nonformat_training_rows"]
        lines.append(f"| {row['iteration']} | {episode.get('n_rows', 0)} | {trained.get('n_rows', 0)} | "
                     f"{cf['n_sign_flips']} / {cf['original']['n_rows']} | "
                     f"{row['n_groups_format_variance_share_ge_075']} / 16 | "
                     f"{row['kl']['mean']:.3f} / {row['kl']['median']:.3f} | "
                     f"{row['kl']['largest_row_share_of_total_kl']:.1%} |")
    lines += ["", "The strongest credit-assignment distortion is iteration 6: 105 of 455 "
              "retained nonformat rows change advantage sign, and 23 become zero, "
              "when format-dead episodes leave the baseline. Their net coefficient "
              "mass changes from +326.14 to -5.99. Six groups have at least 75% of "
              "floor variance between format-death and ordinary-death classes, but "
              "13 groups still contain observable variation among nonformat outcomes. "
              "Calling the whole iteration purely format learning would overstate the evidence.", "",
              "The exact failed completion is never trained: the shared SFT skip "
              "rules remove fatal unexecuted records and retries. In iteration 6 "
              "that means 56 fatal records and 485 retries. Earlier valid decisions "
              "from those 56 doomed episodes supply 543 trained rows; 186 have "
              "positive advantage. Sixteen of the 56 format-dead episodes themselves "
              "have positive advantage because their floor still exceeds their "
              "seed-group mean. This is outcome credit, not a direct format penalty.", "",
              "Required formatting tokens receive 36–42% of the absolute "
              "coefficient allocation despite the action-only mask. This is "
              "coefficient accounting, not a claim that those tokens cause that "
              "fraction of the parameter update.", "",
              "Capping materially reduces the long-episode weighting of uncapped "
              "decision means. It does not remove class imbalance when many short "
              "format-dead trajectories exhaust their eligible rows. Detailed "
              "positive, negative, signed, and absolute masses are in report.json "
              "by screen, action class, retention stage, and stop class.", "",
              "The extreme sampled k3 KL values are attached to short ordinary "
              "action responses, not long reasoning. A single row supplies 79.4% "
              "of iteration 0's summed KL and 82.4% of iteration 6's. Mean KL is "
              "therefore not a stable summary of typical behavior. The stored "
              "logs cannot identify which individual token or numerical mechanism "
              "caused each outlier; fresh forward passes would be required.", "",
              "| Iter | Step | Episode / decision | Action | Completion tokens | KL |", "|---|---:|---|---|---:|---:|"]
    for row in top_kl[:8]:
        lines.append(f"| {row['iteration']} | {row['step']} | {row['stem']} / {row['decision_index']} | "
                     f"{row['selected_action']} | {row['completion_tokens']} | {row['mean_kl']:.2f} |")
    lines += ["", "Reproduce:", "", "```sh",
              "PYTHONPATH=src .venv/bin/python scripts/experiments/astra_credit_audit.py --self-test",
              "PYTHONPATH=src .venv/bin/python scripts/experiments/astra_credit_audit.py", "```", "",
              "Outputs: report.json (aggregates, definitions, input hashes), episodes.csv, "
              "groups.csv, decision_coefficients.csv, kl_rows.csv (all exact joins), "
              "and kl_top30.csv. The original data and trainer remain unchanged.", ""]
    return "\n".join(lines)


def audit(source: Path, out: Path, iterations: list[int]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = json.loads((source / "run_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    max_seq_len = int(config["max_seq_len"])
    all_episodes, all_groups, all_rows, all_kl, results = [], [], [], [], []
    provenance = []
    for iteration in iterations:
        directory = source / f"iter_{iteration}"
        manifest = json.loads((directory / "pg.jsonl.manifest.json").read_text())
        assert chat_template_probe_hash(tokenizer, enable_thinking=False) == manifest["chat_template_hash"]
        episodes, groups, records = [], defaultdict(list), {}
        for meta_path in sorted((directory / "rollouts").glob("*.meta.json")):
            meta = json.loads(meta_path.read_text())
            stem = meta_path.name.removesuffix(".meta.json")
            episode = {"iteration": iteration, "stem": stem,
                       **{key: meta[key] for key in ("world_seed", "rollout_index", "final_floor", "stopped_reason")}}
            episode["stop_class"] = "format_death" if meta["stopped_reason"] == "agent_invalid" else "ordinary_death" if meta["stopped_reason"] == "terminal" and "LOSS" in str(meta.get("outcome")) else meta["stopped_reason"]
            episodes.append(episode)
            groups[meta["world_seed"]].append(episode)
            trace = read_jsonl(directory / "rollouts" / f"{stem}.jsonl")
            assert trajectory_is_contiguous(trace), stem
            for record in trace:
                records[(stem, record["decision_index"])] = record
            for path in (meta_path, directory / "rollouts" / f"{stem}.jsonl"):
                provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        group_rows = [group_diagnostic(group) for group in groups.values()]
        episode_by_stem = {row["stem"]: row for row in episodes}
        examples = read_jsonl(directory / "pg.jsonl")
        trained, truncated = training_rows(examples, tokenizer, max_seq_len)
        logs = json.loads((directory / "adapter" / "trainer_log.json").read_text())
        assert len(trained) == len(logs), (iteration, len(trained), len(logs))
        capped_keys = {(row["stem"], row["decision_index"]) for row in examples}
        trained_keys = {(row["stem"], row["decision_index"]) for row in trained}
        iteration_rows = []
        for (stem, decision_index), record in records.items():
            episode = episode_by_stem[stem]
            skip = _skip_reason(record, ())
            stage = "record_excluded" if skip else "cap_excluded" if (stem, decision_index) not in capped_keys else "truncation_excluded" if (stem, decision_index) not in trained_keys else "trained"
            row = {"iteration": iteration, "stem": stem, "decision_index": decision_index,
                   "screen": record["state"].get("screen_state", "unknown").split(".")[-1],
                   "floor": record["state"].get("floor"), "action_class": action_class(record),
                   "stop_class": episode["stop_class"], "stage": stage, "skip_reason": skip,
                   "advantage": episode["advantage"], "counterfactual_advantage": episode["counterfactual_advantage"],
                   "completion_chars": len((record.get("agent") or {}).get("raw_response") or ""),
                   "selected_action": (record.get("selected_action") or {}).get("description", "")}
            iteration_rows.append(row)
        row_lookup = {(row["stem"], row["decision_index"]): row for row in iteration_rows}
        kl_rows = []
        for example, log in zip(trained, logs):
            for key in ("action_token_count", "supervised_token_count"):
                assert example[key] == log[key], (iteration, log["step"], key)
            assert math.isclose(example["advantage"], log["mean_advantage"], rel_tol=1e-5, abs_tol=1e-6)
            row = dict(row_lookup[(example["stem"], example["decision_index"])])
            assert math.isclose(row["advantage"], example["advantage"], rel_tol=1e-8, abs_tol=1e-8)
            row.update({key: log[key] for key in ("step", "mean_kl", "loss", "action_token_count", "supervised_token_count")})
            row["prompt_tokens"] = example["token_counts"]["n_prompt_tokens"]
            row["completion_tokens"] = example["token_counts"]["n_completion_tokens"]
            row["was_truncated"] = example["was_truncated"]
            row["completion"] = example["completion"]
            row["kl_penalty"] = config["kl_beta"] * row["mean_kl"]
            row["format_coefficient_mass"] = row["advantage"] * (1 - row["action_token_count"] / row["supervised_token_count"])
            kl_rows.append(row)
        trained_records = [row for row in iteration_rows if row["stage"] == "trained"]
        for episode in episodes:
            episode_rows = [row for row in iteration_rows if row["stem"] == episode["stem"]]
            episode["n_recorded"] = len(episode_rows)
            for stage in ("record_excluded", "cap_excluded", "truncation_excluded", "trained"):
                episode[f"n_{stage}"] = sum(row["stage"] == stage for row in episode_rows)
        counterfactual_rows = [row for row in trained_records if row["counterfactual_advantage"] is not None]
        sorted_by_length = sorted(kl_rows, key=lambda row: row["completion_tokens"])
        deciles = []
        for decile in range(10):
            subset = sorted_by_length[len(sorted_by_length) * decile // 10:len(sorted_by_length) * (decile + 1) // 10]
            deciles.append({"decile": decile + 1, "n": len(subset),
                           "min_completion_tokens": min(row["completion_tokens"] for row in subset),
                           "max_completion_tokens": max(row["completion_tokens"] for row in subset),
                           "mean_kl": statistics.mean(row["mean_kl"] for row in subset),
                           "median_kl": statistics.median(row["mean_kl"] for row in subset),
                           "max_kl": max(row["mean_kl"] for row in subset)})
        result = {"iteration": iteration, "n_episodes": len(episodes),
                  "group_classes": dict(Counter(row["classification"] for row in group_rows)),
                  "n_groups_format_variance_share_ge_075": sum(row["format_vs_nonformat_variance_share"] >= .75 for row in group_rows),
                  "episode_by_stop": grouped_coefficients(episodes, ("stop_class",)),
                  "row_by_stage_and_stop": grouped_coefficients(iteration_rows, ("stage", "stop_class")),
                  "trained_by_screen_and_stop": grouped_coefficients(trained_records, ("screen", "stop_class")),
                  "trained_by_action_and_stop": grouped_coefficients(trained_records, ("action_class", "stop_class")),
                  "counterfactual_on_identical_nonformat_training_rows": {
                      "original": coefficient_summary(counterfactual_rows),
                      "excluded_format_baseline": coefficient_summary(counterfactual_rows, "counterfactual_advantage"),
                      "n_sign_flips": sum(row["advantage"] * row["counterfactual_advantage"] < -1e-10 for row in counterfactual_rows),
                      "n_advantage_zeroed": sum(abs(row["advantage"]) > 1e-10 and abs(row["counterfactual_advantage"]) < 1e-10 for row in counterfactual_rows)},
                  "n_tokenization_dropped": len(truncated),
                  "kl": {"mean": statistics.mean(row["mean_kl"] for row in kl_rows),
                         "median": statistics.median(row["mean_kl"] for row in kl_rows),
                         "max": max(row["mean_kl"] for row in kl_rows),
                         "largest_row_share_of_total_kl": max(row["mean_kl"] for row in kl_rows) / sum(row["mean_kl"] for row in kl_rows),
                         "n_kl_penalty_exceeds_abs_advantage": sum(row["kl_penalty"] > abs(row["advantage"]) for row in kl_rows),
                         "n_kl_gt_10": sum(row["mean_kl"] > 10 for row in kl_rows),
                         "top_1pct_share_of_total_kl": sum(sorted((row["mean_kl"] for row in kl_rows), reverse=True)[:max(1, len(kl_rows)//100)]) / sum(row["mean_kl"] for row in kl_rows),
                         "completion_length_deciles": deciles},
                  "format_token_fraction_of_absolute_coefficient_mass": sum(abs(row["format_coefficient_mass"]) for row in kl_rows) / sum(abs(row["advantage"]) for row in kl_rows)}
        results.append(result)
        all_episodes.extend(episodes)
        all_groups.extend(group_rows)
        all_rows.extend(iteration_rows)
        all_kl.extend(kl_rows)
        for path in (directory / "pg.jsonl", directory / "pg.jsonl.manifest.json", directory / "adapter" / "trainer_log.json"):
            provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({"iteration": iteration, "trained": len(trained), "groups": result["group_classes"], "max_kl": result["kl"]["max"]}), flush=True)
    report = {"source": str(source), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "semantics": {"coefficient_mass": "Sum of per-example advantages under row-mean token loss; NOT gradient norms. Token lengths redistribute each row coefficient across supervised tokens.",
                            "counterfactual": "Remove agent_invalid episodes from each seed group, recenter and renormalize surviving final floors, compare identical retained surviving rows. Post-treatment selection: diagnostic only, not an unbiased estimator.",
                            "kl_join": "Cached tokenizer, exact action masks/truncation, Random(17) epoch-zero shuffle; every log row checked against token counts and advantage.",
                            "ordinary_death": "terminal episode with PLAYER_LOSS outcome; other terminal outcomes kept separately.",
                            "cap_excluded": "Eligible under shared skip rules but absent from saved capped PG dataset; may include action-mask-unavailable rows (verify manifest).",
                            "format_only_observed_separation": "Positive full-group floor variance, >=2 nonformat episodes, zero variance among those episodes. Observed designation, not proof of sole causal mechanism."},
              "provenance": provenance, "iterations": results}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "episodes.csv", all_episodes)
    write_csv(out / "groups.csv", all_groups)
    write_csv(out / "decision_coefficients.csv", all_rows)
    write_csv(out / "kl_rows.csv", all_kl)
    top_kl = sorted(all_kl, key=lambda row: row["mean_kl"], reverse=True)[:30]
    write_csv(out / "kl_top30.csv", top_kl)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if iterations == list(range(7)):
        (out / "README.md").write_text(render_summary(report, top_kl))
    return report


class AuditTests(unittest.TestCase):
    def test_zero_variance(self) -> None:
        self.assertEqual(normalized_advantages([3, 3, 3]), [0, 0, 0])

    def test_format_only_and_counterfactual(self) -> None:
        rows = [{"iteration": 0, "world_seed": 1, "final_floor": floor,
                 "stopped_reason": stop} for floor, stop in
                [(4, "agent_invalid"), (16, "terminal"), (16, "terminal")]]
        result = group_diagnostic(rows)
        self.assertEqual(result["classification"], "format_only_observed_separation")
        self.assertAlmostEqual(result["format_vs_nonformat_variance_share"], 1)
        self.assertIsNone(rows[0]["counterfactual_advantage"])
        self.assertEqual(rows[1]["counterfactual_advantage"], 0)
        self.assertEqual(result["n_nonformat_advantage_zeroed"], 2)

    def test_sign_flip(self) -> None:
        rows = [{"iteration": 0, "world_seed": 1, "final_floor": floor,
                 "stopped_reason": stop} for floor, stop in
                [(0, "agent_invalid"), (0, "agent_invalid"), (10, "terminal"), (20, "terminal")]]
        result = group_diagnostic(rows)
        self.assertEqual(result["n_nonformat_sign_flips"], 1)

    def test_mass_is_not_token_weighted(self) -> None:
        result = coefficient_summary([{"advantage": 2}, {"advantage": -1}])
        self.assertEqual(result["signed_mass"], 1)
        self.assertEqual(result["absolute_mass"], 3)

    def test_single_nonformat_is_not_called_format_only(self) -> None:
        rows = [{"iteration": 0, "world_seed": 1, "final_floor": floor,
                 "stopped_reason": stop} for floor, stop in
                [(0, "agent_invalid"), (20, "terminal")]]
        result = group_diagnostic(rows)
        self.assertEqual(result["classification"], "insufficient_nonformat_repeats")

    def test_snapshot_repacking_and_changed_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "left", root / "right"
            for directory in (left, right):
                directory.mkdir()
                (directory / "config.json").write_text("{}")

            def write_tensor_file(directory: Path, reverse: bool, changed: bool = False) -> None:
                values = [("language_model.a", b"ab"), ("language_model.b", b"XY" if changed else b"cd")]
                if reverse:
                    values.reverse()
                header = {name: {"dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
                          for i, (name, _value) in enumerate(values)}
                encoded = json.dumps(header).encode()
                (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(value for _name, value in values))

            write_tensor_file(left, False)
            write_tensor_file(right, True)
            audit_snapshot_pair(left, right, root / "result")
            result = json.loads((root / "result" / "snapshot_audit.json").read_text())
            self.assertEqual(result["n_changed_common_language_tensors"], 0)
            write_tensor_file(right, True, changed=True)
            audit_snapshot_pair(left, right, root / "result")
            result = json.loads((root / "result" / "snapshot_audit.json").read_text())
            self.assertEqual(result["n_changed_common_language_tensors"], 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/rl/ooc_grpo_v1"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/astra_5sep/credit"))
    parser.add_argument("--iterations", nargs="+", type=int, default=list(range(7)))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot-left", type=Path)
    parser.add_argument("--snapshot-right", type=Path)
    parser.add_argument("--prepare-failure-cohort", action="store_true")
    parser.add_argument("--failure-iteration", type=int, default=6)
    parser.add_argument("--compare-cohort", type=Path, default=Path("data/experiments/astra_5sep/policy/cohort.jsonl"))
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.prepare_failure_cohort:
        prepare_failure_cohort(args.source, args.out, args.compare_cohort, args.failure_iteration)
        return
    if args.snapshot_left or args.snapshot_right:
        if not (args.snapshot_left and args.snapshot_right):
            parser.error("snapshot comparison requires both --snapshot-left and --snapshot-right")
        audit_snapshot_pair(args.snapshot_left, args.snapshot_right, args.out)
        return
    audit(args.source, args.out, args.iterations)


if __name__ == "__main__":
    main()
