#!/usr/bin/env python3
"""Export auditable plotting data from completed, frozen fresh-start results.

No model inference, training, model selection, gate verdicts or report writing.
Run analysis only in the frozen study runtime; render the JSON separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_study import read_json

DEFAULT_STUDY = Path("/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start/data/rl_runs/fresh_start_20261015")
SCHEMA = "fresh_gate_plot_data_v1"
RUNS = ("new_combat_5000", "new_combat_20000", "new_combat_full", "new_ooc_full",
        "old_combat_lr3e4", "old_combat_lr1e3")
SCORES = ("new_combat_5000", "new_combat_20000", "new_combat_full", "old_combat_full", "new_ooc_full")
G2_ARMS = ("combat_only", "ooc_only", "fresh_composite")
BEHAVIORS = ("smith_hp>=.85", "elite_choice_hp>=.9", "card_skip", "removal",
             "combat_potion_drink", "combat_potion_discard", "shop_spend")
CHOICE_NOTE = ("Observed shares of menus with >=2 legal actions and the matching WorldMetrics opportunity; "
               "other options may be unavailable. Policy-dependent cohorts are descriptive, not causal; "
               "these are not pure free-choice propensities.")
BEHAVIOR_OPPORTUNITIES = {
    "smith_hp>=.85": "HP >=85% menus offering smith (normally rest/smith); rest need not be available",
    "elite_choice_hp>=.9": "HP >=90% menus offering both elite and non-elite map nodes",
    "card_skip": ("explicit 'skip card reward' only, per rewards menu offering it; "
                  "proceeding past an untaken card is not counted; each gold/potion/relic pick "
                  "on the same screen adds an opportunity"),
    "removal": "shop menus offering buy card remove; excludes event removal",
    "combat_potion_drink": "combat menus offering drink potion; per menu, not per game or fight",
    "combat_potion_discard": "combat menus offering discard potion; per menu, not per game or fight",
    "shop_spend": "menus offering any buy action; chosen means positive gold decrease across the transition",
}
SELECTION_CRITERION = {"metric": "validation_soft_ce", "direction": "min", "tie_break": "lower_lr"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), "Nonfinite/non-numeric metric")
    return float(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON: " + value)


def decode(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_nonfinite)


class Inputs:
    """Hash inputs without loading large manifests; remember/recheck stable stats."""
    def __init__(self):
        self.identities, self.stamps = {}, {}

    @staticmethod
    def stamp(path):
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.identities:
            before = self.stamp(path)
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            require(self.stamp(path) == before, "Input changed while hashing: " + key)
            self.identities[key] = {"path": key, "bytes": before[0], "sha256": sha.hexdigest()}
            self.stamps[key] = before
        require(self.stamp(path) == self.stamps[key], "Input changed: " + key)
        identity = self.identities[key]
        if expected is not None:
            require(identity == expected, "Frozen file identity mismatch: " + key)
        return identity

    def json(self, path, expected=None):
        self.bind(path, expected)
        result = read_json(path, object_pairs_hook=_object, parse_constant=_nonfinite)
        self.bind(path, expected)
        return result

    def jsonl(self, path):
        self.bind(path)
        with Path(path).open() as stream:
            result = [decode(line) for line in stream if line.strip()]
        self.bind(path)
        return result

    def recheck(self):
        for path in self.identities:
            self.bind(path)


def required_paths(study):
    paths = [study / "all_warmup_selections_v1.json", study / "teacher_consistency_final_v1.json",
             study / "dev_cohort_4096.json", study / "comparisons/g1_panel_new_minus_old_v1.json",
             study / "comparisons/g1_labels_full_v1.json", study / "expectations/g1_labels_full_v1.json"]
    for name in SCORES:
        paths.extend(study / f"score_{name}_v1/{f}" for f in ("summary.json", "receipt.json", "test_rows.jsonl"))
        paths.append(study / f"selections/{name}_v1.json")
    for name in RUNS:
        paths.extend(study / f"fit_{name}_v1/{f}" for f in ("summary.json", "config.json", "metrics.jsonl"))
    for name in ("new_combat_full", "new_ooc_full"):
        paths.append(study / f"fit_{name}_v1/probe_selection.json")
    for arm in G2_ARMS:
        paths.extend([study / f"comparisons/g2_{arm}_v1.json", study / f"expectations/g2_{arm}_v1.json"])
    paths.append(study / "expectations/g1_panel_full_v1.json")
    for cohort in ("dev_champion_4096_v1_merged", *(f"g2_{a}_4096_v1_merged" for a in G2_ARMS)):
        paths.extend(study / cohort / f for f in ("summary.json", "receipt.json", "completion.json"))
    return paths


def check_ready(study):
    """Inventory only: no statistics, TEST-row reads, native imports or output files."""
    missing = [str(p) for p in required_paths(study) if not p.is_file()]
    return {"inventory_ready": not missing, "validation_performed": False, "missing": missing,
            "note": "Presence only; the export performs full binding/cohort validation. No partial export."}


def frozen_runtime(study):
    import numpy as np
    import torch
    expected = (study.parents[2] / ".venv").resolve()
    require(Path(sys.prefix).resolve() == expected, "Analysis requires the study's frozen .venv")
    require(sys.version_info[:2] == (3, 12) and np.__version__.startswith("2.5.")
            and torch.__version__.startswith("2.14."), "Require frozen Python 3.12 / NumPy 2.5 / Torch 2.14")
    return {"python": platform.python_version(), "prefix": str(expected), "numpy": np.__version__,
            "torch": torch.__version__}


def metric_stats(value):
    """Retain estimates, never propagate comparison gate/verdict fields."""
    keys = ("point", "paired_difference", "ci95", "se", "rows", "worlds", "weighting", "cluster_unit",
            "bootstrap_seed", "bootstrap_resamples", "bootstrap_valid", "bootstrap_degenerate",
            "new_mean", "old_mean", "denominator", "student_wins", "champion_wins", "student_only",
            "champion_only", "exact_mcnemar_p", "zero_discordance")
    require("point" in value and len(value["ci95"]) == 2, "Missing comparison estimate/CI")
    if value["point"] is not None:
        finite(value["point"])
    lo, hi = value["ci95"]
    require((lo is None and hi is None) or (finite(lo) <= finite(hi)), "Invalid CI")
    return {k: value[k] for k in keys if k in value}


def aggregate_labels(rows, *, resamples=10000):
    """Label-weighted means; bootstrap complete WORLD blocks within each stratum."""
    from sts_rl.fresh_statistics import clustered_mean
    clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        group = row["group"]
        require(group[-1:] in ("1", "2", "3"), "Missing act/room group")
        tags = ["all_rows"]
        if row["choice"]:
            tags.extend(["choice", "act:" + group[-1], "room:" + group[:-1], "group:" + group])
            if row["confident"]:
                tags.append("high_confidence_choice")
        for tag in tags:
            clusters[tag]["agreement"][row["world_seed"]].append(int(row["correct"]))
            clusters[tag]["soft_ce"][row["world_seed"]].append(finite(row["ce"]))
    return {tag: {metric: clustered_mean(worlds, resamples=resamples) for metric, worlds in metrics.items()}
            for tag, metrics in sorted(clusters.items())}


def validate_label_rows(rows, *, receipt, condition, expected_count, aligned=None):
    """Bind every row to the completed score and its checkpoint/split/targets."""
    from sts_rl.fresh_study import INFERENCE
    from sts_rl.fresh_training import world_split
    require(len(rows) == expected_count and len(rows) > 0, "Incomplete TEST row count")
    alignment = {}
    for row in rows:
        world = row["world_seed"]
        require(type(world) is int and 90_000_000_000 <= world < 91_000_000_000
                and world_split(world) == "TEST" and row["split"] == "TEST", "TEST world/split identity mismatch")
        for field, expected in (("data_identity", receipt["data_identity"]), ("inference", INFERENCE),
                                ("selected_checkpoint", receipt["selected_checkpoint"]["sha256"]),
                                ("architecture", receipt["architecture"]), ("phase", receipt["phase"]),
                                ("training_condition", condition)):
            require(row.get(field) == expected, "TEST row binding mismatch: " + field)
        require(type(row["correct"]) is bool and type(row["choice"]) is bool
                and type(row["confident"]) is bool and row.get("has_target") is True
                and row.get("abstention") is False, "Invalid TEST target/status")
        target = row["target"]
        require(target and all(finite(x) >= 0 for x in target) and abs(sum(target) - 1) < 1e-6,
                "Invalid TEST target distribution")
        require(digest(target) == row["target_sha256"] and row["confident"] == (max(target) >= .75)
                and row["choice"] == (len(target) > 1), "TEST target checksum/confidence/menu mismatch")
        finite(row["ce"])
        require(row["ce"] >= -1e-6, "Negative cross entropy")
        identity = row["source_row_id"]
        require(isinstance(identity, str) and len(identity) == 64 and identity not in alignment,
                "Duplicate/invalid TEST row identity")
        require(isinstance(row["source_identity"], str) and len(row["source_identity"]) == 64,
                "Missing scientific source identity")
        alignment[identity] = {k: row[k] for k in ("world_seed", "group", "phase", "public_hash",
            "target_sha256", "choice", "confident", "source_identity", "data_identity", "split")}
    if aligned is not None:
        require(alignment == aligned, "TEST row-ID/target/source cohort differs across combat scores")
    return alignment


def _checkpoint(inputs, identity):
    from sts_policy.training import load_checkpoint
    from sts_rl.fresh_training import checkpoint_training_condition
    inputs.bind(identity["path"], identity)
    payload = load_checkpoint(identity["path"])
    meta = payload["metadata"]
    require(meta.get("initialization") == "from_scratch" and meta.get("parent_checkpoint") is None
            and meta["config"].get("defer_test") is True, "Not a frozen deferred from-scratch checkpoint")
    require(payload["step"] == meta["training_state"]["best_step"], "Checkpoint not validation-selected")
    return checkpoint_training_condition(payload), meta, payload["step"]


def score_data(study, inputs, freeze, runs, *, paired_rows=None):
    from sts_rl.fresh_study import INFERENCE
    output, aligned, common = {}, None, None
    for name in SCORES:
        directory = study / f"score_{name}_v1"
        selection_id = freeze["selections"][name]
        selection = inputs.json(study / f"selections/{name}_v1.json", selection_id)
        require(selection["selection_criterion"] == SELECTION_CRITERION, "Unrecognized frozen selection criterion")
        summary = inputs.json(directory / "summary.json")
        receipt = inputs.json(directory / "receipt.json")
        require(summary["status"] == "completed" and summary.get("test_evaluated") is True
                and summary.get("training") is False and summary.get("weight_updates") == 0
                and receipt.get("training") is False and receipt.get("weight_updates") == 0
                and receipt["stage"] == "deferred_test_scoring" and receipt["device"] == "cpu",
                "Require completed inference-only TEST scoring")
        for value in (summary, receipt):
            require(value["selection"] == selection_id and value["selected_checkpoint"] == selection["selected_checkpoint"],
                    "Scoring differs from pre-TEST selection freeze")
        require(receipt["manifest"] == selection["manifest"] and receipt["selection_criterion"] == SELECTION_CRITERION,
                "Scoring manifest/selection mismatch")
        # Stream hash the 393 MB manifest once; do not parse/load its row inventory.
        inputs.bind(selection["manifest"]["path"], selection["manifest"])
        launch = inputs.json(receipt["launch"]["path"], receipt["launch"])
        effective = launch["effective_config"]
        for field in ("data_identity", "manifest", "selection", "selected_checkpoint", "architecture", "phase"):
            require(effective[field] == receipt[field], "Scoring launch mismatch: " + field)
        condition, metadata, selected_step = _checkpoint(inputs, selection["selected_checkpoint"])
        from sts_rl.fresh_training import FILTERS
        require(receipt["filters"] == metadata["filters"] == FILTERS
                and receipt["vocabulary_sha256"] == metadata["vocabulary_sha256"]
                and receipt["source_receipt"] == metadata["source_receipt"], "Checkpoint/scoring source/filter binding differs")
        data_id = metadata["data_identity"]
        require(data_id == receipt["data_identity"] == summary["data_identity"], "Checkpoint/scoring data identity differs")
        binding = {"data_identity": data_id, "manifest": receipt["manifest"]}
        require(common is None or common == binding, "Scores do not share one frozen dataset")
        common = binding
        architecture, phase, size = name.split("_", 2)
        require(receipt["architecture"] == architecture and receipt["phase"] == phase
                and condition["phase"] == phase and condition["fit_size"] == size,
                "Score architecture/phase/FIT size mismatch")
        if architecture == "old":
            require(condition["old_size"] == "large" and condition["old_context"] == "none",
                    "Old reference must be LARGE/NONE")
            run_name = Path(freeze["old_winner"]["run"]).name.removeprefix("fit_").removesuffix("_v1")
            require(selection["selected_checkpoint"]["sha256"] == freeze["old_winner"]["checkpoint"]["sha256"],
                    "Old control differs from frozen winner")
        else:
            run_name = name
        run = runs[run_name]
        require(run["summary"]["best_step"] == selected_step
                and run["summary"]["selected_checkpoint"]["sha256"] == selection["selected_checkpoint"]["sha256"],
                "FIT summary/checkpoint differs from score selection")
        for key, value in condition.items():
            require(run["config"].get(key, 128 if key == "batch" else None) == value,
                    "Checkpoint/FIT config mismatch: " + key)
        rows = inputs.jsonl(directory / "test_rows.jsonl")
        count = 28676 if phase == "combat" else 44704
        require(summary["test_rows"] == count and summary["test_protocol"] == INFERENCE
                and summary["test_device"] == "cpu", "TEST count/protocol/device mismatch")
        row_alignment = validate_label_rows(rows, receipt=receipt, condition=condition,
            expected_count=count, aligned=aligned if phase == "combat" else None)
        if paired_rows is not None and name in ("new_combat_full", "old_combat_full"):
            paired_rows[name] = rows
        if phase == "combat":
            aligned = row_alignment
        groups = aggregate_labels(rows)
        for tag, stats in summary["test"]["groups"].items():
            require(tag in groups, "Summary group missing from TEST rows")
            for metric in ("agreement", "soft_ce"):
                require(groups[tag][metric]["rows"] == stats["rows"] and
                        math.isclose(groups[tag][metric]["point"], stats[metric], abs_tol=1e-7),
                        "TEST summary/rows mismatch")
        output[name] = {"phase": phase, "architecture": architecture, "fit_size": size,
            "actual_fit_rows": run["summary"]["actual_fit_rows"], "selected_step": selected_step,
            "checkpoint": selection["selected_checkpoint"], "training_condition": condition,
            "test_rows": count, "test_worlds": len({r["world_seed"] for r in rows}),
            "test_row_ids_sha256": digest(sorted(row_alignment)), "groups": groups,
            "inference": INFERENCE, **binding}
    require([output[n]["actual_fit_rows"] for n in SCORES[:3]] == [5000, 20000, 113042],
            "Unexpected combat FIT learning-curve sizes")
    return output


def training_data(study, inputs):
    from sts_rl.fresh_study import INFERENCE
    result = {}
    for name in RUNS:
        directory = study / f"fit_{name}_v1"
        summary, config = (inputs.json(directory / f) for f in ("summary.json", "config.json"))
        events = inputs.jsonl(directory / "metrics.jsonl")
        require(summary["status"] == "completed" and summary["test_deferred"] is True
                and config["defer_test"] is True and config["eval_every"] == 100, "Incomplete/non-deferred FIT run")
        require(events and [e["step"] for e in events] == list(range(100, summary["updates"] + 1, 100)),
                "Supervised optimizer-step log is incomplete or has unexpected cadence")
        inputs.bind(summary["selected_checkpoint"]["path"], summary["selected_checkpoint"])
        series = []
        probe_selection = None
        if name in ("new_combat_full", "new_ooc_full"):
            probe_selection = inputs.json(directory / "probe_selection.json")
            require(probe_selection["selection_identity"] == config["probe_selection"]["selection_identity"],
                    "Probe selection/config mismatch")
            inputs.bind(directory / "probe_selection.json", {k: config["probe_selection"][k] for k in ("path", "bytes", "sha256")})
        for event in events:
            require(event["evaluation_mode"] == "supervised_validation_dropout_disabled"
                    and event["fit_diagnostic"]["groups"]["all_rows"]["rows"] == 128,
                    "FIT diagnostic protocol/subset differs")
            record = {"step": event["step"], "validation": event["validation"],
                      "fit_diagnostic": event["fit_diagnostic"],
                      "parameter_relative_delta": finite(event["parameter_relative_delta"])}
            for scope in ("validation", "fit_diagnostic"):
                for metric in ("agreement", "soft_ce"):
                    finite(record[scope]["groups"]["choice"][metric])
            if probe_selection:
                probe = event["fixed93e9_probes"]
                require(probe["selection_identity"] == probe_selection["selection_identity"]
                        and probe["inference"] == INFERENCE and probe["phase"] == config["phase"]
                        and probe["actor_only"] is True and probe["single_observation"] is True
                        and probe["device"] == "cpu", "Fixed93e9 probe identity/protocol differs")
                for act, meta in probe_selection["by_act"].items():
                    require(probe["by_act"][act]["source_worlds"] == meta["source_worlds"]
                            and probe["by_act"][act]["count"] == meta["selected"], "Probe act coverage differs")
                record["fixed93e9_probes"] = probe
            series.append(record)
        result[name] = {"config": {k: config[k] for k in ("architecture", "phase", "fit_size", "lr", "eval_every", "seed", "steps", "min_steps",
                    "patience", "device", "batch", "defer_test", "old_size", "old_context", "width", "layers", "heads") if k in config},
            "summary": {k: summary[k] for k in ("actual_fit_rows", "best_step", "updates", "selected_checkpoint")},
            "series": series, "probe_selection": None if not probe_selection else {
                k: probe_selection[k] for k in ("selection_identity", "by_act", "selected_count", "selected_worlds")}}
    return result


def aggregate_behaviors(student, champion, seeds, *, resamples=10000):
    """Ratio of summed components, paired WORLD bootstrap; never average world ratios."""
    from sts_rl.fresh_statistics import bootstrap_components, BOOTSTRAP_SEED
    seeds = list(seeds)
    require(len(seeds) == len(set(seeds)) and seeds, "Duplicate/empty requested worlds")
    def index(rows):
        indexed = {r["world_seed"]: r for r in rows}
        require(len(indexed) == len(rows) and set(indexed) == set(seeds), "Missing/duplicate/unexpected behavior worlds")
        return indexed
    new, old = index(student), index(champion)
    output = {}
    for behavior in BEHAVIORS:
        components, totals = {}, {"student": [0., 0., 0.], "champion": [0., 0., 0.]}
        worlds_with_opps = {"student": 0, "champion": 0}
        for seed in seeds:
            pieces = []
            for arm, row in (("student", new[seed]), ("champion", old[seed])):
                b = row["behavior"][behavior]
                chosen, opps, amount = (finite(b[k]) for k in ("chosen", "opportunities", "amount"))
                require(chosen.is_integer() and opps.is_integer() and 0 <= chosen <= opps
                        and amount >= 0 and (opps > 0 or amount == 0), "Invalid behavior components")
                pieces.extend([chosen, opps, amount])
                totals[arm] = [a + v for a, v in zip(totals[arm], (chosen, opps, amount))]
                worlds_with_opps[arm] += int(opps > 0)
            components[seed] = pieces
        def estimate(metric):
            numerator = 2 if metric == "gold_per_opportunity" else 0
            def arm(t, offset):
                return t[offset + numerator] / t[offset + 1] if t[offset + 1] else None
            estimates = {}
            for name, fn in (("student", lambda t: arm(t, 0)), ("champion", lambda t: arm(t, 3)),
                             ("student_minus_champion", lambda t: arm(t, 0) - arm(t, 3)
                              if t[1] and t[4] else None)):
                estimates[name] = {**bootstrap_components(components, fn, resamples=resamples),
                                  "bootstrap_seed": BOOTSTRAP_SEED, "cluster_unit": "paired_source_world"}
            return estimates
        output[behavior] = {"chosen_per_opportunity": estimate("rate"), "denominators": {
            arm: {"chosen": int(t[0]), "opportunities": int(t[1]), "amount_gold": t[2],
                  "worlds_with_opportunities": worlds_with_opps[arm], "cohort_worlds": len(seeds)}
            for arm, t in totals.items()}, "note": CHOICE_NOTE,
            "opportunity_definition": BEHAVIOR_OPPORTUNITIES[behavior]}
        if output[behavior]["chosen_per_opportunity"]["student_minus_champion"].get("bootstrap_degenerate"):
            output[behavior]["uncertainty_note"] = (
                "Degenerate empirical bootstrap cannot quantify unobserved-event uncertainty. "
                + ("Zero observed shop card-removal purchases in both arms; nonzero opportunity denominators. "
                   "This does not establish population absence or zero uncertainty."
                   if behavior == "removal" and all(t[0] == 0 and t[1] > 0 for t in totals.values()) else
                   "Constant observed contrasts do not establish zero population uncertainty."))
        if behavior == "shop_spend":
            output[behavior]["gold_per_opportunity"] = estimate("gold_per_opportunity")
    return output


def crosscheck_game_records(comparison, student, champion, seeds):
    """Check win counts/points and the floor point against the validated cohort."""
    seeds = list(seeds)
    indexed = []
    for rows in (student, champion):
        by_seed = {r["world_seed"]: r for r in rows}
        require(len(seeds) == len(set(seeds)) > 0 and len(by_seed) == len(rows) == len(seeds)
                and set(by_seed) == set(seeds) and all(type(r["win"]) is bool for r in rows),
                "G2 win cross-check requires exact paired cohort")
        indexed.append(by_seed)
    new, old = indexed
    win = comparison["win"]
    counts = {"student_wins": sum(new[s]["win"] for s in seeds),
              "champion_wins": sum(old[s]["win"] for s in seeds),
              "student_only": sum(new[s]["win"] and not old[s]["win"] for s in seeds),
              "champion_only": sum(old[s]["win"] and not new[s]["win"] for s in seeds),
              "denominator": len(seeds), "worlds": len(seeds)}
    require(all(type(win.get(k)) is int and win[k] == v for k, v in counts.items()),
            "G2 comparison win counts/denominator differ from validated cohort records")
    require(math.isclose(finite(win["point"]),
                        (counts["student_wins"] - counts["champion_wins"]) / len(seeds), abs_tol=1e-12),
            "G2 comparison win point differs from validated cohort records")
    floor = comparison["floor"]
    floor_point = math.fsum(finite(new[s]["floor"]) - finite(old[s]["floor"]) for s in seeds) / len(seeds)
    require(type(floor.get("worlds")) is int and floor["worlds"] == len(seeds)
            and math.isclose(finite(floor["point"]), floor_point, rel_tol=0, abs_tol=1e-12),
            "G2 comparison floor point/worlds differ from validated cohort records")


def game_data(study, inputs, scores):
    from sts_rl.fresh_cohort import validate_cohorts
    from sts_rl.fresh_statistics import validate_expectations
    seeds = inputs.json(study / "dev_cohort_4096.json")
    require(seeds == list(range(91_000_000_000, 91_000_000_000 + 4096)), "Require exact frozen 4096 DEV seeds")
    cohorts = {}
    def cohort(name):
        if name not in cohorts:
            directory = study / name
            value = validate_cohorts(directory, world_start=seeds[0], n=len(seeds))
            require(value["config"]["mode"] == "eval", "Behavior source must be greedy full-game eval")
            for path in value["inputs"]:
                inputs.bind(path)
            # Validator verifies every actual world shard, summary, launch and archive.
            cohorts[name] = value
        return cohorts[name]
    champion = cohort("dev_champion_4096_v1_merged")
    results = {}
    for arm in G2_ARMS:
        expectations_path = study / f"expectations/g2_{arm}_v1.json"
        expected = inputs.json(expectations_path)
        validate_expectations(expected, "games")
        for policy in (expected["student_policy"], expected["champion_policy"]):
            for phase in ("champion", "combat", "ooc"):
                if policy[phase] is not None:
                    inputs.bind(policy[phase]["path"], policy[phase])
        for phase in ("combat", "ooc"):
            identity = expected["student_policy"][phase]
            if identity is not None:
                require(identity == scores[f"new_{phase}_full"]["checkpoint"], "G2 replacement differs from frozen scored model")
        student = cohort(f"g2_{arm}_4096_v1_merged")
        require(student["science_identity"]["policy"] == expected["student_policy"]
                and champion["science_identity"]["policy"] == expected["champion_policy"], "Cohort policy differs from frozen expectations")
        require(student["science_identity"]["expected_provenance"] == champion["science_identity"]["expected_provenance"],
                "Behavior arms differ in environment/native provenance")
        comparison = inputs.json(study / f"comparisons/g2_{arm}_v1.json")
        require(comparison["cohort_complete"] is True and comparison["expected_bindings"] == expected
                and comparison["provenance"]["expectations_file"] == inputs.bind(expectations_path)
                and comparison["win"]["denominator"] == 4096, "G2 comparison frozen binding/completeness mismatch")
        for key in ("win", "floor"):
            require(comparison[key]["worlds"] == 4096, "G2 comparison lacks all 4096 worlds")
        crosscheck_game_records(comparison, student["records"], champion["records"], seeds)
        bosses = {}
        for act in ("1", "2", "3"):
            b = comparison["bosses"][act]
            require(b["cohort_denominator"] == 4096, "G2 boss cohort incomplete")
            bosses[act] = {k: metric_stats(b[k]) if isinstance(b[k], dict) else b[k] for k in b}
        results[arm] = {"win": metric_stats(comparison["win"]), "floor": metric_stats(comparison["floor"]),
            "bosses": bosses, "behaviors": aggregate_behaviors(student["records"], champion["records"], seeds),
            "expected_bindings": expected, "source": "completed greedy exact-argmax full-game DEV evaluation",
            "science_identities": {"student": student["science_identity"], "champion": champion["science_identity"]},
            "cohort_worlds": 4096, "world_start": seeds[0], "conditional_note": comparison["conditional_note"]}
    return results


def validate_paired_labels(value, expected, scores, rows, *, resamples=10000):
    """Reproduce the strict paired result from the already validated score rows.

    The comparison format has no row/data hashes, so expectation binding alone
    cannot establish its source. Reuse the existing WORLD statistic, unchanged.
    """
    from sts_rl.fresh_statistics import compare_labels, validate_expectations
    validate_expectations(expected, "labels")
    require(value["expected_bindings"] == expected, "G1 labels frozen expectations differ")
    models = [scores[name] for name in ("new_combat_full", "old_combat_full")]
    for arm, model in zip(("student", "control"), models):
        require(all(expected[arm + "_" + field] == model[key] for field, key in (
            ("checkpoint", "checkpoint"), ("architecture", "architecture"), ("condition", "training_condition"))),
            "G1 labels checkpoint/condition differs from selected score inputs")
    for key in ("data_identity", "manifest", "test_rows", "test_row_ids_sha256"):
        require(models[0][key] == models[1][key], "G1 labels score data/row cohort differs: " + key)
    new, old = (rows[name] for name in ("new_combat_full", "old_combat_full"))
    for model, scored in zip(models, (new, old)):
        require(len(scored) == model["test_rows"]
                and digest(sorted(r["source_row_id"] for r in scored)) == model["test_row_ids_sha256"]
                and all(r["data_identity"] == model["data_identity"] for r in scored),
                "G1 labels rows differ from selected score data identity")
    reproduced = compare_labels(new, old, expectations=expected,
        expected_ids=[r["source_row_id"] for r in new], resamples=resamples)
    require(all(value[k] == reproduced[k] for k in ("metric", "paired_rows", "groups")),
            "G1 labels paired statistics differ from selected score rows")
    return {"metric": value["metric"], "contrast": "new_minus_old", "paired_rows": value["paired_rows"],
        "expected_bindings": expected, "data_identity": models[0]["data_identity"],
        "manifest": models[0]["manifest"], "test_row_ids_sha256": models[0]["test_row_ids_sha256"],
        "groups": {tag: metric_stats(stat) for tag, stat in value["groups"].items()}}


def paired_label_data(study, inputs, scores, rows):
    expected_path = study / "expectations/g1_labels_full_v1.json"
    expected = inputs.json(expected_path)
    value = inputs.json(study / "comparisons/g1_labels_full_v1.json")
    require(value["provenance"]["expectations_file"] == inputs.bind(expected_path),
            "G1 labels frozen expectations file differs")
    return validate_paired_labels(value, expected, scores, rows)


def panel_data(study, inputs, scores):
    from sts_rl.fresh_panel_grid import HP_DEFINITION
    value = inputs.json(study / "comparisons/g1_panel_new_minus_old_v1.json")
    expected_path = study / "expectations/g1_panel_full_v1.json"
    expected = inputs.json(expected_path)
    require(value["schema"] == "fresh_panel_student_contrast_v1"
            and all(value.get(k) is True for k in ("confidence_claims_allowed", "filesystem_validated", "control_exact_match", "grid_complete"))
            and value["contrast"] == "new_replacement_minus_old_replacement"
            and value["hp_definition"] == HP_DEFINITION and value["expected_bindings"] == expected
            and value["input_file_identities"]["expectations"] == inputs.bind(expected_path),
            "G1 requires strict, complete frozen direct new-minus-old comparison")
    for arm in ("new", "old"):
        require(expected[arm + "_policy"]["combat"] == scores[arm + "_combat_full"]["checkpoint"],
                "G1 policy differs from frozen score selection")
    require(value["groups"]["BOSS2:resampled"]["role"] == "primary", "Missing primary BOSS2 direct contrast")
    return {"contrast": value["contrast"], "grid_hash": value["grid_hash"], "hp_definition": value["hp_definition"],
        "cluster_unit": value["cluster_unit"], "weighting": value["weighting"],
        "groups": {tag: {"role": group["role"], "few_worlds_descriptive": group["few_worlds_descriptive"],
                   **{m: metric_stats(group[m]) for m in ("win", "hp_lost")}}
                   for tag, group in value["groups"].items()}}


def export(study):
    runtime = frozen_runtime(study)
    readiness = check_ready(study)
    require(readiness["inventory_ready"], "Required inputs absent:\n" + "\n".join(readiness["missing"]))
    inputs = Inputs()
    freeze = inputs.json(study / "all_warmup_selections_v1.json")
    require(freeze["all_before_TEST"] is True and set(freeze["selections"]) == set(SCORES), "Missing global pre-TEST selection freeze")
    runs = training_data(study, inputs)
    paired_rows = {}
    scores = score_data(study, inputs, freeze, runs, paired_rows=paired_rows)
    labels = paired_label_data(study, inputs, scores, paired_rows)
    del paired_rows
    teacher = inputs.json(study / "teacher_consistency_final_v1.json")
    require(teacher["no_student_selection_or_training_from_repeats"] is True, "Teacher repeats not independent of student training/selection")
    references = {"overall": teacher["overall"],
                  "high_confidence": teacher["high_confidence_primary_max_vote_share_ge_.75"],
                  "role": "REFERENCE: independent teacher repeat agreement; not a strict ceiling"}
    games = game_data(study, inputs, scores)
    panel = panel_data(study, inputs, scores)
    inputs.bind(Path(__file__))
    import sts_rl.fresh_statistics as statistics_module
    inputs.bind(statistics_module.__file__)
    inputs.recheck()
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "study": str(study),
        "protocol": "fresh_start_v1", "runtime": runtime, "statistics": {
            "method": "percentile 95% whole-WORLD bootstrap", "resamples": 10000, "seed": 1919,
            "labels": "label-weighted clustered_mean; complete TEST rows, never rowwise resampling",
            "behaviors": "ratio of summed chosen/opportunities (gold amount/opportunities); same WORLD draws per arm",
            "parameter_relative_delta": "L2 norm of actor parameter displacement since initialization / initial actor parameter L2 norm",
            "critic_explained_variance": "N/A: supervised actor warm-up", "ppo_ratio_clip": "N/A: no PPO updates"},
        "notes": ["All selected checkpoints frozen before TEST. TEST does not tune any setting.",
                  "FIT diagnostic uses fixed 128-row subset, not all training rows; VALIDATION is separate.",
                  "Probe cadence: 100 supervised optimizer steps per block, not PPO updates; per-act source-world coverage is exported.",
                  "Full-game source is greedy evaluation; no stochastic training game win-rate claim.", CHOICE_NOTE,
                  "Only G2 wins have a predeclared 1 percentage-point NI margin. No margin for floor, HP, boss rates or G1.",
                  "Intervals are supplied directly or computed in frozen runtime, never subtracted from arm CIs.",
                  "Constant observed differences can yield degenerate bootstrap CIs; undefined conditional draws are counted/omitted."],
        "scores": scores, "training": runs, "teacher_reference": references, "g2": games, "g1": panel,
        "g1_labels": labels,
        "selection_freeze": freeze, "input_files": sorted(inputs.identities.values(), key=lambda x: x["path"])}


def fresh_output(path, study):
    path = Path(path).resolve()
    require(not path.is_relative_to(study), "Outputs must be outside the read-only study")
    require(not path.exists(), "--out must be a fresh directory")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--out", type=Path, help="Fresh directory outside read-only study; contains plot_data.json")
    parser.add_argument("--check-ready", action="store_true", help="Presence-only inventory; no TEST rows or statistics")
    args = parser.parse_args(argv)
    study = args.study.resolve()
    try:
        if args.check_ready:
            readiness = check_ready(study)
            print(json.dumps(readiness, indent=2))
            return 0 if readiness["inventory_ready"] else 2
        require(args.out is not None, "--out is required for export")
        out = fresh_output(args.out, study)
        value = export(study)
        # No output directory or partial dataset until all required inputs validate.
        out.mkdir(parents=True, exist_ok=False)
        with (out / "plot_data.json").open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"plot_data": str(out / "plot_data.json"), "schema": SCHEMA}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"fresh_gate_report_data: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
