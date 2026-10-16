"""Strict G1 new-minus-old combat-student contrast on a frozen C6 panel.

No inference, training, native imports, or gate verdicts. Filesystem validation
precedes statistics; the in-memory metrics entry point is explicitly unvalidated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from pickle import UnpicklingError
import statistics
import tempfile

from sts_policy.curriculum import public_hash
from sts_rl.fresh_panel_grid import (HP_DEFINITION, SAMPLING_SPEC, SELECTION_PROTOCOL,
                                     canonical_grid_hash)
from sts_rl.fresh_statistics import BOOTSTRAPS, BOOTSTRAP_SEED, clustered_mean, _file_binding
from sts_rl.fresh_study import (DEAD_FILTER_ID, FILTER_ID, INFERENCE, PROTOCOL,
                                PUBLIC_CONTEXT_ID)
from sts_rl.recording import file_identity

EXPECTATIONS_SCHEMA = "fresh_panel_student_contrast_expectations_v1"
INVALID = ("fault", "cut", "ub", "censored", "programming_error", "programming_errors",
           "truncated", "undefined_behavior_evoked", "grid_validation_error")
CONTROL_FIELDS = ("win", "hp_after", "hp_lost", "entry_hp", "source_entry_hp", "decisions",
                  "first_combat_hash", "cut", "ub")
GRID_FIELDS = ("grid_hash", "selection_protocol", "targets", "resamples", "coverage",
               "coverage_sufficient", "coverage_shortfall", "eligible_worlds", "absent_groups")
COMPLETION_FIELDS = ("status", "cohort_complete", "confidence_claims_allowed", "diagnostic_only",
                     "grid_complete", "faults", "cuts", "censored", "unstarted", "stop_reasons",
                     "valid", "results_complete", "missing_results", "invalid_results",
                     "expected_results", "received", "grid_input", *GRID_FIELDS, "hp_definition",
                     "run_control")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _same(a, b):
    # JSON equality must distinguish true from 1 (Python dictionary equality does not).
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class _Inputs:
    """Check identities before/after reads and again before returning a report."""
    def __init__(self):
        self.files = {}
        self.directories = {}

    def verify(self, identity):
        _file_binding(identity)
        path = identity["path"]
        previous = self.files.get(path)
        _require(previous is None or _same(previous, identity), f"Conflicting file identity: {path}")
        if previous is None:
            _require(_same(file_identity(path), identity), f"File identity mismatch: {path}")
            self.files[path] = identity
        return identity

    def read(self, path, expected=None):
        identity = file_identity(path)
        if expected is not None:
            _require(_same(identity, expected), f"File identity mismatch: {path}")
        self.verify(identity)
        value = json.loads(Path(path).read_text(), object_pairs_hook=_json_object,
                           parse_constant=_nonfinite)
        _require(_same(file_identity(path), identity), f"Input changed while reading: {path}")
        _require(isinstance(value, dict), f"Require JSON object: {path}")
        return value, identity

    def directory(self, path, expected):
        path = Path(path)
        self.directories[path] = set(expected)
        _require(set(path.rglob("*")) == self.directories[path], f"Missing/unexpected result or quarantine files: {path}")

    def recheck(self):
        for identity in self.files.values():
            _require(_same(file_identity(identity["path"]), identity),
                     f"Input changed during validation: {identity['path']}")
        for path, expected in self.directories.items():
            _require(set(path.rglob("*")) == expected, f"Result/quarantine directory changed during validation: {path}")


def _policy(policy):
    _require(isinstance(policy, dict) and set(policy) == {
        "protocol", "filter_ids", "public_context_id", "inference", "champion", "combat", "ooc"},
        "Require full frozen StudyPolicy.identity")
    _require(policy["protocol"] == PROTOCOL and policy["filter_ids"] == [FILTER_ID, DEAD_FILTER_ID]
             and policy["public_context_id"] == PUBLIC_CONTEXT_ID and policy["inference"] == INFERENCE,
             "Frozen policy protocol/filter/context/inference mismatch")
    _file_binding(policy["champion"])
    _file_binding(policy["combat"])
    _require(policy["ooc"] is None, "Combat contrast requires ooc=None")
    _require(policy["combat"]["sha256"] != policy["champion"]["sha256"],
             "Combat student must differ from champion")


def _expectations(value, grid_id, inputs):
    _require(value.get("schema") == EXPECTATIONS_SCHEMA and type(value.get("version")) is int
             and value["version"] == 1, "Require frozen contrast expectations schema/version 1")
    _require(_same(value.get("grid"), grid_id), "Frozen expectations grid file identity mismatch")
    for arm in ("new", "old"):
        policy = value.get(arm + "_policy")
        _policy(policy)
        for phase in ("champion", "combat"):
            inputs.verify(policy[phase])
    new, old = value["new_policy"], value["old_policy"]
    _require(all(_same(new[k], old[k]) for k in new if k != "combat"),
             "New/old policies must share champion/filter/context/inference")
    _require(new["combat"]["sha256"] != old["combat"]["sha256"],
             "New/old students require distinct checkpoints")
    conditions, data = {}, []
    for arm in ("new", "old"):
        conditions[arm], data_id = _checkpoint(value[arm + "_policy"]["combat"], arm, inputs)
        _require(_same(conditions[arm], value.get(arm + "_condition")),
                 f"Frozen {arm}_condition differs from actual checkpoint training config")
        data.append(data_id)
    _require(data[0] == data[1], "New/old checkpoint raw data_identity differs")
    _require(conditions["new"]["fit_size"] == conditions["old"]["fit_size"],
             "New/old checkpoint fit_size differs")
    return conditions, data[0]


def _checkpoint(identity, architecture, inputs):
    import torch
    from sts_rl.fresh_training import checkpoint_training_condition

    inputs.verify(identity)
    try:
        payload = torch.load(identity["path"], map_location="cpu", weights_only=False)
        meta = payload["metadata"]
        config = meta["config"]
        _require(config["architecture"] == architecture and config["phase"] == "combat",
                 f"Expected {architecture} architecture combat checkpoint (new/old roles cannot reverse)")
        _require(meta.get("initialization") == "from_scratch" and "parent_checkpoint" in meta
                 and meta["parent_checkpoint"] is None, "Checkpoint must be from_scratch with no parent")
        _require(_hash(meta.get("data_identity")), "Checkpoint requires raw data_identity")
        if architecture == "new":
            from sts_policy.entity_model import policy_from_payload
            model = policy_from_payload(payload)
            _require(meta.get("phase", model.config.phase) == "combat", "Checkpoint metadata phase mismatch")
        else:
            from sts_policy.model import CandidateActorCritic
            from sts_policy.model_agent import policy_from_payload
            _require(meta.get("kind") in (None, "candidate") and meta.get("phase") == "combat",
                     "Old checkpoint metadata architecture/phase mismatch")
            model = policy_from_payload(payload)
            _require(isinstance(model, CandidateActorCritic), "Old checkpoint must be a single candidate head")
        _require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()),
                 "Nonfinite checkpoint weights")
        condition = checkpoint_training_condition(payload)
        _require(condition["fit_size"] in ("5000", "20000", "full"), "Invalid checkpoint fit_size")
        _require(_same(file_identity(identity["path"]), identity), "Checkpoint changed during load")
        return condition, meta["data_identity"]
    except (KeyError, TypeError, RuntimeError, EOFError, UnpicklingError, UnicodeError, IndexError) as exc:
        raise ValueError(f"Invalid {architecture} checkpoint: {exc}") from exc


def _grid(frozen, inputs):
    _require(frozen.get("grid_hash") == canonical_grid_hash(frozen), "Frozen grid canonical hash mismatch")
    _require(frozen.get("protocol") == PROTOCOL and frozen.get("selection_protocol") == SELECTION_PROTOCOL
             and _same(frozen.get("sampling_spec"), SAMPLING_SPEC), "Frozen grid protocol mismatch")
    _require(type(frozen.get("resamples")) is int and frozen["resamples"] == 8
             and _same(frozen.get("salts"), [-1, *range(8)]), "Require unsalted -1 and all eight salts 0..7")
    targets = frozen["targets"]
    _require(isinstance(targets, dict) and "BOSS2" in targets and all(
        isinstance(g, str) and g != "all" and _integer(n, 1) for g, n in targets.items()),
        "Invalid frozen WORLD targets")
    _require(frozen.get("coverage_sufficient") is True and frozen.get("coverage_shortfall") == {}
             and _same(frozen.get("coverage"), targets), "Frozen grid coverage shortfall")
    bank = frozen["bank_identity"]
    _require(isinstance(bank, dict) and _integer(bank["world_start"]) and _integer(bank["n"], 1)
             and _hash(bank["science_identity_hash"]), "Missing frozen bank identity")
    provenance = bank["expected_provenance"]
    _require(provenance["environment_version"] == "0.1.5"
             and provenance["observation_version"] == "sts_env_public_v3", "Frozen bank environment mismatch")
    for kind in ("inputs", "shards"):
        _require(isinstance(bank[kind], list) and bool(bank[kind]), f"Missing bank {kind} identities")
        for identity in bank[kind]:
            inputs.verify(identity)
    starts, world_groups, counts = {}, set(), Counter()
    for row in frozen["starts"]:
        key = row["world_seed"], row["start_id"]
        group = row["group"]
        _require(_integer(key[0]) and 92_000_000_000 <= key[0] < 93_000_000_000
                 and bank["world_start"] <= key[0] < bank["world_start"] + bank["n"]
                 and _integer(key[1]) and group in targets, "Invalid frozen start identity")
        _require(key not in starts and (key[0], group) not in world_groups,
                 "Duplicate frozen start or world/group")
        _require(all(_hash(row[k]) for k in ("public_hash", "first_combat_hash", "replay_hash"))
                 and row["provenance_hash"] == public_hash(provenance), "Invalid frozen start hashes/provenance")
        starts[key] = row
        world_groups.add((key[0], group))
        counts[group] += 1
    _require(dict(counts) == targets and frozen["worlds"] == sorted({w for w, _ in starts}),
             "Frozen starts/world coverage differs from targets")
    eligible = frozen["eligible_worlds"]
    _require(all(_integer(n) for n in eligible.values()) and all(eligible[g] >= n for g, n in targets.items())
             and frozen["absent_groups"] == sorted(g for g, n in eligible.items() if not n),
             "Frozen grid eligible-world metadata mismatch")
    expected = [{"world_seed": w, "start_id": s, "group": starts[w, s]["group"], "salt": salt}
                for w, s in sorted(starts) for salt in frozen["salts"]]
    _require(_same(frozen["grid"], expected), "Frozen grid missing/unexpected/duplicate start-salt cells")
    return starts


def _complete(value, frozen, grid_id):
    n = 2 * len(frozen["grid"])
    _require(value["protocol"] == PROTOCOL and value["event"] == "done" and value["status"] == "completed",
             "Require completed C6 panel")
    _require(all(value[k] is True for k in ("cohort_complete", "grid_complete", "results_complete",
                                           "confidence_claims_allowed")) and value["diagnostic_only"] is False,
             "Panel is incomplete or diagnostic-only")
    _require(all(type(value[k]) is int and value[k] == 0 for k in (
        "faults", "cuts", "censored", "unstarted", "missing_results", "invalid_results"))
        and all(not value.get(k) for k in INVALID) and value["stop_reasons"] == [],
        "Panel contains faults/cuts/UB/censor/programming errors or missing results")
    _require(all(type(value[k]) is int and value[k] == n for k in ("valid", "expected_results", "received")),
             "Panel result counts differ from exact grid coverage")
    _require(all(_same(value[k], frozen[k]) for k in GRID_FIELDS)
             and _same(value["hp_definition"], HP_DEFINITION) and _same(value["grid_input"], grid_id),
             "Panel grid/targets/resamples/HP definition/grid_input binding mismatch")


def _row_key(row):
    _require(_integer(row["world_seed"]) and _integer(row["start_id"]) and type(row["salt"]) is int,
             "Invalid result cell identity")
    return row["world_seed"], row["start_id"], row["salt"], row["policy"]


def _index(rows):
    indexed = {}
    for row in rows:
        key = _row_key(row)
        _require(key not in indexed, "Duplicate result cell")
        indexed[key] = row
    return indexed


def _valid_row(row, cell, start, grid_hash):
    _require(not any(row.get(k) for k in INVALID), "Result contains fault/cut/UB/censor/programming error")
    _require(row.get("cut") is False and row.get("ub") is False and type(row.get("win")) is bool,
             "Result lacks valid terminal outcome/cut/ub")
    _require(row["group"] == cell["group"] and row["grid_hash"] == grid_hash,
             "Result group/grid_hash mismatch")
    _require(_hash(row["first_combat_hash"]) and (row["salt"] != -1 or
             row["first_combat_hash"] == start["first_combat_hash"]), "Result first_combat_hash mismatch")
    # C6 player_hp() returns float, while hp_after/decisions are native integers.
    for field in ("entry_hp", "source_entry_hp"):
        _require(_number(row[field]) and row[field] >= 0, f"Invalid result {field}")
    for field in ("hp_after", "decisions"):
        _require(_integer(row[field]), f"Invalid result {field}")
    _require(_number(row["hp_lost"]) and row["hp_lost"] == row["entry_hp"] - row["hp_after"]
             and (row["win"] or row["hp_after"] == 0), "Invalid pre-healing HP/death-zero accounting")


def _run(directory, policy, frozen, grid_id, starts, inputs):
    directory = Path(directory).resolve()
    _require(directory.is_dir(), "Require completed C5/C6 run directories")
    receipt, receipt_id = inputs.read(directory / "receipt.json")
    summary, summary_id = inputs.read(directory / "summary.json")
    completion, completion_id = inputs.read(directory / "completion.json")
    run_grid, run_grid_id = inputs.read(directory / "grid.json")
    _require(_same(run_grid, frozen), "Run grid.json content differs from supplied frozen grid")
    _require(_same(completion["summary"], summary_id), "Completion summary file identity mismatch")
    _require(_same(summary["source"], receipt) and _same(completion["source"], receipt),
             "Summary/completion source differs from actual run receipt")
    for value in (summary, completion):
        _complete(value, frozen, grid_id)
    _require(all(_same(summary[k], completion[k]) for k in COMPLETION_FIELDS),
             "Summary/completion fields differ")
    _require(type(summary["ub"]) is int and summary["ub"] == 0, "Summary contains UB")
    _require(receipt["protocol"] == PROTOCOL and _same(receipt["policy"], policy),
             "Run receipt policy differs from frozen new/old expectations")
    config = receipt["config"]
    _require(Path(config["out"]).resolve() == directory and Path(config["grid"]).resolve() == Path(grid_id["path"])
             and _same(config["grid_input"], grid_id), "Receipt config.grid_input must bind supplied frozen file")
    _require(Path(config["checkpoint"]).resolve() == Path(policy["champion"]["path"])
             and Path(config["combat"]).resolve() == Path(policy["combat"]["path"])
             and config.get("ooc") is None, "Receipt config checkpoint/combat/ooc mismatch")
    _require(all(_same(config[k], frozen[k]) for k in GRID_FIELDS)
             and _same(config["starts_per_group"], frozen["targets"])
             and _same(config["bank_identity"], frozen["bank_identity"])
             and _same(config["hp_definition"], HP_DEFINITION), "Receipt config frozen grid/bank/HP mismatch")
    launch_id = inputs.verify(receipt["launch"])
    launch_path = Path(launch_id["path"])
    _require(launch_path.name == "launch.json" and directory / "records" in launch_path.parents,
             "Receipt launch must be an actual record in this run directory")
    launch, _ = inputs.read(launch_id["path"], launch_id)
    _require(launch["schema"] == "sts_training_launch_v1" and _same(launch["effective_config"], config)
             and _same(launch["parent_checkpoint"], policy["champion"]), "Launch config/champion binding mismatch")
    expected_inputs = [*frozen["bank_identity"]["inputs"], *frozen["bank_identity"]["shards"],
                       grid_id, policy["combat"]]
    as_keys = lambda items: Counter(json.dumps(i, sort_keys=True) for i in items)
    _require(as_keys(launch["data_inputs"]) == as_keys(expected_inputs), "Launch data_inputs grid/bank/combat mismatch")
    for identity in launch["data_inputs"]:
        inputs.verify(identity)
    archive_id = inputs.verify(launch["source_archive"])
    _require(Path(archive_id["path"]) == launch_path.parent / "source.zip",
             "Launch source archive binding mismatch")
    native, runtime = launch["native"], launch["runtime"]
    _require(_same(native, frozen["bank_identity"]["expected_provenance"]["native_provenance"])
             and _hash(native["module_sha256"]) and Path(native["module_file"]).name == native["module_file"],
             "Launch native identity differs from frozen bank")
    snapshot = Path(launch_id["path"]).parent
    native_id = inputs.verify(file_identity(snapshot / native["module_file"]))
    manifest, _ = inputs.read(snapshot / "build_manifest.json")
    _require(native_id["sha256"] == native["module_sha256"] and _same(manifest, native),
             "Recorded native binary/build manifest identity mismatch")
    _require(isinstance(runtime, dict) and all(runtime.get(k) for k in ("python", "platform", "packages"))
             and all(runtime["packages"].get(k) for k in ("torch", "numpy")), "Missing launch runtime identity")
    _require(not any((directory / "quarantine").rglob("*")), "Completed panel contains quarantine attempts")
    inputs.directory(directory / "quarantine", set())
    rows = summary["records"]
    indexed = _index(rows)
    expected = {_row_key({**r, "policy": p}): r for r in frozen["grid"] for p in ("champion", "replacement")}
    _require(set(indexed) == set(expected), "Missing/unexpected result cells; intersection pairing forbidden")
    paths = set()
    for key, cell in expected.items():
        row = indexed[key]
        _valid_row(row, cell, starts[key[:2]], frozen["grid_hash"])
        # Every row resolves to this run's actual receipt, not an embedded claim.
        _require(_same(row["source_ref"], receipt_id), "Result source_ref differs from actual immutable run receipt")
        inputs.verify(row["source_ref"])
        path = directory / "results" / f"{key[0]}-{key[1]}-{key[2]}-{key[3]}.json"
        actual, _ = inputs.read(path)
        _require(_same(actual, row), "Result file differs from bound summary record")
        paths.add(path)
    _require(set((directory / "results").iterdir()) == paths, "Missing/unexpected/duplicate result files")
    inputs.directory(directory / "results", paths)
    return {"rows": indexed, "native": native, "runtime": runtime,
            "environment": launch["environment"],
            "identities": {"receipt": receipt_id, "summary": summary_id, "completion": completion_id,
                           "grid": run_grid_id, "launch": launch_id, "source_archive": archive_id}}


def contrast_metrics(new, old, frozen, *, resamples=BOOTSTRAPS, full_fit=True):
    """In-memory fixture/diagnostic statistics; never certifies filesystem provenance.

    Inputs are replacement rows only. No intersections or fault exclusions. All
    starts and salts within a source world form one resampled block, also pooled.
    """
    new, old = _index(new), _index(old)
    cells = {(r["world_seed"], r["start_id"], r["salt"], "replacement"): r for r in frozen["grid"]}
    expected = set(cells)
    _require(set(new) == set(old) == expected and bool(expected), "Missing/unexpected replacement pairs")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key in sorted(expected):
        a, b = new[key], old[key]
        cell = cells[key]
        # Structural cell validation here is intentionally independent of receipts.
        for row in (a, b):
            _require(not any(row.get(k) for k in INVALID) and type(row.get("win")) is bool,
                     "Invalid replacement pair outcome")
            _require(row["group"] == cell["group"] and row["first_combat_hash"] == a["first_combat_hash"]
                     and row["entry_hp"] == a["entry_hp"], "Replacement pair initial state/entry HP mismatch")
        kind = ":unsalted" if key[2] == -1 else ":resampled"
        for tag in ("all" + kind, a["group"] + kind):
            for metric in ("win", "hp_lost"):
                av, bv = float(a[metric]), float(b[metric])
                _require(math.isfinite(av) and math.isfinite(bv), "Nonfinite contrast metric")
                for name, value in ((metric, av - bv), (metric + "_new", av), (metric + "_old", bv)):
                    groups[tag][name][key[0]].append(value)
    result = {}
    for tag, clusters in sorted(groups.items()):
        primary = tag == "BOSS2:resampled" and full_fit
        metrics = {}
        for metric in ("win", "hp_lost"):
            stats = clustered_mean(clusters[metric], equal_world=True, resamples=resamples, seed=BOOTSTRAP_SEED)
            metrics[metric] = {**stats, "paired_difference": stats["point"],
                "new_mean": statistics.mean(statistics.mean(v) for v in clusters[metric + "_new"].values()),
                "old_mean": statistics.mean(statistics.mean(v) for v in clusters[metric + "_old"].values())}
        result[tag] = {"role": "primary" if primary else "descriptive", "descriptive_only": not primary,
                       "few_worlds_descriptive": tag.startswith("EVENT2:"), **metrics}
    return {"groups": result, "confidence_claims_allowed": False, "filesystem_validated": False,
            "contrast": "new_replacement_minus_old_replacement", "weighting": "equal_world",
            "cluster_unit": "source_world_all_salts_and_starts", "hp_definition": dict(HP_DEFINITION),
            "hp_lost_direction": "negative new-minus-old HP loss favors new student",
            "bootstrap_note": "Source worlds resampled as blocks; salt -1 separate from 0..7. "
                              "Constant observed differences give degenerate intervals, which cannot quantify unobserved discordance.",
            "mde_note": "2.8 * paired bootstrap SE; normal approximation, 80% power at two-sided 5%.",
            "interpretation": "BOSS2 resampled win and HP loss are primary for full fits; other groups and pooled results descriptive. No gate verdict or margin."}


def compare_panel_runs(new, old, grid, expectations, *, resamples=BOOTSTRAPS):
    """Validate two completed directories and frozen bindings before any CI."""
    inputs = _Inputs()
    try:
        _require(Path(new).resolve() != Path(old).resolve(), "Require two independent panel run directories")
        frozen, grid_id = inputs.read(grid)
        expected, expected_id = inputs.read(expectations)
        starts = _grid(frozen, inputs)
        conditions, data_id = _expectations(expected, grid_id, inputs)
        runs = {arm: _run(directory, expected[arm + "_policy"], frozen, grid_id, starts, inputs)
                for arm, directory in (("new", new), ("old", old))}
        for field in ("native", "runtime", "environment"):
            _require(_same(runs["new"][field], runs["old"][field]), f"New/old {field} identity differs")
        for cell in frozen["grid"]:
            key = cell["world_seed"], cell["start_id"], cell["salt"]
            controls = [runs[arm]["rows"][(*key, "champion")] for arm in ("new", "old")]
            _require(all(_same(controls[0][f], controls[1][f]) for f in CONTROL_FIELDS),
                     f"Champion control exact-match failure at {key}")
            rows = [runs[arm]["rows"][(*key, p)] for arm in ("new", "old") for p in ("champion", "replacement")]
            _require(all(all(_same(row[f], rows[0][f]) for f in (
                "first_combat_hash", "entry_hp", "source_entry_hp")) for row in rows),
                f"Paired first_combat_hash/entry HP mismatch at {key}")
        inputs.recheck()
        report = contrast_metrics(*[[r for k, r in runs[arm]["rows"].items() if k[3] == "replacement"]
                                   for arm in ("new", "old")], frozen, resamples=resamples,
                                  full_fit=conditions["new"]["fit_size"] == "full")
        inputs.recheck()
        return {**report, "schema": "fresh_panel_student_contrast_v1", "version": 1, "protocol": PROTOCOL,
                "confidence_claims_allowed": True, "filesystem_validated": True, "control_exact_match": True,
                "grid_complete": True, "grid_hash": frozen["grid_hash"], "targets": frozen["targets"],
                "resamples": frozen["resamples"], "bank_identity": frozen["bank_identity"],
                "checkpoint_conditions": conditions, "data_identity": data_id, "expected_bindings": expected,
                "input_file_identities": {"grid": grid_id, "expectations": expected_id,
                    **{arm: runs[arm]["identities"] for arm in runs}},
                "verified_files": sorted(inputs.files.values(), key=lambda i: i["path"]),
                "provenance": {k: runs["new"][k] for k in ("native", "runtime", "environment")}}
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing/invalid panel comparison input: {exc}") from exc


def write_comparison(path, report):
    """Atomic exclusive publication; neither races nor existing files overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".contrast-", delete=False) as stream:
            name = stream.name
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
