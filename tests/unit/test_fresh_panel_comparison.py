"""Synthetic C6 artifacts and real tiny checkpoints. No simulator or fitting."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sts_policy.curriculum import public_hash
from sts_policy.entity_model import EntityActorCritic
from sts_policy.model import CandidateActorCritic
from sts_policy.training import save_checkpoint
from sts_rl.fresh_panel_comparison import (EXPECTATIONS_SCHEMA, compare_panel_runs,
    contrast_metrics, write_comparison)
from sts_rl.fresh_panel_grid import HP_DEFINITION, canonical_grid_hash, freeze_grid
from sts_rl.fresh_study import (DEAD_FILTER_ID, FILTER_ID, INFERENCE, PROTOCOL, PUBLIC_CONTEXT_ID)
from sts_rl.recording import file_identity

WORLD = 92_000_010_000
ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def row_path(directory, row):
    return directory / "results" / (f"{row['world_seed']}-{row['start_id']}-{row['salt']}-{row['policy']}.json")


class PanelFixture:
    """Writes the actual C6 prepare_run/_finish_panel fields, independently."""
    def __init__(self, root):
        self.root = Path(root)
        self.new, self.old = self.root / "new", self.root / "old"
        self.grid = self.root / "frozen-grid.json"
        self.expectations = self.root / "expectations.json"
        binary = self.root / "native-fixture"
        binary.write_bytes(b"immutable binary fixture; never loaded")
        self.native = {"module_file": "slaythespire.fixture.so", "module_sha256": file_identity(binary)["sha256"],
                       "backend_api": 1, "variant": "sts_rl_panel"}
        self.provenance = {"environment_version": "0.1.5", "observation_version": "sts_env_public_v3",
                           "native_provenance": self.native}
        self.runtime = {"python": "fixture python", "platform": "fixture platform",
                        "packages": {"torch": "fixture torch", "numpy": "fixture numpy"}}
        paths = {}
        self.conditions = {}
        for arm in ("champion", "new", "old"):
            path = self.root / (arm + ".pt")
            arch = "new" if arm == "new" else "old"
            model = (EntityActorCritic(width=8, layers=1, heads=2, phase="combat") if arch == "new"
                     else CandidateActorCritic(size="small", context="none"))
            config = {"architecture": arch, "phase": "combat", "fit_size": "full", "lr": .001,
                      "seed": 7, "steps": 12, "min_steps": 4, "eval_every": 2, "patience": 4,
                      "device": "cpu", "batch": 8, "defer_test": True}
            condition = {k: v for k, v in config.items() if k != "architecture"}
            dimensions = {"width": 8, "layers": 1, "heads": 2} if arch == "new" else {
                "old_size": "small", "old_context": "none"}
            config.update(dimensions)
            condition.update(dimensions)
            metadata = (model.metadata() if arch == "new" else {"encoder_version": "public_candidate_v2",
                                                                     "phase": "combat"})
            metadata.update(config=config, data_identity="d" * 64, initialization="from_scratch", parent_checkpoint=None)
            save_checkpoint(path, model, model_config=model.metadata(), metadata=metadata)
            paths[arm] = file_identity(path)
            self.conditions[arm] = condition
        self.policies = {arm: {"protocol": PROTOCOL, "filter_ids": [FILTER_ID, DEAD_FILTER_ID],
            "public_context_id": PUBLIC_CONTEXT_ID, "inference": INFERENCE,
            "champion": paths["champion"], "combat": paths[arm], "ooc": None} for arm in ("new", "old")}
        bank_input, bank_shard = self.root / "bank-receipt.json", self.root / "bank-shard.json"
        write(bank_input, {"fixture": "frozen bank receipt"})
        write(bank_shard, {"fixture": "frozen bank shard"})
        bank = {"world_start": WORLD, "n": 3, "science_identity_hash": "a" * 64,
                "expected_provenance": self.provenance, "inputs": [file_identity(bank_input)],
                "shards": [file_identity(bank_shard)]}
        starts = []
        # World 0 contributes two groups/starts; pooled clusters are unequal.
        for w, s, group in ((0, 10, "BOSS2"), (1, 11, "BOSS2"), (2, 12, "BOSS2"), (0, 20, "EVENT2")):
            starts.append({"world_seed": WORLD + w, "start_id": s, "group": group,
                "public_hash": public_hash([w, s, "ooc"]), "first_combat_hash": public_hash([w, s, -1]),
                "prefix": list(range(s)), "task": {"seed": WORLD + w}, "entering_action": 0,
                "entering_description": "enter", "expected_provenance": self.provenance})
        _, self.frozen = freeze_grid(starts, {"BOSS2": 3, "EVENT2": 1}, 8, bank_identity=bank)
        write(self.grid, self.frozen)
        self.expected = {"schema": EXPECTATIONS_SCHEMA, "version": 1, "grid": file_identity(self.grid),
                         **{arm + "_policy": self.policies[arm] for arm in ("new", "old")},
                         **{arm + "_condition": self.conditions[arm] for arm in ("new", "old")}}
        write(self.expectations, self.expected)
        for arm in ("new", "old"):
            self.make_run(arm, binary)

    def make_run(self, arm, binary):
        directory = getattr(self, arm)
        policy = self.policies[arm]
        grid_fields = {k: self.frozen[k] for k in ("grid_hash", "selection_protocol", "targets", "resamples",
            "coverage", "coverage_sufficient", "coverage_shortfall", "eligible_worlds", "absent_groups")}
        config = {"out": str(directory), "grid": str(self.grid), "grid_input": file_identity(self.grid),
            "checkpoint": policy["champion"]["path"], "combat": policy["combat"]["path"],
            "bank": [str(self.root)], "starts_per_group": self.frozen["targets"],
            "bank_identity": self.frozen["bank_identity"], "hp_definition": HP_DEFINITION,
            "workers": 2, "timeout": 30., **grid_fields}
        archive = directory / "records" / "launch-fixture" / "source.zip"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"immutable archived source fixture")
        (archive.parent / self.native["module_file"]).write_bytes(binary.read_bytes())
        write(archive.parent / "build_manifest.json", self.native)
        launch_path = archive.parent / "launch.json"
        write(launch_path, {"schema": "sts_training_launch_v1", "effective_config": config,
            "parent_checkpoint": policy["champion"], "source_archive": file_identity(archive),
            "data_inputs": [*self.frozen["bank_identity"]["inputs"], *self.frozen["bank_identity"]["shards"],
                            file_identity(self.grid), policy["combat"]],
            "native": self.native, "runtime": self.runtime, "environment": {"OMP_NUM_THREADS": "1"}})
        receipt = {"protocol": PROTOCOL, "policy": policy, "config": config, "launch": file_identity(launch_path)}
        write(directory / "receipt.json", receipt)
        write(directory / "grid.json", self.frozen)
        rows = []
        for cell in self.frozen["grid"]:
            w = cell["world_seed"] - WORLD
            for p in ("champion", "replacement"):
                # Resampled new wins worlds 0,1; old wins only world 2. Champion
                # wins all: direct contrast has disjoint successes, unlike either vs champion CI.
                win = True if p == "champion" else (w < 2 if arm == "new" else w == 2)
                if cell["salt"] == -1 and p == "replacement":
                    win = arm == "old"  # deliberately opposite unsalted behavior
                after = (30 if arm == "new" else 20) if p == "replacement" and win else 0
                if p == "champion":
                    after = 25
                row = {**cell, "policy": p, "win": win, "entry_hp": 50., "source_entry_hp": 52.,
                    "hp_after": after, "hp_lost": 50. - after, "decisions": 7, "cut": False, "ub": False,
                    "first_combat_hash": public_hash([w, cell["start_id"], cell["salt"]]),
                    "grid_hash": self.frozen["grid_hash"], "source_ref": file_identity(directory / "receipt.json"),
                    "timing_fixture": arm}
                rows.append(row)
                write(row_path(directory, row), row)
        n = len(rows)
        summary = {"protocol": PROTOCOL, "event": "done", "status": "completed", **grid_fields,
            "grid_input": file_identity(self.grid), "hp_definition": HP_DEFINITION,
            "cohort_complete": True, "grid_complete": True, "results_complete": True,
            "confidence_claims_allowed": True, "diagnostic_only": False, "faults": 0, "cuts": 0,
            "ub": 0, "censored": 0, "unstarted": 0, "missing_results": 0, "invalid_results": 0,
            "stop_reasons": [], "valid": n, "expected_results": n, "received": n, "records": rows,
            "source": receipt, "run_control": {"fixture": "control stats"}, "groups": {}}
        write(directory / "summary.json", summary)
        completion = {k: v for k, v in summary.items() if k not in {"records", "groups", "ub"}}
        completion["summary"] = file_identity(directory / "summary.json")
        write(directory / "completion.json", completion)

    def compare(self, resamples=128):
        return compare_panel_runs(self.new, self.old, self.grid, self.expectations, resamples=resamples)

    def reseal(self, arm, *, rows=True):
        directory = getattr(self, arm)
        summary = read(directory / "summary.json")
        if rows:
            for row in summary["records"]:
                write(row_path(directory, row), row)
        write(directory / "summary.json", summary)
        completion = read(directory / "completion.json")
        completion["summary"] = file_identity(directory / "summary.json")
        write(directory / "completion.json", completion)

    def change_row(self, arm, change, *, policy="replacement"):
        directory = getattr(self, arm)
        summary = read(directory / "summary.json")
        row = next(r for r in summary["records"] if r["policy"] == policy and r["salt"] == 0)
        change(row)
        write(directory / "summary.json", summary)
        self.reseal(arm)

    def change_receipt(self, arm, change):
        directory = getattr(self, arm)
        receipt = read(directory / "receipt.json")
        change(receipt)
        # Keep all outer hashes and sources valid to reach internal validation.
        launch_path = Path(receipt["launch"]["path"])
        launch = read(launch_path)
        launch["effective_config"] = receipt["config"]
        write(launch_path, launch)
        receipt["launch"] = file_identity(launch_path)
        write(directory / "receipt.json", receipt)
        summary = read(directory / "summary.json")
        summary["source"] = receipt
        for row in summary["records"]:
            row["source_ref"] = file_identity(directory / "receipt.json")
        write(directory / "summary.json", summary)
        completion = read(directory / "completion.json")
        completion["source"] = receipt
        write(directory / "completion.json", completion)
        self.reseal(arm)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = PanelFixture(self.temp.name)

    def test_strict_valid_known_direct_contrast_hp_direction_and_separate_unsalted(self):
        result = self.f.compare()
        self.assertTrue(result["confidence_claims_allowed"])
        self.assertTrue(result["control_exact_match"])
        boss = result["groups"]["BOSS2:resampled"]
        self.assertEqual(boss["role"], "primary")
        self.assertEqual((boss["win"]["rows"], boss["win"]["worlds"]), (24, 3))
        self.assertAlmostEqual(boss["win"]["new_mean"], 2 / 3)
        self.assertAlmostEqual(boss["win"]["old_mean"], 1 / 3)
        self.assertAlmostEqual(boss["win"]["paired_difference"], 1 / 3)
        self.assertAlmostEqual(boss["hp_lost"]["paired_difference"], -40 / 3)
        self.assertIn("negative", result["hp_lost_direction"])
        self.assertEqual(result["hp_definition"]["death_hp"], 0)
        self.assertIn("before_end_of_fight", result["hp_definition"]["hp_after"])
        self.assertEqual(result["groups"]["BOSS2:unsalted"]["win"]["point"], -1)
        self.assertTrue(result["groups"]["EVENT2:resampled"]["few_worlds_descriptive"])
        self.assertIsNone(result["groups"]["EVENT2:resampled"]["win"]["se"])
        self.assertNotIn("gate", result)
        self.assertEqual(result["checkpoint_conditions"]["old"]["old_size"], "small")
        self.assertEqual(result["input_file_identities"]["grid"], file_identity(self.f.grid))

    def test_manual_world_block_bootstrap_disjoint_outcomes_and_unequal_clusters(self):
        result = self.f.compare(resamples=512)
        metric = result["groups"]["all:resampled"]["win"]
        # WORLD 0 has 16 fights, other worlds 8. Mean of world contrasts
        # [1,1,-1] is 1/3; fight-weighted answer would be 1/2.
        self.assertEqual(metric["rows"], 32)
        self.assertAlmostEqual(metric["point"], 1 / 3)
        self.assertNotAlmostEqual(metric["point"], .5)
        draws = np.random.default_rng(1919).integers(0, 3, (512, 3))
        direct = np.array([1., 1., -1.])[draws].mean(axis=1)
        self.assertEqual(metric["ci95"], np.quantile(direct, [.025, .975]).tolist())
        self.assertAlmostEqual(metric["se"], statistics.stdev(direct.tolist()))
        self.assertAlmostEqual(metric["approx_80pct_power_mde"], 2.8 * metric["se"])
        independent_draws = np.random.default_rng(1919).integers(0, 32, (512, 32))
        independent = np.array([1.] * 24 + [-1.] * 8)[independent_draws].mean(axis=1)
        self.assertGreater(metric["se"], statistics.stdev(independent.tolist()) * 2)
        new_champ = np.array([0., 0., -1.])[draws].mean(axis=1)
        old_champ = np.array([-1., -1., 0.])[draws].mean(axis=1)
        for vs_champ in (new_champ, old_champ):
            self.assertNotEqual(metric["ci95"], np.quantile(vs_champ, [.025, .975]).tolist())
        # Subtracting CI endpoints also cannot recover the direct paired interval.
        wrong = (np.quantile(new_champ, [.025, .975]) - np.quantile(old_champ, [.025, .975])).tolist()
        self.assertNotEqual(metric["ci95"], wrong)

    def test_exact_controls_rejected_before_any_ci_all_relevant_fields(self):
        originals = {p: p.read_bytes() for p in self.f.old.rglob("*") if p.is_file()}
        changes = {"win": False, "hp_after": 24, "hp_lost": 26, "entry_hp": 51,
                   "source_entry_hp": 53, "decisions": 8, "first_combat_hash": "f" * 64,
                   "cut": True, "ub": True}
        for field, value in changes.items():
            with self.subTest(field=field):
                def change(row):
                    row[field] = value
                    if field in {"hp_after", "entry_hp", "hp_lost", "win"}:
                        if field == "win":
                            row["hp_after"] = 0
                        elif field == "hp_lost":
                            row["hp_after"] = int(row["entry_hp"] - value)
                        row["hp_lost"] = row["entry_hp"] - row["hp_after"]
                self.f.change_row("old", change, policy="champion")
                with patch("sts_rl.fresh_panel_comparison.clustered_mean", side_effect=AssertionError("CI ran")):
                    with self.assertRaisesRegex(ValueError, "Champion control|fault/cut"):
                        self.f.compare()
                for path, content in originals.items():
                    path.write_bytes(content)

    def test_replacement_initial_state_entry_hp_and_invalid_outcomes(self):
        original = {p: p.read_bytes() for p in self.f.new.rglob("*") if p.is_file()}
        for change, error in ((lambda r: r.update(first_combat_hash="f" * 64), "first_combat_hash/entry HP"),
            (lambda r: r.update(entry_hp=51, hp_lost=51-r["hp_after"]), "first_combat_hash/entry HP"),
            (lambda r: r.update(source_entry_hp=51), "first_combat_hash/entry HP"),
            (lambda r: r.update(group="BOSS1"), "group/grid_hash"),
            (lambda r: r.update(grid_hash="f" * 64), "group/grid_hash"),
            (lambda r: r.update(hp_lost=999), "HP/death-zero"),
            (lambda r: r.update(win=None), "terminal outcome"),
            *[(lambda r, flag=flag: r.update({flag: True}), "fault/cut") for flag in
              ("fault", "cut", "ub", "censored", "programming_error")]):
            with self.subTest(error=error):
                self.f.change_row("new", change)
                with self.assertRaisesRegex(ValueError, error):
                    self.f.compare()
                for path, content in original.items():
                    path.write_bytes(content)

    def test_missing_unexpected_duplicate_rows_and_files_fail_closed(self):
        directory = self.f.new
        original = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        for variant in ("missing", "duplicate", "unexpected", "missing-file", "extra-file", "quarantine"):
            with self.subTest(variant=variant):
                summary = read(directory / "summary.json")
                if variant == "missing":
                    summary["records"].pop()
                elif variant == "duplicate":
                    summary["records"].append(summary["records"][0])
                elif variant == "unexpected":
                    summary["records"][0]["salt"] = 99
                elif variant == "missing-file":
                    row_path(directory, summary["records"][0]).unlink()
                elif variant == "extra-file":
                    write(directory / "results" / "extra.json", summary["records"][0])
                else:
                    write(directory / "quarantine" / "fault.json", {"fault": "fixture"})
                write(directory / "summary.json", summary)
                self.f.reseal("new", rows=False)
                with self.assertRaises(ValueError):
                    self.f.compare()
                for path in directory.rglob("*"):
                    if path.is_file() and path not in original:
                        path.unlink()
                for path, content in original.items():
                    path.write_bytes(content)
                if (directory / "quarantine").exists():
                    (directory / "quarantine").rmdir()

    def test_source_ref_missing_tampered_or_other_run_rejected(self):
        for identity in (None, {**file_identity(self.f.new / "receipt.json"), "sha256": "f" * 64},
                         file_identity(self.f.old / "receipt.json")):
            self.f.change_row("new", lambda r: r.update(source_ref=identity))
            with self.assertRaisesRegex(ValueError, "source_ref"):
                self.f.compare()

    def test_missing_frozen_inputs_checkpoint_champion_and_bank_identities(self):
        paths = [self.f.grid, self.f.expectations,
                 *[Path(self.f.policies[a][p]["path"]) for a, p in
                   (("new", "combat"), ("old", "combat"), ("new", "champion"))],
                 self.f.root / "bank-receipt.json", self.f.root / "bank-shard.json"]
        for path in paths:
            with self.subTest(path=path.name):
                content = path.read_bytes()
                path.unlink()
                with self.assertRaises(ValueError):
                    self.f.compare()
                path.write_bytes(content)
        for path in paths[2:]:
            with self.subTest(tampered=path.name):
                content = path.read_bytes()
                path.write_bytes(content + b"tampered")
                with self.assertRaisesRegex(ValueError, "File identity mismatch"):
                    self.f.compare()
                path.write_bytes(content)
        copy = self.f.root / "copied-grid.json"
        copy.write_bytes(self.f.grid.read_bytes())
        with self.assertRaisesRegex(ValueError, "expectations grid file identity"):
            compare_panel_runs(self.f.new, self.f.old, copy, self.f.expectations, resamples=16)

    def test_valid_summary_counts_do_not_hide_result_file_tampering(self):
        summary = read(self.f.new / "summary.json")
        row = summary["records"][0]
        path = row_path(self.f.new, row)
        modified = dict(row, decisions=row["decisions"] + 1)
        write(path, modified)
        with self.assertRaisesRegex(ValueError, "Result file differs"):
            self.f.compare()

    def test_nonfinite_bool_hp_unsalted_hash_and_death_accounting_reject(self):
        original = {p: p.read_bytes() for p in self.f.new.rglob("*") if p.is_file()}
        for change in (lambda r: r.update(entry_hp=True),
                       lambda r: r.update(hp_lost=float("inf")),
                       lambda r: r.update(win=False, hp_after=1, hp_lost=49.),
                       lambda r: r.update(first_combat_hash="f" * 64)):
            for path, content in original.items():
                path.write_bytes(content)
            summary = read(self.f.new / "summary.json")
            row = next(r for r in summary["records"] if r["policy"] == "replacement" and r["salt"] == -1)
            change(row)
            write(self.f.new / "summary.json", summary)
            self.f.reseal("new")
            with self.assertRaises(ValueError):
                self.f.compare()

    def test_no_claims_if_input_or_directory_changes_during_statistics(self):
        from sts_rl.fresh_statistics import clustered_mean
        called = False
        def changed(*args, **kwargs):
            nonlocal called
            if not called:
                write(self.f.new / "results" / "late-extra.json", {"fault": "late artifact"})
                called = True
            return clustered_mean(*args, **kwargs)
        with patch("sts_rl.fresh_panel_comparison.clustered_mean", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "directory changed"):
                self.f.compare()
        (self.f.new / "results" / "late-extra.json").unlink()
        called = False
        def changed_file(*args, **kwargs):
            nonlocal called
            if not called:
                with self.f.expectations.open("a") as stream:
                    stream.write(" ")
                called = True
            return clustered_mean(*args, **kwargs)
        with patch("sts_rl.fresh_panel_comparison.clustered_mean", side_effect=changed_file):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                self.f.compare()

    def test_receipt_config_bindings_none_grid_input_mixed_policy_and_ooc(self):
        original = {p: p.read_bytes() for p in self.f.new.rglob("*") if p.is_file()}
        changes = ((lambda r: r["config"].update(grid_input=None), "config.grid_input"),
            (lambda r: r["config"].update(combat=self.f.policies["old"]["combat"]["path"]), "config checkpoint"),
            (lambda r: r["policy"].update(combat=self.f.policies["old"]["combat"]), "receipt policy"),
            (lambda r: r["policy"].update(ooc=self.f.policies["new"]["combat"]), "receipt policy"),
            (lambda r: r["config"]["hp_definition"].update(death_hp=1), "grid/bank/HP"),
            (lambda r: r["config"].update(bank_identity={}), "grid/bank/HP"))
        for change, error in changes:
            with self.subTest(error=error):
                self.f.change_receipt("new", change)
                with self.assertRaisesRegex(ValueError, error):
                    self.f.compare()
                for path, content in original.items():
                    path.write_bytes(content)

    def test_tampered_and_missing_outer_artifacts(self):
        for relative in ("receipt.json", "completion.json", "summary.json", "grid.json",
                         "records/launch-fixture/launch.json", "records/launch-fixture/source.zip",
                         "records/launch-fixture/slaythespire.fixture.so", "records/launch-fixture/build_manifest.json"):
            path = self.f.new / relative
            content = path.read_bytes()
            with self.subTest(missing=relative):
                path.unlink()
                with self.assertRaises(ValueError):
                    self.f.compare()
                path.write_bytes(content)
            with self.subTest(tampered=relative):
                path.write_bytes(content + b" ")
                # Unbound completion is validated semantically, so whitespace there is harmless.
                if relative in {"completion.json", "grid.json", "records/launch-fixture/build_manifest.json"}:
                    obj = read(path)
                    obj["status" if relative == "completion.json" else "tampered"] = "bad"
                    write(path, obj)
                with self.assertRaises(ValueError):
                    self.f.compare()
                path.write_bytes(content)

    def test_frozen_grid_and_expectations_tampering_and_reversed_arms(self):
        original = self.f.expectations.read_bytes()
        for change in (lambda e: e.update(schema="wrong"), lambda e: e.update(version=True),
            lambda e: e.update(grid={**e["grid"], "sha256": "f" * 64}),
            lambda e: e["new_condition"].update(lr=.5),
            lambda e: e["old_condition"].update(batch=999),
            lambda e: e["new_policy"].update(inference="rounded"),
            lambda e: e["old_policy"].update(ooc=e["old_policy"]["combat"]),
            lambda e: e.update(new_policy=e["old_policy"], old_policy=e["new_policy"]),
            lambda e: e.update(new_condition=e["old_condition"], old_condition=e["new_condition"])):
            value = json.loads(original)
            change(value)
            write(self.f.expectations, value)
            with self.assertRaises(ValueError):
                self.f.compare()
        self.f.expectations.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "receipt policy"):
            compare_panel_runs(self.f.old, self.f.new, self.f.grid, self.f.expectations, resamples=16)
        # Canonical rehash does not excuse incomplete grid cells.
        frozen = read(self.f.grid)
        frozen["grid"].pop()
        frozen["grid_hash"] = canonical_grid_hash(frozen)
        write(self.f.grid, frozen)
        with self.assertRaisesRegex(ValueError, "grid missing"):
            self.f.compare()

    def test_checkpoint_identities_and_actual_metadata_conditions_fail_closed(self):
        path = Path(self.f.policies["new"]["combat"]["path"])
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "File identity"):
            self.f.compare()
        path.write_bytes(original)
        expected_original = self.f.expectations.read_bytes()
        for field, value in (("initialization", "warm_start"), ("parent_checkpoint", {"path": "parent"}),
                             ("data_identity", "e" * 64), ("phase", "ooc")):
            with self.subTest(field=field):
                payload = torch.load(path, weights_only=False)
                payload["metadata"][field] = value
                torch.save(payload, path)
                expected = json.loads(expected_original)
                expected["new_policy"]["combat"] = file_identity(path)
                write(self.f.expectations, expected)
                with self.assertRaises(ValueError):
                    self.f.compare()
                path.write_bytes(original)
        # Actual configs, rather than the requested expectations, determine fit size.
        payload = torch.load(path, weights_only=False)
        payload["metadata"]["config"]["fit_size"] = "5000"
        torch.save(payload, path)
        expected = json.loads(expected_original)
        expected["new_policy"]["combat"] = file_identity(path)
        expected["new_condition"]["fit_size"] = "5000"
        write(self.f.expectations, expected)
        with self.assertRaisesRegex(ValueError, "fit_size differs"):
            self.f.compare()

    def test_launch_runtime_native_inputs_config_archive_and_completion_bindings(self):
        originals = {p: p.read_bytes() for p in self.f.new.rglob("*") if p.is_file()}
        launch_path = self.f.new / "records/launch-fixture/launch.json"
        for change in (lambda l: l["runtime"]["packages"].update(torch="different"),
            lambda l: l["native"].update(module_sha256="f" * 64),
            lambda l: l["data_inputs"].pop(), lambda l: l["effective_config"].update(resamples=7),
            lambda l: l.update(parent_checkpoint=self.f.policies["old"]["combat"]),
            lambda l: l.update(source_archive={**l["source_archive"], "sha256": "f" * 64})):
            launch = read(launch_path)
            change(launch)
            write(launch_path, launch)
            # Refresh receipt.launch and outer source/ref identities while retaining changed launch.
            receipt = read(self.f.new / "receipt.json")
            receipt["launch"] = file_identity(launch_path)
            write(self.f.new / "receipt.json", receipt)
            summary = read(self.f.new / "summary.json")
            summary["source"] = receipt
            for row in summary["records"]:
                row["source_ref"] = file_identity(self.f.new / "receipt.json")
            write(self.f.new / "summary.json", summary)
            completion = read(self.f.new / "completion.json")
            completion["source"] = receipt
            write(self.f.new / "completion.json", completion)
            self.f.reseal("new")
            with self.assertRaises(ValueError):
                self.f.compare()
            for path, content in originals.items():
                path.write_bytes(content)
        completion = read(self.f.new / "completion.json")
        completion["summary"]["sha256"] = "f" * 64
        write(self.f.new / "completion.json", completion)
        with self.assertRaisesRegex(ValueError, "summary file identity"):
            self.f.compare()

    def test_resealed_summary_completion_status_counts_and_source_mismatch(self):
        originals = {p: p.read_bytes() for p in self.f.new.rglob("*") if p.is_file()}
        for field, value in (("status", "incomplete"), ("results_complete", False),
            ("confidence_claims_allowed", False), ("diagnostic_only", True),
            ("ub", 1), ("faults", 1), ("programming_error", "fixture error"),
            ("valid", 999), ("received", 999), ("missing_results", 1),
            ("stop_reasons", ["interrupted"]), ("grid_input", None),
            ("hp_definition", {}), ("source", {})):
            with self.subTest(field=field):
                for path, content in originals.items():
                    path.write_bytes(content)
                summary = read(self.f.new / "summary.json")
                summary[field] = value
                write(self.f.new / "summary.json", summary)
                self.f.reseal("new")
                with self.assertRaises(ValueError):
                    self.f.compare()

    def test_actual_checkpoint_architecture_config_and_malformed_payload_rejected(self):
        path = Path(self.f.policies["new"]["combat"]["path"])
        original, frozen = path.read_bytes(), self.f.expectations.read_bytes()
        for config_update in ({"architecture": "old"}, {"phase": "ooc"}, {"width": 16}):
            with self.subTest(config=config_update):
                path.write_bytes(original)
                payload = torch.load(path, weights_only=False)
                payload["metadata"]["config"].update(config_update)
                torch.save(payload, path)
                expected = json.loads(frozen)
                expected["new_policy"]["combat"] = file_identity(path)
                write(self.f.expectations, expected)
                with self.assertRaisesRegex(ValueError, "architecture|phase"):
                    self.f.compare()
        # Even an expectations identity pointing to invalid checkpoint bytes fails cleanly.
        path.write_bytes(b"invalid checkpoint payload")
        expected = json.loads(frozen)
        expected["new_policy"]["combat"] = file_identity(path)
        write(self.f.expectations, expected)
        with self.assertRaisesRegex(ValueError, "Invalid new checkpoint"):
            self.f.compare()

    def test_pure_metrics_cannot_certify_provenance(self):
        rows = [[r for r in read(d / "summary.json")["records"] if r["policy"] == "replacement"]
                for d in (self.f.new, self.f.old)]
        result = contrast_metrics(*rows, self.f.frozen, resamples=16, full_fit=False)
        self.assertFalse(result["confidence_claims_allowed"])
        self.assertFalse(result["filesystem_validated"])
        self.assertTrue(all(g["descriptive_only"] for g in result["groups"].values()))

    def test_cli_real_process_strict_input_identities_no_overwrite_or_output_on_failure(self):
        out = self.f.root / "contrast.json"
        command = [sys.executable, str(ROOT / "scripts/rl/fresh_panel_compare.py"), "--new", str(self.f.new),
            "--old", str(self.f.old), "--grid", str(self.f.grid), "--expectations", str(self.f.expectations),
            "--out", str(out), "--bootstraps", "16"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["control_exact_match"])
        report = read(out)
        for name, path in (("grid", self.f.grid), ("expectations", self.f.expectations)):
            self.assertEqual(report["input_file_identities"][name], file_identity(path))
        for arm in ("new", "old"):
            for name in ("receipt", "completion", "summary", "grid"):
                self.assertEqual(report["input_file_identities"][arm][name], file_identity(getattr(self.f, arm) / (name + ".json")))
        original = out.read_bytes()
        repeat = subprocess.run(command, capture_output=True, text=True, timeout=40)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn("already exists", repeat.stderr)
        self.assertEqual(out.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            write_comparison(out, {"bad": "overwrite"})
        self.assertFalse(list(out.parent.glob(".contrast-*")))
        out.unlink()
        (self.f.new / "receipt.json").unlink()
        failed = subprocess.run(command, capture_output=True, text=True, timeout=40)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
