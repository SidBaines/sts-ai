"""C5 scientific grid/receipt regressions, with no models or native execution."""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sts_policy.curriculum import public_hash
from sts_rl.fresh_panel import freeze_panel_grid, paired_summary, run_panel
from sts_rl.fresh_panel_grid import (DEFAULT_TARGETS, HP_DEFINITION, SAMPLING_SPEC, SELECTION_PROTOCOL,
    canonical_grid_hash, freeze_grid, load_grid, save_grid)
from sts_rl.fresh_study import atomic_json
from sts_rl.recording import file_identity
from tests.unit.fresh_cohort_fixture import write_cohort
from tests.unit.test_fresh_collection import (Control, EXPECTED, Policy, Progress, SEED, SOURCE,
                                             Worker, config, observation, play_world)


def rows(worlds=4):
    return [{"world_seed": SEED+w, "start_id": start, "group": group,
             "public_hash": f"ooc-{w}-{start}", "first_combat_hash": f"combat-{w}-{start}",
             "expected_provenance": EXPECTED, "prefix": [], "task": {},
             "entering_action": 2, "entering_description": "enter boss"}
            for w in range(worlds) for start, group in ((1, "BOSS2"), (2, "BOSS2"), (3, "MONSTER2"))]


def results(frozen):
    return [{**item, "policy": policy, "win": True, "hp_lost": 3.,
             "first_combat_hash": str(item["salt"]), "grid_hash": frozen["grid_hash"]}
            for item in frozen["grid"] for policy in ("champion", "replacement")]


def fixture_bank(path):
    world = play_world(Worker([observation(), observation("combat", "BOSS"),
                       observation("combat", "BOSS", hp=20, actions=[])]),
                       Policy(), SEED, config(mode="eval"), Control(), Progress(), SOURCE, lambda _: None)
    return write_cohort(path, [world])


class GridTests(unittest.TestCase):
    def test_two_independent_hash_stages_ignore_order_outcomes_and_model_keys(self):
        targets = {"BOSS2": 3, "MONSTER2": 2}
        selected, frozen = freeze_grid(rows(), targets, 8)
        changed = [{**r, "win": False, "baseline_win": False, "baseline_hp_lost": -999,
                    "model_score": 1e9, "checkpoint": "different-model"} for r in reversed(rows())]
        other, repeated = freeze_grid(changed, dict(reversed(list(targets.items()))), 8)
        self.assertEqual(frozen, repeated)
        keys = lambda rs: [(r["group"], r["world_seed"], r["start_id"]) for r in rs]
        self.assertEqual(keys(selected), keys(other))
        self.assertEqual(Counter(r["group"] for r in selected), targets)
        self.assertEqual(len({(r["group"], r["world_seed"]) for r in selected}), 5)
        self.assertEqual(len({(r["world_seed"], r["start_id"]) for r in selected}), 5)
        for group, count in targets.items():
            ranked = sorted(range(SEED, SEED+4), key=lambda world: (public_hash({
                "namespace": SAMPLING_SPEC["namespace"], "stage": "world", "group": group,
                "world_seed": world}), world))[:count]
            self.assertEqual({r["world_seed"] for r in selected if r["group"] == group}, set(ranked))
            for row in [r for r in selected if r["group"] == group]:
                candidates = [r for r in rows() if r["group"] == group and r["world_seed"] == row["world_seed"]]
                expected = min(candidates, key=lambda r: (public_hash({
                    "namespace": SAMPLING_SPEC["namespace"], "stage": "start", "group": group,
                    "world_seed": r["world_seed"], "start_id": r["start_id"]}), r["start_id"]))
                self.assertEqual(row, expected)
        self.assertEqual(frozen["salts"], [-1, *range(8)])
        self.assertEqual(len(frozen["grid"]), 45)
        self.assertEqual(frozen["grid_hash"], canonical_grid_hash(frozen))

    def test_approved_catalog_yields_1006_starts_18108_results_and_300_boss2_worlds(self):
        catalog = {"BOSS1": 1004, "BOSS2": 766, "BOSS3": 195, "ELITE2": 651,
                   "ELITE3": 167, "MONSTER2": 966, "MONSTER3": 326, "EVENT1": 52, "EVENT2": 4}
        base = rows(1)[0]
        bank = [{**base, "world_seed": SEED+w, "group": group, "start_id": 2*i+j}
                for i, (group, n) in enumerate(catalog.items()) for w in range(n) for j in (0, 1)]
        selected, frozen = freeze_grid(bank, DEFAULT_TARGETS, 8)
        self.assertEqual(len(selected), 1006)
        self.assertEqual(len(frozen["grid"])*2, 18108)
        self.assertEqual(len({r["world_seed"] for r in selected if r["group"] == "BOSS2"}), 300)
        self.assertEqual({g: frozen["eligible_worlds"][g] for g in catalog}, catalog)
        self.assertEqual(frozen["coverage"], DEFAULT_TARGETS)
        self.assertTrue(frozen["coverage_sufficient"])
        self.assertIn("EVENT3", frozen["absent_groups"])
        self.assertNotIn("EVENT3", frozen["targets"])

    def test_world_shortage_is_not_hidden_by_many_starts_or_complete_results(self):
        _, frozen = freeze_grid(rows(1), {"BOSS2": 2, "EVENT3": 1}, 8)
        self.assertEqual(frozen["coverage"], {"BOSS2": 1, "EVENT3": 0})
        self.assertEqual(frozen["coverage_shortfall"], {"BOSS2": 1, "EVENT3": 1})
        report = paired_summary(results(frozen), frozen)
        self.assertTrue(report["results_complete"])
        self.assertEqual((report["missing_results"], report["invalid_results"]), (0, 0))
        self.assertFalse(report["coverage_sufficient"])
        self.assertFalse(report["grid_complete"])
        self.assertFalse(report["confidence_claims_allowed"])
        for name, metrics in report["groups"].items():
            self.assertTrue(metrics["diagnostic_only"], name)
            self.assertTrue(metrics["win"]["diagnostic_only"])
        self.assertEqual(report["groups"]["EVENT3:resampled"]["win"]["worlds"], 0)
        _, adequate = freeze_grid(rows(2), {"BOSS2": 2}, 8)
        missing = paired_summary(results(adequate)[:-1], adequate)
        self.assertTrue(missing["coverage_sufficient"])
        self.assertFalse(missing["results_complete"])
        self.assertEqual(missing["missing_results"], 1)

    def test_positive_integer_counts_and_unique_start_keys_required(self):
        for targets, resamples in (({}, 8), ({"BOSS2": 0}, 8), ({"BOSS2": True}, 8),
                                   ({"BOSS2": 2.5}, 8), ({"BOSS2": 1}, True), ({"BOSS2": 1}, 0)):
            with self.subTest(targets=targets, resamples=resamples), self.assertRaises(ValueError):
                freeze_grid(rows(), targets, resamples)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            freeze_grid(rows() + [{**rows()[0], "group": "EVENT2"}], {"BOSS2": 1}, 8)

    def test_world_clustering_keeps_all_eight_salts_and_control_separate(self):
        _, frozen = freeze_grid(rows(2), {"BOSS2": 2}, 8)
        records = results(frozen)
        for r in records:
            if r["policy"] == "replacement":
                r["hp_lost"] = 103 if r["salt"] == -1 else (5 if r["world_seed"] == SEED else 9)
        report = paired_summary(records, frozen)
        metric = report["groups"]["BOSS2:resampled"]["hp_lost"]
        self.assertEqual(metric["worlds"], 2)
        self.assertEqual(metric["paired_difference"], 4.)  # World means 2 and 6.
        self.assertAlmostEqual(metric["se"], 2.)  # Two clusters, not sixteen independent salts.
        self.assertEqual(report["groups"]["BOSS2:unsalted"]["hp_lost"]["paired_difference"], 100.)

    def test_compare_panel_equal_world_matches_collector_summary(self):
        # D4 test_fresh_statistics fixture: three starts in one world, one in the
        # other. Its equal-world points are 0 (win) and -4 (HP), not row means.
        grid = [dict(world_seed=w, start_id=s, salt=t, group="ELITE2")
                for w, starts in ((1, range(3)), (2, range(1)))
                for s in starts for t in (-1, 0, 1, 2)]
        records = [{**item, "policy": p, "win": (p == "replacement") == (item["world_seed"] == 1),
                    "hp_lost": (10 if item["world_seed"] == 1 else 22) if p == "replacement" else 20,
                    "first_combat_hash": f"synthetic:{item['world_seed']}:{item['start_id']}"}
                   for item in grid for p in ("champion", "replacement")]
        frozen = {"grid": grid, "salts": [-1, 0, 1, 2], "coverage_sufficient": True}
        report = paired_summary(records, frozen)
        self.assertTrue(report["grid_complete"])
        self.assertTrue(report["results_complete"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["confidence_claims_allowed"])
        self.assertIsNone(report["grid_hash"])
        self.assertIsNone(report["targets"])
        self.assertEqual(set(report["groups"]), {"ELITE2:unsalted", "ELITE2:resampled"})
        for suffix in (":unsalted", ":resampled"):
            group = report["groups"]["ELITE2"+suffix]
            self.assertTrue(group["diagnostic_only"])
            for metric, difference, champion, replacement, se in (
                    ("win", 0., .5, .5, 1.), ("hp_lost", -4., 20., 16., 6.)):
                actual = group[metric]
                self.assertEqual(actual["paired_difference"], difference)
                self.assertEqual(actual["champion_mean"], champion)
                self.assertEqual(actual["replacement_mean"], replacement)
                self.assertEqual(actual["worlds"], 2)
                self.assertAlmostEqual(actual["se"], se)
                self.assertTrue(actual["diagnostic_only"])

    def test_missing_optional_metadata_is_diagnostic_without_changing_means(self):
        _, frozen = freeze_grid(rows(2), {"BOSS2": 2}, 8)
        records = results(frozen)
        report = paired_summary(records, frozen)
        self.assertTrue(report["confidence_claims_allowed"])
        for key in ("targets", "selection_protocol", "coverage"):
            with self.subTest(missing=key):
                legacy = {k: v for k, v in frozen.items() if k != key}
                diagnostic = paired_summary(records, legacy)
                self.assertTrue(diagnostic["grid_complete"])
                self.assertTrue(diagnostic["diagnostic_only"])
                self.assertFalse(diagnostic["confidence_claims_allowed"])
                for name, group in diagnostic["groups"].items():
                    self.assertTrue(group["diagnostic_only"])
                    for metric in ("win", "hp_lost"):
                        self.assertTrue(group[metric]["diagnostic_only"])
                        self.assertEqual(group[metric]["paired_difference"],
                                         report["groups"][name][metric]["paired_difference"])

    def test_manifest_rejects_tampering_bank_mismatch_and_changed_replay_states(self):
        target, identity = {"BOSS2": 2}, {"shards": [{"path": "bank", "sha256": "original"}]}
        _, frozen = freeze_grid(rows(), target, 8, bank_identity=identity)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"grid.json"
            original_file = save_grid(path, frozen)
            self.assertEqual(load_grid(path, rows(), target, 8, bank_identity=identity)[1:],
                             (frozen, original_file))
            with self.assertRaises(FileExistsError):
                save_grid(path, frozen)
            self.assertEqual(file_identity(path), original_file)
            with self.assertRaisesRegex(ValueError, "bank identity"):
                load_grid(path, rows(), target, 8, bank_identity={"shards": []})
            for change in ("public_hash", "first_combat_hash", "expected_provenance", "prefix", "start_id"):
                changed = deepcopy(rows())
                chosen = frozen["starts"][0]
                row = next(r for r in changed if (r["world_seed"], r["start_id"]) ==
                           (chosen["world_seed"], chosen["start_id"]))
                row[change] = {"changed": True} if change == "expected_provenance" else (
                    [77] if change == "prefix" else 99 if change == "start_id" else "changed")
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "selection/start"):
                    load_grid(path, changed, target, 8, bank_identity=identity)
            for change, rehash in (("grid", False), ("grid", True), ("starts", True),
                                   ("selection_protocol", True), ("targets", True), ("resamples", True)):
                altered = deepcopy(frozen)
                if change in ("grid", "starts"):
                    altered[change][0]["start_id"] = 999
                else:
                    altered[change] = {"BOSS2": 1} if change == "targets" else 7 if change == "resamples" else "old"
                if rehash:
                    altered["grid_hash"] = canonical_grid_hash(altered)
                # A separate file simulates an externally altered/rehashed artifact.
                bad = Path(directory)/"tampered.json"
                atomic_json(bad, altered)
                with self.subTest(change=change, rehash=rehash), self.assertRaises(ValueError):
                    load_grid(bad, rows(), target, 8, bank_identity=identity)


class PanelReceiptTests(unittest.TestCase):
    def test_freeze_without_models_then_reuse_exact_grid_in_multiple_model_runs(self):
        calls, launches = [], []
        class PanelWorker:
            def __init__(self, **kwargs):
                pass
            def fresh_fight(self, bank, salt, champion, combat, expected_policy):
                calls.append((combat, bank["world_seed"], bank["start_id"], salt))
                return {"win": True, "hp_lost": 2, "first_combat_hash": str(salt), "decisions": 1}
            def close(self):
                pass
        class PanelControl(Control):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def stats(self):
                return {"active_ac_seconds": 1.}
            def close(self):
                pass
        def launch(out, effective, checkpoint=None, inputs=()):
            launches.append(deepcopy(effective))
            self.assertEqual(effective["grid_hash"], frozen["grid_hash"])
            self.assertEqual(effective["selection_protocol"], SELECTION_PROTOCOL)
            self.assertEqual(effective["bank_identity"], frozen["bank_identity"])
            self.assertEqual(effective["grid_input"], file_identity(grid))
            self.assertEqual(effective["hp_definition"], HP_DEFINITION)
            self.assertIn(str(grid), inputs)
            path = Path(out)/"launch.json"
            atomic_json(path, {"effective_config": effective})
            return file_identity(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = fixture_bank(root/"bank")
            grid = root/"frozen.json"
            base = {"bank": [str(bank)], "starts_per_group": {"BOSS2": 1}, "resamples": 8}
            with patch("sts_rl.fresh_panel.StudyPolicy", side_effect=AssertionError("freeze loaded model")), \
                 patch("sts_rl.fresh_study.begin_run", side_effect=AssertionError("freeze launched run")):
                frozen = freeze_panel_grid({**base, "out": str(grid)})
            grid_before = file_identity(grid)
            champion = root/"champion.pt"
            champion.write_bytes(b"fixture champion")
            for name in ("old", "new"):
                combat = root/(name+".pt")
                combat.write_bytes(name.encode())
                policy = SimpleNamespace(identity={"champion": file_identity(champion),
                                                     "combat": file_identity(combat), "ooc": None})
                out = root/name
                with patch("sts_rl.fresh_panel.StudyPolicy", return_value=policy), \
                     patch("sts_rl.fresh_study.begin_run", side_effect=launch), \
                     patch("sts_rl.fresh_panel.RunControl", PanelControl):
                    report = run_panel({**base, "grid": str(grid), "out": str(out),
                        "checkpoint": str(champion), "combat": str(combat), "workers": 2, "timeout": 1.},
                        worker_factory=PanelWorker)
                source = json.loads((out/"receipt.json").read_text())
                completion = json.loads((out/"completion.json").read_text())
                self.assertEqual(report["source"], source)
                self.assertEqual(completion["source"], source)
                self.assertEqual(completion["summary"], file_identity(out/"summary.json"))
                for key in ("grid_hash", "selection_protocol", "targets", "coverage", "coverage_sufficient",
                            "hp_definition", "grid_input"):
                    self.assertEqual(report[key], completion[key])
                    self.assertEqual(report[key], source["config"][key])
                receipt_ref = file_identity(out/"receipt.json")
                for record in report["records"]:
                    self.assertNotIn("source", record)
                    self.assertEqual(record["source_ref"], receipt_ref)
                    self.assertEqual(record["grid_hash"], frozen["grid_hash"])
                    self.assertLess(len(json.dumps(record)), 800)
                recovered = [json.loads(p.read_text()) for p in (out/"results").glob("*.json")]
                self.assertEqual({json.dumps(r, sort_keys=True) for r in recovered},
                                 {json.dumps(r, sort_keys=True) for r in report["records"]})
                self.assertTrue(paired_summary(recovered, frozen)["grid_complete"])
                self.assertEqual(json.loads((out/"grid.json").read_text()), frozen)
                self.assertEqual(len(report["records"]), 18)
            self.assertEqual(file_identity(grid), grid_before)
            self.assertEqual(launches[0]["grid_hash"], launches[1]["grid_hash"])
            self.assertEqual({(w, s, t) for combat, w, s, t in calls if combat is None},
                             {(w, s, t) for combat, w, s, t in calls if combat is not None})
            self.assertEqual({(w, s, t) for combat, w, s, t in calls if combat == str(root/"old.pt")},
                             {(w, s, t) for combat, w, s, t in calls if combat == str(root/"new.pt")})

            # Any mismatch rejects before a policy can be loaded/scored/launched.
            with patch("sts_rl.fresh_panel.StudyPolicy", side_effect=AssertionError("loaded before validation")):
                with self.assertRaisesRegex(ValueError, "targets/resamples"):
                    run_panel({**base, "grid": str(grid), "starts_per_group": {"BOSS2": 2},
                               "out": str(root/"bad"), "workers": 1, "timeout": 1.})
                other_bank = fixture_bank(root/"other_bank")
                with self.assertRaisesRegex(ValueError, "bank identity"):
                    run_panel({**base, "bank": [str(other_bank)], "grid": str(grid),
                               "out": str(root/"bad"), "workers": 1, "timeout": 1.})

    def test_cli_model_free_freeze_and_grid_argument(self):
        path = Path(__file__).resolve().parents[2]/"scripts"/"rl"/"fresh_panel.py"
        spec = importlib.util.spec_from_file_location("c5_panel_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory:
            bank = fixture_bank(Path(directory)/"bank")
            grid = Path(directory)/"grid.json"
            output = io.StringIO()
            with patch("sts_rl.fresh_panel.StudyPolicy", side_effect=AssertionError("model loaded")), redirect_stdout(output):
                cli.main(["--freeze-grid-only", "--bank", str(bank), "--out", str(grid),
                          "--starts-per-group", '{"BOSS2":1}', "--resamples", "8"])
            self.assertEqual(json.loads(output.getvalue())["grid_hash"], json.loads(grid.read_text())["grid_hash"])
            self.assertEqual(cli.parser().parse_args(["--bank", str(bank), "--out", "new", "--grid", str(grid)]).grid,
                             str(grid))
