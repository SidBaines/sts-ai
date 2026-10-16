"""Small native-free regressions for reporting estimands and cohort binding."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rl/fresh_gate_report_data.py"
spec = importlib.util.spec_from_file_location("fresh_gate_report_data", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportStatisticsTests(unittest.TestCase):
    def test_labels_are_label_weighted_with_whole_world_intervals(self):
        rows = [dict(world_seed=1, group="BOSS1", choice=True, confident=True, correct=True, ce=2.)] * 3
        rows += [dict(world_seed=2, group="ELITE2", choice=True, confident=False, correct=False, ce=0.)]
        result = report.aggregate_labels(rows, resamples=1000)
        agreement = result["choice"]["agreement"]
        self.assertEqual(agreement["point"], .75)  # Equal-world averaging would incorrectly give .5.
        self.assertEqual(agreement["ci95"], [0., 1.])
        self.assertEqual((agreement["worlds"], agreement["rows"]), (2, 4))
        self.assertEqual(agreement["weighting"], "label_weighted")
        self.assertEqual(result["choice"]["soft_ce"]["point"], 1.5)
        self.assertEqual(result["room:BOSS"]["agreement"]["rows"], 3)
        self.assertEqual(result["act:2"]["agreement"]["point"], 0.)
        self.assertEqual(result["high_confidence_choice"]["agreement"]["rows"], 3)

    @staticmethod
    def behavior_row(seed, chosen, opps, amount=0):
        return {"world_seed": seed, "behavior": {key: {
            "chosen": chosen, "opportunities": opps, "amount": amount if key == "shop_spend" else 0}
            for key in report.BEHAVIORS}}

    def test_paired_ratio_uses_each_arms_summed_opportunities(self):
        new = [self.behavior_row(1, 1, 1, 10), self.behavior_row(2, 0, 3)]
        old = [self.behavior_row(1, 0, 2), self.behavior_row(2, 1, 1, 6)]
        result = report.aggregate_behaviors(new, old, [1, 2], resamples=1000)
        rate = result["card_skip"]["chosen_per_opportunity"]
        self.assertEqual(rate["student"]["point"], .25)
        self.assertAlmostEqual(rate["champion"]["point"], 1/3)
        self.assertAlmostEqual(rate["student_minus_champion"]["point"], -1/12)
        # Two-world resamples have differences +1, -1/12, -1: pairing and ratio
        # normalization must both survive resampling (never subtract arm CIs).
        self.assertEqual(rate["student_minus_champion"]["ci95"], [-1., 1.])
        gold = result["shop_spend"]["gold_per_opportunity"]
        self.assertEqual(gold["student"]["point"], 2.5)  # /4 opps, not /1 purchase.
        self.assertEqual(gold["champion"]["point"], 2.)
        self.assertEqual(gold["student_minus_champion"]["point"], .5)
        self.assertEqual(result["shop_spend"]["denominators"]["student"]["opportunities"], 4)
        self.assertEqual(rate["student_minus_champion"]["worlds"], 2)

    def test_zero_opportunities_are_undefined_and_draws_counted(self):
        new = [self.behavior_row(1, 0, 0), self.behavior_row(2, 1, 1)]
        old = [self.behavior_row(1, 0, 0), self.behavior_row(2, 0, 1)]
        result = report.aggregate_behaviors(new, old, [1, 2], resamples=200)
        diff = result["card_skip"]["chosen_per_opportunity"]["student_minus_champion"]
        self.assertEqual(diff["point"], 1.)
        self.assertGreater(diff["bootstrap_valid"], 0)
        self.assertLess(diff["bootstrap_valid"], 200)
        zeros = [self.behavior_row(1, 0, 0), self.behavior_row(2, 0, 0)]
        diff = report.aggregate_behaviors(zeros, old, [1, 2], resamples=20)["card_skip"]["chosen_per_opportunity"]["student_minus_champion"]
        self.assertIsNone(diff["point"])
        self.assertEqual(diff["ci95"], [None, None])
        self.assertEqual(diff["bootstrap_valid"], 0)

    def test_behavior_intersection_pairing_and_bad_components_rejected(self):
        row = self.behavior_row(1, 1, 1)
        for student in ([row, row], [row], [self.behavior_row(1, 2, 1), self.behavior_row(2, 0, 1)]):
            with self.assertRaises(ValueError):
                report.aggregate_behaviors(student, [row, self.behavior_row(2, 0, 1)], [1, 2], resamples=2)

    def test_comparison_verdicts_are_not_exported(self):
        value = report.metric_stats({"point": 0., "ci95": [-.01, .01], "gate": "PASS", "evidence_of_decline": False})
        self.assertNotIn("gate", value)
        self.assertNotIn("evidence_of_decline", value)

    def test_zero_observed_purchases_have_nonzero_denominators_and_degenerate_flag(self):
        rows = [self.behavior_row(1, 0, 7), self.behavior_row(2, 0, 11)]
        result = report.aggregate_behaviors(rows, rows, [1, 2], resamples=100)["removal"]
        self.assertEqual(result["denominators"]["student"]["opportunities"], 18)
        self.assertEqual(result["denominators"]["champion"]["chosen"], 0)
        for stat in result["chosen_per_opportunity"].values():
            self.assertEqual(stat["ci95"], [0., 0.])
            self.assertTrue(stat["bootstrap_degenerate"])
        self.assertIn("Zero observed shop card-removal purchases", result["uncertainty_note"])
        self.assertIn("does not establish population absence", result["uncertainty_note"])
        self.assertIn("excludes event removal", result["opportunity_definition"])


class ReportGameCrosscheckTests(unittest.TestCase):
    def setUp(self):
        self.student = [dict(world_seed=i, win=i in (1, 2), floor=10*i) for i in (1, 2, 3, 4)]
        self.champion = [dict(world_seed=i, win=i in (2, 3, 4), floor=10*i+2) for i in (1, 2, 3, 4)]
        self.comparison = {"win": dict(student_wins=2, champion_wins=3, student_only=1,
            champion_only=2, denominator=4, worlds=4, point=-.25), "floor": dict(worlds=4, point=-2.)}

    def test_counts_denominators_discordance_and_points_must_match_records(self):
        report.crosscheck_game_records(self.comparison, self.student, list(reversed(self.champion)), [1, 2, 3, 4])
        for key, value in (("student_wins", 3), ("champion_wins", 2), ("denominator", 3),
                           ("worlds", 3), ("student_only", 0), ("champion_only", 1), ("point", .25)):
            bad = deepcopy(self.comparison)
            bad["win"][key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "validated cohort records"):
                report.crosscheck_game_records(bad, self.student, self.champion, [1, 2, 3, 4])
        for key, value in (("point", 2.), ("point", -1.99), ("worlds", 3), ("worlds", 4.)):
            bad = deepcopy(self.comparison)
            bad["floor"][key] = value
            with self.subTest(floor_field=key, value=value), self.assertRaisesRegex(ValueError, "validated cohort records"):
                report.crosscheck_game_records(bad, self.student, self.champion, [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "exact paired cohort"):
            report.crosscheck_game_records(self.comparison, self.student[:-1], self.champion, [1, 2, 3, 4])

    def test_stale_g2_comparison_fails_export_before_output_directory(self):
        import contextlib
        import io
        import tempfile
        from sts_rl.fresh_study import PROTOCOL, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID, INFERENCE
        identity = {"path": "/synthetic/champion.pt", "bytes": 1, "sha256": "c"*64}
        replacement = {**identity, "path": "/synthetic/replacement.pt", "sha256": "d"*64}
        champion_policy = dict(protocol=PROTOCOL, filter_ids=[FILTER_ID, DEAD_FILTER_ID],
            public_context_id=PUBLIC_CONTEXT_ID, inference=INFERENCE, champion=identity, combat=None, ooc=None)
        student_policy = {**champion_policy, "combat": replacement}
        expected = dict(schema="fresh_comparison_expectations_v1", version=1, mode="games",
                        student_policy=student_policy, champion_policy=champion_policy)
        seeds = list(range(91_000_000_000, 91_000_000_000 + 4096))
        comparison = dict(cohort_complete=True, expected_bindings=expected,
            provenance={"expectations_file": {}}, floor={"worlds": 4096},
            win=dict(student_wins=2, champion_wins=1, student_only=1, champion_only=0,
                     denominator=4096, worlds=4096, point=1/4096))
        class FixtureInputs:
            def bind(self, *args):
                return {} if len(args) == 1 else args[1]
            def json(self, path):
                path = str(path)
                if path.endswith("all_warmup_selections_v1.json"):
                    return dict(all_before_TEST=True, selections=dict.fromkeys(report.SCORES))
                if path.endswith("teacher_consistency_final_v1.json"):
                    return {"no_student_selection_or_training_from_repeats": True, "overall": {},
                            "high_confidence_primary_max_vote_share_ge_.75": {}}
                if path.endswith("dev_cohort_4096.json"):
                    return seeds
                return expected if "/expectations/" in path else comparison
        def cohort(path, **kwargs):
            policy = champion_policy if Path(path).name.startswith("dev_champion") else student_policy
            return {"config": {"mode": "eval"}, "inputs": [], "science_identity": {
                "policy": policy, "expected_provenance": {"fixture": True}},
                "records": [dict(world_seed=s, win=i == 0) for i, s in enumerate(seeds)]}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()) as errors, \
                patch.object(report, "frozen_runtime", return_value={}), \
                patch.object(report, "check_ready", return_value={"inventory_ready": True, "missing": []}), \
                patch.object(report, "Inputs", FixtureInputs), \
                patch.object(report, "training_data", return_value={}), \
                patch.object(report, "score_data", return_value={"new_combat_full": {"checkpoint": replacement}}), \
                patch.object(report, "paired_label_data", return_value={}), \
                patch("sts_rl.fresh_cohort.validate_cohorts", side_effect=cohort), \
                patch.object(report, "aggregate_behaviors") as aggregate:
            out = Path(temporary) / "must-not-exist"
            with self.assertRaises(SystemExit) as stopped:
                report.main(["--study", str(Path(temporary) / "study"), "--out", str(out)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("win counts/denominator differ from validated cohort records", errors.getvalue())
            self.assertFalse(out.exists())
            aggregate.assert_not_called()


class ReportBindingTests(unittest.TestCase):
    def setUp(self):
        from sts_rl.fresh_study import INFERENCE
        from sts_rl.fresh_training import world_split
        self.world = next(w for w in range(90_000_000_000, 90_000_000_100) if world_split(w) == "TEST")
        self.condition = {"phase": "combat", "fit_size": "5000"}
        self.receipt = {"data_identity": "d"*64, "selected_checkpoint": {"sha256": "c"*64},
                        "architecture": "new", "phase": "combat"}
        self.row = dict(world_seed=self.world, split="TEST", data_identity="d"*64,
            selected_checkpoint="c"*64, architecture="new", phase="combat", inference=INFERENCE,
            training_condition=self.condition, correct=True, choice=True, confident=True,
            has_target=True, abstention=False, target=[.25, .75], target_sha256=report.digest([.25, .75]),
            ce=.5, source_row_id="a"*64, source_identity="b"*64, public_hash="f"*64, group="BOSS1")

    def validate(self, rows, aligned=None):
        return report.validate_label_rows(rows, receipt=self.receipt, condition=self.condition,
                                          expected_count=1, aligned=aligned)

    def test_checkpoint_target_split_and_complete_row_identity_bindings(self):
        alignment = self.validate([self.row])
        self.assertEqual(self.validate([self.row], alignment), alignment)
        for key, value in (("selected_checkpoint", "wrong"), ("split", "FIT"), ("confident", False),
                           ("target", [.5, .5]), ("source_identity", ""), ("data_identity", "wrong")):
            bad = {**deepcopy(self.row), key: value}
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.validate([bad])
        with self.assertRaisesRegex(ValueError, "row count"):
            self.validate([])
        with self.assertRaisesRegex(ValueError, "cohort differs"):
            self.validate([{**self.row, "public_hash": "e"*64}], alignment)
        self.assert_stale_ooc_count_rejected()

    def assert_stale_ooc_count_rejected(self):
        """An OOC summary and row list agreeing on a short count must still fail."""
        from sts_rl.fresh_study import INFERENCE
        from sts_rl.fresh_training import FILTERS
        identity = dict(path="/synthetic/ooc.pt", bytes=1, sha256="c"*64)
        manifest = dict(path="/synthetic/manifest.json", bytes=1, sha256="d"*64)
        condition = dict(phase="ooc", fit_size="full")
        selection = dict(selection_criterion=report.SELECTION_CRITERION,
                         selected_checkpoint=identity, manifest=manifest)
        receipt = dict(selection={}, selected_checkpoint=identity, manifest=manifest,
            selection_criterion=report.SELECTION_CRITERION, training=False, weight_updates=0,
            stage="deferred_test_scoring", device="cpu", launch={"path": "/synthetic/launch.json"},
            data_identity="e"*64, architecture="new", phase="ooc", filters=FILTERS,
            vocabulary_sha256="f"*64, source_receipt={})
        summary = dict(status="completed", test_evaluated=True, training=False, weight_updates=0,
            selection={}, selected_checkpoint=identity, data_identity="e"*64,
            test_rows=1, test_protocol=INFERENCE, test_device="cpu")
        metadata = {k: receipt[k] for k in ("data_identity", "filters", "vocabulary_sha256", "source_receipt")}
        class FixtureInputs:
            def bind(self, *args):
                return None
            def json(self, path, *args):
                path = str(path)
                if "/selections/" in path:
                    return selection
                if path.endswith("summary.json"):
                    return summary
                if path.endswith("receipt.json"):
                    return receipt
                return {"effective_config": receipt}
            def jsonl(self, path):
                return [self_row]
        self_row = {**self.row, "phase": "ooc", "training_condition": condition}
        runs = {"new_ooc_full": {"summary": dict(best_step=1, selected_checkpoint=identity), "config": condition}}
        with patch.object(report, "SCORES", ("new_ooc_full",)), \
                patch.object(report, "_checkpoint", return_value=(condition, metadata, 1)), \
                patch.object(report, "validate_label_rows") as validate:
            with self.assertRaisesRegex(ValueError, "TEST count/protocol/device mismatch"):
                report.score_data(Path("/synthetic"), FixtureInputs(),
                    {"selections": {"new_ooc_full": {}}}, runs)
            validate.assert_not_called()

    def test_readiness_does_not_open_test_or_create_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            status = report.check_ready(path)
            self.assertFalse(status["inventory_ready"])
            self.assertFalse(status["validation_performed"])
            self.assertEqual(list(path.iterdir()), [])

    def paired_fixture(self):
        from sts_rl.fresh_statistics import compare_labels
        common = dict(phase="combat", fit_size="full", lr=.0003, seed=1919, steps=12000,
            min_steps=2000, eval_every=100, patience=6, device="mps", batch=128, defer_test=True)
        conditions = ({**common, "width": 128, "layers": 2, "heads": 4},
                      {**common, "old_size": "large", "old_context": "none"})
        expected = dict(schema="fresh_comparison_expectations_v1", version=1, mode="labels")
        scores, rows = {}, {}
        for arm, architecture, condition, wins in zip(("student", "control"), ("new", "old"),
                                                      conditions, ((True, False), (False, True))):
            checkpoint = dict(path=f"/synthetic/{architecture}.pt", bytes=1,
                              sha256=("c" if architecture == "new" else "d")*64)
            expected.update({arm+"_checkpoint": checkpoint, arm+"_architecture": architecture,
                             arm+"_condition": condition})
            name = architecture + "_combat_full"
            rows[name] = [{**self.row, "source_row_id": str(i)*64, "world_seed": self.world+i,
                "selected_checkpoint": checkpoint["sha256"], "training_condition": condition,
                "architecture": architecture, "correct": win} for i, win in enumerate(wins)]
            scores[name] = dict(checkpoint=checkpoint, architecture=architecture, training_condition=condition,
                data_identity=self.receipt["data_identity"], manifest={"sha256": "f"*64}, test_rows=2,
                test_row_ids_sha256=report.digest(sorted(r["source_row_id"] for r in rows[name])))
        value = compare_labels(rows["new_combat_full"], rows["old_combat_full"],
                               expectations=expected, resamples=100)
        return value, expected, scores, rows

    def test_paired_label_export_binds_expectations_scores_data_and_rows(self):
        value, expected, scores, rows = self.paired_fixture()
        exported = report.validate_paired_labels(value, expected, scores, rows, resamples=100)
        self.assertEqual(exported["groups"]["choice"]["point"], 0.)
        self.assertEqual(exported["groups"]["choice"]["ci95"], [-1., 1.])
        self.assertEqual(exported["paired_rows"], 2)
        for change in ("expectation", "checkpoint", "data", "rows", "point", "ci"):
            bad_value, bad_expected, bad_scores, bad_rows = map(deepcopy, (value, expected, scores, rows))
            if change == "expectation":
                bad_value["expected_bindings"]["student_condition"]["lr"] = 1.
            elif change == "checkpoint":
                bad_scores["new_combat_full"]["checkpoint"]["sha256"] = "e"*64
            elif change == "data":
                bad_rows["new_combat_full"][0]["data_identity"] = "e"*64
            elif change == "rows":
                bad_rows["new_combat_full"][0]["source_row_id"] = "e"*64
            elif change == "point":
                bad_value["groups"]["choice"]["point"] = .1
            else:
                bad_value["groups"]["choice"]["ci95"] = [0., 0.]
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_paired_labels(bad_value, bad_expected, bad_scores, bad_rows, resamples=100)

    def test_paired_label_ci_catches_changed_pairing_with_unchanged_arm_agreement(self):
        value, expected, scores, rows = self.paired_fixture()
        for row in rows["old_combat_full"]:
            row["correct"] = not row["correct"]  # Both arms still 50%; paired CI now [0,0].
        with self.assertRaisesRegex(ValueError, "paired statistics differ"):
            report.validate_paired_labels(value, expected, scores, rows, resamples=100)


if __name__ == "__main__":
    unittest.main()
