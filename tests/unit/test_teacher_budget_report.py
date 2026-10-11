from __future__ import annotations

from dataclasses import replace
import unittest

from scripts.benchmark_teacher_budget import summarize, scenario_identity
from sts_policy.curriculum import TaskSpec


class TeacherBudgetReportTests(unittest.TestCase):
    def test_disagreement_uses_identical_paired_frontiers(self):
        rows = [{"world_seed": 1, "labels": {"50": 0, "200": 1, "1000": 1},
                 "latency_seconds": {"50": .001, "200": .002, "1000": .004}},
                {"world_seed": 2, "labels": {"50": 2, "200": 2, "1000": 1},
                 "latency_seconds": {"50": .003, "200": .004, "1000": .006}}]
        report = summarize(rows)
        self.assertEqual(report["paired_disagreement"]["50_vs_200"]["disagreement_rate"], .5)
        self.assertEqual(report["paired_disagreement"]["50_vs_1000"]["disagreement_rate"], 1.)
        self.assertEqual(report["query_rpc_latency"]["200"]["mean_ms"], 3.)
        self.assertEqual(report["worlds"], 2)

    def test_scenario_matching_excludes_only_teacher_budget(self):
        task = TaskSpec(seed=3, prefix=(0,), expected_public_hash="public")
        self.assertEqual(scenario_identity(replace(task, battle_simulations=50)),
                         scenario_identity(replace(task, battle_simulations=1000)))
        self.assertNotEqual(scenario_identity(task), scenario_identity(replace(task, prefix=(1,))))
        self.assertNotEqual(scenario_identity(task), scenario_identity(replace(task, expected_public_hash="other")))
