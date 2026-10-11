from __future__ import annotations

import unittest

from scripts.report_nonllm import summarize_live, world_bootstrap, summarize_metrics


class NonllmReportTests(unittest.TestCase):
    def test_world_clusters_and_equal_world_weight(self):
        result = world_bootstrap([(1, 1.)] * 20 + [(2, 0.)], replicates=1000, seed=3)
        self.assertEqual(result["worlds"], 2)
        self.assertEqual(result["observations"], 21)
        self.assertEqual(result["world_mean"], .5)
        self.assertEqual(result["ci95"], [0., 1.])
        self.assertEqual(result, world_bootstrap([(1, 1.)] * 20 + [(2, 0.)], replicates=1000, seed=3))

    def test_faults_and_truncations_are_not_losses(self):
        rows = [{"world_seed": 1, "task_kind": "battle", "terminated": True, "task_outcome": "battle_victory"},
                {"world_seed": 2, "task_kind": "battle", "terminated": True, "task_outcome": "battle_defeat"},
                {"world_seed": 3, "task_kind": "battle", "truncated": True, "task_outcome": "battle_defeat"},
                {"world_seed": 4, "task_kind": "battle", "fault": True, "task_outcome": "battle_defeat"}]
        faults = [{"world_seed": 5, "task_kind": "battle", "error": "SimulatorTimeout"}]
        result = summarize_live(rows, faults)["battle"]
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["faults"], 2)
        self.assertEqual(result["truncated"], 1)
        self.assertEqual(result["success_among_decisive_episodes"], .5)
        self.assertEqual(result["decisive_success"]["worlds"], 2)

    def test_unknown_terminal_and_single_world_no_false_interval(self):
        self.assertIsNone(world_bootstrap([(1, 1.)])["ci95"])
        result = summarize_live([{"world_seed": 1, "task_kind": "run", "terminated": True,
                                  "outcome": "unknown", "floor": 8}], [])["run"]
        self.assertEqual(result["unresolved"], 1)
        self.assertIsNone(result["success_among_decisive_episodes"])
        self.assertEqual(result["floor_all_valid"]["world_mean"], 8)

    def test_missing_world_identity_rejected(self):
        with self.assertRaises(ValueError):
            world_bootstrap([(None, 1.)])

    def test_aggregate_bc_cannot_invent_world_confidence_interval(self):
        result = summarize_metrics({"development": {"accuracy": .7, "examples": 20}})
        self.assertEqual(result["bc_development"]["accuracy"], .7)
        self.assertIn("per-world", result["bc_ci_note"])
