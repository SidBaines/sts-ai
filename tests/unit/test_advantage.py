"""Unit tests for policy-gradient advantage calculations."""
from __future__ import annotations

import unittest

from sts_ai.train.advantage import offline_advantages, group_relative_advantages
from sts_ai.train.reward import TrajectoryLabel


OFFLINE_REPORT_KEYS = {
    "mode",
    "baseline_kind",
    "baseline_value",
    "n_pool",
    "n_simulator_error_excluded",
    "advantage_min",
    "advantage_max",
    "advantage_mean",
}

GROUP_RELATIVE_REPORT_KEYS = {
    "mode",
    "std_norm",
    "eps",
    "n_groups",
    "group_sizes",
    "n_simulator_error_excluded",
    "advantage_min",
    "advantage_max",
    "advantage_mean",
}


def _label(
    stem: str,
    final_floor: int,
    *,
    world_seed: int = 0,
    rollout_index: int = 0,
    stopped_reason: str = "terminal",
) -> TrajectoryLabel:
    return TrajectoryLabel(
        stem=stem,
        world_seed=world_seed,
        rollout_index=rollout_index,
        outcome="GameOutcome.UNDECIDED",
        final_act=1,
        final_floor=final_floor,
        stopped_reason=stopped_reason,
        n_invalid=0,
        n_decisions=1,
        kept=False,
        keep_reason="not_kept",
    )


class OfflineAdvantagesTest(unittest.TestCase):
    def test_median_baseline_returns_floor_minus_median(self):
        labels = [
            _label("low", 5),
            _label("mid", 10),
            _label("high", 15),
        ]

        advantages, report = offline_advantages(labels)

        self.assertEqual(advantages, {"low": -5.0, "mid": 0.0, "high": 5.0})
        self.assertTrue(OFFLINE_REPORT_KEYS <= report.keys())
        self.assertEqual(report["mode"], "offline")
        self.assertEqual(report["baseline_kind"], "median")
        self.assertEqual(report["baseline_value"], 10)
        self.assertEqual(report["n_pool"], 3)
        self.assertEqual(report["n_simulator_error_excluded"], 0)
        self.assertEqual(report["advantage_min"], -5.0)
        self.assertEqual(report["advantage_max"], 5.0)
        self.assertEqual(report["advantage_mean"], 0.0)

    def test_mean_baseline_variant(self):
        labels = [
            _label("low", 5),
            _label("mid", 10),
            _label("high", 15),
            _label("top", 30),
        ]

        advantages, report = offline_advantages(labels, baseline="mean")

        self.assertEqual(report["baseline_kind"], "mean")
        self.assertEqual(report["baseline_value"], 15)
        self.assertEqual(
            advantages,
            {"low": -10.0, "mid": -5.0, "high": 0.0, "top": 15.0},
        )

    def test_bad_baseline_raises(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            offline_advantages([_label("x", 1)], baseline="mode")

    def test_simulator_error_is_excluded_from_pool_and_advantages(self):
        labels = [
            _label("low", 5),
            _label("crashed", 999, stopped_reason="simulator_error"),
            _label("high", 15),
        ]

        advantages, report = offline_advantages(labels)

        self.assertEqual(advantages, {"low": -5.0, "high": 5.0})
        self.assertNotIn("crashed", advantages)
        self.assertEqual(report["baseline_value"], 10.0)
        self.assertEqual(report["n_pool"], 2)
        self.assertEqual(report["n_simulator_error_excluded"], 1)

    def test_empty_pool_returns_no_advantages_and_none_baseline(self):
        labels = [
            _label("crashed", 999, stopped_reason="simulator_error"),
        ]

        advantages, report = offline_advantages(labels)

        self.assertEqual(advantages, {})
        self.assertIsNone(report["baseline_value"])
        self.assertEqual(report["n_pool"], 0)
        self.assertEqual(report["n_simulator_error_excluded"], 1)
        self.assertIsNone(report["advantage_min"])
        self.assertIsNone(report["advantage_max"])
        self.assertIsNone(report["advantage_mean"])


class GroupRelativeAdvantagesTest(unittest.TestCase):
    def test_std_normalized_advantages_are_relative_to_world_seed_group(self):
        labels = [
            _label("a_low", 10, world_seed=1, rollout_index=0),
            _label("a_high", 20, world_seed=1, rollout_index=1),
            _label("b_only", 7, world_seed=2, rollout_index=0),
        ]

        advantages, report = group_relative_advantages(labels, eps=0.0)

        self.assertEqual(advantages, {"a_low": -1.0, "a_high": 1.0, "b_only": 0.0})
        self.assertTrue(GROUP_RELATIVE_REPORT_KEYS <= report.keys())
        self.assertEqual(report["mode"], "group_relative")
        self.assertTrue(report["std_norm"])
        self.assertEqual(report["eps"], 0.0)
        self.assertEqual(report["n_groups"], 2)
        self.assertEqual(report["group_sizes"], {1: 2, 2: 1})
        self.assertEqual(report["n_simulator_error_excluded"], 0)
        self.assertEqual(report["advantage_min"], -1.0)
        self.assertEqual(report["advantage_max"], 1.0)
        self.assertEqual(report["advantage_mean"], 0.0)

    def test_raw_group_relative_advantages_when_std_norm_is_false(self):
        labels = [
            _label("a_low", 10, world_seed=1, rollout_index=0),
            _label("a_high", 20, world_seed=1, rollout_index=1),
        ]

        advantages, report = group_relative_advantages(labels, std_norm=False)

        self.assertEqual(advantages, {"a_low": -5.0, "a_high": 5.0})
        self.assertFalse(report["std_norm"])
        self.assertEqual(report["advantage_mean"], 0.0)

    def test_zero_variance_group_returns_zero_advantages(self):
        labels = [
            _label("same_a", 12, world_seed=1, rollout_index=0),
            _label("same_b", 12, world_seed=1, rollout_index=1),
        ]

        advantages, report = group_relative_advantages(labels)

        self.assertEqual(advantages, {"same_a": 0.0, "same_b": 0.0})
        self.assertEqual(report["advantage_min"], 0.0)
        self.assertEqual(report["advantage_max"], 0.0)
        self.assertEqual(report["advantage_mean"], 0.0)

    def test_simulator_error_is_excluded_from_group_stats_and_advantages(self):
        labels = [
            _label("low", 10, world_seed=1, rollout_index=0),
            _label(
                "crashed",
                999,
                world_seed=1,
                rollout_index=1,
                stopped_reason="simulator_error",
            ),
            _label("high", 20, world_seed=1, rollout_index=2),
        ]

        advantages, report = group_relative_advantages(labels, eps=0.0)

        self.assertEqual(advantages, {"low": -1.0, "high": 1.0})
        self.assertNotIn("crashed", advantages)
        self.assertEqual(report["n_groups"], 1)
        self.assertEqual(report["group_sizes"], {1: 2})
        self.assertEqual(report["n_simulator_error_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
