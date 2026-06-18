"""Unit tests for rollout reward labels.

Pure Python: operates on rollout metadata dicts, no simulator required.
"""
from __future__ import annotations

import unittest

from sts_ai.train.reward import (
    TrajectoryLabel,
    is_act_boss_clear,
    label_trajectories,
    rwr_multiplicities,
)


REPORT_KEYS = {
    "n_total",
    "n_positives",
    "n_kept",
    "min_act",
    "min_positives",
    "fallback_floor_quantile",
    "fallback_engaged",
    "threshold_floor",
    "stopped_reason_counts",
    "kept_stopped_reason_counts",
    "n_victory",
    "n_act_boss_clear",
    "agent_invalid_rate",
}


RWR_REPORT_KEYS = {
    "mode",
    "beta",
    "baseline_kind",
    "baseline_value",
    "max_multiplier",
    "n_pool",
    "n_dropped_zero",
    "n_capped",
    "multiplicity_histogram",
    "n_simulator_error_excluded",
    "total_trajectory_multiplicity",
}


def _meta(
    *,
    world_seed: int,
    rollout_index: int = 0,
    outcome: str = "GameOutcome.UNDECIDED",
    final_act: int = 1,
    final_floor: int = 0,
    stopped_reason: str = "",
    n_invalid: int = 0,
    n_decisions: int = 1,
) -> dict:
    return {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "outcome": outcome,
        "final_act": final_act,
        "final_floor": final_floor,
        "stopped_reason": stopped_reason,
        "n_invalid": n_invalid,
        "n_decisions": n_decisions,
    }


def _rwr_label(
    stem: str,
    final_floor: int,
    stopped_reason: str = "terminal",
) -> TrajectoryLabel:
    return TrajectoryLabel(
        stem=stem,
        world_seed=0,
        rollout_index=0,
        outcome="GameOutcome.UNDECIDED",
        final_act=1,
        final_floor=final_floor,
        stopped_reason=stopped_reason,
        n_invalid=0,
        n_decisions=1,
        kept=False,
        keep_reason="not_kept",
    )


class ActBossClearTest(unittest.TestCase):
    def test_victory_is_positive(self):
        self.assertTrue(is_act_boss_clear(_meta(world_seed=1, outcome="GameOutcome.VICTORY", final_act=1)))

    def test_later_act_is_positive(self):
        self.assertTrue(is_act_boss_clear(_meta(world_seed=1, final_act=2), min_act=1))

    def test_stopped_reason_does_not_gate_later_act(self):
        for reason in ("simulator_error", "agent_invalid", "max_decisions"):
            with self.subTest(reason=reason):
                self.assertTrue(is_act_boss_clear(_meta(world_seed=1, final_act=2, stopped_reason=reason)))

    def test_final_act_one_is_not_positive_regardless_of_stopped_reason(self):
        for reason in ("simulator_error", "agent_invalid", "max_decisions"):
            with self.subTest(reason=reason):
                self.assertFalse(is_act_boss_clear(_meta(world_seed=1, final_act=1, stopped_reason=reason)))


class LabelTrajectoriesStrictTest(unittest.TestCase):
    def test_strict_path_keeps_only_act_boss_clears(self):
        metas = [
            _meta(
                world_seed=11,
                rollout_index=0,
                outcome="GameOutcome.VICTORY",
                final_act=3,
                final_floor=50,
                stopped_reason="victory",
                n_invalid=1,
                n_decisions=10,
            ),
            _meta(
                world_seed=12,
                rollout_index=1,
                final_act=2,
                final_floor=18,
                stopped_reason="max_decisions",
                n_invalid=2,
                n_decisions=20,
            ),
            _meta(
                world_seed=13,
                rollout_index=2,
                final_act=1,
                final_floor=14,
                stopped_reason="agent_invalid",
                n_invalid=3,
                n_decisions=30,
            ),
        ]

        labels, report = label_trajectories(metas, min_positives=2)

        self.assertEqual(len(labels), 3)
        self.assertIsInstance(labels[0], TrajectoryLabel)
        self.assertEqual([label.stem for label in labels], ["seed_11_r0", "seed_12_r1", "seed_13_r2"])
        self.assertEqual([label.kept for label in labels], [True, True, False])
        self.assertEqual([label.keep_reason for label in labels], ["victory", "act_boss_clear", "not_kept"])

        self.assertTrue(REPORT_KEYS <= report.keys())
        self.assertEqual(report["n_total"], 3)
        self.assertEqual(report["n_positives"], 2)
        self.assertEqual(report["n_kept"], 2)
        self.assertFalse(report["fallback_engaged"])
        self.assertIsNone(report["threshold_floor"])
        self.assertEqual(report["stopped_reason_counts"], {"victory": 1, "max_decisions": 1, "agent_invalid": 1})
        self.assertEqual(report["kept_stopped_reason_counts"], {"victory": 1, "max_decisions": 1})
        self.assertEqual(report["n_victory"], 1)
        self.assertEqual(report["n_act_boss_clear"], 2)
        self.assertAlmostEqual(report["agent_invalid_rate"], 6 / 60)


class LabelTrajectoriesFallbackTest(unittest.TestCase):
    def test_fallback_keeps_top_floor_quantile_and_excludes_simulator_errors(self):
        metas = [
            _meta(world_seed=21, rollout_index=0, final_floor=1, stopped_reason="agent_invalid"),
            _meta(world_seed=22, rollout_index=0, final_floor=10, stopped_reason="max_decisions"),
            _meta(world_seed=23, rollout_index=0, final_floor=999, stopped_reason="simulator_error"),
            _meta(world_seed=24, rollout_index=0, final_floor=5, stopped_reason="agent_invalid"),
            _meta(world_seed=25, rollout_index=0, final_floor=10, stopped_reason=""),
        ]

        labels, report = label_trajectories(metas, min_positives=20, fallback_floor_quantile=0.8)

        self.assertTrue(report["fallback_engaged"])
        self.assertEqual(report["threshold_floor"], 10)
        self.assertEqual(report["n_positives"], 0)
        self.assertEqual(report["n_kept"], 2)
        self.assertEqual([label.kept for label in labels], [False, True, False, False, True])
        self.assertEqual(
            [label.keep_reason for label in labels],
            ["not_kept", "fallback_topq_floor", "not_kept", "not_kept", "fallback_topq_floor"],
        )
        self.assertEqual(report["kept_stopped_reason_counts"], {"max_decisions": 1, "": 1})

    def test_fallback_with_no_eligible_trajectories_keeps_none(self):
        metas = [
            _meta(world_seed=31, stopped_reason="simulator_error", final_floor=100),
            _meta(world_seed=32, stopped_reason="simulator_error", final_floor=200),
        ]

        labels, report = label_trajectories(metas, min_positives=20)

        self.assertTrue(report["fallback_engaged"])
        self.assertIsNone(report["threshold_floor"])
        self.assertEqual(report["n_kept"], 0)
        self.assertEqual([label.kept for label in labels], [False, False])


class AgentInvalidRateTest(unittest.TestCase):
    def test_zero_denominator_reports_zero(self):
        labels, report = label_trajectories(
            [
                _meta(world_seed=41, n_invalid=4, n_decisions=0),
                _meta(world_seed=42, n_invalid=2, n_decisions=0),
            ]
        )

        self.assertEqual(len(labels), 2)
        self.assertEqual(report["agent_invalid_rate"], 0.0)


class RWRMultiplicitiesTest(unittest.TestCase):
    def test_median_baseline_anchors_median_floor_at_one(self):
        labels = [
            _rwr_label("low", 5),
            _rwr_label("mid", 10),
            _rwr_label("high", 15),
        ]

        multiplicities, report = rwr_multiplicities(labels, beta=5.0)

        self.assertEqual(multiplicities, {"low": 0, "mid": 1, "high": 3})
        self.assertTrue(RWR_REPORT_KEYS <= report.keys())
        self.assertEqual(report["mode"], "rwr")
        self.assertEqual(report["beta"], 5.0)
        self.assertEqual(report["baseline_kind"], "median")
        self.assertEqual(report["baseline_value"], 10)
        self.assertEqual(report["max_multiplier"], 8)
        self.assertEqual(report["n_pool"], 3)
        self.assertEqual(report["n_dropped_zero"], 1)
        self.assertEqual(report["n_capped"], 0)
        self.assertEqual(report["multiplicity_histogram"], {0: 1, 1: 1, 3: 1})
        self.assertEqual(report["n_simulator_error_excluded"], 0)
        self.assertEqual(report["total_trajectory_multiplicity"], 4)

    def test_tiny_beta_drops_below_baseline_and_caps_above_baseline(self):
        labels = [
            _rwr_label("low", 1),
            _rwr_label("mid", 10),
            _rwr_label("high", 20),
        ]

        multiplicities, report = rwr_multiplicities(
            labels,
            beta=0.01,
            max_multiplier=4,
        )

        self.assertEqual(multiplicities, {"low": 0, "mid": 1, "high": 4})
        self.assertEqual(report["n_dropped_zero"], 1)
        self.assertEqual(report["n_capped"], 1)
        self.assertEqual(report["multiplicity_histogram"], {0: 1, 1: 1, 4: 1})

    def test_large_beta_is_near_uniform(self):
        labels = [
            _rwr_label("low", 1),
            _rwr_label("mid", 10),
            _rwr_label("high", 20),
        ]

        multiplicities, report = rwr_multiplicities(labels, beta=1e6)

        self.assertEqual(multiplicities, {"low": 1, "mid": 1, "high": 1})
        self.assertEqual(report["multiplicity_histogram"], {1: 3})
        self.assertEqual(report["total_trajectory_multiplicity"], 3)

    def test_simulator_error_is_excluded_with_zero_multiplicity(self):
        labels = [
            _rwr_label("ok", 10),
            _rwr_label("crashed", 999, stopped_reason="simulator_error"),
        ]

        multiplicities, report = rwr_multiplicities(labels, beta=5.0)

        self.assertEqual(multiplicities, {"ok": 1, "crashed": 0})
        self.assertEqual(report["baseline_value"], 10)
        self.assertEqual(report["n_pool"], 1)
        self.assertEqual(report["n_simulator_error_excluded"], 1)
        self.assertEqual(report["n_dropped_zero"], 0)
        self.assertEqual(report["multiplicity_histogram"], {1: 1, 0: 1})

    def test_beta_must_be_positive(self):
        for beta in (0, -1):
            with self.subTest(beta=beta):
                with self.assertRaisesRegex(ValueError, "beta"):
                    rwr_multiplicities([_rwr_label("x", 1)], beta=beta)


if __name__ == "__main__":
    unittest.main()
