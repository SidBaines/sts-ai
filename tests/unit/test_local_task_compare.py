from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.local_task_compare import build_report
from sts_ai.local_tasks.runner import window_rollout_indices


def _write_meta(
    root: Path,
    *,
    world_seed: int,
    rollout_index: int,
    window_id: str,
    reward: float,
    hp_loss: int,
    survived: bool,
) -> None:
    meta = {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "n_decisions": 10,
        "n_invalid": 0,
        "extra": {
            "local_task": {
                "task_id": "gremlin_nob",
                "window_id": window_id,
                "label": "won" if survived else "loss",
                "reward": reward,
                "metrics": {
                    "reward": reward,
                    "hp_loss": hp_loss,
                    "survived": survived,
                    "action_counts": {"attack": 5, "block": 3, "skill": 1, "other": 1},
                },
            }
        },
    }
    path = root / f"seed_{world_seed}_r{rollout_index}.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")


class LocalTaskCompareTest(unittest.TestCase):
    def test_multiple_rollouts_per_window_are_averaged_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            trained = Path(tmp) / "trained"
            base.mkdir()
            trained.mkdir()
            # Window w1: base rollouts 0.0 and 1.0 (mean 0.5); trained 1.0 and 1.0.
            # The pre-fix _by_window kept only the lexically-last rollout per
            # window, which here would report a base of 1.0 and a delta of 0.
            _write_meta(base, world_seed=8, rollout_index=0, window_id="seed_8_r0_w0",
                        reward=0.0, hp_loss=40, survived=False)
            _write_meta(base, world_seed=8, rollout_index=1, window_id="seed_8_r0_w0",
                        reward=1.0, hp_loss=0, survived=True)
            _write_meta(trained, world_seed=8, rollout_index=0, window_id="seed_8_r0_w0",
                        reward=1.0, hp_loss=0, survived=True)
            _write_meta(trained, world_seed=8, rollout_index=1, window_id="seed_8_r0_w0",
                        reward=1.0, hp_loss=0, survived=True)

            report = build_report(base, trained, metric="reward")

            self.assertEqual(report["paired"]["n"], 1)
            self.assertAlmostEqual(report["paired"]["mean_delta"], 0.5)
            counts = report["paired"]["rollouts_per_window"]
            self.assertEqual(counts["base"], {"min": 2, "max": 2})
            self.assertEqual(counts["trained"], {"min": 2, "max": 2})
            # Per-episode aggregates still count episodes, not windows.
            self.assertEqual(report["arms"]["base"]["n"], 2)

    def test_pairing_uses_window_intersection_and_reports_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            trained = Path(tmp) / "trained"
            base.mkdir()
            trained.mkdir()
            for k, reward in enumerate((0.2, 0.4)):
                _write_meta(base, world_seed=4, rollout_index=k, window_id="seed_4_r0_w0",
                            reward=reward, hp_loss=20, survived=True)
                _write_meta(trained, world_seed=4, rollout_index=k, window_id="seed_4_r0_w0",
                            reward=reward + 0.3, hp_loss=10, survived=True)
            # Unpaired window present only in the base arm must be dropped.
            _write_meta(base, world_seed=12, rollout_index=0, window_id="seed_12_r0_w0",
                        reward=-1.0, hp_loss=80, survived=False)

            report = build_report(base, trained, metric="reward")

            self.assertEqual(report["paired"]["window_ids"], ["seed_4_r0_w0"])
            self.assertAlmostEqual(report["paired"]["mean_delta"], 0.3)
            self.assertIn("bootstrap_ci_95", report["paired"])
            self.assertIn("p_value", report["paired"]["sign_test"])

    def test_hp_loss_metric_reads_nested_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            trained = Path(tmp) / "trained"
            base.mkdir()
            trained.mkdir()
            _write_meta(base, world_seed=4, rollout_index=0, window_id="w",
                        reward=0.0, hp_loss=40, survived=True)
            _write_meta(base, world_seed=4, rollout_index=1, window_id="w",
                        reward=0.5, hp_loss=20, survived=True)
            _write_meta(trained, world_seed=4, rollout_index=0, window_id="w",
                        reward=0.75, hp_loss=10, survived=True)

            report = build_report(base, trained, metric="hp_loss")

            self.assertAlmostEqual(report["paired"]["mean_delta"], 10.0 - 30.0)


class WindowRolloutIndicesTest(unittest.TestCase):
    def test_k1_matches_historical_ordinal_indexing(self):
        self.assertEqual(window_rollout_indices({"ordinal": 0}, 1), [0])
        self.assertEqual(window_rollout_indices({"ordinal": 2}, 1), [2])

    def test_k_gt_1_is_collision_free_across_ordinals(self):
        w0 = window_rollout_indices({"ordinal": 0}, 4)
        w1 = window_rollout_indices({"ordinal": 1}, 4)
        self.assertEqual(w0, [0, 1, 2, 3])
        self.assertEqual(w1, [4, 5, 6, 7])
        self.assertFalse(set(w0) & set(w1))

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            window_rollout_indices({"ordinal": 0}, 0)


if __name__ == "__main__":
    unittest.main()
