from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_paired import (
    aggregate_arm,
    build_report,
    format_report,
    load_metas,
    per_seed_metric_means,
)


def _write_meta(
    root: Path,
    *,
    world_seed: int,
    rollout_index: int,
    outcome: str = "GameOutcome.LOSS",
    final_act: int = 1,
    final_floor: int = 0,
    stopped_reason: str = "terminal",
    n_invalid: int = 0,
    n_decisions: int = 1,
) -> None:
    meta = {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "outcome": outcome,
        "final_act": final_act,
        "final_floor": final_floor,
        "stopped_reason": stopped_reason,
        "n_invalid": n_invalid,
        "n_decisions": n_decisions,
    }
    path = root / f"seed_{world_seed}_r{rollout_index}.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")


class ComparePairedHelpersTest(unittest.TestCase):
    def test_loads_aggregates_and_pairs_intersecting_seed_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            trained = root / "trained"
            base.mkdir()
            trained.mkdir()

            _write_meta(
                base,
                world_seed=1,
                rollout_index=0,
                final_floor=10,
                stopped_reason="max_decisions",
                n_invalid=1,
                n_decisions=10,
            )
            _write_meta(
                base,
                world_seed=2,
                rollout_index=0,
                outcome="GameOutcome.VICTORY",
                final_floor=20,
                n_decisions=10,
            )
            _write_meta(
                base,
                world_seed=2,
                rollout_index=1,
                final_act=2,
                final_floor=30,
                stopped_reason="agent_invalid",
                n_invalid=2,
                n_decisions=20,
            )
            _write_meta(
                base,
                world_seed=3,
                rollout_index=0,
                final_floor=99,
                n_decisions=0,
            )

            _write_meta(
                trained,
                world_seed=1,
                rollout_index=0,
                final_act=2,
                final_floor=12,
                n_decisions=5,
            )
            _write_meta(
                trained,
                world_seed=2,
                rollout_index=0,
                outcome="GameOutcome.VICTORY",
                final_floor=18,
                n_invalid=1,
                n_decisions=5,
            )
            _write_meta(
                trained,
                world_seed=2,
                rollout_index=1,
                final_floor=28,
                stopped_reason="max_decisions",
                n_decisions=5,
            )
            _write_meta(
                trained,
                world_seed=4,
                rollout_index=0,
                final_floor=40,
                n_invalid=3,
                n_decisions=15,
            )

            base_metas = load_metas(base)
            trained_metas = load_metas(trained)

            self.assertEqual(
                per_seed_metric_means(base_metas, "final_floor"),
                {1: 10.0, 2: 25.0, 3: 99.0},
            )
            self.assertEqual(
                per_seed_metric_means(trained_metas, "final_floor"),
                {1: 12.0, 2: 23.0, 4: 40.0},
            )

            base_agg = aggregate_arm(base_metas, min_act=1)
            self.assertEqual(base_agg["n_rollouts"], 4)
            self.assertEqual(base_agg["win_rate"], 0.25)
            self.assertEqual(base_agg["act_boss_clear_rate"], 0.5)
            self.assertEqual(base_agg["mean_floor"], 39.75)
            self.assertEqual(base_agg["agent_invalid_rate"], 3 / 40)
            self.assertEqual(base_agg["mean_decisions"], 10.0)
            self.assertEqual(base_agg["budget_truncated_rate"], 0.25)

            trained_agg = aggregate_arm(trained_metas, min_act=1)
            self.assertEqual(trained_agg["n_rollouts"], 4)
            self.assertEqual(trained_agg["win_rate"], 0.25)
            self.assertEqual(trained_agg["act_boss_clear_rate"], 0.5)
            self.assertEqual(trained_agg["mean_floor"], 24.5)
            self.assertEqual(trained_agg["agent_invalid_rate"], 4 / 30)
            self.assertEqual(trained_agg["mean_decisions"], 7.5)
            self.assertEqual(trained_agg["budget_truncated_rate"], 0.25)

            report = build_report(base, trained, n_resamples=200, bootstrap_seed=7)
            self.assertEqual(report["paired"]["n_seeds"], 2)
            self.assertEqual(report["paired"]["deltas_by_seed"], {1: 2.0, 2: -2.0})
            self.assertEqual(report["paired"]["mean_delta"], 0.0)

    def test_load_metas_descends_into_agent_label_subdir(self) -> None:
        # run_until writes sidecars under output_dir/<agent_label>/, so load_metas
        # must recurse — a non-recursive glob at the arm root silently finds zero.
        with tempfile.TemporaryDirectory() as tmp:
            arm = Path(tmp) / "base"
            label_dir = arm / "vllm_gemma_4_E4B_it_thinking_8192"
            label_dir.mkdir(parents=True)
            _write_meta(label_dir, world_seed=47, rollout_index=0, final_floor=12)
            _write_meta(label_dir, world_seed=48, rollout_index=0, final_floor=15)

            metas = load_metas(arm)
            self.assertEqual(len(metas), 2)
            self.assertEqual({m["world_seed"] for m in metas}, {47, 48})

    def test_format_report_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            trained = root / "trained"
            base.mkdir()
            trained.mkdir()
            _write_meta(base, world_seed=1, rollout_index=0, final_floor=10)
            _write_meta(trained, world_seed=1, rollout_index=0, final_floor=12)

            report = build_report(base, trained, n_resamples=200, bootstrap_seed=7)
            rendered = format_report(report)
            self.assertIn("Paired base-vs-trained eval", rendered)
            self.assertIn("paired_seeds: 1", rendered)
            # Per-arm rows and the sign-test line are present in the rendered table.
            self.assertIn("agent_invalid_rate", rendered)
            self.assertIn("sign-test:", rendered)
            self.assertGreater(len(rendered.splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
