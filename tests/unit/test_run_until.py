from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_until import existing_rollout_specs, generate_specs, load_split_seeds


class GenerateSpecsTest(unittest.TestCase):
    def test_fresh_k_one_matches_old_behavior(self) -> None:
        specs = generate_specs(3, 5)

        self.assertEqual(specs, [(5, 0), (6, 0), (7, 0)])

    def test_fresh_k_two_expands_each_counted_seed(self) -> None:
        specs = generate_specs(2, 5, rollouts_per_seed=2)

        self.assertEqual(specs, [(5, 0), (5, 1), (6, 0), (6, 1)])

    def test_fresh_excludes_world_seed(self) -> None:
        specs = generate_specs(2, 5, excluded={6})

        self.assertEqual(specs, [(5, 0), (7, 0)])

    def test_fresh_skips_done_seed_and_resumes_partial_seed(self) -> None:
        specs = generate_specs(
            2,
            5,
            already_done_specs={(5, 0), (5, 1), (6, 0)},
            rollouts_per_seed=2,
        )

        self.assertEqual(specs, [(6, 1), (7, 0), (7, 1)])

    def test_fresh_overwrite_ignores_done_specs(self) -> None:
        specs = generate_specs(
            2,
            5,
            already_done_specs={(5, 0), (5, 1), (6, 0)},
            rollouts_per_seed=2,
            overwrite=True,
        )

        self.assertEqual(specs, [(5, 0), (5, 1), (6, 0), (6, 1)])

    def test_split_sorts_and_expands_seeds(self) -> None:
        specs = generate_specs(None, 1, seeds=[10, 12, 11], rollouts_per_seed=2)

        self.assertEqual(
            specs,
            [(10, 0), (10, 1), (11, 0), (11, 1), (12, 0), (12, 1)],
        )

    def test_split_removes_excluded_seed_and_done_pair(self) -> None:
        specs = generate_specs(
            None,
            1,
            excluded={12},
            already_done_specs={(11, 1)},
            seeds=[10, 12, 11],
            rollouts_per_seed=2,
        )

        self.assertEqual(specs, [(10, 0), (10, 1), (11, 0)])

    def test_rejects_nonpositive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "target must be >= 1"):
            generate_specs(target=0, seed_start=1)

    def test_rejects_nonpositive_rollouts_per_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollouts_per_seed must be >= 1"):
            generate_specs(1, 1, rollouts_per_seed=0)


class ExistingRolloutSpecsTest(unittest.TestCase):
    def test_parses_rollout_pairs_from_meta_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "seed_5_r0.meta.json").write_text("{}", encoding="utf-8")
            (out_dir / "seed_5_r1.meta.json").write_text("{}", encoding="utf-8")
            (out_dir / "seed_8_r2.meta.json").write_text("{}", encoding="utf-8")
            (out_dir / "seed_9_rx.meta.json").write_text("{}", encoding="utf-8")
            (out_dir / "seed_10_r0.jsonl").write_text("", encoding="utf-8")

            self.assertEqual(
                existing_rollout_specs(out_dir),
                {(5, 0), (5, 1), (8, 2)},
            )


class LoadSplitSeedsTest(unittest.TestCase):
    def test_returns_sorted_split_seed_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "frozen_seeds.json"
            config_path.write_text(
                json.dumps({"splits": {"dev": [12, 10, 11]}}),
                encoding="utf-8",
            )

            self.assertEqual(load_split_seeds(config_path, "dev"), [10, 11, 12])

    def test_errors_on_missing_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "frozen_seeds.json"
            config_path.write_text(json.dumps({"splits": {"dev": [1]}}), encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "eval"):
                load_split_seeds(config_path, "eval")


if __name__ == "__main__":
    unittest.main()
