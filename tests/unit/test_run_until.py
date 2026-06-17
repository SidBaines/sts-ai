from __future__ import annotations

import unittest

from scripts.run_until import generate_specs


class GenerateSpecsTest(unittest.TestCase):
    def test_returns_exactly_target_distinct_ascending_specs(self) -> None:
        specs = generate_specs(target=5, seed_start=10, excluded=set(), already_done=set())

        self.assertEqual(specs, [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0)])
        self.assertEqual(len({world_seed for world_seed, _ in specs}), 5)
        self.assertTrue(all(rollout_index == 0 for _, rollout_index in specs))

    def test_skips_excluded_and_already_done_until_target_is_met(self) -> None:
        specs = generate_specs(
            target=5,
            seed_start=1,
            excluded={2, 5},
            already_done={1, 4, 9},
        )

        self.assertEqual(specs, [(3, 0), (6, 0), (7, 0), (8, 0), (10, 0)])

    def test_overwrite_allows_already_done_but_still_skips_excluded(self) -> None:
        specs = generate_specs(
            target=4,
            seed_start=1,
            excluded={2},
            already_done={1, 3},
            overwrite=True,
        )

        self.assertEqual(specs, [(1, 0), (3, 0), (4, 0), (5, 0)])

    def test_rejects_nonpositive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "target must be >= 1"):
            generate_specs(target=0, seed_start=1)


if __name__ == "__main__":
    unittest.main()
