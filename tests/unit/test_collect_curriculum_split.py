from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from sts_policy.curriculum import world_split


def load_script():
    path = Path(__file__).resolve().parents[2] / "scripts/collect_curriculum.py"
    spec = importlib.util.spec_from_file_location("collect_curriculum_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CollectionSplitTests(unittest.TestCase):
    def test_exact_requested_count_excludes_every_other_split(self):
        script = load_script()
        for split in ("train", "dev", "final"):
            worlds = script.selected_worlds(300000000, 32, split)
            self.assertEqual(len(worlds), 32)
            self.assertEqual(len(set(worlds)), 32)
            self.assertTrue(all(world_split(seed) == split for seed in worlds))
            self.assertEqual(worlds, script.selected_worlds(300000000, 32, split))

    def test_all_preserves_existing_contiguous_collection_and_bad_input_rejects(self):
        script = load_script()
        self.assertEqual(script.selected_worlds(1000, 4, "all"), [1000, 1001, 1002, 1003])
        for count, split in ((0, "train"), (1, "unknown")):
            with self.assertRaises(ValueError):
                script.selected_worlds(1000, count, split)


if __name__ == "__main__":
    unittest.main()
