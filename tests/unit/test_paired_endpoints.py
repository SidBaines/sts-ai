import json
import tempfile
import unittest
from pathlib import Path

from sts_rl.paired_endpoints import compare, digest


class PairedEndpointTests(unittest.TestCase):
    def test_missing_outcomes_keep_known_side_and_reject_wrong_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ck = root / "policy.pt"
            ck.write_bytes(b"policy")
            config = dict(world_start=10, n=6, processes=7, games_per_process=8)
            first, second = root / "first", root / "second"
            for path, outcomes in [(first, [True, False, False, None, True, None]),
                                   (second, [False, True, True, False, None, True])]:
                path.mkdir()
                records = [dict(world_seed=10+i, win=x, status="completed" if x is not None else "censored")
                           for i,x in enumerate(outcomes)]
                (path / "run.json").write_text(json.dumps(dict(config=config, native={"module_sha256":"one"}, checkpoints={"checkpoint":{"sha256":digest(ck)}})))
                summary = dict(records=records, completed=6-outcomes.count(None), wins=outcomes.count(True), faults=0,
                               cuts=0, censored=outcomes.count(None), mean_floor=40, requeues=0)
                (path / "summary.json").write_text(json.dumps(summary))
            result = compare(first, second, 10, "a", "b", ck, ck, 6)
            self.assertEqual(result["paired_completed"], 3)
            self.assertAlmostEqual(result["full_cohort_gain_bounds_pp"][0], -100/6)
            self.assertAlmostEqual(result["full_cohort_gain_bounds_pp"][1], 200/6)
            summary["records"][0]["world_seed"] = 99
            (second / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "roster"):
                compare(first, second, 10, "a", "b", ck, ck, 6)
