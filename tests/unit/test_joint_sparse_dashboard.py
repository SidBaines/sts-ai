import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "joint_dashboard",
    Path(__file__).resolve().parents[2] / "scripts/joint_sparse_dashboard.py",
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class JointDashboardTests(unittest.TestCase):
    def test_filters_pagination_and_private_fault_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with gzip.open(out / "collection_0001.json.gz", "wt") as f:
                json.dump(
                    {
                        "episodes": [
                            dict(
                                world_seed=1,
                                iteration=1,
                                act=3,
                                classification="win",
                                decision_counts={"combat:act3": 2},
                                action_prefix=[99],
                                native_provenance="private",
                            ),
                            dict(
                                world_seed=2,
                                iteration=1,
                                act=1,
                                classification="loss",
                                decision_counts={"ooc:act1": 1},
                            ),
                        ],
                        "faults": [
                            dict(
                                world_seed=3,
                                error="SimulatorError",
                                fault_record_path="/private/fault.json",
                                message="sensitive",
                            )
                        ],
                    },
                    f,
                )
            (out / "log.jsonl").write_text(
                json.dumps(
                    {
                        "event": "update",
                        "iteration": 1,
                        "accepted": True,
                        "phases": {"combat": {"entropy": 0.3}},
                    }
                )
                + "\n"
                + "{unfinished"
            )
            t = dashboard.Telemetry(out)
            all_ = t.snapshot({})
            self.assertEqual(all_["summary"]["filtered"], 3)
            self.assertEqual(all_["summary"]["valid"], 2)
            self.assertEqual(all_["summary"]["win_rate_valid"], 0.5)
            self.assertNotIn("private", json.dumps(all_))
            self.assertNotIn("sensitive", json.dumps(all_))
            self.assertIn("phases.combat.entropy", all_["metric_keys"])
            filtered = t.snapshot({"phase": ["combat"], "classification": ["win"]})
            self.assertEqual(filtered["summary"]["filtered"], 1)
            page = t.snapshot({"limit": ["1"], "offset": ["1"]})
            self.assertEqual(page["episodes"][0]["world_seed"], 2)
            with self.assertRaises(ValueError):
                t.snapshot({"offset": ["garbage"]})

    def test_missing_is_null_and_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            result = dashboard.Telemetry(d).snapshot({})
            self.assertIsNone(result["summary"]["win_rate_valid"])
            self.assertIsNone(result["baseline"])


if __name__ == "__main__":
    unittest.main()
