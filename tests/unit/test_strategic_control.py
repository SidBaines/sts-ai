from __future__ import annotations

import unittest
import hashlib
from tempfile import TemporaryDirectory
from pathlib import Path

from sts_policy.strategic_control import power_record, resource_gate, paired_retention, retention_gate, unchanged_sources


class StrategicControlTests(unittest.TestCase):
    def test_loaded_code_identity_rejects_changed_or_deleted_sources(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"model.py"
            path.write_text("original")
            identity = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertTrue(unchanged_sources(identity))
            path.write_text("replacement")
            self.assertFalse(unchanged_sources(identity))
            path.unlink()
            self.assertFalse(unchanged_sources(identity))
    def test_ac_presence_does_not_hide_discharging_battery(self):
        record = power_record("Now drawing from 'AC Power'\n 24%; discharging; 1:30 remaining")
        record["disk_free_bytes"] = 20*1024**3
        self.assertTrue(record["ac_present"])
        self.assertFalse(record["charging"])
        self.assertEqual(resource_gate(record), "battery_reserve")
        record = power_record("Now drawing from 'AC Power'\n 24%; charging; 1:30 remaining")
        record["disk_free_bytes"] = 20*1024**3
        self.assertIsNone(resource_gate(record))
        record["disk_free_bytes"] = 7*1024**3
        self.assertEqual(resource_gate(record), "storage_reserve")
        record = power_record("Now drawing from 'AC Power'\n 24%; AC attached; not charging")
        record["disk_free_bytes"] = 20*1024**3
        self.assertFalse(record["charging"])
        self.assertEqual(resource_gate(record), "battery_reserve")

    def test_paired_retention_preserves_win_gains_and_flags_missing_games(self):
        def episode(i, act=2, outcome="defeat", floor=20):
            return {"task_id": str(i), "terminated": True, "truncated": False, "task_outcome": outcome,
                    "max_act_reached": act, "floor": floor, "action_trace_sha256": str(act)+outcome}
        baseline = {"episodes": [episode(i) for i in range(10)]}
        candidate = {"episodes": [episode(i, act=3, outcome="victory", floor=51) if i == 0 else episode(i) for i in range(10)]}
        paired = paired_retention(baseline, candidate)
        self.assertEqual(paired["victory"], {"gains": 1, "losses": 0, "net": 1, "reference": 0, "candidate": 1})
        self.assertIsNone(retention_gate(paired, 10))
        candidate["episodes"].pop()
        self.assertEqual(retention_gate(paired_retention(baseline, candidate), 10), "evaluation_incomplete")
        candidate = {"episodes": [episode(i, act=1) for i in range(10)]}
        self.assertEqual(retention_gate(paired_retention(baseline, candidate), 10), "act1_retention_loss")


if __name__ == "__main__":
    unittest.main()
