from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.monitor_fight_runs import refresh_checkpoint, terminal_results
from sts_ai import fight_lab as lab


class FightMonitorTest(unittest.TestCase):
    def test_refreshes_first_fights_completion_and_budget_changes(self):
        state = {"arms": {
            "new_no_thinking": {"progress": {"finished": 200, "status": "running"}},
            "new_thinking": {"progress": {"finished": 0},
                             "budget_status.json": {"max_tokens": 16384}}}}
        previous = refresh_checkpoint(state, False)
        state["arms"]["new_no_thinking"]["progress"].update(finished=202, status="finished")
        self.assertNotEqual(previous, refresh_checkpoint(state, False))
        previous = refresh_checkpoint(state, False)
        for n in range(1, 6):
            state["arms"]["new_thinking"]["progress"]["finished"] = n
            current = refresh_checkpoint(state, False)
            self.assertNotEqual(previous, current)
            previous = current
        state["arms"]["new_thinking"]["progress"]["finished"] = 9
        self.assertEqual(previous, refresh_checkpoint(state, False))
        state["arms"]["new_thinking"]["progress"]["finished"] = 10
        self.assertNotEqual(previous, refresh_checkpoint(state, False))
        previous = refresh_checkpoint(state, False)
        state["arms"]["new_thinking"]["budget_status.json"]["max_tokens"] = 32768
        self.assertNotEqual(previous, refresh_checkpoint(state, False))
        self.assertNotEqual(refresh_checkpoint(state, False), refresh_checkpoint(state, True))

    def test_finished_requires_all_result_files_and_both_new_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {a: root / a for a in ("new_no_thinking", "new_thinking")}
            state = {"arms": {a: {"progress": {"status": "finished", "finished": 1}} for a in runs}}
            for path in runs.values():
                lab.write_new(path / "manifest.json", {"order": ["one"]})
            self.assertFalse(terminal_results(runs, state))
            for path in runs.values():
                lab.write_new(path / "one.result.json", {})
            self.assertTrue(terminal_results(runs, state))
            state["arms"]["new_thinking"]["queue_status.json"] = {"status": "running"}
            self.assertFalse(terminal_results(runs, state))
            state["arms"]["new_thinking"].pop("queue_status.json")
            state["arms"]["new_thinking"]["progress"]["status"] = "running"
            self.assertFalse(terminal_results(runs, state))
