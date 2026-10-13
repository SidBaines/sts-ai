from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import queue_thinking_fights as queue
from scripts import sample_fight_bank as sample
from scripts import thinking_fight_budget as budget
from sts_ai import fight_lab as lab
from tests.unit import test_queue_thinking_fights as queue_tests


def response(tokens: int, valid: bool = True) -> dict:
    return {"agent": {"completion_tokens": tokens, "valid": valid, "metadata": {}}}


class ThinkingBudgetTest(unittest.TestCase):
    def test_threshold_waits_then_doubles_only_for_frequent_caps(self):
        limit = budget.LIMITS[0]
        self.assertEqual(budget.assessment([response(limit)] * 2, limit)["action"], "observe")
        self.assertEqual(budget.assessment([response(limit)] * 2 + [response(50)] * 3, limit)["action"], "raise")
        self.assertEqual(budget.assessment([response(limit)] + [response(50)] * 19, limit)["action"], "accept")
        self.assertEqual(budget.assessment([response(5000)] * 20, limit)["cap_hits"], 0)
        self.assertEqual(budget.assessment([response(budget.LIMITS[-1])] * 20,
                                          budget.LIMITS[-1])["action"], "ceiling_reached")

    def test_incomplete_near_cap_counts_but_short_format_failure_does_not(self):
        self.assertTrue(budget.cap_hit(response(16380, False), 16384))
        self.assertFalse(budget.cap_hit(response(16380, True), 16384))
        self.assertFalse(budget.cap_hit(response(100, False), 16384))

    def test_full_cohort_restarts_and_keeps_calibration_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            after, output, original = queue_tests.ThinkingQueueTest().setup_run(Path(directory))
            budget.configure(output)
            queue.validate_dependency(output)
            manifest, selection, settings = budget.validated(output)
            self.assertEqual(settings["max_tokens"], 16384)
            self.assertGreater(manifest["stall_seconds"], 16384 / 10)
            state_id = manifest["order"][0]
            trace = output / f"{state_id}_r0.jsonl"
            text = json.dumps(response(16384, False)) + "\n"
            trace.write_text(text)
            sample.atomic(output / f"{state_id}.result.json", {"state_id": state_id, "status": "timeout"})
            budget.raise_budget(output, {"action": "raise", "generations": 5, "cap_hits": 2})
            new, selected, cfg = budget.validated(output)
            queue.validate_dependency(output)
            self.assertEqual(cfg["max_tokens"], 32768)
            self.assertEqual(selected, selection)
            self.assertEqual(new["order"], original["order"])
            self.assertEqual(new["sampling_seeds"], original["sampling_seeds"])
            self.assertFalse(trace.exists())
            self.assertFalse(list(output.glob("*.result.json")))
            self.assertEqual(next((output / "budget_calibration").glob("*/*_r0.jsonl")).read_text(), text)
            self.assertEqual(lab.read_json(after / "settings.json")["max_tokens"], 4096)

    def test_reader_ignores_only_the_partial_last_append(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, manifest = queue_tests.ThinkingQueueTest().setup_run(Path(directory))
            path = output / f"{manifest['order'][0]}_r0.jsonl"
            path.write_text(json.dumps(response(100)) + '\n{"unfinished"')
            self.assertEqual(budget.observations(output), [response(100)])

    def test_result_counts_actual_budget_not_frozen_4096_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            lab.write_new(output / "a_r0.meta.json", {"max_tokens": 16384})
            (output / "a_r0.jsonl").write_text(json.dumps(response(5000)) + "\n")
            with patch.object(budget, "_base_result", return_value={"generation_cap_hits": 1}):
                result = budget.result(output, "a", 1.0)
            self.assertEqual(result["generation_cap_hits"], 0)
            self.assertEqual(result["max_tokens"], 16384)

    def test_queue_stops_archives_and_relaunches_at_higher_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = queue_tests.ThinkingQueueTest()
            after, output, manifest = setup.setup_run(Path(directory))
            setup.finish(after, manifest)
            budget.configure(output)
            limits = []
            state_id = manifest["order"][0]

            def launch(*args, **kwargs):
                limit = lab.read_json(output / "settings.json")["max_tokens"]
                limits.append(limit)
                first = len(limits) == 1
                records = ([response(limit, False)] * 2 + [response(20)] * 3
                           if first else [response(20)] * 20)
                (output / f"{state_id}_r0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                if not first:
                    sample.atomic(output / f"{state_id}.result.json", {"state_id": state_id, "status": "timeout"})
                sample.progress(output, manifest, "running" if first else "finished")

                class Child:
                    returncode = None if first else 0

                    def poll(self):
                        return self.returncode

                    def send_signal(self, signal):
                        self.returncode = 0
                        sample.progress(output, manifest, "paused")

                    def wait(self, timeout):
                        return self.returncode
                return Child()

            with patch.object(queue.subprocess, "Popen", side_effect=launch):
                with patch.object(queue.signal, "signal"):
                    queue.watch(output)
            self.assertEqual(limits, [16384, 32768])
            self.assertEqual(lab.read_json(output / "budget_status.json")["action"], "accept")
            self.assertEqual(lab.read_json(output / "queue_status.json")["status"], "finished")
