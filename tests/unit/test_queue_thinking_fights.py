from __future__ import annotations

import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import queue_thinking_fights as queue
from scripts import sample_fight_bank as sample
from sts_ai import fight_lab as lab
from tests.unit.test_fight_lab import record


class ThinkingQueueTest(unittest.TestCase):
    def setup_run(self, root: Path) -> tuple[Path, Path, dict]:
        bank, after, output = root / "bank", root / "after", root / "thinking"
        rows = lab.index_records([record()], "seed_1004_r0.jsonl")
        lab.write_new(bank / "collection.json", {"states": rows})
        sample.prepare(bank, after, combat_observation="combat_public_v4")
        manifest = lab.read_json(after / "manifest.json")
        sample.progress(after, manifest, "running")
        queue.prepare(after, output)
        return after, output, manifest

    def finish(self, after: Path, manifest: dict) -> None:
        state_id = manifest["order"][0]
        sample.atomic(after / f"{state_id}.result.json", {
            "state_id": state_id, "status": "timeout"})
        sample.progress(after, manifest, "finished")

    def test_only_thinking_changes_and_entire_scheduled_cohort_is_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            after, output, original = self.setup_run(Path(directory))
            manifest, selected, settings = sample.validate(output)
            previous = lab.read_json(after / "settings.json")
            self.assertEqual(settings, {**previous, "thinking": True})
            self.assertEqual(selected, lab.read_json(after / "selection.json"))
            for name in ("order", "sampling_seeds", "implementation", "runtime_versions"):
                self.assertEqual(manifest[name], original[name])
            self.assertFalse(queue.ready(output))

    def test_waits_for_finished_status_all_results_and_lock_release(self):
        with tempfile.TemporaryDirectory() as directory:
            after, output, manifest = self.setup_run(Path(directory))
            self.finish(after, manifest)
            with (after / "run.lock").open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(queue.ready(output))
            self.assertTrue(queue.ready(output))
            state_id = manifest["order"][0]
            (after / f"{state_id}.result.json").unlink()
            self.assertFalse(queue.ready(output))
            self.finish(after, manifest)
            for state in ("paused", "error", "running"):
                sample.progress(after, manifest, state)
                self.assertFalse(queue.ready(output))

    def test_changed_predecessor_configuration_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            after, output, _ = self.setup_run(Path(directory))
            settings = lab.read_json(after / "settings.json")
            settings["max_tokens"] = 99
            sample.atomic(after / "settings.json", settings)
            with self.assertRaisesRegex(ValueError, "predecessor configuration changed"):
                queue.ready(output)

    def test_pending_stop_never_launches_model(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, _ = self.setup_run(Path(directory))
            (output / "STOP").touch()
            with patch.object(queue.subprocess, "Popen") as launch:
                with patch.object(queue.signal, "signal"):
                    queue.watch(output)
            launch.assert_not_called()
            self.assertEqual(lab.read_json(output / "queue_status.json")["status"], "cancelled")

    def test_finished_predecessor_launches_existing_resumable_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            after, output, manifest = self.setup_run(Path(directory))
            self.finish(after, manifest)

            def launched(*args, **kwargs):
                sample.progress(output, manifest, "paused")
                class Child:
                    returncode = 0

                    def poll(self):
                        return 0
                return Child()

            with patch.object(queue.subprocess, "Popen", side_effect=launched) as launch:
                with patch.object(queue.signal, "signal"):
                    queue.watch(output)
            self.assertEqual(launch.call_args.args[0][1:],
                             [str(sample.SCRIPT), "resume", "--output", str(output)])
            self.assertEqual(lab.read_json(output / "queue_status.json")["status"], "paused")
