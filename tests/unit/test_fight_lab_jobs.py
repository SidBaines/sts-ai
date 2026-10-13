from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from sts_ai import fight_lab as lab
from sts_ai.fight_lab_jobs import FightJobs, RunBusyError
from tests.unit.test_fight_lab import record


class FightJobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.collection = {"states": lab.index_records(
            [record(0, combat=False), record(1), record(2)], "seed_1004_r0.jsonl"),
            "sources": [], "config": {}, "reports": []}
        lab.write_new(self.root / "collection.json", self.collection)
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.close()
        self.tmp.cleanup()

    def manager(self, code="import time; time.sleep(30)", timeout=10):
        def command(selected, output):
            return [sys.executable, "-c", code, str(selected), str(output)]
        manager = FightJobs(self.root, self.collection, command_builder=command, timeout=timeout)
        self.managers.append(manager)
        return manager

    def wait_job(self, manager, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = manager.get(job_id)
            if job["status"] != "running":
                return job
            time.sleep(.01)
        self.fail("job did not finish")

    def test_single_target_busy_cancel_and_restart_persistence(self):
        manager = self.manager()
        job = manager.start("seed_1004_r0_d2")
        selected = lab.read_json(manager.directory / job["job_id"] / "selection.json")
        self.assertEqual([r["index"] for r in selected["states"]], [2])
        lab.validate_selection(self.collection, selected)
        with self.assertRaises(RunBusyError):
            manager.start("seed_1004_r0_d1")
        self.assertEqual(manager.cancel(job["job_id"])["status"], "cancelled")
        self.assertIsNone(manager.active)
        manager.close()
        restored = self.manager()
        self.assertEqual(restored.get(job["job_id"])["status"], "cancelled")

    def test_noncombat_and_unknown_state_rejected(self):
        manager = self.manager()
        for state_id in ("seed_1004_r0_d0", "unknown"):
            with self.assertRaises(ValueError):
                manager.start(state_id)
        self.assertFalse(manager.list())

    def test_timeout_and_shutdown_stop_workers(self):
        manager = self.manager(timeout=.05)
        job = manager.start("seed_1004_r0_d1")
        process = manager.process
        self.assertEqual(self.wait_job(manager, job["job_id"])["status"], "timeout")
        self.assertIsNotNone(process.poll())
        manager.close()
        other = self.manager()
        job = other.start("seed_1004_r0_d1")
        process = other.process
        other.close()
        self.assertIsNotNone(process.poll())
        self.assertEqual(other.get(job["job_id"])["status"], "cancelled")

    def test_completion_discovery_progress_and_worker_errors(self):
        code = '''import json, pathlib, sys
root=pathlib.Path(sys.argv[2]);root.mkdir(parents=True)
(root/'fight.jsonl').write_text('{}\\n{}\\n')
(root/'fight.meta.json').write_text('{}')
(root/'report.json').write_text(json.dumps({'completed':1,'episodes':[{'status':'ok','trace':'fight.jsonl','metrics':{'won':True}}]}))
'''
        manager = self.manager(code)
        job = self.wait_job(manager, manager.start("seed_1004_r0_d1")["job_id"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["decisions"], 2)
        self.assertTrue(job["trajectory"].endswith("fight.jsonl"))
        manager.close()
        manager = self.manager("raise ValueError('replay diverged')")
        job = self.wait_job(manager, manager.start("seed_1004_r0_d1")["job_id"])
        self.assertEqual(job["status"], "error")
        self.assertIn("replay diverged", job["error"])

    @unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "requires app extra")
    def test_api_start_poll_cancel_and_exact_export(self):
        from fastapi.testclient import TestClient
        from sts_ai.fight_lab_server import create_app
        with TestClient(create_app(self.root, jobs=self.manager())) as client:
            response = client.post("/api/runs", json={"state_id": "seed_1004_r0_d2"})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["job_id"]
            self.assertEqual(client.get(f"/api/runs/{job_id}").json()["status"], "running")
            self.assertEqual(client.get("/api/collection").status_code, 200)
            self.assertEqual(client.get("/api/runs").json()[0]["job_id"], job_id)
            self.assertEqual(client.post("/api/runs", json={"state_id": "seed_1004_r0_d1"}).status_code, 409)
            self.assertEqual(client.post("/api/runs", json={"state_id": "seed_1004_r0_d0"}).status_code, 400)
            exported = client.get("/api/states?state_id=seed_1004_r0_d2&download=true")
            self.assertEqual([r["index"] for r in exported.json()["states"]], [2])
            self.assertEqual(client.post(f"/api/runs/{job_id}/cancel").json()["status"], "cancelled")
            self.assertEqual(client.get("/api/runs/missing").status_code, 404)
