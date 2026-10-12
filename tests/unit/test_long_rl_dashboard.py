from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/long_rl_dashboard.py"
spec = importlib.util.spec_from_file_location("long_rl_dashboard", MODULE_PATH)
dashboard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dashboard
spec.loader.exec_module(dashboard)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.campaign = Path(self.temp.name)
        (self.campaign / "checkpoints").mkdir()
        self.metric_path = self.campaign / "checkpoints/chunk.metrics.json"
        self.metrics = {"elapsed_seconds": 4, "rounds": [{"elapsed_seconds": 2, "transitions": 10,
            "task_ids": ["a", "b"], "cohort_completed": True, "episodes": [
                {"task_id": "a", "reward": 1, "floor": 17, "truncated": False},
                {"task_id": "b", "reward": -1, "floor": 8, "truncated": False}],
            "faults": [], "update": {"steps": 3, "elapsed_seconds": 1, "policy_loss": .2,
                "config": {"max_grad_norm": .5}}}]}
        self.metric_path.write_text(json.dumps(self.metrics))

    def test_logged_values_missing_metrics_and_exact_coverage(self):
        row = dashboard.normalize_training(self.metrics, {}, 0)
        self.assertEqual(row["transitions_per_second"], 2.5)
        self.assertEqual(row["rollout_transitions_per_second"], 5)
        self.assertEqual(row["return_mean"], 0)
        self.assertEqual(row["return_q10"], -.8)
        self.assertEqual(row["policy_loss"], .2)
        self.assertIsNone(row["gradient_norm"])
        self.assertIsNone(row["entropy"])
        self.assertTrue(row["coverage"]["exact_match"])
        self.metrics["rounds"][0]["episodes"][1]["task_id"] = "a"
        row = dashboard.normalize_training(self.metrics, {}, 0)
        self.assertFalse(row["coverage"]["exact_match"])
        self.assertEqual(row["coverage"]["missing_ids"], 1)
        self.assertEqual(row["coverage"]["unexpected_ids"], 1)

    def test_curated_paths_and_symlink_escape(self):
        self.assertEqual(dashboard.curated_metric_path(self.campaign, str(self.metric_path)), self.metric_path.resolve())
        for path in ("../../secret.metrics.json", "checkpoints/model.pt", "checkpoints/final.metrics.json", "/tmp/private.metrics.json"):
            self.assertIsNone(dashboard.curated_metric_path(self.campaign, path))
        outside = self.campaign / "outside.metrics.json"
        outside.write_text("{}")
        symlink = self.campaign / "checkpoints/link.metrics.json"
        symlink.symlink_to(outside)
        self.assertIsNone(dashboard.curated_metric_path(self.campaign, str(symlink)))

    def test_mtime_cache_and_live_job_metadata(self):
        cache = dashboard.CachedJSON()
        first = cache.read(self.metric_path)
        self.assertIs(cache.read(self.metric_path), first)
        self.assertEqual(cache.reads, 1)
        self.metric_path.write_text(json.dumps({"changed": True}))
        self.assertEqual(cache.read(self.metric_path), {"changed": True})
        self.assertEqual(cache.reads, 2)
        self.metric_path.write_text(json.dumps(self.metrics))
        store = dashboard.TelemetryStore(self.campaign)
        job = {"metrics": str(self.metric_path), "status": "running"}
        self.assertIsNone(store.metric(job, 0)["actor_changed"])
        job.update(status="complete", finished=123, actor_evidence={"before": {"actor_sha256": "a"}, "after": {"actor_sha256": "b"}})
        row = store.metric(job, 0)
        self.assertTrue(row["actor_changed"])
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["finished"], 123)

    def test_paused_allocation_does_not_extrapolate_and_families_separate(self):
        family1 = json.dumps({"start_source": "fresh_scratch", "goal_act": 1, "start_act": 1})
        family2 = json.dumps({"start_source": "native_teacher", "goal_act": 2, "start_act": 2})
        journal = {"status": "paused", "allocation": {"used_seconds": 100, "authorized_seconds": 36000},
                   "jobs": [{"kind": "train", "metrics": str(self.metric_path), "family_strata": {
                       family1: {"requested": 10, "clears": 10}, family2: {"requested": 10, "clears": 1}}}]}
        (self.campaign / "journal.json").write_text(json.dumps(journal))
        (self.campaign / "progress.json").write_text("{}")
        (self.campaign / "config.json").write_text("{}")
        result = dashboard.TelemetryStore(self.campaign).get()
        self.assertEqual(result["active_used_seconds"], 100)
        self.assertEqual(result["active_remaining_seconds"], 35900)
        self.assertEqual(len(result["families"]), 2)
        self.assertFalse(result["families"][0]["conditional"])
        self.assertTrue(result["families"][1]["conditional"])
        self.assertEqual(result["families"][1]["clear_rate"], .1)

    def test_explicit_missing_dev_quality_is_unavailable(self):
        journal = {"current_goal": None, "events": [{"round": 2, "quality": None,
                   "retention": None, "decision": {"action": "hold", "reason": "incomplete_development"}}]}
        for name, value in (("journal.json", journal), ("progress.json", {}), ("config.json", {})):
            (self.campaign / name).write_text(json.dumps(value))
        result = dashboard.TelemetryStore(self.campaign).get()
        self.assertIsNone(result["stage"])
        self.assertIsNone(result["selected_dev"])
        self.assertIsNone(result["decisions"][0]["quality"])
        self.assertIsNone(result["decisions"][0]["goal_act"])

    def test_full_run_missing_is_not_zero_and_roles_bind_exact_sha(self):
        missing = dashboard.normalize_full_run(None, {})
        self.assertIsNone(missing["latest"]["selected_best"])
        self.assertIsNone(missing["latest"]["latest_candidate"])
        self.assertEqual(missing["history"], [])
        ids = [str(i) for i in range(48)]
        sha = "a" * 64
        record = {"roles": ["selected_best", "latest_candidate"], "checkpoint_sha256": sha,
                  "capture_at": 100, "evaluated_at": 110, "captured_global_round": 14,
                  "requested": 48, "clears": 1, "defeats": 47, "faults": 0, "truncated": 0,
                  "unstarted": 0, "complete": True, "floor_mean": 30, "task_ids": ids}
        checkpoints = {"best_checkpoint": {"sha256": sha}, "last_checkpoint": {"sha256": "b" * 64}}
        result = dashboard.normalize_full_run({"monitoring_only": True, "panel": "dev_act3", "histories": [record, record]}, checkpoints, ids)
        self.assertEqual(len(result["history"]), 1)
        row = result["latest"]["selected_best"]
        self.assertTrue(row["complete"])
        self.assertTrue(row["matches_live_role"]["selected_best"])
        self.assertFalse(row["matches_live_role"]["latest_candidate"])
        self.assertEqual(row["clear_rate"], 1 / 48)
        self.assertEqual(row["captured_global_round"], 14)
        self.assertIs(result["latest"]["selected_best"], result["latest"]["latest_candidate"])

    def test_full_run_alias_can_reuse_old_exact_sha_after_rollback(self):
        ids = [str(i) for i in range(48)]
        def record(sha, stamp):
            return {"roles": ["latest_candidate"], "checkpoint_sha256": sha * 64,
                    "evaluated_at": stamp, "captured_at": stamp-10, "captured_global_round": stamp,
                    "requested": 48, "clears": 0, "defeats": 48, "faults": 0, "truncated": 0,
                    "unstarted": 0, "complete": True, "floor_mean": 28, "task_ids": ids}
        old, newer = record("a", 100), record("b", 200)
        alias = dict(old, role="latest_candidate", role_capture_at=300,
                     role_captured_global_round=300, reused_evaluation=True)
        result = dashboard.normalize_full_run({"monitoring_only": True, "histories": [old, newer],
                    "latest": {"latest_candidate": alias}}, {"last_checkpoint": {"sha256": "a"*64}}, ids)
        row = result["latest"]["latest_candidate"]
        self.assertEqual(row["evaluated_at"], 100)
        self.assertEqual(row["role_capture_at"], 300)
        self.assertTrue(row["reused_evaluation"])
        self.assertTrue(row["matches_live_role"]["latest_candidate"])
        self.assertEqual(len(result["history"]), 2)

    def test_full_run_exact_cohort_and_incomplete_outcomes(self):
        ids = [str(i) for i in range(48)]
        row = {"role": "selected_best", "checkpoint_sha256": "a" * 64, "evaluated_at": 110,
               "requested": 48, "clears": 0, "defeats": 48, "faults": 0, "truncated": 0,
               "unstarted": 0, "complete": True, "floor_mean": 28, "task_ids": ids[:-1] + [ids[0]]}
        result = dashboard.normalize_full_run({"monitoring_only": True, "histories": [row]}, {}, ids)
        self.assertFalse(result["history"][0]["matched_panel"])
        self.assertFalse(result["history"][0]["complete"])
        row.update(task_ids=ids, defeats=47, unstarted=1)
        result = dashboard.normalize_full_run({"monitoring_only": True, "histories": [row]}, {}, ids)
        self.assertTrue(result["history"][0]["matched_panel"])
        self.assertFalse(result["history"][0]["complete"])
        row["checkpoint_sha256"] = "short"
        result = dashboard.normalize_full_run({"monitoring_only": True, "histories": [row]}, {}, ids)
        self.assertEqual(result["history"], [])
        self.assertTrue(result["errors"])
        self.assertFalse(dashboard.normalize_full_run({"monitoring_only": False, "histories": [row]}, {}, ids)["history"])

    def test_full_run_owned_path_cache_and_initial_reference(self):
        observer = self.campaign.parent / "full_run_eval"
        # Use a nested owned root so this fixture cannot collide with another test.
        study = self.campaign / "study"
        campaign = study / "campaign"
        campaign.mkdir(parents=True)
        observer = study / "full_run_eval"
        observer.mkdir()
        for name, value in (("journal.json", {}), ("progress.json", {}), ("config.json", {"initial_checkpoint_sha256": "f" * 64})):
            (campaign / name).write_text(json.dumps(value))
        telemetry = observer / "telemetry.json"
        telemetry.write_text(json.dumps({"monitoring_only": True, "histories": []}))
        store = dashboard.TelemetryStore(campaign)
        first = store.get()
        self.assertEqual(first["full_run"]["initial"]["clears"], 0)
        self.assertEqual(first["full_run"]["initial"]["requested"], 48)
        self.assertIsNone(first["full_run"]["latest"]["selected_best"])
        reads = store.cache.reads
        store.get()
        self.assertEqual(store.cache.reads, reads)
        outside = study / "unrelated.json"
        outside.write_text('{"monitoring_only":true}')
        telemetry.unlink()
        telemetry.symlink_to(outside)
        self.assertIsNone(store.observer_telemetry())
        self.assertIn("Rejected observer telemetry symlink", store.errors[-1])

    def test_http_smoke_routes_read_only(self):
        for name in ("journal.json", "progress.json", "config.json"):
            (self.campaign / name).write_text("{}")
        server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler_for(dashboard.TelemetryStore(self.campaign)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/telemetry") as response:
            result = json.loads(response.read())
            self.assertEqual(result["schema"], 1)
            self.assertIn("Content-Security-Policy", response.headers)
        with urlopen(base + "/") as response:
            self.assertIn(b"<!doctype html>", response.read())
        with self.assertRaises(HTTPError) as error:
            urlopen(base + "/api/file?path=journal.json")
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
