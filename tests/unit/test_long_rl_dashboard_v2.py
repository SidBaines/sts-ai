from __future__ import annotations

from collections import Counter
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from tests.unit.test_long_rl_dashboard import dashboard


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V2DashboardTests(unittest.TestCase):
    def test_lineages_keep_learner_and_champion_separate(self):
        initial, first, second = "f"*64, "a"*64, "b"*64
        def job(checkpoint, digest, parent, steps):
            return {"kind": "train", "status": "complete", "checkpoint": checkpoint,
                    "artifacts": {checkpoint: digest}, "parent_sha256": parent,
                    "actor_evidence": {"updates": steps, "transitions": steps*10}}
        journal = {"identity": {"initial_checkpoint_sha256": initial},
                   "jobs": [job("a.pt", first, initial, 7), job("b.pt", second, first, 11)],
                   "learner": {"sha256": second, "cold_optimizer_count": 1, "optimizer_restore_count": 1, "rollback_count": 0},
                   "champion": {"sha256": first, "active_at": 100}, "allocation": {"used_seconds": 2000}}
        data = dashboard.normalize_v2(journal, {})
        self.assertEqual(data["learning"]["learner_lineage"]["updates"], 18)
        self.assertEqual(data["learning"]["champion_lineage"]["updates"], 7)
        self.assertEqual(data["learning"]["discarded_updates"], 0)
        self.assertEqual(data["learning"]["plateau_active_seconds"], 1900)
        self.assertTrue(data["alerts"])
        broken = dashboard.checkpoint_lineage(journal, "c"*64)
        self.assertFalse(broken["complete"])
        self.assertIsNone(broken["updates"])
        journal["jobs"][0]["parent_sha256"] = second
        self.assertFalse(dashboard.checkpoint_lineage(journal, second)["complete"])

    def test_independent_panels_do_not_pool_or_copy_old_baseline(self):
        screen, confirm = [str(i) for i in range(48)], [str(i+100) for i in range(192)]
        def q(ids, clears):
            return {"cohort": ids, "complete": True, "requested": len(ids), "clears": clears,
                    "floor_mean": 29, "faults": 0, "truncated": 0}
        journal = {"identity": {"initial_checkpoint_sha256": "f"*64}, "initial_panels": {
                     "screen_full3": q(screen, 1), "confirm_full3": q(confirm, 4)},
                   "evaluations": [{"role": "learner", "panel_id": "screen_full3", "round": 1,
                       "checkpoint_sha256": "a"*64, "quality": q(screen, 2)}]}
        data = dashboard.normalize_v2(journal, {"current_checkpoint": {"sha256": "b"*64}})
        self.assertEqual(data["panels"]["screen_full3"]["initial"]["clears"], 1)
        self.assertEqual(data["panels"]["confirm_full3"]["initial"]["clears"], 4)
        self.assertEqual(len(data["panels"]["screen_full3"]["history"]), 1)
        self.assertEqual(data["panels"]["confirm_full3"]["history"], [])
        self.assertIsNone(data["panels"]["retention_act1"]["initial"])
        self.assertFalse(data["panels"]["screen_full3"]["history"][0]["matches_live_role"])
        journal["evaluations"][0]["quality"] = q(confirm, 4)
        self.assertFalse(dashboard.normalize_v2(journal,{})["panels"]["screen_full3"]["history"][0]["complete"])

    def test_logged_health_fields_missing_are_not_invented(self):
        metrics = {"rounds": [{"task_ids": ["a", "b"], "cohort_completed": True, "episodes": [
            {"task_id": "a", "terminated": True, "end_hp": 0, "end_max_hp": 90,
             "start_hp": 75, "max_act_reached": 3, "termination_state": {"phase": "terminal", "room": "BOSS", "done": True}},
            {"task_id": "b", "terminated": True}], "faults": []}]}
        rows = dashboard.compact_episodes(metrics, {}, 1, {})
        self.assertEqual(rows[0]["start_hp"], 75)
        self.assertEqual(rows[0]["end_hp_fraction"], 0)
        self.assertEqual(rows[0]["max_act_reached"], 3)
        self.assertEqual(rows[0]["termination_phase"], "terminal")
        self.assertIsNone(rows[1]["start_hp"])
        self.assertIsNone(rows[1]["end_hp"])
        self.assertIsNone(rows[1]["max_act_reached"])

    def test_v2_episode_panel_and_stable_page_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve(); campaign = parent / "campaign"; (campaign/"metrics").mkdir(parents=True)
            task_path = parent / "screen_full3.jsonl"
            tasks = [{"seed": i, "split": "dev", "prefix": [], "max_act": 3, "horizon": 4096} for i in range(48)]
            task_path.write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in tasks))
            ids = [hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest() for row in tasks]
            checkpoint = str(parent / "candidate.pt")
            metric = campaign/"metrics/screen.metrics.json"
            metrics = {"config": {"checkpoint": checkpoint, "tasks": str(task_path), "seed": 777,
                       "max_act": 3, "horizon": 4096, "split": "dev", "simulations": 200},
                       "training_environment": {"native_module_sha256": "native"}, "rounds": [{"task_ids": ids,
                       "cohort_completed": True, "episodes": [{"task_id": task_id,"world_seed": index,
                       "outcome": "defeat", "terminated": True, "floor": 22, "end_hp": 0,
                       "end_max_hp": 80, "start_hp": 80, "max_act_reached": 2} for index,task_id in enumerate(ids)], "faults": []}]}
            metric.write_text(json.dumps(metrics))
            quality = {"cohort": ids, "clears": 0, "requested": 48, "floor_mean": 22,
                       "faults": 0, "truncated": 0, "complete": True}
            job = {"kind": "eval", "panel_id": "screen_full3", "role": "learner", "round": 5,
                   "metrics": str(metric), "status": "complete", "started": 1, "finished": 2,
                   "command": ["--tasks",str(task_path)], "signature": {"inputs": {
                   "--tasks": {"path":str(task_path),"sha256":sha(task_path)}, "--checkpoint": {"path":checkpoint,"sha256":"a"*64}}},
                   "artifacts": {str(metric):sha(metric)}}
            journal = {"version":"sts_long_run_v2", "initial_panels":{"screen_full3":quality}, "jobs":[job],
                       "identity":{"initial_checkpoint_sha256":"f"*64,"backend":{"provenance":{"module_sha256":"native"}}},
                       "learner":{"sha256":"a"*64},"champion":{"sha256":"f"*64}}
            for name,value in (("journal.json",journal),("progress.json",{}),("config.json",{})):
                (campaign/name).write_text(json.dumps(value))
            store=dashboard.TelemetryStore(campaign)
            first=store.episodes({"kind":["full3"],"panel_id":["screen_full3"],"limit":["20"]})
            second=store.episodes({"kind":["full3"],"panel_id":["screen_full3"],"limit":["28"],"offset":["20"]})
            self.assertEqual(first["total"],48)
            self.assertEqual(first["dataset_id"],second["dataset_id"])
            self.assertEqual(Counter(row["task_id"]for row in first["episodes"]+second["episodes"]),Counter(ids))
            self.assertEqual(first["episodes"][0]["end_hp"],0)
            self.assertEqual(first["episodes"][0]["roles"],["learner"])
            self.assertEqual(store.episodes({"kind":["full3"],"panel_id":["confirm_full3"]})["total"],0)

    def test_registered_campaigns_reject_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for name in ("journal.json","progress.json","config.json"):(path/name).write_text("{}")
            store=dashboard.TelemetryStore(path,condition_id="v2")
            server=ThreadingHTTPServer(("127.0.0.1",0),dashboard.handler_for(store,{"v2":store,"v1":store}))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                base=f"http://127.0.0.1:{server.server_port}"
                response=json.loads(urlopen(base+"/api/telemetry?campaign=v2").read())
                self.assertEqual(response["selected_campaign"],"v2")
                self.assertEqual(len(response["available_campaigns"]),2)
                with self.assertRaises(HTTPError)as error:urlopen(base+"/api/telemetry?campaign=/etc/passwd")
                self.assertEqual(error.exception.code,400)
            finally:server.shutdown();server.server_close()


if __name__=="__main__":unittest.main()
