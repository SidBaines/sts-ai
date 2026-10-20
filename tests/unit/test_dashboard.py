import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rl_dashboard", ROOT / "scripts/rl/dashboard.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dashboard-test-", dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "default"
        self.extra = Path(self.tmp.name) / "extra"
        self.root.mkdir()
        self.extra.mkdir()
        self.runs_patch = patch.object(dashboard, "RUNS", self.root)
        self.runs_patch.start()
        self.addCleanup(self.runs_patch.stop)
        dashboard._CACHE.clear()
        self.addCleanup(dashboard._CACHE.clear)

    def log(self, root, name, records=None):
        path = root / name / "log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if records is None:
            records = [{"event": "start", "config": {"iterations": 24, "games": 3072}},
                       {"event": "train", "iteration": 1, "iteration_seconds": 480},
                       {"event": "eval", "iteration": 0, "n": 1024, "wins": 17, "mean_floor": 30.8}]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def queue(self, root, *jobs):
        (root / "queue.json").write_text(json.dumps({"runs": list(jobs)}))

    def job(self, root, name, out, status="running"):
        return {"name": name, "status": status, "cmd": ["python", "train.py", "--out", str(root / out)],
                "started": 900, "lane": "train"}

    def test_nested_discovery_keeps_direct_names_and_stops_at_two_levels(self):
        for name in ("ooc_r2_lam1", "batch/recipe", "batch/deeper/ignored", "_scratch/run", "batch/_scratch"):
            self.log(self.root, name)
        runs = dashboard.state()["runs"]
        self.assertEqual(set(runs), {"ooc_r2_lam1", "batch/recipe"})
        self.assertTrue(runs["ooc_r2_lam1"]["key"])
        self.assertEqual(runs["batch/recipe"]["name"], "batch/recipe")

    def test_extra_roots_merge_runs_queues_history_and_runners(self):
        self.log(self.root, "direct")
        self.log(self.extra, "batch/recipe")
        self.queue(self.root, self.job(self.root, "default-job", "direct"))
        self.queue(self.extra, self.job(self.extra, "recipe-job", "batch/recipe"),
                   self.job(self.extra, "next-job", "batch/next", "pending"),
                   self.job(self.extra, "cancelled-job", "batch/old", "cancelled"))
        default_status = {"heartbeat": 990, "on_battery": False}
        extra_status = {"heartbeat": 995, "on_battery": True}
        (self.root / "runner_status.json").write_text(json.dumps(default_status))
        (self.extra / "runner_status.json").write_text(json.dumps(extra_status))
        with patch.object(dashboard.time, "time", return_value=1000):
            state = dashboard.state(extra_roots=[self.extra])
        self.assertEqual(set(state["runs"]), {"direct", "batch/recipe"})
        self.assertEqual(state["runner"], default_status)
        self.assertEqual(state["runners"], [{"root": str(self.root), "status": default_status},
                                             {"root": str(self.extra), "status": extra_status}])
        jobs = {j["name"]: j for j in state["active"]}
        self.assertEqual(set(jobs), {"default-job", "recipe-job", "next-job"})
        self.assertEqual(jobs["recipe-job"]["run"], "batch/recipe")
        self.assertEqual(jobs["recipe-job"]["root"], str(self.extra))
        self.assertEqual(jobs["recipe-job"]["progress"], [1, 24])
        self.assertEqual(jobs["recipe-job"]["last_eval"]["wins"], 17)
        self.assertIsNone(jobs["next-job"]["run"])
        self.assertEqual(jobs["next-job"]["out"], "batch/next")
        self.assertEqual(state["history"], {"cancelled": 1})
        self.assertEqual(state["runs"]["batch/recipe"]["status"], "running")

    def test_duplicate_roots_and_names_do_not_drop_or_misassign_runs(self):
        self.log(self.root, "same")
        self.log(self.extra, "same")
        self.queue(self.root, self.job(self.root, "first", "same"))
        self.queue(self.extra, self.job(self.extra, "second", "same", "paused"))
        state = dashboard.state(extra_roots=[self.root, self.extra, self.extra])
        self.assertEqual(len(state["runners"]), 2)
        self.assertEqual(len(state["runs"]), 2)
        self.assertIn("same", state["runs"])
        for job in state["active"]:
            self.assertEqual(state["runs"][job["run"]]["root"], job["root"])
            self.assertEqual(state["runs"][job["run"]]["status"], job["status"])

    def test_out_mapping_handles_nested_relative_and_elsewhere_paths(self):
        nested = self.extra / "batch/recipe"
        self.assertEqual(dashboard._out_dir(["--out", str(nested)], self.extra), "batch/recipe")
        self.assertEqual(dashboard._out_dir(["--out", str(nested.relative_to(ROOT))], self.extra), "batch/recipe")
        self.assertEqual(dashboard._out_dir(["--out", str(self.root / "recipe")], self.extra), "recipe")
        self.assertEqual(dashboard._out_dir(["--out", str(self.root / "direct")]), "direct")
        self.assertIsNone(dashboard._out_dir([]))
        self.assertIsNone(dashboard._out_dir(["--out"]))

    def test_progress_uses_iteration_numbers_and_last_three_times_before_thinning(self):
        records = [{"event": "start", "config": {"iterations": 24}}]
        records += [{"event": "train", "iteration": i, "iteration_seconds": seconds}
                    for i, seconds in ((16, 900), (17, 30), (18, 60), (19, 90))]
        self.log(self.root, "resumed", records)
        self.queue(self.root, self.job(self.root, "job", "resumed"))
        with patch.object(dashboard.time, "time", return_value=1000):
            state = dashboard.state(max_points=2)
        run, job = state["runs"]["resumed"], state["active"][0]
        for row in (run, job):
            self.assertEqual(row["iterations_done"], 19)
            self.assertEqual(row["mean_iteration_seconds"], 60)
            self.assertEqual(row["remaining_seconds"], 300)
            self.assertEqual(row["eta"], 1300)
            self.assertEqual(row["last_train"]["iteration"], 19)
        self.assertEqual(job["progress"], [19, 24])
        self.assertEqual(run["n_train"], 4)
        self.assertEqual(len(run["train"]), 2)

    def test_progress_handles_missing_times_and_completed_or_extended_runs(self):
        run = {"config": {"iterations": 4}, "train": [], "evals": [], "done": False}
        progress = dashboard._progress(run, 1000)
        self.assertEqual(progress["iterations_done"], 0)
        self.assertIsNone(progress["mean_iteration_seconds"])
        self.assertIsNone(progress["eta"])
        run["train"] = [{"iteration": 3, "iteration_seconds": None},
                        {"iteration": 4, "iteration_seconds": 20},
                        {"iteration": 4, "iteration_seconds": 40}]
        self.assertEqual(dashboard._progress(run, 1000)["mean_iteration_seconds"], 30)
        self.assertEqual(dashboard._progress(run, 1000)["remaining_seconds"], 0)
        self.assertIsNone(dashboard._progress(run, 1000)["eta"])
        run["done"] = True
        run["config"]["iterations"] = 6
        self.assertIsNone(dashboard._progress(run, 1000)["eta"])

    def test_health_fields_and_potion_protocol_are_retained(self):
        train = {"event": "train", "iteration": 1, "value_loss": 0.15, "clip_fraction": 0.05,
                 "early_stop": False, "collector_restarts": 2, "faults": 3, "update_seconds": 300,
                 "collect_seconds": 180, "ooc_choice_transitions": 260000, "return_scale": 1.2,
                 "steps": 192, "games": 3072, "collect_wins": 40, "approx_kl": 0.01,
                 "entropy": 0.18, "card_skip_rate": 0.02}
        evaluation = {"event": "eval", "iteration": 0, "n": 1024, "wins": 17, "mean_floor": 30.8,
                      "act2_boss_reach": 0.58, "faults": 4, "collector_restarts": 1}
        self.log(self.root, "visible", [{"event": "start", "config": {"iterations": 24}}, train, evaluation])
        self.log(self.extra, "hidden", [{"event": "start", "config": {"hide_ooc_potions": True}}])
        state = dashboard.state(extra_roots=[self.extra])
        self.assertEqual(state["runs"]["visible"]["last_train"], {k: v for k, v in train.items() if k != "event"})
        self.assertEqual(state["runs"]["visible"]["last_eval"], {k: v for k, v in evaluation.items() if k != "event"})
        self.assertNotIn("hide_ooc_potions", state["runs"]["visible"]["config"])
        self.assertTrue(state["runs"]["hidden"]["config"]["hide_ooc_potions"])

    def test_resume_reopens_finished_run_and_ignores_partial_log_line(self):
        path = self.log(self.root, "resumed", [
            {"event": "start", "config": {"iterations": 4}}, {"event": "done"},
            {"event": "resume", "config": {"iterations": 8}}, {"event": "train", "iteration": 5}])
        with path.open("a") as log:
            log.write('{"event":"train",')
        run = dashboard.state()["runs"]["resumed"]
        self.assertFalse(run["done"])
        self.assertEqual(run["iterations"], 8)
        self.assertEqual(run["iterations_done"], 5)

    def test_state_and_payload_keep_default_root_behavior(self):
        self.log(self.root, "direct")
        self.log(self.extra, "not-included")
        self.queue(self.root, self.job(self.root, "job", "direct"))
        evaluation = self.root / "win_eval/result.json"
        evaluation.parent.mkdir()
        evaluation.write_text(json.dumps({"summary": {"n": 4096, "wins": 156}}))
        state = dashboard.state()
        payload = json.loads(dashboard.state_payload("/api/state?max_points=2"))
        self.assertEqual(set(state["runs"]), {"direct"})
        self.assertEqual(set(payload["runs"]), {"direct"})
        self.assertEqual(payload["win"][0]["wins"], 156)
        self.assertEqual(payload["sig"], state["sig"])
        self.assertIsNone(payload["runner"])
        self.assertEqual(len(payload["runners"]), 1)
        idle = json.loads(dashboard.state_payload("/api/state?since=" + state["sig"]))
        self.assertIsNone(idle["runs"])
        self.assertIsNone(idle["win"])
        self.assertEqual(idle["active"][0]["name"], "job")
        self.assertIn("direct", json.loads(dashboard.state_payload("/api/state?max_points=bad"))["runs"])

    def test_payload_merges_extra_roots_and_queue_changes_refresh_run_status(self):
        self.log(self.extra, "batch/recipe")
        self.queue(self.extra, self.job(self.extra, "job", "batch/recipe", "pending"))
        first = json.loads(dashboard.state_payload("/api/state", [self.extra]))
        self.queue(self.extra, self.job(self.extra, "job", "batch/recipe", "running"))
        second = dashboard.state(since=first["sig"], extra_roots=[self.extra])
        self.assertNotEqual(first["sig"], second["sig"])
        self.assertEqual(second["runs"]["batch/recipe"]["status"], "running")
        (self.extra / "runner_status.json").write_text(json.dumps({"heartbeat": 123, "on_battery": False}))
        third = dashboard.state(since=second["sig"], extra_roots=[self.extra])
        self.assertIsNone(third["runs"])
        self.assertEqual(third["runners"][1]["status"]["heartbeat"], 123)

    def test_new_nested_log_and_appended_iteration_refresh_idle_payload(self):
        path = self.log(self.extra, "batch/recipe")
        first = dashboard.state(extra_roots=[self.extra])
        with path.open("a") as log:
            log.write(json.dumps({"event": "train", "iteration": 2, "iteration_seconds": 420}) + "\n")
        second = dashboard.state(since=first["sig"], extra_roots=[self.extra])
        self.assertEqual(second["runs"]["batch/recipe"]["iterations_done"], 2)
        self.log(self.extra, "batch/recipe_hidden", [
            {"event": "start", "config": {"iterations": 24, "hide_ooc_potions": True}}])
        third = dashboard.state(since=second["sig"], extra_roots=[self.extra])
        self.assertEqual(set(third["runs"]), {"batch/recipe", "batch/recipe_hidden"})
        self.assertTrue(third["runs"]["batch/recipe_hidden"]["config"]["hide_ooc_potions"])

    def test_handler_passes_roots_without_opening_a_socket(self):
        self.log(self.extra, "batch/recipe")
        handler = dashboard.Handler.__new__(dashboard.Handler)
        handler.path = "/api/state?max_points=2"
        handler.extra_roots = (self.extra,)
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.do_GET()
        self.assertEqual(set(json.loads(handler.wfile.getvalue())["runs"]), {"batch/recipe"})
        handler.send_response.assert_called_once_with(200)

    def test_missing_roots_are_empty(self):
        with patch.object(dashboard, "RUNS", self.root / "missing"):
            state = dashboard.state(extra_roots=[self.extra / "missing"])
        self.assertEqual(state["runs"], {})
        self.assertEqual(state["active"], [])
        self.assertEqual(len(state["runners"]), 2)


class DashboardCommandTests(unittest.TestCase):
    def test_server_pids_match_exact_ports_with_root_arguments(self):
        output = "\n".join([
            "PID ARGS", "101 python scripts/rl/dashboard.py serve --port 8795 --root /tmp/extra",
            "102 python scripts/rl/dashboard.py serve --root /tmp/other --port 8795",
            "103 python scripts/rl/dashboard.py serve --port 8791 --root /tmp/extra",
            "104 python scripts/rl/dashboard.py serve --port 87950",
            "105 python scripts/rl/long_rl_dashboard.py serve --port 8795",
            "106 python scripts/rl/dashboard.py start --port 8795",
            "107 python scripts/rl/dashboard.py serve --port=8795 --root /tmp/extra",
            "108 python scripts/rl/dashboard.py serve --port 8795"])
        with patch.object(dashboard.subprocess, "run", return_value=MagicMock(stdout=output)), \
                patch.object(dashboard.os, "getpid", return_value=108):
            self.assertEqual(dashboard._server_pids(8795), [101, 102, 107])

    def test_start_forwards_every_root_to_detached_serve(self):
        runs = MagicMock()
        log = mock_open()
        runs.__truediv__.return_value.open = log
        roots = [ROOT / "first root", ROOT / "second"]
        with patch.object(dashboard, "RUNS", runs), patch.object(dashboard.subprocess, "Popen") as spawn, \
                patch("sys.stdout", new_callable=io.StringIO):
            dashboard._start(8795, roots)
        self.assertEqual(spawn.call_args.args[0], [dashboard.sys.executable, dashboard.__file__, "serve", "--port", "8795",
                                                 "--root", str(roots[0]), "--root", str(roots[1])])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertEqual(spawn.call_args.kwargs["cwd"], ROOT)

    def test_start_is_idempotent_when_endpoint_responds(self):
        with patch.object(dashboard.urllib.request, "urlopen") as request, \
                patch.object(dashboard, "_start") as start, patch("sys.stdout", new_callable=io.StringIO):
            dashboard.main(["start", "--port", "8795", "--root", "/extra"])
        start.assert_not_called()
        request.assert_called_once_with("http://127.0.0.1:8795/api/state", timeout=3)

    def test_start_does_not_duplicate_a_server_that_is_still_starting(self):
        with patch.object(dashboard.urllib.request, "urlopen", side_effect=OSError), \
                patch.object(dashboard, "_server_pids", return_value=[101]) as pids, \
                patch.object(dashboard, "_start") as start, patch("sys.stdout", new_callable=io.StringIO):
            dashboard.main(["start", "--port", "8795", "--root", "/extra"])
        pids.assert_called_once_with(8795)
        start.assert_not_called()

    def test_start_and_restart_cli_preserve_roots_and_target_only_requested_port(self):
        args = ["--port", "8795", "--root", "/extra", "--root", "/another"]
        with patch.object(dashboard.urllib.request, "urlopen", side_effect=OSError), \
                patch.object(dashboard, "_server_pids", return_value=[]), patch.object(dashboard, "_start") as start:
            dashboard.main(["start", *args])
        start.assert_called_once_with(8795, [Path("/extra"), Path("/another")])
        with patch.object(dashboard, "_server_pids", return_value=[101]) as pids, \
                patch.object(dashboard.os, "kill") as kill, patch.object(dashboard.time, "sleep"), \
                patch.object(dashboard, "_start") as start:
            dashboard.main(["restart", *args])
        pids.assert_called_once_with(8795)
        kill.assert_called_once_with(101, dashboard.signal.SIGTERM)
        start.assert_called_once_with(8795, [Path("/extra"), Path("/another")])


if __name__ == "__main__":
    unittest.main()
