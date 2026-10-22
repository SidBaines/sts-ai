import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
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
        self.assertFalse(runs["ooc_r2_lam1"]["key"])
        self.assertEqual(runs["ooc_r2_lam1"]["label"], dashboard.LABELS["ooc_r2_lam1"])
        self.assertEqual(runs["batch/recipe"]["name"], "batch/recipe")

    def test_current_lineage_is_ordered_and_old_labels_do_not_make_runs_key(self):
        lineage = ("round4_20261021/round4", "continued2_20261021/train", "continued3_20261022/train",
                   "continued4_20261022/train", "joint_20261022/train")
        self.assertEqual(dashboard.KEY_RUNS, lineage)
        present = [n for n in lineage if "continued4" not in n]
        old = ["large_batch_20261020/" + n for n in ("recipe", "recipe_hidden", "hygiene_hidden")]
        for name in reversed(present + old + ["active"]):
            self.log(self.root, name)
        self.queue(self.root, self.job(self.root, "job", "active"))
        runs = dashboard.state()["runs"]
        self.assertEqual(list(runs)[:len(present)], present)
        self.assertEqual([n for n, r in runs.items() if r["key"]], present)
        self.assertNotIn(lineage[3], runs)
        for name in present + old:
            self.assertEqual(runs[name]["label"], dashboard.LABELS[name])
        self.assertTrue(runs["active"]["active"])
        self.assertFalse(runs["active"]["key"])

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

    def test_lineage_mapping_resolves_paths_and_parent_labels_for_ppo_only(self):
        nodes = [
            {"id": "ppo:root", "kind": "ppo", "label": "Root training", "path": str(self.root / "a"),
             "x0": 0, "parent": None},
            {"id": "splice:student", "kind": "splice", "label": "Student splice", "path": "student.pt",
             "x0": 24, "parent": {"node": "ppo:root"}},
            {"id": "ppo:continued", "kind": "ppo", "label": "Continued", "path": str(self.extra / "a/../b"),
             "x0": 24, "parent": {"node": "splice:student"}},
            {"id": "distill:rule", "kind": "distill", "label": "Rule", "path": "rule", "x0": 24,
             "parent": {"node": "ppo:continued"}},
        ]
        self.assertEqual(dashboard._lineage_mapping({"nodes": nodes}), {
            str(self.root / "a"): {"x0": 0, "parent": None},
            str(self.extra / "b"): {"x0": 24, "parent": "Student splice"}})

    def test_lineage_refresh_reuses_and_saves_hash_cache_without_changing_other_fields(self):
        cache = Path(self.tmp.name) / "cache.json"
        cache.write_text(json.dumps({"file_hashes": {"old.pt": {"sha256": "old"}}, "other": [1, 2]}))
        extra = Path(self.tmp.name) / "edges.json"
        extra.write_text(json.dumps({"labels": {"a": "A"}}))
        lineage = dashboard.Lineage([self.root], extra, ["_smoke"], cache)
        self.assertIsNone(lineage.snapshot())
        caches = []

        def reconstruct(roots, extras, excludes, hashes):
            self.assertEqual(roots, [self.root])
            self.assertEqual(extras, {"labels": {"a": "A"}})
            self.assertEqual(excludes, ["_smoke"])
            self.assertEqual(hashes["old.pt"], {"sha256": "old"})
            caches.append(hashes)
            hashes["new.pt"] = {"sha256": "new"}
            return {"nodes": [{"id": "ppo:a", "kind": "ppo", "path": str(self.root / "a"),
                               "x0": 48, "parent": None}]}

        with patch.object(dashboard, "_reconstruct_lineage", side_effect=reconstruct):
            lineage.refresh()
            self.assertEqual(lineage.snapshot(), {str(self.root / "a"): {"x0": 48, "parent": None}})
            lineage.refresh()
        self.assertIs(caches[0], caches[1])
        saved = json.loads(cache.read_text())
        self.assertEqual(saved, {"file_hashes": {"old.pt": {"sha256": "old"}, "new.pt": {"sha256": "new"}},
                                 "other": [1, 2]})

    def test_lineage_import_and_reconstruction_work_on_tiny_run(self):
        checkpoint = Path(self.tmp.name) / "root.pt"
        checkpoint.write_bytes(b"root checkpoint")
        self.log(self.root, "a", [{"event": "start", "config": {"credit": "macro", "checkpoint": str(checkpoint)}},
                                  {"event": "train", "iteration": 1, "games": 10}])
        lineage = dashboard.Lineage([self.root])
        lineage.refresh()
        self.assertEqual(lineage.snapshot(), {str(self.root / "a"): {"x0": 0, "parent": "root.pt"}})

    def test_lineage_worker_retries_after_failure_and_keeps_last_snapshot(self):
        lineage = dashboard.Lineage([self.root])
        initial = {"nodes": [{"id": "ppo:a", "kind": "ppo", "path": str(self.root / "a"),
                               "x0": 24, "parent": None}]}
        updated = {"nodes": [{**initial["nodes"][0], "x0": 48}]}
        with patch.object(dashboard, "_reconstruct_lineage", side_effect=[initial, RuntimeError("broken"), updated]):
            lineage.refresh()
            previous = lineage.snapshot()
            snapshots = []

            def wait(seconds):
                self.assertEqual(seconds, 600)
                snapshots.append(lineage.snapshot())
                return len(snapshots) >= 2

            with patch.object(lineage._stop, "wait", side_effect=wait) as waits, \
                    self.assertLogs(level="ERROR") as logs:
                lineage._run()
        self.assertIn("keeping the previous result", logs.output[0])
        self.assertEqual(waits.call_count, 2)
        self.assertEqual(waits.call_args_list[0].args, (600,))
        self.assertEqual(waits.call_args_list[1].args, (600,))
        self.assertIs(snapshots[0], previous)
        self.assertEqual(previous[str(self.root / "a")]["x0"], 24)
        self.assertEqual(lineage.snapshot()[str(self.root / "a")]["x0"], 48)

    def test_state_lineage_fields_and_signature_only_change_with_relevant_mapping(self):
        self.log(self.root, "same")
        self.log(self.extra, "same")
        off = dashboard.state(extra_roots=[self.extra])
        self.assertEqual(off["lineage"], {"enabled": False, "ready": False})
        self.assertIsNone(off["runs"]["same"]["x0"])
        self.assertIsNone(off["runs"]["same"]["lineage_parent"])
        lineage = dashboard.Lineage([self.root, self.extra])
        pending = dashboard.state(extra_roots=[self.extra], lineage=lineage)
        self.assertEqual(pending["lineage"], {"enabled": True, "ready": False})
        nodes = [{"id": "ppo:parent", "kind": "ppo", "path": str(self.extra / "same"),
                  "label": "Parent training", "x0": 24, "parent": None},
                 {"id": "ppo:child", "kind": "ppo", "path": str(self.root / "same"),
                  "x0": 48, "parent": {"node": "ppo:parent"}}]
        with patch.object(dashboard, "_reconstruct_lineage", return_value={"nodes": nodes}) as reconstruct:
            lineage.refresh()
            handler = dashboard.Handler.__new__(dashboard.Handler)
            handler.path = "/api/state?since=" + pending["sig"]
            handler.extra_roots = (self.extra,)
            handler.lineage = lineage
            handler.wfile = io.BytesIO()
            handler.send_response, handler.send_header, handler.end_headers = MagicMock(), MagicMock(), MagicMock()
            handler.do_GET()
            ready = json.loads(handler.wfile.getvalue())
            self.assertEqual(ready["lineage"], {"enabled": True, "ready": True})
            self.assertEqual(ready["runs"]["same"]["x0"], 48)
            self.assertEqual(ready["runs"]["same"]["lineage_parent"], "Parent training")
            self.assertEqual(ready["runs"][f"same [{self.extra}]"]["x0"], 24)
            self.assertIsNone(ready["runs"][f"same [{self.extra}]"]["lineage_parent"])
            reconstruct.assert_called_once()  # the HTTP request never reconstructs
            lineage.refresh()
            idle = dashboard.state(since=ready["sig"], extra_roots=[self.extra], lineage=lineage)
            self.assertIsNone(idle["runs"])
            nodes.append({"id": "ppo:unrelated", "kind": "ppo", "path": str(self.root / "elsewhere"),
                          "x0": 99, "parent": None})
            lineage.refresh()
            self.assertIsNone(dashboard.state(since=ready["sig"], extra_roots=[self.extra], lineage=lineage)["runs"])
            nodes[1]["x0"] = 72
            lineage.refresh()
            changed = dashboard.state(since=ready["sig"], extra_roots=[self.extra], lineage=lineage)
            self.assertEqual(changed["runs"]["same"]["x0"], 72)
            nodes[0]["label"] = "Renamed parent"
            lineage.refresh()
            renamed = dashboard.state(since=changed["sig"], extra_roots=[self.extra], lineage=lineage)
            self.assertEqual(renamed["runs"]["same"]["lineage_parent"], "Renamed parent")

    def test_ready_empty_lineage_exposes_unknown_starts(self):
        self.log(self.root, "unmatched")
        lineage = dashboard.Lineage([self.root])
        pending = dashboard.state(lineage=lineage)
        with patch.object(dashboard, "_reconstruct_lineage", return_value={"nodes": []}):
            lineage.refresh()
        ready = json.loads(dashboard.state_payload("/api/state?since=" + pending["sig"], lineage=lineage))
        self.assertEqual(ready["lineage"], {"enabled": True, "ready": True})
        self.assertIsNone(ready["runs"]["unmatched"]["x0"])

    def test_missing_roots_are_empty(self):
        with patch.object(dashboard, "RUNS", self.root / "missing"):
            state = dashboard.state(extra_roots=[self.extra / "missing"])
        self.assertEqual(state["runs"], {})
        self.assertEqual(state["active"], [])
        self.assertEqual(len(state["runners"]), 2)


class DashboardPageTests(unittest.TestCase):
    def test_page_loads_pinned_plugins_after_chartjs_and_has_chart_resets(self):
        self.assertEqual(re.findall(r'<script src="([^"]+)"', dashboard.PAGE), [
            "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
            "https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js",
            "https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"])
        canvases = re.findall(r'<canvas id="([^"]+)"', dashboard.PAGE)
        self.assertEqual(re.findall(r'<button data-reset="([^"]+)"', dashboard.PAGE), canvases)
        for control in ("all", "none", "default", "arch", "reset-all"):
            self.assertIn(f'<button id="{control}">', dashboard.PAGE)
        self.assertIn("Drag to zoom · ctrl/⌘ + wheel to zoom · shift + drag to pan · double-click or reset to restore",
                      dashboard.PAGE)

    def run_js(self, body, *, saved=None, saved_axis=None, blocked=False, zoom=True, lineage=None):
        if not shutil.which("node"):
            self.skipTest("node is required for dashboard JavaScript checks")
        runs = {name: {"name": name, "root": "/tmp/data/rl_runs", "label": name, "config": {},
                       "key": name == "key", "active": name == "active", "status": "done",
                       "train": [{"iteration": 1, "collect_floor": 20}], "evals": []}
                for name in ("key", "active", "old")}
        harness = r"""
const assert=require('node:assert/strict'), vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{id,innerHTML:'',textContent:'',dataset:{},handlers:{},
 addEventListener(type,fn){this.handlers[type]=fn}});return elements.get(id)}
const buttons=input.canvases.map(id=>{const b=element('reset_'+id);b.dataset.reset=id;return b});
const saved=new Map([['sts-rl-dashboard-runs',input.saved],['sts-rl-dashboard-axis',input.saved_axis]]);
const storage={read:(key='sts-rl-dashboard-runs')=>saved.get(key)};
class Chart {
 static register(plugin){assert.equal(plugin.id,'zoom')}
 constructor(canvas,config){this.data=config.data;this.options=config.options;this.updates=[];this.resets=0;
  if(input.zoom)this.resetZoom=mode=>{assert.equal(mode,'none');this.resets++;
   for(const s of Object.values(this.options.scales)){delete s.min;delete s.max}}}
 update(mode){this.updates.push(mode)}
}
const context=vm.createContext({assert,Chart,buttons,element,storage,state:input.state,
 window:{ChartZoom:input.zoom?{id:'zoom'}:undefined},
 document:{hidden:false,getElementById:element,querySelectorAll:()=>buttons,addEventListener(){}},
 localStorage:{getItem(key){if(input.blocked)throw Error('blocked');return saved.get(key)},
  setItem(key,value){if(input.blocked)throw Error('blocked');saved.set(key,value)}},
 fetch:async()=>({ok:true,json:async()=>input.state}),AbortSignal:{timeout(){}},setTimeout:()=>0,clearTimeout(){}});
vm.runInContext(input.script,context);
(async()=>{await new Promise(setImmediate);await vm.runInContext('(async()=>{'+input.body+'})()',context)})()
 .catch(e=>{console.error(e);process.exitCode=1});
"""
        payload = {"script": dashboard.PAGE.split("<script>")[1].split("</script>")[0], "body": body,
                   "canvases": re.findall(r'<canvas id="([^"]+)"', dashboard.PAGE),
                   "state": {"now": 1000, "sig": "test", "runs": runs, "runners": [],
                             "active": [], "history": {}, "win": [], "fresh": [],
                             "lineage": lineage or {"enabled": False, "ready": False}},
                   "saved": saved, "saved_axis": saved_axis, "blocked": blocked, "zoom": zoom}
        result = subprocess.run(["node", "-e", harness], input=json.dumps(payload), capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_global_controls_and_hidden_archived_selection(self):
        self.run_js(r"""
assert.equal(Object.keys(charts).length,11);
assert.deepEqual(visible(),['key','active']);
const shown=n=>Object.values(charts).every(c=>c.data.datasets.some(d=>d.label===n||d.label.startsWith(n+' ')));
assert(shown('key')&&shown('active'));assert(!shown('old'));
const oldColor=color('old');
element('all').handlers.click();assert(shown('old'));
assert(element('arch').textContent.includes('1 checked'));
element('arch').handlers.click();assert(visible().includes('old'));
element('arch').handlers.click();assert(!visible().includes('old'));assert(shown('old'));
assert(!element('runs').innerHTML.includes('data-run="old"'));
element('runs').handlers.change({target:{matches:()=>true,dataset:{run:'old'}}});
assert(!shown('old'));assert.equal(color('old'),oldColor);
element('runs').handlers.click({target:{closest:()=>({dataset:{run:'active'}})}});
assert(!shown('active'));
element('none').handlers.click();assert(Object.values(charts).every(c=>c.data.datasets.length===0));
element('default').handlers.click();assert(shown('key')&&shown('active'));assert(!shown('old'));
assert.deepEqual(JSON.parse(storage.read()),{key:true,active:true,old:false});
""")

    def test_selection_restores_and_new_active_runs_default_to_selected(self):
        self.run_js(r"""
assert.equal(selected.get('key'),false);assert.equal(selected.get('old'),true);
assert(!visible().includes('old'));assert(element('arch').textContent.includes('1 checked'));
assert(Object.values(charts).every(c=>c.data.datasets.some(d=>d.label==='old'||d.label.startsWith('old '))));
element('none').handlers.click();assert(Object.values(JSON.parse(storage.read())).every(v=>v===false));
state.runs.later={...state.runs.active,name:'later'};await load();
assert.equal(selected.get('later'),true);assert.equal(selected.get('key'),false);
""", saved=json.dumps({"key": False, "active": True, "old": True}))

    def test_invalid_or_unavailable_storage_uses_defaults(self):
        for saved, blocked in (("broken", False), ('["old"]', False), ('{"old":"yes"}', False),
                               ('{"key":false}', True)):
            with self.subTest(saved=saved, blocked=blocked):
                self.run_js(r"""
assert.equal(selected.get('key'),true);assert.equal(selected.get('active'),true);assert.equal(selected.get('old'),false);
element('none').handlers.click();assert(Object.values(charts).every(c=>c.data.datasets.length===0));
""", saved=saved, blocked=blocked)

    def test_zoom_ranges_survive_refresh_and_reset_controls_work(self):
        self.run_js(r"""
for(const c of Object.values(charts)){
 const z=c.options.plugins.zoom;
 assert.equal(z.zoom.mode,'xy');assert.equal(z.zoom.drag.enabled,true);assert.equal(z.zoom.wheel.enabled,true);
 assert.equal(z.pan.enabled,true);assert.equal(z.pan.mode,'xy');assert.equal(z.pan.modifierKey,'shift');
 assert.equal(!!z.zoom.onZoomStart({event:{type:'wheel'}}),false);
 assert.equal(z.zoom.onZoomStart({event:{type:'wheel',ctrlKey:true}}),true);
 assert.equal(z.zoom.onZoomStart({event:{type:'wheel',metaKey:true}}),true);
 assert.equal(z.zoom.onZoomStart({event:{type:'mousedown'}}),true);
 assert.equal(c.options.plugins.legend.onClick(),undefined);
 c.options.scales.x.min=2;c.options.scales.x.max=4;c.options.scales.y={min:5,max:10};
}
const c=charts.c_train, scales=c.options.scales, data=c.data.datasets;
await load();assert.equal(charts.c_train,c);assert.equal(c.options.scales,scales);assert.notEqual(c.data.datasets,data);
for(const chart of Object.values(charts)){
 assert.equal(chart.options.scales.x.min,2);assert.equal(chart.options.scales.x.max,4);
 assert.equal(chart.options.scales.y.min,5);assert.equal(chart.options.scales.y.max,10);
 assert.equal(chart.resets,0);assert.equal(chart.updates.at(-1),'none');
}
buttons.find(b=>b.dataset.reset==='c_train').handlers.click();assert.equal(c.resets,1);assert.equal(charts.c_wins.resets,0);
element('c_train').handlers.dblclick();assert.equal(c.resets,2);
element('reset-all').handlers.click();assert(Object.values(charts).every(chart=>chart.resets>0));
assert.equal(c.options.scales.x.min,undefined);
""")

    def test_missing_zoom_plugin_keeps_charts_and_selection_working(self):
        self.run_js(r"""
assert.equal(Object.keys(charts).length,11);assert(Object.values(charts).every(c=>c.options.plugins.zoom===false));
assert(element('reset-all').disabled);assert(buttons.every(b=>b.disabled));
element('c_train').handlers.dblclick();element('reset-all').handlers.click();
element('all').handlers.click();assert(Object.values(charts).every(c=>c.data.datasets.some(d=>d.label==='old'||d.label.startsWith('old '))));
await load();assert(Object.values(charts).every(c=>c.updates.at(-1)==='none'));
""", zoom=False)

    def test_cumulative_axis_shifts_all_series_and_bounds_and_resets_zoom(self):
        self.run_js(r"""
state.lineage={enabled:true,ready:true};
for(const [name,r] of Object.entries(state.runs)){
 r.x0=name==='key'?144:name==='active'?0:null;r.lineage_parent='<parent>';
 r.train=[{iteration:1,collect_floor:20,smith_rate:.5,map_elite:.2,card_skip_rate:.1,grad_cosine_half:.3,
  entropy:.4,approx_kl:.01,value_loss:.02,iteration_seconds:60,collect_seconds:40,update_seconds:20,
  collect_wins:10,games:100}];
 r.evals=[{iteration:2,wins:12,n:100,paired:{diff:2,se:.1},act2_boss_pass:.4,act3_boss_reach:.2}];
}
await load();element('all').handlers.click();element('arch').handlers.click();
const before=Object.fromEntries(Object.entries(charts).map(([id,c])=>[id,JSON.parse(JSON.stringify(c.data.datasets))]));
const references=Object.values(charts);
for(const c of references){c.options.scales.x.min=1;c.options.scales.x.max=2;c.options.scales.y={min:0,max:1}}
element('axis-mode').handlers.change({target:{value:'cumulative'}});
assert.equal(storage.read(AXIS_STORAGE_KEY),'cumulative');
assert.equal(axisMode,'cumulative');assert(!element('axis-cumulative').disabled);
for(const [id,c] of Object.entries(charts)){
 assert.equal(c.options.scales.x.title.text,'cumulative update');assert.equal(c.resets,1);
 assert.equal(c.options.scales.x.min,undefined);assert.equal(c.options.scales.y.min,undefined);
 c.data.datasets.forEach((d,i)=>{
  assert(d.data.length>0);
  d.data.forEach((p,j)=>{assert.equal(p.x,before[id][i].data[j].x+(d.label.startsWith('key')?144:0));
   assert.equal(p.y,before[id][i].data[j].y)});
 });
}
assert(element('runs').innerHTML.includes('from &lt;parent&gt; at update 144'));
assert(element('runs').innerHTML.includes('from &lt;parent&gt; at update 0'));
assert(element('runs').innerHTML.includes('(start unknown)'));
assert.equal(state.runs.key.train[0].iteration,1);assert.equal(state.runs.key.evals[0].iteration,2);
await load();assert(Object.values(charts).every(c=>c.resets===1));assert.deepEqual(Object.values(charts),references);
element('axis-mode').handlers.change({target:{value:'per-run'}});
assert.equal(storage.read(AXIS_STORAGE_KEY),'per-run');
for(const [id,c] of Object.entries(charts)){
 assert.equal(c.options.scales.x.title.text,'update');assert.equal(c.resets,2);
 assert.equal(JSON.stringify(c.data.datasets),JSON.stringify(before[id]));
}
assert(!element('runs').innerHTML.includes('<th>start</th>'));
""")

    def test_saved_cumulative_axis_waits_for_lineage_then_restores(self):
        self.run_js(r"""
assert.equal(axisMode,'per-run');assert.equal(preferredAxisMode,'cumulative');
assert(element('axis-cumulative').disabled);assert.equal(element('lineage-note').textContent,'computing lineage…');
assert.equal(charts.c_train.data.datasets[0].data[0].x,1);
state.runs.key.x0=168;state.lineage.ready=true;await load();
assert.equal(axisMode,'cumulative');assert.equal(element('axis-mode').value,'cumulative');
assert(!element('axis-cumulative').disabled);assert.equal(element('lineage-note').textContent,'');
assert.equal(charts.c_train.data.datasets[0].data[0].x,169);
assert(Object.values(charts).every(c=>c.resets===1));
await load();assert(Object.values(charts).every(c=>c.resets===1));
""", saved_axis="cumulative", lineage={"enabled": True, "ready": False})

    def test_axis_mode_handles_unavailable_storage_and_zoom_plugin(self):
        self.run_js(r"""
assert.equal(axisMode,'per-run');assert(!element('axis-cumulative').disabled);
state.runs.key.x0=24;await load();
element('axis-mode').handlers.change({target:{value:'cumulative'}});
assert.equal(charts.c_train.data.datasets[0].data[0].x,25);
assert.equal(charts.c_train.options.scales.x.title.text,'cumulative update');
await load();assert.equal(axisMode,'cumulative');
""", saved_axis="cumulative", blocked=True, zoom=False, lineage={"enabled": True, "ready": True})

    def test_saved_cumulative_mode_is_disabled_when_feature_is_off(self):
        self.run_js(r"""
assert.equal(axisMode,'per-run');assert(element('axis-cumulative').disabled);
assert.equal(element('lineage-note').textContent,'lineage unavailable');
assert.equal(charts.c_train.data.datasets[0].data[0].x,1);
assert.equal(charts.c_train.options.scales.x.title.text,'update');
""", saved_axis="cumulative")


class DashboardCommandTests(unittest.TestCase):
    def test_serve_starts_background_lineage_only_when_roots_are_configured(self):
        args = ["serve", "--port", "8899", "--root", "/runs", "--lineage-roots", "/one", "/two",
                "--lineage-extra", "/extra.json", "--lineage-exclude", "repro", "--lineage-exclude", "smoke", "other",
                "--lineage-cache", "/cache.json"]
        with patch.object(dashboard, "ThreadingHTTPServer") as server, patch.object(dashboard, "Lineage") as lineage, \
                patch.object(dashboard.Handler, "lineage", None), patch.object(dashboard.Handler, "extra_roots", ()):
            dashboard.main(args)
            lineage.assert_called_once_with([Path("/one"), Path("/two")], Path("/extra.json"),
                                            ["repro", "smoke", "other"], Path("/cache.json"))
            lineage.return_value.start.assert_called_once_with()
            lineage.return_value.stop.assert_called_once_with()
            server.assert_called_once_with(("127.0.0.1", 8899), dashboard.Handler)
            server.return_value.__enter__.return_value.serve_forever.assert_called_once_with()
        with patch.object(dashboard, "ThreadingHTTPServer"), patch.object(dashboard, "Lineage") as lineage, \
                patch.object(dashboard.Handler, "lineage", None), patch.object(dashboard.Handler, "extra_roots", ()):
            dashboard.main(["serve", "--port", "8899"])
            lineage.assert_not_called()

    def test_lineage_thread_is_daemon_and_can_be_stopped(self):
        lineage = dashboard.Lineage([ROOT])
        with patch.object(dashboard.threading, "Thread") as thread:
            lineage.start()
        thread.assert_called_once_with(target=lineage._run, name="dashboard-lineage", daemon=True)
        thread.return_value.start.assert_called_once_with()
        lineage.stop()
        self.assertTrue(lineage._stop.is_set())

    def test_start_forwards_lineage_cli_options(self):
        flags = ["--lineage-roots", "/one", "/two", "--lineage-extra", "/extra.json",
                 "--lineage-cache", "/cache.json", "--lineage-exclude", "repro", "--lineage-exclude", "/_smoke"]
        with patch.object(dashboard.urllib.request, "urlopen", side_effect=OSError), \
                patch.object(dashboard, "_server_pids", return_value=[]), patch.object(dashboard, "_start") as start:
            dashboard.main(["start", "--port", "8899", *flags])
        start.assert_called_once_with(8899, [], lineage_args=flags)
        runs = MagicMock()
        runs.__truediv__.return_value.open = mock_open()
        with patch.object(dashboard, "RUNS", runs), patch.object(dashboard.subprocess, "Popen") as spawn, \
                patch("sys.stdout", new_callable=io.StringIO):
            dashboard._start(8899, lineage_args=flags)
        self.assertEqual(spawn.call_args.args[0][5:], flags)

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
