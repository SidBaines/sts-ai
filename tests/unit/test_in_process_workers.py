"""Worker contracts and collector crash recovery without a native simulator."""
from concurrent.futures import Future
from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from sts_env.env import SimulatorError, SimulatorTimeout
from sts_policy.losses import PPOConfig
from sts_rl.common import FilteringWorker, close_workers, make_workers, train_tasks
from sts_rl.in_process import InProcessWorker
from sts_rl.parallel_collect import CollectorPool


class FakeEpisode:
    def __init__(self, config):
        self.config = config
        self.error = None

    def reset(self, spec):
        if self.error:
            raise self.error
        return {"legal_actions": [{"index": 0}, {"index": 1}]}, {"seed": spec}

    def step(self, action):
        if self.error:
            raise self.error
        return {"legal_actions": []}, 1., True, False, {"action": action}

    def teacher_action(self, simulations=50, search_seed=None):
        if self.error:
            raise self.error
        return 1


class InProcessWorkerTests(unittest.TestCase):
    def setUp(self):
        self.native = SimpleNamespace(_fresh_native_guards=False)
        self.load = self.enterContext(patch("sts_rl.in_process.import_lightspeed", return_value=self.native))
        self.guards = self.enterContext(patch("sts_rl.in_process.install_native_guards",
            side_effect=lambda native: setattr(native, "_fresh_native_guards", True)))
        self.factory = self.enterContext(patch("sts_rl.in_process.CurriculumEpisode", side_effect=FakeEpisode))
        self.worker = InProcessWorker()
        self.addCleanup(self.worker.close)

    def test_lazy_construction_config_and_guards_once_for_all_slots(self):
        config = {"fault_dir": "unused", "run_record": "unused", "gamma": 1.0}
        with patch.dict(os.environ, STS_LIGHTSPEED_BUILD="fake-build"), InProcessWorker(config) as other:
            self.load.assert_not_called()
            self.factory.assert_not_called()
            self.assertTrue(other.finished)
            self.assertIsNone(other.observation)
            self.assertIsNone(other.info)
            self.assertIsNone(other.last_fault_path)
            self.assertEqual(other.config, {"gamma": 1., "build_dir": "fake-build"})
            self.assertIn("fault_dir", config)
            other.reset(10)
            other.reset(11)
            self.load.assert_called_once_with("fake-build")
            self.worker.reset(12)
            self.guards.assert_called_once_with(self.native)
            self.assertEqual(self.factory.call_count, 2)
        self.assertTrue(other.finished)
        with self.assertRaises(RuntimeError):
            other.step(0)

    def test_submit_resolves_results_and_exceptions_immediately(self):
        future = self.worker.submit("reset", 10)
        self.assertIsInstance(future, Future)
        self.assertTrue(future.done())
        self.assertEqual(future.result(), (self.worker.observation, self.worker.info))
        self.assertFalse(self.worker.finished)
        self.assertEqual(self.worker.submit("teacher_action", 50, 42).result(), 1)
        error = ValueError("programming defect")
        episode = self.worker._episode
        episode.error = error
        future = self.worker.submit("step", 0)
        self.assertTrue(future.done())
        self.assertIs(future.exception(), error)
        with self.assertRaises(ValueError) as caught:
            future.result()
        self.assertIs(caught.exception, error)
        self.assertTrue(self.worker.finished)
        self.assertIs(self.worker._episode, episode)
        with self.assertRaisesRegex(RuntimeError, "reset before step"):
            self.worker.step(0)

    def test_simulator_fault_requires_reset_and_fresh_episode(self):
        for error in (SimulatorError("native fault"), SimulatorTimeout("timeout")):
            self.worker.reset(10)
            old = self.worker._episode
            old.error = error
            future = self.worker.submit("step", 0)
            self.assertIs(future.exception(), error)
            self.assertTrue(self.worker.finished)
            self.assertIsNone(self.worker._episode)
            self.assertIsNone(self.worker.last_fault_path)
            with self.assertRaisesRegex(RuntimeError, "reset before step"):
                self.worker.step(0)
            self.worker.reset(11)
            self.assertIsNot(self.worker._episode, old)
            self.assertFalse(self.worker.finished)
            self.assertEqual(self.worker.step(1)[1:], (1., True, False, {"action": 1}))
            self.assertTrue(self.worker.finished)

    def test_faults_on_reset_and_teacher_also_discard_episode(self):
        for method, args in (("reset", (10,)), ("teacher_action", ())):
            self.worker.reset(10)
            self.worker._episode.error = SimulatorError("native fault")
            with self.assertRaises(SimulatorError):
                getattr(self.worker, method)(*args)
            self.assertTrue(self.worker.finished)
            self.assertIsNone(self.worker._episode)

    def test_programming_error_propagates_unchanged(self):
        self.worker.reset(10)
        episode = self.worker._episode
        episode.error = ValueError("programming defect")
        with self.assertRaises(ValueError) as caught:
            self.worker.step(0)
        self.assertIs(caught.exception, episode.error)
        self.assertIs(self.worker._episode, episode)
        self.assertTrue(self.worker.finished)
        with self.assertRaisesRegex(RuntimeError, "reset before step"):
            self.worker.step(0)

    def test_unknown_methods_and_invalid_actions(self):
        with self.assertRaisesRegex(ValueError, "unknown public RPC"):
            self.worker.submit("close")
        with self.assertRaises(RuntimeError):
            self.worker.step(0)
        self.worker.reset(10)
        for action in (True, False, -1, 2):
            with self.assertRaises(ValueError):
                self.worker.step(action)
        for action in (0.0, "0", None):
            with self.assertRaises(TypeError):
                self.worker.step(action)

    def test_make_workers_uses_identical_filter_wrapper(self):
        workers = make_workers(2, in_process=True, dead_action_filter=True)
        self.addCleanup(close_workers, workers)
        self.factory.assert_not_called()
        for worker in workers:
            self.assertIsInstance(worker, FilteringWorker)
            self.assertIsInstance(worker.worker, InProcessWorker)
            self.assertTrue(worker.dead_actions)
            self.assertTrue(worker.slim)


def _crashing_collector(conn, checkpoint, workers, worker_options, threads):
    """The checkpoint path is a shared death counter for spawned fixture children."""
    marker = Path(checkpoint)
    while True:
        kind, payload = conn.recv()
        if kind == "close":
            conn.close()
            return
        deaths = int(marker.read_text()) if marker.exists() else 0
        limit = worker_options["fixture_deaths"]
        if deaths < limit:
            marker.write_text(str(deaths + 1))
            os._exit(1)
        # A complete, simulator-free collection reply; echo inputs to prove replay.
        episodes = [{"world_seed": task.seed, "seed": payload["seed"], "state": payload["state"],
                     "in_process": worker_options["in_process"], "workers": workers}
                    for task in payload["tasks"]]
        conn.send(("ok", {"kept": [], "game_episodes": episodes, "faults": 0,
                          "fault_records": [], "stats": {}}))


class CollectorRecoveryTests(unittest.TestCase):
    def collect(self, pool, model, tasks):
        return pool.collect_ooc(model, tasks, combat="greedy", temperature=1., ppo=PPOConfig(),
                                credit="macro", gae_lambda=1., chunk=128, seed=7)

    def test_collector_restarts_once_with_same_request_for_both_worker_kinds(self):
        for in_process in (False, True):
            with self.subTest(in_process=in_process), tempfile.TemporaryDirectory() as directory, \
                 patch("sts_rl.parallel_collect._collector_main", _crashing_collector):
                pool = CollectorPool(1, workers=4, checkpoint=str(Path(directory) / "deaths"),
                                     in_process=in_process, fixture_deaths=1)
                try:
                    # No tensors need sharing to exercise process death/replay.
                    model = torch.nn.Identity()
                    tasks = train_tasks(100, 4)
                    result = self.collect(pool, model, tasks)
                    self.assertEqual(result["collector_restarts"], 1)
                    self.assertEqual(pool.restarts, 1)
                    self.assertEqual(result["faults"], 0)
                    expected = [{"world_seed": task.seed, "seed": 7 * 1009,
                                 "state": {}, "in_process": in_process, "workers": 4}
                                for task in tasks]
                    self.assertEqual(result["game_episodes"], expected)
                    result = self.collect(pool, model, tasks)
                    self.assertEqual(result["game_episodes"], expected)
                    self.assertEqual(result["collector_restarts"], 0)
                    self.assertEqual(pool.restarts, 1)
                finally:
                    pool.close()

    def test_second_death_raises_instead_of_retrying_forever(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("sts_rl.parallel_collect._collector_main", _crashing_collector):
            pool = CollectorPool(1, workers=1, checkpoint=str(Path(directory) / "deaths"), fixture_deaths=2)
            try:
                with self.assertRaisesRegex(RuntimeError, "failed twice"):
                    self.collect(pool, torch.nn.Identity(), train_tasks(100, 1))
                self.assertEqual(pool.restarts, 1)
                self.assertEqual((Path(directory) / "deaths").read_text(), "2")
                self.assertIsNone(pool.procs[0])
            finally:
                pool.close()

    def test_pipe_failures_and_dead_process_resend_the_exact_request(self):
        for stage in ("send", "recv", "dead"):
            for error_type in (EOFError, BrokenPipeError, OSError):
                with self.subTest(stage=stage, error=error_type):
                    pool = CollectorPool.__new__(CollectorPool)
                    original, replacement = Mock(), Mock()
                    process = Mock()
                    process.is_alive.return_value = True
                    pool.conns, pool.procs, pool.restarts = [original], [process], 0
                    if stage == "dead":
                        original.recv.side_effect = EOFError("collector is no longer alive")
                    else:
                        method = "send_bytes" if stage == "send" else "recv"
                        getattr(original, method).side_effect = error_type("connection failed")
                    replacement.recv.return_value = ("ok", "complete")

                    def launch(index):
                        pool.conns[index] = replacement
                        pool.procs[index] = Mock(is_alive=Mock(return_value=True))

                    with patch.object(pool, "_launch", side_effect=launch) as restart, \
                         patch("sts_rl.parallel_collect.mp.reduction.ForkingPickler.dumps",
                               wraps=ForkingPickler.dumps) as dumps:
                        result = pool._scatter("ooc", torch.nn.Identity(), [10, 11], 7, credit="macro")
                    self.assertEqual(result, ["complete"])
                    self.assertEqual(pool.restarts, 1)
                    restart.assert_called_once_with(0)
                    original.close.assert_called_once()
                    self.assertEqual(dumps.call_count, 2)
                    self.assertIs(dumps.call_args_list[0].args[0], dumps.call_args_list[1].args[0])
                    request = pickle.loads(replacement.send_bytes.call_args.args[0])
                    self.assertEqual(request, ("ooc", {"state": {}, "seed": 7 * 1009,
                                                       "tasks": [10, 11], "credit": "macro"}))
                    self.assertEqual(request, pickle.loads(original.send_bytes.call_args.args[0]))

    def test_local_pickling_failure_propagates_without_restarting_collector(self):
        pool = CollectorPool.__new__(CollectorPool)
        conn = Mock()
        pool.conns, pool.procs, pool.restarts = [conn], [Mock(is_alive=Mock(return_value=True))], 0
        error = OSError("local pickling failure")
        with patch("sts_rl.parallel_collect.mp.reduction.ForkingPickler.dumps", side_effect=error), \
             patch.object(pool, "_launch") as restart:
            with self.assertRaises(OSError) as caught:
                pool._scatter("ooc", torch.nn.Identity(), [], 7)
        self.assertIs(caught.exception, error)
        conn.send_bytes.assert_not_called()
        conn.close.assert_not_called()
        restart.assert_not_called()
        self.assertEqual(pool.restarts, 0)

    def test_finished_reply_is_kept_when_collector_dies_after_sending(self):
        pool = CollectorPool.__new__(CollectorPool)
        conn, process = Mock(), Mock()
        process.is_alive.side_effect = [True, False]
        conn.recv.return_value = ("ok", "complete")
        pool.conns, pool.procs, pool.restarts = [conn], [process], 0
        with patch.object(pool, "_launch") as restart:
            self.assertEqual(pool._scatter("ooc", torch.nn.Identity(), [], 7), ["complete"])
        process.is_alive.assert_called_once()
        restart.assert_not_called()
        self.assertEqual(pool.restarts, 0)

    def test_reported_programming_error_does_not_restart_collector(self):
        pool = CollectorPool.__new__(CollectorPool)
        conn = Mock()
        conn.recv.return_value = ("error", "programming defect")
        pool.conns, pool.procs, pool.restarts = [conn], [Mock(is_alive=Mock(return_value=True))], 0
        with patch.object(pool, "_launch") as restart:
            with self.assertRaisesRegex(RuntimeError, "programming defect"):
                pool._scatter("ooc", torch.nn.Identity(), [], 7)
        restart.assert_not_called()
        self.assertEqual(pool.restarts, 0)

    def test_training_cli_exposes_both_boolean_flags(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, str(root / "scripts/rl/train_ooc.py"), "--help"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--in-process-games", result.stdout)
        self.assertIn("--no-in-process-games", result.stdout)


if __name__ == "__main__":
    unittest.main()
