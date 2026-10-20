from collections import deque
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from sts_env.env import SimulatorError
from sts_policy.curriculum import public_hash
from sts_policy.model import CandidateActorCritic
from sts_rl.batched_runner import (BatchedPolicy, _Lane, compare_records, evaluation_summary,
                                   fault_record, run_in_process, run_multiprocess)
from sts_rl.fresh_study import StudyPolicy
from tests.integration.test_fresh_policy import observation


def study_policy():
    policy = StudyPolicy.__new__(StudyPolicy)
    policy.champion = CandidateActorCritic().eval()
    policy.replacements = {}
    policy._lock = threading.RLock()
    return policy


def game_observation(step=0):
    obs = observation("combat")
    obs["state"].update(floor=step, room="BOSS", act=1)
    obs["legal_actions"] = [{"index": 0, "description": "end turn"}]
    return obs


class FakeEpisode:
    def __init__(self, config):
        self.step_count = 0

    def reset(self, spec):
        self.seed = spec.seed
        if self.seed == 12:
            raise SimulatorError("reset failed")
        return game_observation(), {"environment_version": "0.1.5", "observation_version": "v3",
            "native_provenance": {}, "python_interface_sha256": "test", "runtime_python_version": "test"}

    def step(self, action):
        if self.seed == 13:
            raise SimulatorError("undefined behavior")
        if self.seed == 14:
            raise ValueError("programming defect")
        self.step_count += 1
        terminal = self.step_count == 2 and self.seed != 15
        cut = self.seed == 16
        return game_observation(self.step_count), 0, terminal, cut, {
            "task_outcome": "victory" if terminal and self.seed == 10 else "defeat" if terminal else "",
            "termination_reason": "task_horizon" if cut else ""}


class BatchedPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(234)
        self.study = study_policy()

    def test_filter_mapping_mixed_phases_padding_and_forced_skips(self):
        raw = [observation("combat"), observation("ooc"), observation("combat")]
        raw[1]["legal_actions"].extend([
            {"index": 89, "description": "discard potion Fire Potion (slot 1)"},
            {"index": 93, "description": "take key"}])
        raw[2]["legal_actions"] = raw[2]["legal_actions"][:1]
        original = deepcopy(raw)
        expected = [self.study.choose_action(obs) for obs in raw]
        forward = self.study.champion.actor_logits
        with patch.object(self.study.champion, "actor_logits", wraps=forward) as actor:
            self.assertEqual(BatchedPolicy(self.study).choose_actions(raw), expected)
            actor.assert_called_once()
            self.assertEqual(actor.call_args.args[0].shape[0], 2)
        self.assertEqual(raw, original)

    def test_first_tie_wins_and_singleton_is_bit_identical(self):
        obs = observation("combat")
        details = []
        BatchedPolicy(self.study).choose_actions([obs], details=details)
        self.assertTrue(torch.equal(details[0], self.study._logits(obs)))
        with torch.no_grad():
            for parameter in self.study.champion.parameters():
                parameter.zero_()
        self.assertEqual(BatchedPolicy(self.study).choose_actions([obs] * 4), [3] * 4)

    def test_entity_and_controlled_phase_replacements(self):
        from sts_policy.entity_model import EntityActorCritic
        from sts_policy.controlled_entity import ControlledEntityActorCritic
        from sts_rl.controlled_policy import ControlledStudyPolicy
        for controlled in (False, True):
            if controlled:
                policy = ControlledStudyPolicy.__new__(ControlledStudyPolicy)
                policy.__dict__.update(self.study.__dict__)
                policy._controlled_phases = {"combat"}
                model = ControlledEntityActorCritic(width=8, layers=1, heads=2, phase="combat")
            else:
                policy = self.study
                model = EntityActorCritic(width=8, layers=1, heads=2, phase="combat")
            policy.replacements = {"combat": model.eval()}
            raw = [observation("combat"), observation("ooc"), observation("combat")]
            raw[2]["legal_actions"] = raw[2]["legal_actions"][:2]
            expected = [policy.choose_action(obs) for obs in raw]
            self.assertEqual(BatchedPolicy(policy).choose_actions(raw), expected)

    def test_invalid_logits_and_empty_menus_propagate(self):
        with patch.object(self.study.champion, "actor_logits", return_value=torch.full((1, 3), float("nan"))):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                BatchedPolicy(self.study).choose_actions([observation("combat")])
        obs = observation("combat")
        obs["legal_actions"] = []
        with self.assertRaisesRegex(ValueError, "legal decision"):
            BatchedPolicy(self.study).choose_actions([obs])


class RunnerTests(unittest.TestCase):
    def run_games(self, seeds, k=4, **kwargs):
        return run_in_process(seeds, study_policy(), k, episode_factory=FakeEpisode, **kwargs)

    def test_metrics_hashes_refill_and_faults_are_not_losses(self):
        events = []
        rows = self.run_games([10, 11, 12, 13, 16], on_event=lambda kind, value: events.append(kind))
        self.assertEqual([r["status"] for r in rows], ["completed", "completed", "fault", "fault", "censored"])
        for row in rows:
            self.assertEqual(row["action_hash"], public_hash({"actions": row["action_prefix"]}))
            if row["status"] != "completed":
                self.assertIsNone(row["win"])
        self.assertEqual(rows[0]["bosses"]["1"]["pass"], True)
        self.assertEqual(rows[1]["decisions"], {"combat:1": 2})
        self.assertEqual(rows[0]["terminal_hash"], public_hash(game_observation(2)))
        summary = evaluation_summary(rows)
        self.assertEqual([summary[k] for k in ("completed", "wins", "faults", "cuts")], [2, 1, 2, 1])
        self.assertEqual(events.count("start"), 5)
        self.assertEqual(events.count("record"), 5)
        self.assertEqual(events.count("decision"), 5)

    def test_decision_limit_is_a_cut_and_python_errors_propagate(self):
        row = self.run_games([15], 1)[0]
        self.assertEqual(row["status"], "censored")
        self.assertEqual(len(row["action_prefix"]), 4096)
        self.assertIsNone(row["win"])
        with self.assertRaisesRegex(ValueError, "programming defect"):
            self.run_games([14])

    def test_cohort_validation(self):
        for seeds, k in (([10, 10], 1), ([True], 1), ([10], 0)):
            with self.assertRaises(ValueError):
                self.run_games(seeds, k)
        with self.assertRaises(ValueError):
            compare_records([fault_record(10, "x")], [fault_record(11, "x")])
        with self.assertRaisesRegex(ValueError, "Diagnostic references"):
            self.run_games([10], reference_records=[])

    def test_construction_failure_is_a_fault(self):
        def factory(config):
            raise SimulatorError("construction failure")
        rows = run_in_process([10], study_policy(), episode_factory=factory)
        self.assertEqual(rows[0]["status"], "fault")
        self.assertIsNone(rows[0]["win"])

    def test_native_guards_restore_original_api_even_on_error(self):
        from sts_rl.fresh_native import _GUARDED_METHODS, native_guard_scope
        def original(*args):
            return None
        native = SimpleNamespace(**{name: type(name, (), {method: original for method in methods})
                                    for name, methods in _GUARDED_METHODS.items()})
        with self.assertRaisesRegex(ValueError, "test failure"):
            with native_guard_scope(native):
                self.assertTrue(native._fresh_native_guards)
                self.assertIsNot(native.GameAction.execute, original)
                with native_guard_scope(native):
                    self.assertIsNot(native.GameAction.execute, original)
                raise ValueError("test failure")
        self.assertIs(native.GameAction.execute, original)
        self.assertFalse(hasattr(native, "_fresh_native_guards"))

    def test_lane_quarantine_retries_active_once_and_preserves_completed(self):
        lane, records = _Lane([10, 11, 12]), []
        lane.accept("start", 10, records, 1.)
        lane.accept("start", 11, records, 1.)
        completed = self.run_games([10], 1)[0]
        lane.accept("record", completed, records, 2.)
        partial = fault_record(11, "snapshot")
        partial["action_prefix"] = [0]
        partial["action_hash"] = public_hash({"actions": [0]})
        lane.accept("decision", [partial], records, 3.)
        lane.quarantine(records, "watchdog timeout")
        self.assertEqual(lane.last_progress, 3.)
        self.assertEqual(lane.remaining, [11, 12])
        self.assertFalse(lane.active)
        self.assertEqual(records, [completed])
        self.assertEqual(lane.requeues, {11: 1})
        lane.accept("start", 11, records, 4.)
        lane.accept("decision", [partial], records, 5.)
        lane.quarantine(records, "second watchdog timeout")
        self.assertEqual(lane.remaining, [12])
        self.assertEqual([r["status"] for r in records], ["completed", "fault"])
        self.assertEqual(records[-1]["action_prefix"], [0])
        self.assertIsNone(records[-1]["win"])
        self.assertEqual(records[-1]["requeues"], 1)
        self.assertEqual(evaluation_summary(records)["requeues"], 1)

    def test_first_divergence_records_the_actual_batch_logit_gap(self):
        from sts_rl.batched_runner import _World, _diagnose
        obs = observation("combat")
        policy = BatchedPolicy(study_policy())
        scores = policy.study._logits(obs)
        best = int(scores.argmax())
        other = (best + 1) % len(scores)
        world = _World(10, None, observation=obs)
        refs = {10: {"action_prefix": [obs["legal_actions"][best]["index"]]}}
        batched = scores.clone()
        batched[other] = scores[best] + 1.
        _diagnose(world, obs["legal_actions"][other]["index"], batched, policy, refs)
        self.assertEqual(world.divergence["decision"], 0)
        self.assertAlmostEqual(world.divergence["singleton_logit_gap"], float(scores[best] - scores[other]))
        self.assertAlmostEqual(world.divergence["batched_logit_gap"], 1., places=5)


def _crash_fixture(connection, seeds, checkpoint, combat, ooc, games_per_process, reference_records=None):
    for seed in seeds:
        connection.send(("start", seed))
        if seed == 10:
            os._exit(7)
        if seed == 11:
            # Start a second concurrent world, then stop making decisions.
            connection.send(("start", seeds[1]))
            time.sleep(30)
        connection.send(("record", fault_record(seed, "fixture simulator fault")))
    connection.send(("done", None))
    connection.close()


def _fault_fixture(connection, seeds, checkpoint, combat, ooc, games_per_process, reference_records=None):
    connection.send(("start", seeds[0]))
    connection.send(("record", fault_record(seeds[0], "fixture reset fault")))
    connection.close()  # Retire after the fault; untouched worlds need a new process.


class DriverProcessTests(unittest.TestCase):
    def run_fake_children(self, scripts, *, gaps=(), seeds=(10, 11), **kwargs):
        """Inject pipe events and loop times without starting native processes."""
        clock = SimpleNamespace(now=0., iterations=0)
        scripts, gaps, assignments = deque(scripts), iter(gaps), []

        class Connection:
            def __init__(self, batches):
                self.batches, self.events = deque(batches), deque()

            def recv(self):
                event = self.events.popleft()
                if isinstance(event, Exception):
                    raise event
                return event

            def poll(self):
                return bool(self.events)

            def close(self):
                pass

        def pipe(**kwargs):
            self.assertTrue(scripts, "Unexpected child restart")
            return Connection(scripts.popleft()), Mock()

        def process(*, args, **kwargs):
            assignments.append(list(args[1]))
            return Mock(is_alive=Mock(return_value=False), exitcode=7)

        def wait_for_events(connections, **kwargs):
            clock.iterations += 1
            self.assertLess(clock.iterations, 100, "Parent loop failed to finish")
            clock.now += next(gaps, .1)
            for connection in connections:
                if connection.batches:
                    connection.events.extend(connection.batches.popleft())
            return [connection for connection in connections if connection.poll()]

        context = Mock(Pipe=Mock(side_effect=pipe), Process=Mock(side_effect=process))
        with patch("sts_rl.batched_runner.multiprocessing.get_context", return_value=context), \
             patch("sts_rl.batched_runner.wait", side_effect=wait_for_events):
            rows = run_multiprocess(seeds, "unused", clock=lambda: clock.now, **kwargs)
        return rows, assignments

    @staticmethod
    def completed(seed):
        return {**fault_record(seed, "fixture"), "status": "completed", "win": False, "floor": 2}

    def test_long_parent_pause_does_not_stall_any_lane(self):
        scripts = [
            [[("start", seed)], [], [], [("record", self.completed(seed)), ("done", None)]]
            for seed in (10, 11)
        ]
        rows, assignments = self.run_fake_children(scripts, processes=2, timeout=1.,
                                                   gaps=(.1, 3600., .1, .1))
        self.assertEqual(assignments, [[10], [11]])
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        self.assertEqual(evaluation_summary(rows)["requeues"], 0)

    def test_genuinely_stalled_lane_is_retried_once_then_faulted(self):
        scripts = [[[("start", 10)]], [[("start", 10)]]]
        rows, assignments = self.run_fake_children(scripts, seeds=(10,), timeout=.25)
        self.assertEqual(assignments, [[10], [10]])
        self.assertEqual(rows[0]["status"], "fault")
        self.assertIsNone(rows[0]["win"])
        self.assertIn("no completed decision", rows[0]["reason"])
        self.assertEqual(evaluation_summary(rows)["requeues"], 1)

    def test_known_culprit_faults_but_peer_restarts_and_completes(self):
        scripts = [
            [[("start", 10), ("start", 11), ("record", fault_record(10, "simulator fault")), EOFError()]],
            [[("start", 11), ("record", self.completed(11)),
              ("start", 12), ("record", self.completed(12)), ("done", None)]],
        ]
        rows, assignments = self.run_fake_children(scripts, seeds=(10, 11, 12))
        self.assertEqual(assignments, [[10, 11, 12], [11, 12]])
        self.assertEqual([r["status"] for r in rows], ["fault", "completed", "completed"])
        self.assertEqual([r["requeues"] for r in rows], [0, 1, 0])
        summary = evaluation_summary(rows)
        self.assertEqual([summary[k] for k in ("completed", "faults", "requeues")], [2, 1, 1])
        self.assertIsNone(rows[0]["win"])

    def test_unknown_process_death_or_retirement_retries_each_active_world_once(self):
        for end in (EOFError(), ("done", None)):
            with self.subTest(end=end):
                scripts = [
                    [[("start", 10), ("start", 11), end]],
                    [[("start", 10), ("start", 11), EOFError()]],
                ]
                rows, assignments = self.run_fake_children(scripts)
                self.assertEqual(assignments, [[10, 11], [10, 11]])
                self.assertTrue(all(r["status"] == "fault" and r["win"] is None for r in rows))
                self.assertEqual([r["requeues"] for r in rows], [1, 1])
                self.assertEqual(evaluation_summary(rows)["requeues"], 2)

    def test_repeated_startup_death_still_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "repeatedly died before starting") as caught:
            self.run_fake_children([[[EOFError()]], [[EOFError()]]])
        self.assertEqual(caught.exception.records, [])
        self.assertEqual(caught.exception.requeues, 0)

    def test_abort_keeps_requeue_count_before_restarted_world_starts(self):
        scripts = [[[("start", 10), EOFError()]], [[("error", "programming defect")]]]
        with self.assertRaisesRegex(RuntimeError, "programming defect") as caught:
            self.run_fake_children(scripts, seeds=(10,))
        self.assertEqual(caught.exception.records, [])
        self.assertEqual(caught.exception.requeues, 1)

    def test_native_fault_retires_child_after_publishing_record(self):
        from sts_rl.batched_runner import _runner_child
        connection = Mock()
        def run(*args, on_event, **kwargs):
            on_event("start", 10)
            on_event("record", fault_record(10, "undefined behavior"))
            self.fail("Child must retire after a native fault")
        with patch("sts_rl.controlled_policy.policy_factory", return_value=object()), \
             patch("sts_rl.batched_runner.run_in_process", side_effect=run):
            _runner_child(connection, [10, 11], "unused", None, None, 2)
        self.assertEqual([call.args[0][0] for call in connection.send.call_args_list], ["start", "record"])
        connection.close.assert_called_once()

    def test_spawn_death_timeout_and_restart(self):
        try:
            parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
        except PermissionError as exc:
            self.skipTest(f"Sandbox denies multiprocessing pipes: {exc}")
        parent.close()
        child.close()
        with patch("sts_rl.batched_runner._runner_child", _crash_fixture):
            rows = run_multiprocess([10, 11, 12, 13], "unused", timeout=1., games_per_process=2)
        self.assertEqual([r["world_seed"] for r in rows], [10, 11, 12, 13])
        self.assertTrue(all(r["status"] == "fault" and r["win"] is None for r in rows))
        self.assertIn("no completed decision", rows[1]["reason"])
        self.assertIn("no completed decision", rows[2]["reason"])
        self.assertEqual(rows[3]["reason"], "fixture simulator fault")

    def test_programming_error_preserves_completed_records_and_propagates(self):
        try:
            parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
        except PermissionError as exc:
            self.skipTest(f"Sandbox denies multiprocessing pipes: {exc}")
        parent.close()
        child.close()
        with patch("sts_rl.batched_runner._runner_child", _error_fixture):
            with self.assertRaisesRegex(RuntimeError, "programming defect") as caught:
                run_multiprocess([20, 21, 22], "unused", timeout=10.)
        rows = caught.exception.records
        self.assertEqual([r["world_seed"] for r in rows], [20, 21])
        self.assertTrue(all(r["win"] is None for r in rows))
        self.assertIn("runner aborted", rows[1]["reason"])

    def test_consecutive_reset_faults_do_not_count_as_startup_failures(self):
        try:
            parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
        except PermissionError as exc:
            self.skipTest(f"Sandbox denies multiprocessing pipes: {exc}")
        parent.close()
        child.close()
        with patch("sts_rl.batched_runner._runner_child", _fault_fixture):
            rows = run_multiprocess([20, 21, 22], "unused", timeout=10.)
        self.assertEqual([r["world_seed"] for r in rows], [20, 21, 22])
        self.assertTrue(all(r["status"] == "fault" and r["win"] is None for r in rows))


def _error_fixture(connection, seeds, checkpoint, combat, ooc, games_per_process, reference_records=None):
    connection.send(("start", seeds[0]))
    connection.send(("record", fault_record(seeds[0], "fixture simulator fault")))
    connection.send(("start", seeds[1]))
    connection.send(("error", "programming defect"))
    connection.close()


class ToolSmokeTests(unittest.TestCase):
    def test_aborted_summary_counts_requeues_of_unstarted_worlds(self):
        root = Path(__file__).resolve().parents[2]
        main = runpy.run_path(str(root / "scripts/rl/fast_eval.py"))["main"]
        error = RuntimeError("fixture abort")
        error.records, error.requeues = [], 1
        with tempfile.TemporaryDirectory(prefix=".batched-abort-test-", dir=root) as directory:
            with patch.dict(main.__globals__, prepare_output=lambda _: Path(directory),
                            run_multiprocess=Mock(side_effect=error)):
                with self.assertRaisesRegex(RuntimeError, "fixture abort"):
                    main(["--checkpoint", "unused", "--world-start", "10", "--n", "1",
                          "--out", directory])
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["requeues"], 1)
            self.assertEqual(summary["unstarted"], 1)
            self.assertFalse(summary["confidence_claims_allowed"])

    def test_run_record_includes_batch_size_process_count_and_world_assignment(self):
        root = Path(__file__).resolve().parents[2]
        prepare = runpy.run_path(str(root / "scripts/rl/fast_eval.py"))["prepare_output"]
        with tempfile.TemporaryDirectory(prefix=".batched-run-record-test-", dir=root) as directory:
            config = {"out": directory, "checkpoint": "unused", "world_start": 10, "n": 5,
                      "games_per_process": 8, "processes": 2}
            with patch("sts_env._backend.lightspeed_import.native_metadata", return_value={}), \
                 patch.dict(prepare.__globals__, file_identity=lambda _: {}):
                prepare(config)
            run = json.loads((Path(directory) / "run.json").read_text())
            self.assertEqual(run["config"]["games_per_process"], 8)
            self.assertEqual(run["config"]["processes"], 2)
            self.assertEqual(run["batching"], {"games_per_process": 8, "processes": 2,
                                              "world_seeds_by_process": [[10, 12, 14], [11, 13]]})

    def test_help(self):
        root = Path(__file__).resolve().parents[2]
        for name in ("fast_eval.py", "benchmark_batched_runner.py"):
            result = subprocess.run([sys.executable, str(root / "scripts/rl" / name), "--help"],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--checkpoint", result.stdout)

    def test_nonempty_output_is_refused_before_loading_checkpoint(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix=".batched-output-test-", dir=root) as directory:
            marker = Path(directory) / "existing"
            marker.write_text("preserve")
            result = subprocess.run([sys.executable, str(root / "scripts/rl/fast_eval.py"),
                "--checkpoint", "absent.pt", "--world-start", "10", "--n", "1", "--out", directory],
                capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output directory must be empty", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [marker])
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
