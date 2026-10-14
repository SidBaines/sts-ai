from __future__ import annotations
import unittest
from sts_env.env import SimulatorError, SimulatorTimeout
from sts_policy.curriculum import TaskSpec, public_hash
from sts_policy.data import collect_world
from sts_policy.workers import CurriculumWorker
from tests.support import requires_simulator


@requires_simulator
class NativeCurriculumTests(unittest.TestCase):
    def test_saved_battle_replays_in_fresh_persistent_worker(self):
        with CurriculumWorker() as first:
            observation, info = first.reset(TaskSpec(seed=3))
            spec = TaskSpec(seed=3, prefix=tuple(info["task_start_prefix"]),
                            expected_public_hash=public_hash(observation), seek_combat=False,
                            ooc_previews=tuple(info["task_start_ooc_previews"]))
            with CurriculumWorker() as second:
                replay, _ = second.reset(spec)
                self.assertEqual(observation, replay)
                self.assertEqual(first.teacher_action(50), second.teacher_action(50))
            original_pid = first._process.pid
            first.reset(TaskSpec(seed=4))
            self.assertEqual(first._process.pid, original_pid)

    def test_battle_stops_before_reward_screen_and_cutoff_distinct(self):
        with CurriculumWorker() as worker:
            worker.reset(TaskSpec(seed=3))
            for _ in range(128):
                obs, reward, terminated, truncated, info = worker.step(worker.teacher_action(50))
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["task_outcome"], "battle_victory")
            self.assertEqual(obs["phase"], "out_of_combat")
            self.assertGreater(reward, 0)
            self.assertEqual(info["battle_outcome"], "PLAYER_VICTORY")
            worker.reset(TaskSpec(seed=3, horizon=1))
            _, _, terminated, truncated, info = worker.step(worker.teacher_action(50))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["termination_reason"], "task_horizon")
            self.assertEqual(info["task_outcome"], "")

    def test_hash_fault_kills_worker_and_next_reset_recovers(self):
        with CurriculumWorker() as worker:
            with self.assertRaises(SimulatorError):
                worker.reset(TaskSpec(seed=3, expected_public_hash="wrong"))
            self.assertIsNone(worker._process)
            obs, _ = worker.reset(TaskSpec(seed=3))
            self.assertEqual(obs["phase"], "combat")

    def test_native_timeout_is_not_a_loss_and_recovers(self):
        with CurriculumWorker(timeout=0.000001) as worker:
            with self.assertRaises(SimulatorTimeout):
                worker.reset(TaskSpec(seed=3))
            self.assertIsNone(worker._process)
            worker.timeout = 10
            obs, _ = worker.reset(TaskSpec(seed=3))
            self.assertEqual(obs["phase"], "combat")

    def test_ooc_teacher_labels_are_legal_and_async_rpc(self):
        with CurriculumWorker() as worker:
            obs, _ = worker.submit("reset", TaskSpec(seed=3, kind="ooc", horizon=3)).result()
            for _ in range(3):
                index = worker.submit("teacher_action", 50).result()
                self.assertLess(index, len(obs["legal_actions"]))
                obs, _, terminated, truncated, _ = worker.submit("step", index).result()
                self.assertEqual(obs["phase"], "out_of_combat")
                if terminated or truncated:
                    break
            self.assertTrue(terminated or truncated)

    def test_later_battle_replay_preserves_teacher_preview_rng(self):
        with CurriculumWorker() as worker:
            world = collect_world(worker, 1000, epsilon_ooc=0.05, epsilon_combat=0.02)
            self.assertEqual(world["status"], "ok")
            self.assertGreater(len(world["battles"]), 2)
            for row in world["battles"]:
                observation, _ = worker.reset(TaskSpec.from_dict(row))
                self.assertEqual(public_hash(observation), row["expected_public_hash"])

    def test_act_clear_curriculum_success_and_logged_progress(self):
        with CurriculumWorker({"reward_mode": "progress", "gamma": 1.0}) as worker:
            worker.reset(TaskSpec(seed=1000, kind="ooc", max_act=1))
            for _ in range(256):
                _, reward, terminated, truncated, info = worker.step(worker.teacher_action(50))
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["task_outcome"], "act_clear")
            self.assertEqual(info["termination_reason"], "curriculum_act_clear")
            self.assertEqual(info["reward_components"]["terminal"], 1.0)
            self.assertEqual(info["reward_settings"]["mode"], "progress")

    def test_bank_native_provenance_mismatch_fails_closed(self):
        with CurriculumWorker() as worker:
            obs, info = worker.reset(TaskSpec(seed=3))
            identity = {k: info[k] for k in ("python_interface_sha256", "runtime_python_version",
                       "environment_version", "observation_version", "native_provenance")}
            identity["native_provenance"] = {**identity["native_provenance"], "module_sha256": "wrong"}
            with self.assertRaises(SimulatorError):
                worker.reset(TaskSpec(seed=3, expected_provenance=identity))
            self.assertIsNone(worker._process)

    def test_fast_cache_preserves_multiple_world_traces_and_encoded_features(self):
        import numpy as np
        import random
        from sts_policy.encoding import encode_observation
        with CurriculumWorker() as reference, CurriculumWorker({"fast_observations": True}) as fast:
            for seed in range(1000, 1004):
                rng = random.Random(seed)
                left, _ = reference.reset(TaskSpec(seed=seed, kind="run"))
                right, _ = fast.reset(TaskSpec(seed=seed, kind="run"))
                for _ in range(1500):
                    self.assertEqual(left, right)
                    a, b = encode_observation(left), encode_observation(right)
                    np.testing.assert_array_equal(a.state_features, b.state_features)
                    np.testing.assert_array_equal(a.action_features, b.action_features)
                    index = reference.teacher_action(50)
                    self.assertEqual(index, fast.teacher_action(50))
                    epsilon = 0.05 if left["phase"] == "out_of_combat" else 0.02
                    if rng.random() < epsilon:
                        index = rng.randrange(len(left["legal_actions"]))
                    left, reward_a, term_a, trunc_a, info_a = reference.step(index)
                    right, reward_b, term_b, trunc_b, info_b = fast.step(index)
                    self.assertEqual((reward_a, term_a, trunc_a), (reward_b, term_b, trunc_b))
                    self.assertEqual(info_a, info_b)
                    if term_a or trunc_a:
                        self.assertEqual(left, right)
                        break
                self.assertTrue(term_a or trunc_a)
            # Discovery is supported in 0.1.2; this former rejection must now
            # complete identically with and without observation caches.
            bad_ref = collect_world(reference, 2005, epsilon_ooc=0.05, epsilon_combat=0.02)
            bad_fast = collect_world(fast, 2005, epsilon_ooc=0.05, epsilon_combat=0.02)
            self.assertEqual(bad_ref["status"], "ok")
            self.assertEqual(bad_fast["status"], bad_ref["status"])
            self.assertEqual(bad_fast["records"], bad_ref["records"])
