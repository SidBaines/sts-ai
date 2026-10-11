from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from sts_policy.curriculum import TaskSpec, public_hash, task_reward, reward_components, validate_provenance, world_split
from sts_policy.data import atomic_json, atomic_jsonl, collect_world
from sts_env.env import SimulatorError
from sts_policy.workers import CurriculumWorker


class CurriculumTests(unittest.TestCase):
    def test_replay_spec_round_trip_and_world_split(self):
        spec = TaskSpec(seed=3, prefix=(0, 2, 0), expected_public_hash="abc")
        self.assertEqual(TaskSpec.from_dict(spec.to_dict()), spec)
        self.assertEqual(world_split(spec.seed), world_split(TaskSpec(seed=3, kind="ooc").seed))
        self.assertEqual(public_hash({"b": 2, "a": 1}), public_hash({"a": 1, "b": 2}))
        self.assertEqual({world_split(s) for s in range(100)}, {"train", "dev", "final"})

    def test_reward_cutoff_is_not_defeat_and_terminal_potential_zero(self):
        obs = {"state": {"cur_hp": 80, "max_hp": 80}}
        cutoff = task_reward(obs, obs, kind="battle", task_outcome="", terminated=False)
        loss = task_reward(obs, obs, kind="battle", task_outcome="battle_defeat", terminated=True)
        win = task_reward(obs, obs, kind="battle", task_outcome="battle_victory", terminated=True)
        self.assertAlmostEqual(cutoff, 0.0)
        self.assertAlmostEqual(loss, -1.25)
        self.assertAlmostEqual(win, 0.75)

    def test_atomic_journal_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            atomic_json(path, {"complete": True})
            atomic_json(path, {"complete": False})
            self.assertEqual(json.loads(path.read_text()), {"complete": False})
            atomic_jsonl(path, [{"n": 1}, {"n": 2}])
            self.assertEqual([json.loads(s) for s in path.read_text().splitlines()], [{"n": 1}, {"n": 2}])
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_invalid_action_rejected_before_native_rpc(self):
        with CurriculumWorker() as worker:
            worker.finished = False
            worker.observation = {"legal_actions": [{"index": 0}]}
            for bad in (True, -1, 2):
                with self.assertRaises(ValueError):
                    worker.step(bad)
            self.assertIsNone(worker._process)

    def test_failed_world_discards_labels_instead_of_scoring_loss(self):
        class FailingWorker:
            def reset(self, spec):
                return {"phase": "out_of_combat", "legal_actions": [{"index": 0}],
                        "state": {"act": 1, "floor": 0, "cur_hp": 80}}, {}
            def teacher_action(self, simulations):
                return 0
            def step(self, action):
                raise SimulatorError("simulated native failure")
        result = collect_world(FailingWorker(), 3)
        self.assertEqual(result["status"], "simulator_fault")
        self.assertEqual(result["records"], [])
        self.assertEqual(result["battles"], [])
        self.assertEqual(result["discarded_records"], 1)
        self.assertNotIn("result", result)

    def test_logged_progress_and_optional_hp_terminal_reward(self):
        old = {"state": {"cur_hp": 40, "max_hp": 80, "floor": 1}}
        new = {"state": {"cur_hp": 40, "max_hp": 80, "floor": 3}}
        components = reward_components(old, new, task_outcome="act_clear", terminated=True,
                                       mode="progress", terminal_hp_weight=0.2)
        self.assertEqual(components, {"terminal": 1.0, "potential": 0.0, "progress": 0.1, "terminal_hp": 0.1})
        self.assertEqual(reward_components(old, old, task_outcome="", terminated=False, mode="sparse")["terminal"], 0)

    def test_provenance_check_requires_native_and_public_interface_identity(self):
        identity = {"python_interface_sha256": "a", "runtime_python_version": "3.12",
                    "environment_version": "0.1.0", "observation_version": "v1",
                    "native_provenance": {"module_sha256": "b"}}
        validate_provenance(identity, identity)
        self.assertEqual(TaskSpec.from_dict({**TaskSpec(seed=3).to_dict(), "provenance": identity}).expected_provenance, identity)
        for key in identity:
            changed = dict(identity)
            changed[key] = "other"
            with self.assertRaises(ValueError):
                validate_provenance(identity, changed)
