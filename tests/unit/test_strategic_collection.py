from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
import unittest

import torch

from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec
from sts_policy.entity_model import ResidualOOCPolicy
from sts_policy.strategic_collection import collect
from tests.unit.test_public_entities import observation


class ScriptedWorker:
    def __init__(self, truncated=False, fault_step=None):
        self.truncated, self.fault_step = truncated, fault_step

    def submit(self, method, *args):
        future = Future()
        if method == "reset":
            self.task, self.position = args[0], 0
            self.frames = [observation(False), observation(), observation(), observation(False), observation(False)]
            self.frames[-1]["legal_actions"] = [] if not self.truncated else self.frames[-1]["legal_actions"]
            future.set_result((deepcopy(self.frames[0]), {}))
        elif method == "step":
            if self.fault_step == self.position:
                future.set_exception(SimulatorError("injected native fault"))
            else:
                self.position += 1
                end = self.position == 4
                reward = [.05, .1, .15, 1.][self.position - 1]
                future.set_result((deepcopy(self.frames[self.position]), reward, end and not self.truncated,
                    end and self.truncated, {"reward_components": {"progress": reward},
                    "task_outcome": "victory" if end and not self.truncated else "",
                    "termination_reason": "task_horizon" if end and self.truncated else "victory" if end else ""}))
        else:
            raise AssertionError("native teacher/control must never be invoked")
        return future


class StrategicCollectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.model = ResidualOOCPolicy({"size": "small", "context": "none"}, width=32, embedding=4)
        self.tasks = [TaskSpec(seed=10, kind="run", max_act=3), TaskSpec(seed=11, kind="run", max_act=3)]

    def test_macro_and_micro_share_trajectory_and_reward_with_task_rng(self):
        outputs = [collect(self.model, [ScriptedWorker(), ScriptedWorker()], self.tasks,
                           credit_unit=unit, sampling_seed=42) for unit in ("micro", "macro")]
        micro, macro = outputs
        self.assertEqual([e["action_trace_sha256"] for e in micro["episodes"]],
                         [e["action_trace_sha256"] for e in macro["episodes"]])
        self.assertEqual([e["reward"] for e in micro["episodes"]], [e["reward"] for e in macro["episodes"]])
        self.assertEqual(len(micro["transitions"]), 8)
        self.assertEqual(len(macro["transitions"]), 4)
        first = macro["transitions"][0]
        self.assertEqual(first.primitive_duration, 3)
        self.assertAlmostEqual(first.reward, .3)
        self.assertAlmostEqual(first.completed_return, 1.3)
        self.assertEqual(first.completed_return, micro["transitions"][0].completed_return)
        self.assertTrue(all(r.observation["phase"] == "out_of_combat" for r in macro["transitions"]))
        # Different lane layout/task order and unrelated torch randomness preserve
        # each task's OOC choices. No random draw is consumed by combat plays.
        torch.rand(100)
        reordered = collect(self.model, [ScriptedWorker()], list(reversed(self.tasks)),
                            credit_unit="macro", sampling_seed=42)
        self.assertEqual({e["task_id"]: e["action_trace_sha256"] for e in macro["episodes"]},
                         {e["task_id"]: e["action_trace_sha256"] for e in reordered["episodes"]})

    def test_fault_quarantines_whole_episode_in_both_modes(self):
        for unit in ("micro", "macro"):
            result = collect(self.model, [ScriptedWorker(fault_step=2)], self.tasks[:1], credit_unit=unit)
            self.assertEqual(result["transitions"], [])
            self.assertEqual(result["episodes"], [])
            self.assertEqual(len(result["faults"]), 1)
            self.assertEqual(result["excluded_primitive_steps"], 2)

    def test_horizon_bootstraps_actual_endpoint_without_completed_return(self):
        for unit in ("micro", "macro"):
            result = collect(self.model, [ScriptedWorker(truncated=True)], self.tasks[:1], credit_unit=unit)
            row = result["transitions"][-1]
            self.assertTrue(row.truncated)
            self.assertFalse(row.terminated)
            self.assertNotEqual(row.next_value, 0.)
            self.assertTrue(all(r.completed_return is None for r in result["transitions"]))
            self.assertTrue(result["episodes"][0]["truncated"])

    def test_learned_combat_evaluator_routes_through_public_model(self):
        # A sharply preferred second combat candidate must be used when the
        # learned-combat path is enabled; the frozen legacy prior remains intact.
        original = self.model.forward
        def changed(*batch):
            logits, values = original(*batch)
            combat = batch[3][:, 0] > .5
            logits = logits.clone()
            if combat.any():
                logits[combat, 1] = logits[combat].max()+100
            return logits, values
        self.model.forward = changed
        first = collect(self.model,[ScriptedWorker()],self.tasks[:1],deterministic=True,retain_transitions=False)
        second = collect(self.model,[ScriptedWorker()],self.tasks[:1],deterministic=True,retain_transitions=False,frozen_combat=False)
        self.assertNotEqual(first["episodes"][0]["action_trace_sha256"],second["episodes"][0]["action_trace_sha256"])
        self.assertEqual(first["episodes"][0]["reward"],second["episodes"][0]["reward"])


if __name__ == "__main__":
    unittest.main()
