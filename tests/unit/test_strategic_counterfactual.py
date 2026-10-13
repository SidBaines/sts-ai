from __future__ import annotations

from copy import deepcopy
import unittest

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.strategic_counterfactual import evaluate_menu
from tests.unit.test_public_entities import observation


class SourceWorker:
    def __init__(self, combat=False):
        self.combat = combat

    def reset(self, task):
        return observation(self.combat), {}


def fake_collector(*args, **kwargs):
    action = next(iter(kwargs["initial_actions"].values()), 7)
    kwargs["decision_observer"]([observation(False)], [0], [0], [action])
    episode = {"terminated": True, "truncated": False, "task_outcome": "victory" if action == 4 else "defeat",
        "max_act_reached": 3, "floor": 55 if action == 4 else 50, "reward": 1. if action == 4 else -1.,
        "action_trace_sha256": str(action), "termination_state": {"floor": 55 if action == 4 else 50}}
    return {"transitions": [], "episodes": [episode], "faults": [], "cohort_completed": True}


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskSpec(seed=next(seed for seed in range(100) if world_split(seed) == "train"),
                             kind="run", max_act=3)

    def test_complete_menu_has_paired_outcomes_and_control_identity(self):
        result = evaluate_menu(None, SourceWorker(), self.task, collector=fake_collector)
        self.assertTrue(result["usable_complete_menu"])
        self.assertIsNone(result["teacher_target"])
        panel = result["panels"][0]
        self.assertTrue(panel["control_trace_matches"])
        self.assertEqual([branch["index"] for branch in panel["branches"]], [7, 4, 20])
        self.assertEqual(panel["branches"][1]["paired"]["victory_delta"], 1)
        self.assertEqual(panel["branches"][1]["paired"]["floor_delta"], 5)

    def test_fault_and_cut_are_not_losses_or_usable_partial_menus(self):
        for fault in (True, False):
            def incomplete(*args, **kwargs):
                result = fake_collector(*args, **kwargs)
                if 20 in kwargs["initial_actions"].values():
                    if fault:
                        result.update(episodes=[], faults=[{"error": "native fault"}])
                    else:
                        result["episodes"][0].update(terminated=False, truncated=True)
                return result
            result = evaluate_menu(None, SourceWorker(), self.task, collector=incomplete)
            self.assertFalse(result["usable_complete_menu"])
            self.assertIsNone(result["panels"][0]["branches"][-1]["paired"])

    def test_control_mismatch_invalidates_otherwise_completed_panel(self):
        def changed(*args, **kwargs):
            result = deepcopy(fake_collector(*args, **kwargs))
            if 7 in kwargs["initial_actions"].values():
                result["episodes"][0]["action_trace_sha256"] = "changed"
            return result
        result = evaluate_menu(None, SourceWorker(), self.task, collector=changed)
        self.assertFalse(result["usable_complete_menu"])
        self.assertFalse(result["panels"][0]["control_trace_matches"])

    def test_forbid_nontrain_combat_and_fake_greedy_replications(self):
        final = TaskSpec(seed=next(seed for seed in range(100) if world_split(seed) == "final"), kind="run", max_act=3)
        for task, worker, settings in ((final, SourceWorker(), {}),
                                      (self.task, SourceWorker(True), {}),
                                      (self.task, SourceWorker(), {"sampling_seeds": (0, 1)})):
            with self.assertRaises(ValueError):
                evaluate_menu(None, worker, task, collector=fake_collector, **settings)

    def test_privileged_comparator_is_explicit_and_uses_same_branch_controller(self):
        calls = []
        def inspected(*args, **kwargs):
            calls.append(kwargs["native_combat_simulations"])
            return fake_collector(*args, **kwargs)
        result = evaluate_menu(None, SourceWorker(), self.task, collector=inspected,
                               native_combat_simulations=1000)
        self.assertTrue(result["usable_complete_menu"])
        self.assertIn("privileged", result["continuation"]["policy"])
        self.assertEqual(calls, [1000] * 4)
        self.assertIsNone(result["teacher_target"])
        with self.assertRaises(ValueError):
            evaluate_menu(None, SourceWorker(), self.task, collector=inspected,
                          native_combat_simulations=0)


if __name__ == "__main__":
    unittest.main()
