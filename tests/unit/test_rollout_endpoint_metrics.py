from __future__ import annotations

from concurrent.futures import Future
import json
import unittest

from sts_policy.rollout_training import collect_rollouts, public_endpoint_state
from tests.unit.test_nonllm_training import FakeTask, TinyModel, tiny_collate, tiny_forward


def observed(hp, maximum, act, floor, *, combat=False, done=False):
    state = {"cur_hp": hp, "max_hp": maximum, "act": act, "floor": floor,
             "done": done, "room": "MONSTER", "screen_state": "COMBAT", "hidden_deck": ["never logged"]}
    if combat:
        state["cur_hp"] = 999  # Game-context HP is stale while the battle is live.
        state["combat"] = {"player_cur_hp": hp, "player_max_hp": maximum}
    return {"phase": "combat" if combat else "terminal" if done else "out_of_combat",
            "state": state, "legal_actions": [] if done else [{"index": 7, "description": "End turn"}]}


class EndpointWorker:
    def __init__(self, fail=False):
        self.fail = fail

    def submit(self, method, *args):
        future = Future()
        try:
            if method == "reset":
                self.step = 0
                future.set_result((observed(75, 88, 2, 18, combat=True), {}))
            else:
                self.step += 1
                if self.fail and self.step == 2:
                    raise RuntimeError("simulator fault")
                if self.step == 1:
                    future.set_result((observed(68, 88, 3, 35), .5, False, False, {}))
                else:
                    future.set_result((observed(0, 90, 2, 37, done=True), -1., True, False,
                                       {"outcome": "defeat", "hp_cost": 75., "termination_reason": "defeat"}))
        except Exception as exc:
            future.set_exception(exc)
        return future


class EndpointMetricTests(unittest.TestCase):
    def collect(self, worker, seconds=10):
        return collect_rollouts(TinyModel(), [worker], [FakeTask(7, kind="run")],
                                collate=tiny_collate, forward=tiny_forward,
                                deterministic=True, max_seconds=seconds)

    def test_public_health_and_actual_max_act_not_floor_heuristic(self):
        result = self.collect(EndpointWorker())
        episode = result["episodes"][0]
        self.assertEqual(episode["start_hp"], 75)
        self.assertEqual(episode["start_max_hp"], 88)
        self.assertEqual(episode["end_hp"], 0)
        self.assertEqual(episode["end_max_hp"], 90)
        self.assertEqual(episode["max_act_reached"], 3)
        self.assertEqual(episode["act"], 2)  # Actual last act differs from observed maximum.
        self.assertEqual(episode["termination_state"]["phase"], "terminal")
        self.assertEqual(episode["termination_state"]["floor"], 37)
        self.assertNotIn("hidden_deck", json.dumps(episode))
        self.assertEqual(len(result["transitions"]), 2)
        self.assertEqual([t.reward for t in result["transitions"]], [.5, -1.])
        self.assertEqual(episode["hp_cost"], 75.)

    def test_fault_has_last_observation_not_invented_terminal_endpoint(self):
        result = self.collect(EndpointWorker(fail=True))
        self.assertEqual(result["episodes"], [])
        fault = result["faults"][0]
        self.assertEqual(fault["start_hp"], 75)
        self.assertEqual(fault["max_act_reached"], 3)
        self.assertEqual(fault["last_observed_state"]["hp"], 68)
        self.assertIsNone(fault["end_hp"])
        self.assertIsNone(fault["end_max_hp"])
        self.assertIsNone(fault["termination_state"])
        self.assertEqual(result["transitions"], [])

    def test_budget_endpoint_is_actual_partial_state_and_explicit_cutoff(self):
        episode = self.collect(EndpointWorker(), seconds=0)["episodes"][0]
        self.assertEqual(episode["end_hp"], 75)
        self.assertTrue(episode["truncated"])
        self.assertFalse(episode["terminated"])
        self.assertEqual(episode["termination_reason"], "collector_budget")
        self.assertEqual(episode["max_act_reached"], 2)

    def test_missing_public_numbers_stay_unavailable(self):
        self.assertIsNone(public_endpoint_state(None))
        summary = public_endpoint_state({"phase": "combat", "state": {"cur_hp": 80,
                   "max_hp": 90, "act": True, "floor": float("nan"), "combat": {"player_cur_hp": 0}}})
        self.assertEqual(summary["hp"], 0)
        self.assertIsNone(summary["max_hp"])
        self.assertIsNone(summary["act"])
        self.assertIsNone(summary["floor"])
        self.assertIsNone(public_endpoint_state({"state": {}})["hp"])


if __name__ == "__main__":
    unittest.main()
