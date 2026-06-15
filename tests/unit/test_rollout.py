"""Pure-Python rollout unit tests — no native simulator required.

These exercise rollout/env logic via fakes or ``object.__new__`` so the fast
suite runs on any checkout. Tests that drive the real simulator live in
``tests/integration``.
"""
import unittest

from sts_ai.agents import FirstLegalAgent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import run_rollout
from sts_ai.schemas import LegalAction


class FakeStepErrorEnv:
    seed = 123

    def advance_to_decision(self):
        return 0

    def is_terminal(self):
        return False

    def phase(self):
        return "out_of_combat"

    def legal_actions(self):
        return [LegalAction(index=0, bits=1, description="only action")]

    def describe_state(self):
        return "fake state"

    def step(self, action_index):
        raise RuntimeError("simulator exploded")

    def summary(self):
        return {"seed": self.seed, "done": False}

    @staticmethod
    def action_dict(action):
        return {"index": action.index, "bits": action.bits, "description": action.description}


class FakeCombatEnv:
    """Records exactly one in-combat decision, then terminates.

    Lets the rollout loop be exercised without the native simulator to confirm
    the ``phase`` field is threaded from ``env.phase()`` into the DecisionRecord.
    """

    seed = 7

    def __init__(self):
        self._steps = 0

    def advance_to_decision(self):
        return 0

    def is_terminal(self):
        return self._steps > 0

    def phase(self):
        return "combat"

    def legal_actions(self):
        return [
            LegalAction(index=0, bits=0, description="play Strike (cost 1) -> Cultist"),
            LegalAction(index=1, bits=1, description="end turn"),
        ]

    def describe_state(self):
        return "Battle turn 0\nPlayer HP: 80/80"

    def step(self, action_index):
        self._steps += 1
        actions = self.legal_actions()
        return actions[action_index]

    def summary(self):
        return {"seed": self.seed, "phase": self.phase(), "done": self.is_terminal()}

    @staticmethod
    def action_dict(action):
        return {"index": action.index, "bits": action.bits, "description": action.description}


class TerminalBoundaryTest(unittest.TestCase):
    # object.__new__ exercises is_terminal() without constructing the real env,
    # so no native module is loaded.
    def test_act_equal_to_max_act_is_not_terminal(self):
        env = object.__new__(LightspeedHybridEnv)
        env.max_act = 1
        env.sts = type("Sts", (), {"GameOutcome": type("Outcome", (), {"UNDECIDED": "undecided"})})
        env.gc = type("Gc", (), {"outcome": "undecided", "act": 1})
        self.assertFalse(env.is_terminal())

    def test_act_greater_than_max_act_is_terminal(self):
        env = object.__new__(LightspeedHybridEnv)
        env.max_act = 1
        env.sts = type("Sts", (), {"GameOutcome": type("Outcome", (), {"UNDECIDED": "undecided"})})
        env.gc = type("Gc", (), {"outcome": "undecided", "act": 2})
        self.assertTrue(env.is_terminal())


class RolloutSimulatorErrorTest(unittest.TestCase):
    def test_simulator_step_error_stops_rollout_with_error_payload(self):
        result = run_rollout(FakeStepErrorEnv(), FirstLegalAgent(), max_decisions=1)
        self.assertEqual(result.stopped_reason, "simulator_error")
        self.assertEqual(len(result.decisions), 0)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error["phase"], "step")
        self.assertEqual(result.error["decision_index"], 0)
        self.assertIn("simulator exploded", result.error["message"])


class RolloutPhaseRecordingTest(unittest.TestCase):
    def test_combat_decision_records_combat_phase(self):
        result = run_rollout(FakeCombatEnv(), FirstLegalAgent(), max_decisions=2)
        self.assertEqual(len(result.decisions), 1)
        self.assertEqual(result.decisions[0].phase, "combat")

    def test_decision_record_phase_defaults_to_out_of_combat(self):
        # Back-compat: pre-combat traces have no `phase`; the field must default so
        # they still construct/load.
        from sts_ai.schemas import DecisionRecord

        record = DecisionRecord(
            seed=1,
            decision_index=0,
            state={},
            state_text="",
            legal_actions=[],
            selected_action={},
            agent={},
            after_state={},
        )
        self.assertEqual(record.phase, "out_of_combat")


if __name__ == "__main__":
    unittest.main()
