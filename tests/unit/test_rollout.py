"""Pure-Python rollout unit tests — no native simulator required.

These exercise rollout/env logic via fakes or ``object.__new__`` so the fast
suite runs on any checkout. Tests that drive the real simulator live in
``tests/integration``.
"""
import inspect
import random
import unittest
from unittest.mock import patch

from sts_ai.agents import FirstLegalAgent
from sts_ai.hinting import HintConfig
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import build_rollout_meta, run_rollout
from sts_ai.schemas import AgentDecision, LegalAction, RolloutResult


class FakeStepErrorEnv:
    world_seed = 123

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

    def map_graph(self):
        return None

    def step(self, action_index):
        raise RuntimeError("simulator exploded")

    def summary(self):
        return {"world_seed": self.world_seed, "done": False}

    @staticmethod
    def action_dict(action):
        return {"index": action.index, "bits": action.bits, "description": action.description}


class FakeInvalidEnv(FakeStepErrorEnv):
    world_seed = 321

    def __init__(self):
        self.step_calls = 0

    def step(self, action_index):
        self.step_calls += 1
        return self.legal_actions()[action_index]

    def summary(self):
        return {"world_seed": self.world_seed, "done": False, "cur_hp": 80}


class InvalidAgent:
    name = "invalid"

    def reseed(self, policy_seed: int) -> None:
        return None

    def choose_action(self, state_text, legal_actions):
        return AgentDecision(
            action_index=0,
            raw_response="not json",
            valid=False,
            metadata={"error": "no json object"},
        )


class FakeCombatEnv:
    """Records exactly one in-combat decision, then terminates.

    Lets the rollout loop be exercised without the native simulator to confirm
    the ``phase`` field is threaded from ``env.phase()`` into the DecisionRecord.
    """

    world_seed = 7

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

    def map_graph(self):
        return None

    def step(self, action_index):
        self._steps += 1
        actions = self.legal_actions()
        return actions[action_index]

    def summary(self):
        return {"world_seed": self.world_seed, "phase": self.phase(), "done": self.is_terminal()}

    @staticmethod
    def action_dict(action):
        return {"index": action.index, "bits": action.bits, "description": action.description}


class HintableCombatEnv:
    world_seed = 77

    def __init__(self):
        self.step_indices: list[int] = []

    def advance_to_decision(self):
        return 0

    def is_terminal(self):
        return bool(self.step_indices)

    def phase(self):
        return "combat"

    def legal_actions(self):
        return [
            LegalAction(index=0, bits=0, description="end turn"),
            LegalAction(
                index=1,
                bits=1,
                description="play Strike (cost 1) -> Cultist (deal 6)",
            ),
        ]

    def describe_state(self):
        return "\n".join(
            [
                "Battle turn 0",
                "Player HP: 70/80, block: 0",
                "Player powers: none",
                "Enemies:",
                "Cultist [enemy 0]: HP 6/48, block 0, intent CULTIST_INCANTATION (no attack)",
                "Hand:",
                "[0] Strike (cost 1)",
            ]
        )

    def map_graph(self):
        return None

    def step(self, action_index):
        self.step_indices.append(action_index)
        return self.legal_actions()[action_index]

    def summary(self):
        return {
            "world_seed": self.world_seed,
            "phase": self.phase(),
            "done": self.is_terminal(),
            "cur_hp": 70,
            "combat": {
                "player_cur_hp": 70,
                "player_block": 0,
                "player_energy": 1,
                "enemies": [
                    {
                        "name": "Cultist",
                        "cur_hp": 6,
                        "block": 0,
                        "alive": True,
                        "intent_damage": 0,
                        "intent_hits": 0,
                    }
                ],
            },
        }

    @staticmethod
    def action_dict(action):
        return {"index": action.index, "bits": action.bits, "description": action.description}


class ScriptedHintAgent:
    name = "scripted-hint"

    def __init__(self, *, hinted_action_index: int = 1, laundered_action_index: int = 1):
        self.hinted_action_index = hinted_action_index
        self.laundered_action_index = laundered_action_index
        self.calls: list[str] = []
        self.reseed_calls: list[int] = []
        self.laundered_raw = (
            f'{{"reasoning":"strike kills","action_index":{laundered_action_index}}}'
        )

    def reseed(self, policy_seed: int) -> None:
        self.reseed_calls.append(policy_seed)

    def choose_action(self, state_text, legal_actions):
        self.calls.append(state_text)
        if "Target action:" in state_text:
            return AgentDecision(
                action_index=self.laundered_action_index,
                raw_response=self.laundered_raw,
                reasoning="strike kills",
            )
        if "Factual hint:" in state_text:
            return AgentDecision(
                action_index=self.hinted_action_index,
                raw_response=(
                    f'{{"reasoning":"hinted","action_index":{self.hinted_action_index}}}'
                ),
                reasoning="hinted",
            )
        return AgentDecision(
            action_index=0,
            raw_response='{"reasoning":"normal","action_index":0}',
            reasoning="normal",
        )


class TerminalBoundaryTest(unittest.TestCase):
    # object.__new__ exercises is_terminal() without constructing the real env,
    # so no native module is loaded.
    def test_constructor_default_max_act_is_full_game(self):
        default = inspect.signature(LightspeedHybridEnv).parameters["max_act"].default
        self.assertEqual(default, 3)

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

    def test_invalid_agent_decision_records_and_stops_without_step(self):
        env = FakeInvalidEnv()
        result = run_rollout(env, InvalidAgent(), max_decisions=3)

        self.assertEqual(result.stopped_reason, "agent_invalid")
        self.assertIsNone(result.error)
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(len(result.decisions), 1)
        record = result.decisions[0]
        self.assertFalse(record.agent["valid"])
        self.assertFalse(record.action_executed)
        self.assertEqual(record.selected_action, {})
        self.assertEqual(record.after_state["cur_hp"], 80)


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
            world_seed=1,
            decision_index=0,
            state={},
            state_text="",
            legal_actions=[],
            selected_action={},
            agent={},
            after_state={},
        )
        self.assertEqual(record.phase, "out_of_combat")


class SerialHintingTest(unittest.TestCase):
    def test_combat_mistake_triggers_hint_launder_and_corrected_step(self):
        env = HintableCombatEnv()
        agent = ScriptedHintAgent()

        result = run_rollout(
            env,
            agent,
            max_decisions=1,
            hint_cfg=HintConfig(enabled=True),
        )

        self.assertEqual(result.stopped_reason, "max_decisions")
        self.assertEqual(env.step_indices, [1])
        self.assertEqual(len(agent.calls), 3)
        self.assertNotIn("Factual hint:", agent.calls[0])
        self.assertIn("Factual hint:", agent.calls[1])
        self.assertIn("Target action: 1: play Strike", agent.calls[2])
        self.assertEqual(len(result.decisions), 1)
        record = result.decisions[0]
        self.assertTrue(record.hint_applied)
        self.assertEqual(record.selected_action["index"], 1)
        self.assertTrue(record.agent["valid"])
        self.assertEqual(record.agent["retries"], 0)
        self.assertEqual(record.agent["raw_response"], agent.laundered_raw)
        hint_meta = record.agent["metadata"]["hint"]
        self.assertEqual(hint_meta["mistake_kind"], "lethal")
        self.assertEqual(hint_meta["launder_outcome"], "laundered")
        self.assertEqual(hint_meta["original_action_index"], 0)
        self.assertEqual(hint_meta["final_action_index"], 1)

    def test_hint_no_change_records_provenance_and_keeps_normal_action(self):
        env = HintableCombatEnv()
        agent = ScriptedHintAgent(hinted_action_index=0)

        result = run_rollout(
            env,
            agent,
            max_decisions=1,
            hint_cfg=HintConfig(enabled=True),
        )

        self.assertEqual(env.step_indices, [0])
        self.assertEqual(len(agent.calls), 2)
        record = result.decisions[0]
        self.assertFalse(record.hint_applied)
        self.assertEqual(record.selected_action["index"], 0)
        self.assertEqual(record.agent["action_index"], 0)
        self.assertEqual(record.agent["raw_response"], '{"reasoning":"normal","action_index":0}')
        hint_meta = record.agent["metadata"]["hint"]
        self.assertFalse(hint_meta["triggered"])
        self.assertEqual(hint_meta["launder_outcome"], "no_change")

    def test_hint_cfg_none_matches_default_path_without_extra_agent_calls(self):
        default_env = HintableCombatEnv()
        default_agent = ScriptedHintAgent()
        explicit_env = HintableCombatEnv()
        explicit_agent = ScriptedHintAgent()

        default_result = run_rollout(default_env, default_agent, max_decisions=1)
        explicit_result = run_rollout(
            explicit_env,
            explicit_agent,
            max_decisions=1,
            hint_cfg=None,
        )

        self.assertEqual(len(default_agent.calls), 1)
        self.assertEqual(len(explicit_agent.calls), 1)
        self.assertEqual(default_env.step_indices, [0])
        self.assertEqual(explicit_env.step_indices, [0])
        self.assertEqual(
            default_result.decisions[0].selected_action,
            explicit_result.decisions[0].selected_action,
        )
        self.assertEqual(
            default_result.decisions[0].agent,
            explicit_result.decisions[0].agent,
        )
        self.assertFalse(explicit_result.decisions[0].hint_applied)
        self.assertNotIn("hint", explicit_result.decisions[0].agent["metadata"])

    def test_hint_path_reuses_one_affordance_compute_for_detection_and_record(self):
        env = HintableCombatEnv()
        agent = ScriptedHintAgent()
        expected_affordances = {"single_target_lethal_available": True}

        with patch(
            "sts_ai.rollout.affordances.compute",
            return_value=expected_affordances,
        ) as compute:
            result = run_rollout(
                env,
                agent,
                max_decisions=1,
                hint_cfg=HintConfig(enabled=True),
            )

        self.assertEqual(compute.call_count, 1)
        self.assertIs(result.decisions[0].affordances, expected_affordances)
        self.assertTrue(result.decisions[0].hint_applied)


class FakeRandomEnv:
    world_seed = 42

    def __init__(self, decisions: int = 12):
        self._decisions = decisions
        self._steps = 0

    def advance_to_decision(self):
        return 0

    def is_terminal(self):
        return self._steps >= self._decisions

    def phase(self):
        return "out_of_combat"

    def legal_actions(self):
        return [
            LegalAction(index=i, bits=i, description=f"action {i}")
            for i in range(5)
        ]

    def describe_state(self):
        return f"fake decision {self._steps}"

    def map_graph(self):
        return None

    def step(self, action_index):
        self._steps += 1
        return self.legal_actions()[action_index]

    def summary(self):
        return {"world_seed": self.world_seed, "step": self._steps, "done": self.is_terminal()}

    @staticmethod
    def action_dict(action):
        return {"index": action.index, "bits": action.bits, "description": action.description}


class StubStochasticAgent:
    name = "stub-stochastic"

    def __init__(self):
        self.rng = random.Random()
        self.reseed_calls: list[int] = []

    def reseed(self, policy_seed: int) -> None:
        self.reseed_calls.append(policy_seed)
        self.rng = random.Random(policy_seed)

    def choose_action(self, state_text, legal_actions):
        return AgentDecision(
            action_index=self.rng.randrange(len(legal_actions)),
            raw_response="stub stochastic",
        )


class StubConfiguredAgent:
    name = "vllm"

    @property
    def config(self):
        return {
            "backend": "vllm",
            "reasoning_mode": "prompted",
            "model_id": "x",
        }


class RolloutMetaProvenanceTest(unittest.TestCase):
    def test_agent_config_is_preserved_in_extra_alongside_run_extra(self):
        result = RolloutResult(
            world_seed=99,
            decisions=[],
            terminal_state={},
            stopped_reason="terminal",
        )

        meta = build_rollout_meta(
            result,
            FakeRandomEnv(),
            StubConfiguredAgent(),
            run_meta={"extra": {"experiment": "unit"}},
        )

        self.assertEqual(meta.extra["experiment"], "unit")
        self.assertEqual(meta.extra["agent_config"]["reasoning_mode"], "prompted")
        self.assertEqual(meta.extra["agent_config"]["backend"], "vllm")


class RolloutPolicySeedTest(unittest.TestCase):
    def test_stochastic_agent_reseeded_by_world_seed_and_rollout_index(self):
        agent_a = StubStochasticAgent()
        result_a = run_rollout(FakeRandomEnv(), agent_a, max_decisions=20, rollout_index=0)
        agent_b = StubStochasticAgent()
        result_b = run_rollout(FakeRandomEnv(), agent_b, max_decisions=20, rollout_index=0)
        agent_c = StubStochasticAgent()
        result_c = run_rollout(FakeRandomEnv(), agent_c, max_decisions=20, rollout_index=1)

        seq_a = [d.selected_action["index"] for d in result_a.decisions]
        seq_b = [d.selected_action["index"] for d in result_b.decisions]
        seq_c = [d.selected_action["index"] for d in result_c.decisions]

        self.assertEqual(seq_a, seq_b)
        self.assertNotEqual(seq_a, seq_c)
        self.assertEqual(len(agent_a.reseed_calls), 1)
        self.assertEqual(result_a.policy_seed, agent_a.reseed_calls[0])
        self.assertEqual(result_a.rollout_index, 0)
        self.assertEqual(result_c.rollout_index, 1)


if __name__ == "__main__":
    unittest.main()
