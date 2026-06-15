"""Unit tests for the per-rollout meta record (provenance + outcome aggregates).

Pure Python: build_rollout_meta operates on a RolloutResult + plain env/agent
objects, no simulator needed."""
from __future__ import annotations

import unittest

from sts_ai.rollout import build_rollout_meta
from sts_ai.schemas import DecisionRecord, RolloutResult, SCHEMA_VERSION


class _FakeEnv:
    ascension = 0
    combat_control = "llm"


class _FakeAgent:
    name = "mlx"
    config = {
        "model_id": "mlx-community/Qwen3-4B-4bit",
        "framing": "RISK-FRAMING-TEXT",
        "temperature": 0.2,
        "max_tokens": 256,
        "thinking": True,
        "max_retries": 1,
    }


def _decision(idx, phase, hp, valid=True):
    after = {"phase": phase, "cur_hp": hp}
    if phase == "combat":
        after["combat"] = {"player_cur_hp": hp}
    return DecisionRecord(
        seed=7, decision_index=idx, state={}, state_text="", legal_actions=[],
        selected_action={}, agent={"valid": valid}, after_state=after, phase=phase,
    )


class BuildRolloutMetaTest(unittest.TestCase):
    def setUp(self):
        decisions = [
            _decision(0, "out_of_combat", 80),
            _decision(1, "combat", 72),
            _decision(2, "combat", 60, valid=False),
        ]
        result = RolloutResult(
            seed=7,
            decisions=decisions,
            terminal_state={"outcome": "GameOutcome.PLAYER_LOSS", "act": 1, "floor": 6,
                            "cur_hp": 0, "max_hp": 80, "undefined_behavior_evoked": False},
            stopped_reason="terminal",
            error=None,
        )
        self.meta = build_rollout_meta(
            result, _FakeEnv(), _FakeAgent(),
            run_meta={"git_sha": "abc123", "battle_simulations": 50, "timestamp": "T"},
        )

    def test_provenance_from_agent_config_and_run_meta(self):
        self.assertEqual(self.meta.model_id, "mlx-community/Qwen3-4B-4bit")
        self.assertEqual(self.meta.framing, "RISK-FRAMING-TEXT")  # the study's IV, captured
        self.assertTrue(self.meta.thinking)
        self.assertEqual(self.meta.git_sha, "abc123")
        self.assertEqual(self.meta.combat_control, "llm")
        self.assertEqual(self.meta.schema_version, SCHEMA_VERSION)

    def test_outcome_and_aggregates(self):
        self.assertEqual(self.meta.outcome, "GameOutcome.PLAYER_LOSS")
        self.assertEqual(self.meta.stopped_reason, "terminal")
        self.assertEqual((self.meta.n_decisions, self.meta.n_combat, self.meta.n_out_of_combat), (3, 2, 1))
        self.assertEqual(self.meta.n_invalid, 1)
        self.assertEqual(self.meta.final_floor, 6)
        self.assertEqual(self.meta.hp_trajectory, [80, 72, 60])


if __name__ == "__main__":
    unittest.main()
