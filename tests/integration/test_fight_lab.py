from __future__ import annotations

from dataclasses import asdict
import unittest

from tests.support import requires_simulator


@requires_simulator
class FightLabReplayTest(unittest.TestCase):
    def test_partial_source_replays_to_arbitrary_mid_fight_state(self):
        from sts_ai.agents import SimpleHeuristicAgent
        from sts_ai.fight_lab import public_digest, replay_to
        from sts_ai.lightspeed import LightspeedHybridEnv
        from sts_ai.rollout import prepare_decision, run_rollout
        config = dict(world_seed=3, combat_control="llm", combat_observation="combat_public_v3",
                      max_act=1, battle_simulations=50)
        result = run_rollout(LightspeedHybridEnv(**config), SimpleHeuristicAgent(), max_decisions=30)
        records = [asdict(r) for r in result.decisions]
        target = next(i for i, r in enumerate(records) if i > 5 and r["phase"] == "combat")
        env = LightspeedHybridEnv(**config)
        replay_to(env, records, target)
        status, view = prepare_decision(env)
        self.assertEqual(status, "ok")
        view["legal_actions"] = view["legal_action_dicts"]
        self.assertEqual(public_digest(view), public_digest(records[target]))
        records[target]["state_text"] += " corrupt"
        with self.assertRaisesRegex(ValueError, "diverged"):
            replay_to(LightspeedHybridEnv(**config), records, target)
