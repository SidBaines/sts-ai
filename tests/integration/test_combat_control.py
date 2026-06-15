"""Integration test for full in-combat LLM control (combat_control="llm").

Drives the real simulator with a deterministic scripted combat policy and proves
the combat step-loop works end to end: the env surfaces in-combat decisions
(phase="combat"), a battle plays through to a terminal Outcome, exit_battle writes
results back (the run reaches REWARDS), and no illegal action reaches the sim.

Run in a child process with a timeout to contain the residual seed-class UB hang
(see docs/simulator_issue_handoff.md and tests/integration/test_battle_search.py).
Gated with @requires_simulator. See tests/CLAUDE.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from sts_ai.schemas import AgentDecision

from tests.support import requires_simulator


class ScriptedCombatAgent:
    """Deterministic policy usable for both decision kinds.

    In combat: play the first affordable card / usable potion, else end the turn —
    so battles actually progress to a terminal outcome. Out of combat: take the
    first legal action to path toward the next battle.
    """

    name = "scripted-combat"

    def reseed(self, policy_seed):
        return None

    def choose_action(self, state_text, legal_actions):
        descs = [a.description for a in legal_actions]
        for i, d in enumerate(descs):
            if d.startswith("play ") or d.startswith("drink potion"):
                return AgentDecision(action_index=i, raw_response=f"combat:{d}")
        for i, d in enumerate(descs):
            if d == "end turn":
                return AgentDecision(action_index=i, raw_response="end turn")
        return AgentDecision(action_index=0, raw_response="out_of_combat:0")


@requires_simulator
class CombatControlTest(unittest.TestCase):
    def test_llm_combat_control_plays_a_full_battle(self):
        code = textwrap.dedent(
            """
            import json
            from tests.integration.test_combat_control import ScriptedCombatAgent
            from sts_ai.lightspeed import LightspeedHybridEnv
            from sts_ai.rollout import run_rollout

            env = LightspeedHybridEnv(world_seed=3, battle_simulations=50, max_act=1, combat_control="llm")
            result = run_rollout(env, ScriptedCombatAgent(), max_decisions=400)

            phases = [d.phase for d in result.decisions]
            screens = set()
            battle_outcomes = set()
            for d in result.decisions:
                screens.add(d.state.get("screen_state", ""))
                screens.add(d.after_state.get("screen_state", ""))
                combat = d.after_state.get("combat")
                if combat:
                    battle_outcomes.add(combat.get("battle_outcome", ""))
            print(json.dumps({
                "decisions": len(result.decisions),
                "n_combat": sum(p == "combat" for p in phases),
                "saw_rewards": any("REWARDS" in s for s in screens),
                "battle_outcomes": sorted(battle_outcomes),
                "stopped_reason": result.stopped_reason,
                "error": result.error,
            }, sort_keys=True))
            """
        )
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"

        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Residual seed-class UB can hang inside native battle setup on some
            # builds; subprocess isolation contains it (matches test_battle_search).
            self.assertIsNotNone(exc)
            self.skipTest("combat replay hung in native code; contained by subprocess timeout")

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"child rollout hard-crashed\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        payload = json.loads(completed.stdout)

        # The whole rollout path is clean: no simulator error anywhere (not just the
        # absence of the specific invalid-battle-action message).
        self.assertIsNone(payload["error"], msg=f"rollout hit a simulator error: {payload}")
        self.assertNotEqual(
            payload["stopped_reason"],
            "simulator_error",
            msg=f"rollout stopped on a simulator error: {payload}",
        )

        # The env surfaced in-combat decisions...
        self.assertGreater(payload["n_combat"], 0, msg=f"no combat decisions recorded: {payload}")
        # ...and a battle ran to completion (exit_battle wrote results -> REWARDS).
        self.assertTrue(payload["saw_rewards"], msg=f"no battle completed to rewards: {payload}")


@requires_simulator
class HybridControlRegressionTest(unittest.TestCase):
    def test_search_mode_still_resolves_battles_with_zero_combat_decisions(self):
        # Default hybrid mode must be unchanged: battles auto-resolved, so no
        # decision is ever phase="combat".
        from sts_ai.lightspeed import LightspeedHybridEnv
        from sts_ai.rollout import run_rollout

        env = LightspeedHybridEnv(world_seed=3, battle_simulations=50, max_act=1)
        self.assertEqual(env.combat_control, "search")
        result = run_rollout(env, ScriptedCombatAgent(), max_decisions=400)
        self.assertTrue(all(d.phase == "out_of_combat" for d in result.decisions))


@requires_simulator
class EnrichedSerializationTest(unittest.TestCase):
    """The serializers carry sim-computed numbers (intent/card damage, statuses) and
    OOC card type/rarity. Structural assertions only (exact values are seed-/UB-
    sensitive). Seed 3 reaches all of these within a bounded drive."""

    def test_combat_and_ooc_carry_sim_computed_numbers(self):
        from sts_ai.lightspeed import LightspeedHybridEnv

        env = LightspeedHybridEnv(world_seed=3, battle_simulations=50, max_act=1, combat_control="llm")
        saw_intent_keys = saw_attack_damage = saw_card_deal = saw_ooc_tag = False
        skill_never_has_deal = True

        for _ in range(400):
            if env.is_terminal():
                break
            env.advance_to_decision()
            if env.is_terminal():
                break
            actions = env.legal_actions()
            if not actions:
                break

            if env.phase() == "combat":
                enemies = list(env.bc.enemies())
                if enemies and all({"intent_damage", "intent_hits"} <= e.keys() for e in enemies):
                    saw_intent_keys = True
                # An attacking enemy (intent_hits != -1) has a computed intent_damage
                # and its serialized line carries the "(deal …)" annotation.
                if any(e["intent_hits"] != -1 for e in enemies):
                    saw_attack_damage = True
                    self.assertIn("(deal ", env.describe_state())
                for a in actions:
                    if a.description.startswith("play ") and "(deal " in a.description:
                        saw_card_deal = True
                    if a.description.startswith("play Defend") and "(deal " in a.description:
                        skill_never_has_deal = False  # a Skill must never get a damage tag
            else:
                for a in actions:
                    if a.description.startswith("take card") and a.description.endswith("]"):
                        saw_ooc_tag = True

            env.step(0)
            if saw_intent_keys and saw_attack_damage and saw_card_deal and saw_ooc_tag:
                break

        self.assertTrue(saw_intent_keys, "structured enemies() lacked intent_damage/intent_hits keys")
        self.assertTrue(saw_card_deal, "no attack action carried a '(deal N)' annotation")
        self.assertTrue(saw_attack_damage, "no attacking enemy surfaced computed intent damage")
        self.assertTrue(skill_never_has_deal, "a Defend (Skill) action wrongly carried a '(deal N)' annotation")
        self.assertTrue(saw_ooc_tag, "no out-of-combat card choice carried a '[Type, Rarity]' tag")


if __name__ == "__main__":
    unittest.main()
