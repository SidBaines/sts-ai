"""Real-engine checks for v4 presentation and unchanged action execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from sts_env._backend.combat_public_v4 import CombatPublicV4Env
from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.native_helpers import prepare_decision

from tests.support import requires_simulator


def _gambling_chip_env() -> LightspeedHybridEnv:
    env = LightspeedHybridEnv(world_seed=4, max_act=1, combat_control="llm",
                             combat_observation="combat_public_v3")
    for _ in range(20):
        if env.gc.screen_state == env.sts.ScreenState.BATTLE:
            break
        actions = list(env.gc.legal_actions())
        if not actions:
            raise AssertionError("could not reach battle")
        actions[0].execute(env.gc)
    else:
        raise AssertionError("could not reach battle within bounded setup")
    env.gc.obtain_relic(env.sts.RelicId.GAMBLING_CHIP)
    env.advance_to_decision()
    return env


@requires_simulator
class CombatPublicV4IntegrationTest(unittest.TestCase):
    def test_gambling_chip_public_surfaces_match_and_preserve_native_execution(self) -> None:
        base = _gambling_chip_env()
        control = _gambling_chip_env()
        wrapped = CombatPublicV4Env(base)
        self.assertEqual(base.summary(), control.summary())
        self.assertEqual(len(base.legal_actions()), 1)
        self.assertEqual(base.legal_actions()[0].description, "{ GAMBLE none")
        action = wrapped.legal_actions()[0]
        self.assertEqual(action.description, "keep current hand (discard no cards)")
        self.assertEqual((action.index, action.bits),
                         (base.legal_actions()[0].index, base.legal_actions()[0].bits))
        self.assertIn("  - " + action.description, wrapped.describe_state())
        self.assertEqual(wrapped.public_combat_cards()["selection"]["candidates"],
                         [action.description])
        self.assertEqual(wrapped.summary()["combat"]["cards"]["selection"]["candidates"],
                         [action.description])
        status, view = prepare_decision(wrapped)
        self.assertEqual(status, "ok")
        self.assertEqual(view["legal_actions"], [action])
        self.assertEqual(view["state"]["combat_observation"], "combat_public_v4")
        self.assertIn("A card selection is pending.", view["state_text"])
        self.assertIn("Gambling Chip:", view["state_text"])
        self.assertNotIn("not enough energy", view["state_text"])
        self.assertNotIn("{ GAMBLE none", view["state_text"])
        hand = base.public_combat_cards()["hand"]
        self.assertEqual(wrapped.step(0), action)
        control.step(0)
        self.assertEqual(base.summary(), control.summary())
        self.assertEqual(base.public_combat_cards()["hand"], hand)
        self.assertIn(action.description, wrapped.summary()["combat"]["recent_actions"])
        self.assertIn(action.description, wrapped.describe_state())
        self.assertEqual(base.combat_observation, "combat_public_v3")
        for _ in range(5):
            wrapped.step(0)
            control.step(0)
            self.assertEqual(base.summary(), control.summary())

    def test_normal_turn_math_is_scoped_and_uses_live_shown_values(self) -> None:
        wrapped = CombatPublicV4Env(_gambling_chip_env())
        self.assertEqual(wrapped._turn_math_lines(), [])
        wrapped.step(0)
        lines = wrapped._turn_math_lines()
        self.assertEqual(len(lines), 2)
        self.assertIn("does not predict HP loss", lines[0])
        self.assertIn(f"current energy {wrapped.bc.player_energy}", lines[1])
        self.assertNotIn("not lethal this turn", "\n".join(lines))
        self.assertNotIn("you would take", "\n".join(lines))

    def test_v4_actions_and_states_are_deterministic_across_processes(self) -> None:
        driver = '''
import json
from sts_env._backend.combat_public_v4 import CombatPublicV4Env
from tests.integration.test_combat_public_v4 import _gambling_chip_env
env = CombatPublicV4Env(_gambling_chip_env())
trace = []
for _ in range(6):
    trace.append([env.summary(), env.describe_state(), env._turn_math_lines(),
                  [env.action_dict(a) for a in env.legal_actions()]])
    env.step(0)
print(json.dumps(trace, sort_keys=True))
'''
        child_env = dict(os.environ, PYTHONPATH="src")
        outputs = []
        for _ in range(2):
            completed = subprocess.run([sys.executable, "-c", driver], env=child_env,
                                       cwd=Path.cwd(), text=True, capture_output=True,
                                       timeout=60, check=True)
            json.loads(completed.stdout)
            outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])
