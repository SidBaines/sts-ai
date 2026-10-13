"""Native parity and determinism checks without model inference or a rebuild."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from sts_ai.combat_public_v5 import CombatPublicV5Env
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import prepare_decision
from tests.support import requires_simulator


def card_fixture(name: str, upgraded: bool = False, smooth: bool = False):
    env = LightspeedHybridEnv(world_seed=4, max_act=1, combat_control="llm",
                             combat_observation="combat_public_v3")
    for _ in range(20):
        if env.gc.screen_state == env.sts.ScreenState.BATTLE:
            break
        list(env.gc.legal_actions())[0].execute(env.gc)
    else:
        raise AssertionError("bounded fixture did not reach a battle")
    for _ in range(len(env.gc.deck)):
        env.gc.remove_card(0)
    for _ in range(5):
        card = env.sts.Card(getattr(env.sts.CardId, name))
        if upgraded:
            card.upgrade()
        env.gc.obtain_card(card)
    if smooth:
        env.gc.obtain_relic(env.sts.RelicId.ODDLY_SMOOTH_STONE)
    env.advance_to_decision()
    return env


@requires_simulator
class CombatPublicV5IntegrationTest(unittest.TestCase):
    def test_block_annotations_agree_with_executed_card_and_preserve_actions(self):
        for name, upgraded, smooth, expected in (("DEFEND_RED", False, False, 5),
                                                 ("DEFEND_RED", True, True, 9),
                                                 ("FLAME_BARRIER", True, False, 16),
                                                 ("IRON_WAVE", False, True, 7)):
            base = card_fixture(name, upgraded, smooth)
            control = card_fixture(name, upgraded, smooth)
            wrapped = CombatPublicV5Env(base)
            _, view = prepare_decision(wrapped)
            self.assertIn(f": {expected} (includes current Dexterity", view["state_text"])
            self.assertEqual([(a.index, a.bits) for a in view["legal_actions"]],
                             [(a.index, a.bits) for a in control.legal_actions()])
            action = next(a for a in wrapped.legal_actions() if a.description.startswith("play "))
            before = base.bc.player_block
            wrapped.step(action.index)
            control.step(action.index)
            self.assertEqual(base.bc.player_block - before, expected)
            self.assertEqual(base.summary(), control.summary())

    def test_optional_exhaust_label_executes_the_same_zero_mask(self):
        base = card_fixture("PURITY")
        control = card_fixture("PURITY")
        wrapped = CombatPublicV5Env(base)
        action = next(a for a in wrapped.legal_actions() if a.description.startswith("play Purity"))
        wrapped.step(action.index)
        control.step(action.index)
        _, view = prepare_decision(wrapped)
        self.assertEqual(view["legal_actions"][0].description, "exhaust no cards")
        self.assertIn("Exhausting no cards finishes", view["state_text"])
        self.assertEqual(wrapped.public_combat_cards()["selection"]["candidates"], ["exhaust no cards"])
        wrapped.step(0)
        control.step(0)
        self.assertEqual(base.summary(), control.summary())

    def test_v5_is_deterministic_across_processes(self):
        driver = '''
import json
from sts_ai.combat_public_v5 import CombatPublicV5Env
from sts_ai.rollout import prepare_decision
from tests.integration.test_combat_public_v5 import card_fixture
env = CombatPublicV5Env(card_fixture("PURITY"))
trace = []
for _ in range(4):
    status, view = prepare_decision(env)
    trace.append([status, view['state'], view['state_text'], view['legal_action_dicts']])
    env.step(0)
print(json.dumps(trace, sort_keys=True))
'''
        outputs = [subprocess.run([sys.executable, "-c", driver], env=dict(os.environ, PYTHONPATH="src"),
                                  cwd=Path.cwd(), capture_output=True, text=True,
                                  check=True, timeout=60).stdout for _ in range(2)]
        self.assertEqual(json.loads(outputs[0]), json.loads(outputs[1]))
