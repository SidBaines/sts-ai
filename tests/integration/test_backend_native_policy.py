from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.support import requires_simulator


@requires_simulator
class NativeOutOfCombatPolicyTest(unittest.TestCase):
    def _reward_env(self, *, require_gold=False):
        env = LightspeedHybridEnv(3, battle_simulations=10, max_act=1)
        for _ in range(100):
            actions = env.legal_actions()
            if (any(action.description.startswith("take card ") for action in actions)
                    and (not require_gold or any(action.description.startswith("take gold") for action in actions))):
                return env
            env.step(0)
        raise AssertionError("bounded fixture did not reach card rewards")

    def test_individual_card_skip_preserves_other_rewards(self):
        env = self._reward_env(require_gold=True)
        before = env.legal_actions()
        others = [(a.bits, a.description) for a in before
                  if not a.description.startswith(("take card ", "skip card reward"))]
        self.assertTrue(any(d.startswith("take gold") for _, d in others))
        skip = next(a for a in before if a.description == "skip card reward")
        deck_size = len(env.gc.deck)
        env.step(skip.index)
        self.assertEqual(len(env.gc.deck), deck_size)
        self.assertEqual([(a.bits, a.description) for a in env.legal_actions()], others)

    def test_singing_bowl_card_skip_grants_two_max_hp(self):
        env = self._reward_env()
        env.gc.obtain_relic(env.sts.RelicId.SINGING_BOWL)
        max_hp = env.gc.max_hp
        skip = next(a for a in env.legal_actions() if a.description.startswith("gain +2 max hp (Singing Bowl)"))
        env.step(skip.index)
        self.assertEqual(env.gc.max_hp, max_hp + 2)

    def test_chest_policy_opens_chest_without_cursed_key(self):
        env = LightspeedHybridEnv(3, battle_simulations=10, max_act=1)
        env.gc.screen_state = env.sts.ScreenState.TREASURE_ROOM
        # Selection-only policy reads no uninitialized chest metadata unless the
        # Cursed Key is held. Execute is intentionally avoided in this fixture.
        choice = env.native_ooc_action()
        self.assertEqual(env.legal_actions()[choice].description, "open chest")

    def test_ooc_preview_preserves_world_and_is_repeatable(self):
        env = LightspeedHybridEnv(3, battle_simulations=10, max_act=1)
        env.seed_native_ooc(17)
        before = deepcopy(env.summary())
        first = env.native_ooc_action()
        self.assertEqual(env.native_ooc_action(), first)
        self.assertEqual(env.summary(), before)
        chosen = env.advance_native_ooc()
        self.assertEqual(chosen.index, first)

    def test_weighted_reward_sampler_can_choose_multiple_cards(self):
        driver = '''
from sts_env._backend.lightspeed import LightspeedHybridEnv
import json
env = LightspeedHybridEnv(3, battle_simulations=10, max_act=1)
for _ in range(100):
    actions = env.legal_actions()
    cards = [a for a in actions if a.description.startswith("take card ")]
    if cards and all(a.description.startswith(("take card ", "skip card reward", "skip rewards")) for a in actions):
        selected = set()
        before = env.summary()
        for seed in range(2, 202):
            env.seed_native_ooc(seed)
            selected.add(env.legal_actions()[env.native_ooc_action()].description)
        assert env.summary() == before
        print(json.dumps(sorted(selected)))
        break
    # Take gold/relic/potion first, remove optional reward actions, then menu
    # contains only card choices and skip; no future state/rollout evaluation.
    take = next((a for a in actions if a.description.startswith(("take gold", "take relic", "take potion"))), actions[0])
    env.step(take.index)
else:
    raise AssertionError("bounded fixture did not reach card rewards")
'''
        result = subprocess.run([sys.executable, "-c", driver],
                                cwd=Path(__file__).resolve().parents[2],
                                env=dict(os.environ, PYTHONPATH="src"),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        selections = [x for x in json.loads(result.stdout) if x.startswith("take card ")]
        self.assertGreater(len(selections), 1, selections)

    def test_visible_map_exists_on_neow_and_after_leaving_map(self):
        env = LightspeedHybridEnv(3, battle_simulations=10, max_act=1)
        self.assertTrue(env.map_graph()["nodes"])
        env.step(0)
        while env.gc.screen_state == env.sts.ScreenState.CARD_SELECT:
            env.step(0)
        self.assertTrue(env.map_graph()["nodes"])
