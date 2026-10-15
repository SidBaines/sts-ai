"""Run resources follow the live battle, then return to GameContext at exit."""
from __future__ import annotations

import unittest

from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.integration.test_public_combat_teacher import _advance_to_first_combat
from tests.support import requires_simulator


@requires_simulator
class LiveCombatResourcesTest(unittest.TestCase):
    def assert_live_resources(self, env: LightspeedHybridEnv) -> None:
        player = env.public_combat_player()
        state = env.summary()
        for name in ("cur_hp", "max_hp", "gold"):
            self.assertEqual(state[name], player[name], name)
        self.assertEqual(state["cur_hp"], state["combat"]["player_cur_hp"])
        self.assertEqual(state["max_hp"], state["combat"]["player_max_hp"])
        self.assertIn(
            f"\nPlayer HP: {player['cur_hp']}/{player['max_hp']},",
            env.describe_state(),
        )

    def assert_run_resources(self, env: LightspeedHybridEnv) -> None:
        self.assertIsNone(env.bc)
        state = env.summary()
        for name in ("cur_hp", "max_hp", "gold"):
            self.assertEqual(state[name], int(getattr(env.gc, name)), name)
        self.assertEqual(env.describe_state(), str(env.gc.describe_state()))
        self.assertIn(
            f"\nYour HP: {state['cur_hp']}/{state['max_hp']}, gold: {state['gold']}",
            env.describe_state(),
        )

    def test_damage_max_hp_and_gold_are_live_until_battle_exit(self):
        for observation in ("legacy", "combat_public_v1", "combat_public_v2", "combat_public_v3"):
            with self.subTest(observation=observation):
                env = LightspeedHybridEnv(
                    world_seed=3,
                    max_act=1,
                    combat_control="llm",
                    combat_observation=observation,
                )
                self.assert_run_resources(env)
                # Seed 3 starts with two slimes. Hand of Greed can kill one and
                # change gold while the other keeps the combat decision live.
                for _ in range(len(env.gc.deck)):
                    env.gc.remove_card(0)
                for _ in range(10):
                    env.gc.obtain_card(env.sts.Card(env.sts.CardId.HAND_OF_GREED))
                _advance_to_first_combat(env)
                entry = {name: int(getattr(env.gc, name))
                         for name in ("cur_hp", "max_hp", "gold")}
                self.assert_live_resources(env)

                for _ in range(5):
                    end_turn = next(a.index for a in env.legal_actions()
                                    if a.description == "end turn")
                    env.step(end_turn)
                    self.assertIsNotNone(env.bc)
                    self.assert_live_resources(env)
                    if int(env.bc.player_cur_hp) < entry["cur_hp"]:
                        break
                else:
                    self.fail("bounded end turns did not cause damage")
                self.assertLess(env.summary()["cur_hp"], entry["cur_hp"])

                env.bc.obtain_potion(env.sts.Potion.FRUIT_JUICE)
                drink = next(a.index for a in env.legal_actions()
                             if "drink" in a.description.lower() and "Fruit Juice" in a.description)
                env.step(drink)
                self.assert_live_resources(env)
                self.assertEqual(env.summary()["max_hp"], entry["max_hp"] + 5)

                kill = next(a.index for a in env.legal_actions()
                            if a.description.startswith("play Hand of Greed")
                            and "ACID_SLIME_S" in a.description)
                env.step(kill)
                self.assertIsNotNone(env.bc)
                self.assert_live_resources(env)
                self.assertEqual(env.summary()["gold"], entry["gold"] + 20)
                # Reading live resources must not synchronize/mutate GameContext.
                self.assertEqual(
                    {name: int(getattr(env.gc, name)) for name in entry}, entry,
                )

                for _ in range(30):
                    env.step(0)
                    if env.bc is None:
                        break
                    self.assert_live_resources(env)
                else:
                    self.fail("bounded actions did not finish the fight")
                self.assertFalse(env.is_terminal(), "fixture must win the fight")
                self.assertEqual(env.gc.screen_state, env.sts.ScreenState.REWARDS)
                self.assert_run_resources(env)
                self.assertEqual(env.summary()["max_hp"], entry["max_hp"] + 5)
                self.assertGreater(env.summary()["gold"], entry["gold"])


if __name__ == "__main__":
    unittest.main()
