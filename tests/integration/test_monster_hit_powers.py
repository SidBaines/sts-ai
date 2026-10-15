"""Flight expires on grounding; Angry requires positive post-block attack damage."""
from __future__ import annotations

import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import load_search_test_fixture, requires_simulator


@requires_simulator
class MonsterHitPowersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sts = import_lightspeed()
        cls.fixture = load_search_test_fixture()

    def battle(self, encounter, ascension: int):
        sts = self.sts
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 3, ascension)
        for _ in range(100):
            if gc.screen_state == sts.ScreenState.BATTLE:
                break
            gc.legal_actions()[0].execute(gc)
        else:
            self.fail("Seed 3 must reach combat")
        for _ in range(len(gc.deck)):
            gc.remove_card(0)
        for _ in range(5):
            gc.obtain_card(sts.Card(sts.CardId.STRIKE_RED))
        # Select the encounter without constructing monster HP, starting powers
        # or moves. Attack damage and subsequent turns use the real engine.
        gc = self.fixture.encounter_game_copy(gc, encounter)
        bc = sts.BattleContext()
        bc.init(gc)
        self.assertFalse(bc.undefined_behavior_evoked)
        return bc

    def strike(self, bc, target: int) -> None:
        action = next(a for a in bc.legal_actions()
                      if a.action_type() == self.sts.BattleActionType.CARD
                      and a.target_idx() == target)
        self.assertIn("Strike", action.describe(bc))
        action.execute(bc)
        self.assertFalse(bc.undefined_behavior_evoked)

    def end_turn(self, bc) -> None:
        next(a for a in bc.legal_actions()
             if a.action_type() == self.sts.BattleActionType.END_TURN).execute(bc)
        self.assertFalse(bc.undefined_behavior_evoked)

    def test_grounded_byrd_stays_grounded_until_fly_reapplies_exact_flight(self) -> None:
        for ascension, flight in ((0, 3), (17, 4)):
            with self.subTest(ascension=ascension):
                bc = self.battle(self.sts.MonsterEncounter.THREE_BYRDS, ascension)
                bc = self.fixture.combat_hit_setup_copy(bc, 0, 0, 10, 0)
                self.assertEqual(bc.enemies()[0]["powers"]["Flight"], flight)
                for hit in range(flight):
                    before = bc.enemies()[0]["cur_hp"]
                    self.strike(bc, 0)
                    enemy = bc.enemies()[0]
                    self.assertEqual(before - enemy["cur_hp"], 3)
                    if hit + 1 < flight:
                        self.assertEqual(enemy["powers"]["Flight"], flight - hit - 1)
                    else:
                        self.assertNotIn("Flight", enemy["powers"])
                        self.assertEqual(enemy["intent"], "BYRD_STUNNED")

                # A further hit on the same turn must receive no Flight reduction.
                copy = self.fixture.copy(bc)
                before = copy.enemies()[0]["cur_hp"]
                self.strike(copy, 0)
                self.assertEqual(before - copy.enemies()[0]["cur_hp"], 6)

                for intent in ("BYRD_HEADBUTT", "BYRD_FLY"):
                    self.end_turn(bc)
                    enemy = bc.enemies()[0]
                    self.assertNotIn("Flight", enemy["powers"])
                    self.assertEqual(enemy["intent"], intent)
                    copy = self.fixture.copy(bc)
                    before = copy.enemies()[0]["cur_hp"]
                    self.strike(copy, 0)
                    self.assertEqual(before - copy.enemies()[0]["cur_hp"], 6)

                self.end_turn(bc)
                self.assertEqual(bc.enemies()[0]["powers"]["Flight"], flight)

    def test_airborne_byrd_still_resets_flight_at_turn_start(self) -> None:
        for ascension, flight in ((0, 3), (17, 4)):
            with self.subTest(ascension=ascension):
                bc = self.battle(self.sts.MonsterEncounter.THREE_BYRDS, ascension)
                self.strike(bc, 0)
                self.assertEqual(bc.enemies()[0]["powers"]["Flight"], flight - 1)
                self.end_turn(bc)
                self.assertEqual(bc.enemies()[0]["powers"]["Flight"], flight)

    def test_mad_gremlin_angry_requires_positive_post_block_attack_damage(self) -> None:
        for ascension, angry in ((0, 1), (17, 2)):
            bc = self.battle(self.sts.MonsterEncounter.GREMLIN_GANG, ascension)
            target = next(e["index"] for e in bc.enemies() if "Angry" in e["powers"])
            self.assertEqual(bc.enemies()[target]["name"], "MAD_GREMLIN")
            self.assertEqual(bc.enemies()[target]["powers"]["Angry"], angry)
            for block, strength, damage in ((8, 0, 0), (6, 0, 0), (0, -6, 0),
                                           (5, 0, 1), (0, 0, 6)):
                with self.subTest(ascension=ascension, block=block, strength=strength):
                    copy = self.fixture.combat_hit_setup_copy(bc, target, block, 3, strength)
                    before = copy.enemies()[target]
                    self.strike(copy, target)
                    after = copy.enemies()[target]
                    self.assertEqual(before["cur_hp"] - after["cur_hp"], damage)
                    self.assertEqual(after["block"], max(0, block - (6 + strength)))
                    self.assertEqual(after["strength"] - before["strength"],
                                     angry if damage > 0 else 0)

            # Explosive Potion follows the non-attack damage path and must not enrage.
            # Avoid killing the target so Strength remains observable on a live enemy.
            copy = self.fixture.copy(bc)
            copy.obtain_potion(self.sts.Potion.EXPLOSIVE_POTION)
            before = copy.enemies()[target]
            next(a for a in copy.legal_actions()
                 if "drink" in a.describe(copy)).execute(copy)
            after = copy.enemies()[target]
            self.assertEqual(before["cur_hp"] - after["cur_hp"], 10)
            self.assertTrue(after["alive"])
            self.assertEqual(after["strength"], before["strength"])
            self.assertFalse(copy.undefined_behavior_evoked)


if __name__ == "__main__":
    unittest.main()
