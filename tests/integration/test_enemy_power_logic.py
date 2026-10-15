"""Vanilla enemy-power regressions exercised through real encounter/card logic."""
from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import load_search_test_fixture, requires_simulator

if TYPE_CHECKING:
    import slaythespire as sts


@requires_simulator
class EnemyPowerLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sts = import_lightspeed()
        cls.fixture = load_search_test_fixture()

    def battle(self, encounter: sts.MonsterEncounter, cards: list[sts.Card],
               act: int, ascension: int = 0) -> sts.BattleContext:
        gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, 3, ascension)
        gc.act = act
        for _ in range(len(gc.deck)):
            gc.remove_card(0)
        for card in cards:
            gc.obtain_card(card)
        gc = self.fixture.encounter_game_copy(gc, encounter)
        bc = self.sts.BattleContext()
        bc.init(gc)
        self.assertFalse(bc.undefined_behavior_evoked)
        return bc

    def cards(self, card_id: sts.CardId, count: int = 5) -> list[sts.Card]:
        return [self.sts.Card(card_id) for _ in range(count)]

    def play(self, bc: sts.BattleContext, name: str,
             target: int | None = None) -> None:
        action = next(a for a in bc.legal_actions()
                      if a.describe(bc).startswith(f"play {name} ")
                      and (target is None or a.target_idx() == target))
        action.execute(bc)
        self.assertFalse(bc.undefined_behavior_evoked)

    def end_turn(self, bc: sts.BattleContext) -> None:
        next(a for a in bc.legal_actions()
             if a.action_type() == self.sts.BattleActionType.END_TURN).execute(bc)
        self.assertFalse(bc.undefined_behavior_evoked)

    def test_curiosity_adds_ascension_scaled_strength_for_each_power(self) -> None:
        for ascension, amount in ((0, 1), (19, 2)):
            with self.subTest(ascension=ascension):
                bc = self.battle(self.sts.MonsterEncounter.AWAKENED_ONE,
                                 self.cards(self.sts.CardId.INFLAME),
                                 act=3, ascension=ascension)
                before = bc.enemies()
                self.assertEqual(before[2]["powers"]["Curiosity"], amount)
                for played in (1, 2):
                    self.play(bc, "Inflame")
                    after = bc.enemies()
                    self.assertEqual(after[2]["strength"],
                                     before[2]["strength"] + played * amount)
                    self.assertEqual([e["strength"] for e in after[:2]],
                                     [e["strength"] for e in before[:2]])

    def test_curiosity_does_not_trigger_on_attacks_or_skills(self) -> None:
        for card_id, name in ((self.sts.CardId.STRIKE_RED, "Strike"),
                              (self.sts.CardId.DEFEND_RED, "Defend")):
            with self.subTest(card=name):
                bc = self.battle(self.sts.MonsterEncounter.AWAKENED_ONE,
                                 self.cards(card_id), act=3)
                strength = bc.enemies()[2]["strength"]
                self.play(bc, name, 2 if name == "Strike" else None)
                self.assertEqual(bc.enemies()[2]["strength"], strength)

    def test_curiosity_skips_holder_half_killed_by_panache_on_same_power(self) -> None:
        for ascension in (0, 19):
            with self.subTest(ascension=ascension):
                cards = [self.sts.Card(card_id) for card_id in (
                    self.sts.CardId.PANACHE, self.sts.CardId.ANGER,
                    self.sts.CardId.FINESSE, self.sts.CardId.DEFEND_RED,
                    self.sts.CardId.INFLAME,
                )]
                bc = self.battle(self.sts.MonsterEncounter.AWAKENED_ONE, cards,
                                 act=3, ascension=ascension)
                self.play(bc, "Panache")
                self.play(bc, "Anger", 0)
                self.play(bc, "Finesse")  # Redraw Anger from the discard pile.
                self.play(bc, "Anger", 0)
                self.play(bc, "Defend")
                # Only HP is constructed: the next real Power play triggers
                # Panache before the queued Curiosity buff can resolve.
                bc = self.fixture.monster_hp_copy(bc, 2, 1)
                before = bc.enemies()[2]
                self.assertIn("Curiosity", before["powers"])
                self.play(bc, "Inflame")
                after = bc.enemies()[2]
                self.assertEqual(after["cur_hp"], 0)
                self.assertNotIn("Curiosity", after["powers"])
                self.assertEqual(after["strength"], before["strength"])
                self.end_turn(bc)
                reborn = bc.enemies()[2]
                self.assertGreater(reborn["cur_hp"], 0)
                self.assertEqual(reborn["strength"], before["strength"])

    def test_curiosity_stops_after_awakened_one_rebirth(self) -> None:
        killing_card = self.sts.Card(self.sts.CardId.SEARING_BLOW)
        for _ in range(30):
            killing_card.upgrade()
        bc = self.battle(self.sts.MonsterEncounter.AWAKENED_ONE,
                         [killing_card] + self.cards(self.sts.CardId.INFLAME, 4), act=3)
        self.play(bc, "Searing Blow+30", 2)
        self.assertEqual(bc.enemies()[2]["cur_hp"], 0)
        self.assertNotIn("Curiosity", bc.enemies()[2]["powers"])
        # Play while half dead as well as after the natural rebirth turn.
        strength = bc.enemies()[2]["strength"]
        self.play(bc, "Inflame")
        self.assertEqual(bc.enemies()[2]["strength"], strength)
        self.end_turn(bc)
        self.assertGreater(bc.enemies()[2]["cur_hp"], 0)
        strength = bc.enemies()[2]["strength"]
        self.play(bc, "Inflame")
        self.assertEqual(bc.enemies()[2]["strength"], strength)

    def test_intangible_caps_annotated_and_executed_attack_damage(self) -> None:
        for card_id, name, damage in ((self.sts.CardId.STRIKE_RED, "Strike", 1),
                                      (self.sts.CardId.BODY_SLAM, "Body Slam", 0)):
            with self.subTest(card=name):
                bc = self.battle(self.sts.MonsterEncounter.NEMESIS,
                                 self.cards(card_id), act=3)
                self.end_turn(bc)  # Nemesis naturally gains Intangible.
                self.assertIn("Intangible", bc.enemies()[0]["powers"])
                self.assertEqual(bc.enemies()[0]["block"], 0)
                action = next(a for a in bc.legal_actions()
                              if a.describe(bc).startswith(f"play {name} "))
                self.assertIn(f"(deal {damage})", action.describe(
                    bc, include_computed_damage=True))
                hp = bc.enemies()[0]["cur_hp"]
                action.execute(bc)
                self.assertEqual(hp - bc.enemies()[0]["cur_hp"], damage)
                self.assertFalse(bc.undefined_behavior_evoked)

    def test_lagavulin_natural_wake_removes_sleep_powers_once(self) -> None:
        bc = self.battle(self.sts.MonsterEncounter.LAGAVULIN,
                         self.cards(self.sts.CardId.DEFEND_RED), act=1)
        initial_hp = bc.enemies()[0]["cur_hp"]
        for turn in range(3):
            self.assertEqual(bc.turn, turn)
            enemy = bc.enemies()[0]
            self.assertIn("Asleep", enemy["powers"])
            self.assertEqual(enemy["powers"]["Metallicize"], 8)
            self.assertEqual(enemy["block"], 8)
            self.assertEqual(enemy["intent"], "LAGAVULIN_SLEEP")
            self.end_turn(bc)
        # Preserve the existing transition: attack is selected after three idles.
        for intent in ("LAGAVULIN_ATTACK", "LAGAVULIN_ATTACK",
                       "LAGAVULIN_SIPHON_SOUL"):
            enemy = bc.enemies()[0]
            self.assertEqual(enemy["cur_hp"], initial_hp)
            self.assertNotIn("Asleep", enemy["powers"])
            self.assertEqual(enemy["powers"].get("Metallicize", 0), 0)
            self.assertEqual(enemy["block"], 0)
            self.assertEqual(enemy["intent"], intent)
            self.end_turn(bc)

    def test_lagavulin_damage_wake_still_requires_unblocked_damage(self) -> None:
        bc = self.battle(self.sts.MonsterEncounter.LAGAVULIN,
                         self.cards(self.sts.CardId.STRIKE_RED), act=1)
        hp = bc.enemies()[0]["cur_hp"]
        self.play(bc, "Strike", 0)  # Six damage is fully blocked by the initial eight.
        self.assertEqual(bc.enemies()[0]["cur_hp"], hp)
        self.assertIn("Asleep", bc.enemies()[0]["powers"])
        self.play(bc, "Strike", 0)
        self.assertEqual(bc.enemies()[0]["cur_hp"], hp - 4)
        self.assertNotIn("Asleep", bc.enemies()[0]["powers"])
        self.assertEqual(bc.enemies()[0]["powers"].get("Metallicize", 0), 0)
        self.assertEqual(bc.enemies()[0]["intent"], "LAGAVULIN_SLEEP")
        self.end_turn(bc)
        self.assertEqual(bc.enemies()[0]["intent"], "LAGAVULIN_ATTACK")
        self.assertEqual(bc.enemies()[0]["powers"].get("Metallicize", 0), 0)
        self.assertEqual(bc.enemies()[0]["block"], 0)


if __name__ == "__main__":
    unittest.main()
