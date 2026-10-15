from __future__ import annotations

import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import requires_simulator


@requires_simulator
class NativeBattleInitializationTests(unittest.TestCase):
    def setUp(self):
        self.sts = import_lightspeed()

    def first_battle(self, character_class):
        sts = self.sts
        gc = sts.GameContext(character_class, 3, 0)
        for _ in range(100):
            if gc.screen_state == sts.ScreenState.BATTLE:
                return gc
            actions = gc.legal_actions()
            self.assertTrue(actions, "seeded fixture must reach its first combat")
            actions[0].execute(gc)
        self.fail("seeded fixture exceeded bounded first-combat horizon")

    def test_entropic_brew_uses_actual_character_class_after_other_classes(self):
        sts = self.sts
        fixtures = [
            (sts.CharacterClass.IRONCLAD, "Heart Of Iron"),
            (sts.CharacterClass.SILENT, "Ghost In A Jar"),
            (sts.CharacterClass.DEFECT, "Essence Of Darkness"),
            (sts.CharacterClass.WATCHER, "Ambrosia"),
        ]
        snapshots = []
        for character_class, expected in fixtures + fixtures:
            with self.subTest(character_class=character_class):
                bc = sts.BattleContext()
                bc.init(self.first_battle(character_class))
                self.assertEqual(bc.player_character_class, character_class)
                bc.obtain_potion(sts.Potion.ENTROPIC_BREW)
                action = next(a for a in bc.legal_actions() if "Entropic Brew" in a.describe(bc))
                action.execute(bc)
                potions = bc.public_potions()
                self.assertEqual(potions["occupied"], 3)
                self.assertEqual(potions["slots"][0]["name"], expected)
                self.assertEqual([p["name"] for p in potions["slots"][1:]],
                                 ["Ancient Potion", "Explosive Potion"])
                snapshots.append(potions)
        self.assertEqual(snapshots[:4], snapshots[4:])

    def test_ironclad_card_potions_offer_matching_card_types(self):
        sts = self.sts
        for name, card_type in (("ATTACK_POTION", "Attack"),
                                ("SKILL_POTION", "Skill"),
                                ("POWER_POTION", "Power")):
            with self.subTest(potion=name):
                bc = sts.BattleContext()
                bc.init(self.first_battle(sts.CharacterClass.IRONCLAD))
                bc.obtain_potion(getattr(sts.Potion, name))
                next(a for a in bc.legal_actions() if "drink" in a.describe(bc).lower()).execute(bc)
                self.assertEqual(bc.input_state, sts.InputState.CARD_SELECT)
                candidates = bc.public_cards(True)["selection"]["candidates"]
                self.assertEqual(len(candidates), 3)
                self.assertTrue(all(f"[{card_type}]" in card for card in candidates))
                self.assertFalse(bc.undefined_behavior_evoked)

    def test_auto_combat_rejects_live_undefined_behavior_before_exit(self):
        sts = self.sts
        gc = self.first_battle(sts.CharacterClass.IRONCLAD)
        for _ in range(len(gc.deck)):
            gc.remove_card(0)
        gc.obtain_card(sts.Card(sts.CardId.DISCOVERY))
        gc.cur_hp = gc.max_hp = 100
        agent = sts.Agent()
        agent.simulation_count_base = 50
        before_hp = gc.cur_hp
        with self.assertRaisesRegex(RuntimeError, "native battle playout evoked undefined behavior"):
            sts.resolve_current_battle(gc, agent)
        self.assertEqual(gc.screen_state, sts.ScreenState.BATTLE)
        self.assertEqual(gc.cur_hp, before_hp)

    def test_potion_setup_rejects_empty_sentinel(self):
        sts = self.sts
        gc = self.first_battle(sts.CharacterClass.IRONCLAD)
        bc = sts.BattleContext()
        bc.init(gc)
        for context in (gc, bc):
            with self.assertRaises(ValueError):
                context.obtain_potion(sts.Potion.EMPTY_POTION_SLOT)
