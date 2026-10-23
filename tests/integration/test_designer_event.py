from __future__ import annotations

import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import requires_simulator


@requires_simulator
class DesignerEventTest(unittest.TestCase):
    def designer(self, seed, parent_x, child_x, ascension=0):
        # Start at a synthetic map frontier, then let the native event-generation
        # path initialize every event-specific field. No private event/state-info
        # writes or test-only native APIs are needed.
        sts = import_lightspeed()
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.act = 2
        gc.screen_state = sts.ScreenState.MAP_SCREEN
        gc.cur_map_node_x = parent_x
        gc.cur_map_node_y = 0
        enter = next(action for action in gc.legal_actions()
                     if action.describe(gc).startswith(f"choose map node x={child_x} "))
        enter.execute(gc)
        self.assertEqual(gc.current_event, "DESIGNER_IN_SPIRE")
        self.assertEqual(gc.screen_state, sts.ScreenState.EVENT_SCREEN)
        return sts, gc

    def test_transform_does_not_fall_into_full_service_or_leave(self):
        sts, gc = self.designer(319, 2, 3)
        self.assertFalse(gc.public_event()["cleanup_is_remove"])
        before_gold, before_hp = gc.gold, gc.cur_hp
        choice = next(action for action in gc.legal_actions()
                      if action.describe(gc) == "event option 3")
        choice.execute(gc)
        self.assertEqual(gc.gold, before_gold - 60)
        self.assertEqual(gc.cur_hp, before_hp)
        self.assertEqual(gc.screen_state, sts.ScreenState.MAP_SCREEN)

    def test_full_service_keeps_card_choice_without_leave_hp_loss(self):
        sts, gc = self.designer(90, 0, 1)
        before_gold, before_hp = gc.gold, gc.cur_hp
        choice = next(action for action in gc.legal_actions()
                      if action.describe(gc) == "event option 4")
        choice.execute(gc)
        self.assertEqual(gc.gold, before_gold - 90)
        self.assertEqual(gc.cur_hp, before_hp)
        self.assertEqual(gc.screen_state, sts.ScreenState.CARD_SELECT)
        self.assertTrue(gc.legal_actions())

    def test_random_adjustments_charge_gold_and_upgrade_two_cards(self):
        for ascension, seed, parent_x, child_x, cost in ((0, 90, 0, 1, 40),
                                                       (15, 2, 3, 4, 50)):
            with self.subTest(ascension=ascension):
                sts, gc = self.designer(seed, parent_x, child_x, ascension)
                self.assertFalse(gc.public_event()["upgrade_one"])
                before_gold, before_hp = gc.gold, gc.cur_hp
                before_deck = [(c.id, c.upgrade_count, c.misc) for c in gc.deck]
                choice = next(a for a in gc.legal_actions()
                              if a.describe(gc) == "event option 1")
                choice.execute(gc)
                self.assertEqual(gc.gold, before_gold - cost)
                self.assertEqual(gc.cur_hp, before_hp)
                self.assertEqual([(c.id, c.misc) for c in gc.deck],
                                 [(card_id, misc) for card_id, _, misc in before_deck])
                self.assertEqual(sum(c.upgrade_count for c in gc.deck),
                                 sum(upgrades for _, upgrades, _ in before_deck) + 2)
                self.assertEqual(gc.screen_state, sts.ScreenState.MAP_SCREEN)

    def test_adjustments_variants_have_the_same_affordability_threshold(self):
        for ascension, cost, variants in ((0, 40, ((319, 2, 3, 0), (90, 0, 1, 1))),
                                          (15, 50, ((162, 4, 5, 0), (2, 3, 4, 1)))):
            for seed, parent_x, child_x, index in variants:
                with self.subTest(ascension=ascension, index=index):
                    _, gc = self.designer(seed, parent_x, child_x, ascension)
                    self.assertEqual(gc.public_event()["upgrade_one"], index == 0)
                    description = f"event option {index}"
                    choice = next(a for a in gc.legal_actions() if a.describe(gc) == description)
                    gc.gold = cost - 1
                    self.assertFalse(choice.is_valid(gc))
                    self.assertNotIn(description, [a.describe(gc) for a in gc.legal_actions()])
                    gc.gold = cost
                    self.assertTrue(choice.is_valid(gc))
                    self.assertIn(description, [a.describe(gc) for a in gc.legal_actions()])
