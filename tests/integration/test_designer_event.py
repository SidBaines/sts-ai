from __future__ import annotations

import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import requires_simulator


@requires_simulator
class DesignerEventTest(unittest.TestCase):
    def designer(self, seed, parent_x, child_x):
        # Start at a synthetic map frontier, then let the native event-generation
        # path initialize every event-specific field. No private event/state-info
        # writes or test-only native APIs are needed.
        sts = import_lightspeed()
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
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
