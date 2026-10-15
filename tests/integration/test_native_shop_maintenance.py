from __future__ import annotations

import re
import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import requires_simulator


@requires_simulator
class ShopMaintenanceTests(unittest.TestCase):
    def setUp(self): self.sts=import_lightspeed()

    def shop(self, *, courier=True, membership=False, sozu=False, seed=3):
        # A constructed native fixture: skip to a real map shop through its
        # normal transition/setup path. This is not a played-world outcome.
        sts=self.sts;gc=sts.GameContext(sts.CharacterClass.IRONCLAD,seed,0)
        for _ in range(32):
            if gc.screen_state==sts.ScreenState.MAP_SCREEN:break
            gc.legal_actions()[0].execute(gc)
        self.assertEqual(gc.screen_state,sts.ScreenState.MAP_SCREEN)
        for active,name in [(courier,'THE_COURIER'),(membership,'MEMBERSHIP_CARD'),(sozu,'SOZU')]:
            if active:gc.obtain_relic(getattr(sts.RelicId,name))
        gc.gold=1000
        nodes=gc.map_graph()['nodes']
        target=next(n for n in nodes if n['room']=='SHOP' and n['y']>0)
        parent=next(n for n in nodes if n['y']==target['y']-1 and target['x'] in n['edges'])
        gc.cur_map_node_x=parent['x'];gc.cur_map_node_y=parent['y'];gc.floor_num=target['y']
        next(a for a in gc.legal_actions() if ('x='+str(target['x'])) in a.describe(gc)).execute(gc)
        self.assertEqual(gc.screen_state,sts.ScreenState.SHOP_ROOM)
        return gc

    def price(self,action,gc):return int(re.search(r'for (\d+)g',action.describe(gc)).group(1))

    def test_courier_potion_restock_keeps_base_price_and_applies_membership(self):
        prices=[]
        for member in (False,True):
            gc=self.shop(membership=member)
            action=next(a for a in gc.legal_actions() if a.describe(gc).startswith('buy potion'))
            before=gc.gold;charged=self.price(action,gc);bits=action.bits
            action.execute(gc)
            self.assertEqual(before-gc.gold,charged)
            restocked=next(a for a in gc.legal_actions() if a.bits==bits)
            price=self.price(restocked,gc);prices.append(price)
            self.assertGreaterEqual(price,18 if member else 36)
        self.assertLessEqual(abs(prices[1]-prices[0]/2),1)

    def test_full_inventory_disables_purchase_and_invalid_execution_keeps_gold(self):
        gc=self.shop()
        action=next(a for a in gc.legal_actions() if a.describe(gc).startswith('buy potion'))
        for _ in range(10):gc.obtain_potion(self.sts.Potion.FIRE_POTION)
        self.assertFalse(any(a.describe(gc).startswith('buy potion') for a in gc.legal_actions()))
        self.assertFalse(action.is_valid(gc))
        gold=gc.gold
        with self.assertRaisesRegex(ValueError,'invalid shop potion purchase'):action.execute(gc)
        self.assertEqual(gc.gold,gold)

    def test_sozu_has_no_futile_potion_purchases(self):
        gc=self.shop(sozu=True)
        self.assertFalse(any(a.describe(gc).startswith('buy potion') for a in gc.legal_actions()))

    def test_restocked_cards_do_not_receive_an_extra_courier_half_discount(self):
        prices=[]
        for member in (False,True):
            gc=self.shop(membership=member)
            action=next(a for a in gc.legal_actions() if a.describe(gc).startswith('buy card ') and 'remove' not in a.describe(gc))
            bits=action.bits;action.execute(gc)
            restock=next(a for a in gc.legal_actions() if a.bits==bits)
            price=self.price(restock,gc);prices.append(price)
            self.assertGreaterEqual(price,18 if member else 36)
        self.assertLessEqual(abs(prices[1]-prices[0]/2),1)

    def test_restocked_relic_charges_its_new_price(self):
        gc=self.shop()
        action=next(a for a in gc.legal_actions() if a.describe(gc).startswith('buy relic'))
        bits=action.bits;old_price=self.price(action,gc);action.execute(gc)
        restock=next(a for a in gc.legal_actions() if a.bits==bits)
        new_price=self.price(restock,gc)
        self.assertGreater(new_price,80)
        self.assertNotEqual(new_price,old_price)
        before=gc.gold;restock.execute(gc)
        self.assertEqual(before-gc.gold,new_price)
