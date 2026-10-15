from __future__ import annotations

import unittest

from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import requires_simulator


@requires_simulator
class EventFixTests(unittest.TestCase):
    def setUp(self):
        self.sts = import_lightspeed()

    def event(self, name: str, seed: int, act: int = 1, ascension: int = 0):
        # Construct a map frontier, then use normal event generation. Seeds were
        # found by a bounded search over the first event node at row 6 or later.
        # These are synthetic fixtures, not outcomes of played worlds. Never set
        # curEvent, event-specific state-info, RNGs or relic pools directly.
        sts = self.sts
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.screen_state = sts.ScreenState.MAP_SCREEN
        if act > 1:
            self.advance_acts(gc, act)
        nodes = gc.map_graph()["nodes"]
        target = next(n for n in nodes if n["room"] == "EVENT" and n["y"] >= 6)
        parent = next(n for n in nodes
                      if n["y"] == target["y"] - 1 and target["x"] in n["edges"])
        gc.cur_map_node_x, gc.cur_map_node_y = parent["x"], parent["y"]
        # Mushrooms only spawns after floor 6; keep the floor consistent with
        # the current map row before entering the event.
        gc.floor_num = target["y"] + (act - 1) * 17
        next(action for action in gc.legal_actions()
             if action.describe(gc).startswith(f"choose map node x={target['x']} ")).execute(gc)
        self.assertEqual(gc.current_event, name)
        self.assertEqual(gc.screen_state, sts.ScreenState.EVENT_SCREEN)
        expected = range(4) if name == "THE_WOMAN_IN_BLUE" else range(2)
        self.assertEqual([a.describe(gc) for a in gc.legal_actions()],
                         [f"event option {i}" for i in expected])
        self.assertTrue(all(a.is_valid(gc) for a in gc.legal_actions()))
        return gc

    def advance_acts(self, gc, act: int):
        # Act-specific event pools are private. Initialize them through real
        # boss victories and act transitions, using a temporary strong deck via
        # public acquisition/removal APIs. Skip all rewards and restore the deck.
        sts = self.sts
        original_deck = gc.deck
        for i in reversed(range(len(gc.deck))):
            gc.remove_card(i)
        card = sts.Card(sts.CardId.SEARING_BLOW)
        for _ in range(100):
            card.upgrade()
        for _ in range(5):
            gc.obtain_card(card)
        for _ in range(act - 1):
            gc.cur_map_node_y = 14
            gc.cur_map_node_x = next(n["x"] for n in gc.map_graph()["nodes"] if n["y"] == 14)
            next(a for a in gc.legal_actions()
                 if a.describe(gc).endswith(" advance to boss")).execute(gc)
            self.assertEqual(gc.screen_state, sts.ScreenState.BATTLE)
            bc = sts.BattleContext()
            bc.init(gc)
            for _ in range(32):
                if bc.outcome != sts.BattleOutcome.UNDECIDED:
                    break
                actions = bc.legal_actions()
                action = next((a for a in actions if "play Searing Blow" in a.describe(bc)), None)
                if action is None:
                    action = next(a for a in actions if "end turn" in a.describe(bc))
                action.execute(bc)
            self.assertFalse(bc.undefined_behavior_evoked)
            self.assertEqual(bc.outcome, sts.BattleOutcome.PLAYER_VICTORY)
            bc.exit_battle(gc)
            next(a for a in gc.legal_actions()
                 if a.describe(gc) == "skip rewards / proceed").execute(gc)
            next(a for a in gc.legal_actions()
                 if a.describe(gc) == "skip boss relic reward").execute(gc)
            self.assertEqual(gc.screen_state, sts.ScreenState.MAP_SCREEN)
        self.assertEqual(gc.act, act)
        for i in reversed(range(len(gc.deck))):
            gc.remove_card(i)
        for card in original_deck:
            gc.obtain_card(card)

    def choose(self, gc, index: int):
        next(action for action in gc.legal_actions()
             if action.describe(gc) == f"event option {index}").execute(gc)

    def deck(self, gc):
        return [(card.id, card.upgrade_count, card.misc) for card in gc.deck]

    def test_mushrooms_eat_heals_floor_quarter_adds_parasite_and_keeps_gold(self):
        for ascension, max_hp, cases in ((0, 80, ((20, 40), (78, 80), (80, 80))),
                                         (15, 75, ((20, 38), (73, 75), (75, 75)))):
            for hp, expected_hp in cases:
                with self.subTest(ascension=ascension, hp=hp):
                    gc = self.event("HYPNOTIZING_COLORED_MUSHROOMS", 35, ascension=ascension)
                    self.assertEqual(gc.max_hp, max_hp)
                    gc.cur_hp = hp
                    before_deck, before_gold = self.deck(gc), gc.gold
                    self.choose(gc, 1)
                    self.assertEqual(gc.cur_hp, expected_hp)
                    self.assertEqual(gc.max_hp, max_hp)
                    self.assertEqual(gc.gold, before_gold)
                    self.assertEqual(self.deck(gc), before_deck + [(self.sts.CardId.PARASITE, 0, 0)])
                    self.assertEqual(gc.screen_state, self.sts.ScreenState.MAP_SCREEN)

    def test_sphere_leave_keeps_gold_hp_and_deck(self):
        for ascension in (0, 15):
            with self.subTest(ascension=ascension):
                gc = self.event("MYSTERIOUS_SPHERE", 11, act=3, ascension=ascension)
                before = gc.gold, gc.cur_hp, gc.max_hp, self.deck(gc), [r.id for r in gc.relics]
                self.choose(gc, 1)
                self.assertEqual((gc.gold, gc.cur_hp, gc.max_hp, self.deck(gc),
                                  [r.id for r in gc.relics]), before)
                self.assertEqual(gc.screen_state, self.sts.ScreenState.MAP_SCREEN)

    def test_mausoleum_always_grants_relic_with_both_curse_rolls(self):
        for ascension in (0, 15):
            for seed, cursed in ((26, True), (54, False)):
                with self.subTest(ascension=ascension, seed=seed):
                    gc = self.event("THE_MAUSOLEUM", seed, act=2, ascension=ascension)
                    before_relics = [r.id for r in gc.relics]
                    before_deck, before_gold = self.deck(gc), gc.gold
                    self.choose(gc, 0)
                    self.assertEqual(len(gc.relics), len(before_relics) + 1)
                    self.assertEqual([r.id for r in gc.relics][:-1], before_relics)
                    self.assertNotIn(gc.relics[-1].id, before_relics)
                    expected_deck = before_deck[:]
                    if cursed or ascension >= 15:
                        expected_deck.append((self.sts.CardId.WRITHE, 0, 0))
                    self.assertEqual(self.deck(gc), expected_deck)
                    self.assertEqual(gc.gold, before_gold)
                    self.assertEqual(gc.screen_state, self.sts.ScreenState.MAP_SCREEN)

    def test_mausoleum_relic_applies_on_obtain_effects(self):
        gc = self.event("THE_MAUSOLEUM", 729, act=2)
        gc.cur_hp = 20
        before_max_hp = gc.max_hp
        self.choose(gc, 0)
        self.assertEqual(gc.relics[-1].id, self.sts.RelicId.PEAR)
        self.assertEqual(gc.max_hp, before_max_hp + 10)
        self.assertEqual(gc.cur_hp, 30)

    def test_woman_in_blue_charges_before_offering_potion_rewards(self):
        for ascension, seed in ((0, 6), (15, 101)):
            for index, cost in enumerate((20, 30, 40)):
                with self.subTest(ascension=ascension, index=index):
                    gc = self.event("THE_WOMAN_IN_BLUE", seed, ascension=ascension)
                    self.assertGreaterEqual(gc.gold, 50)
                    before_gold, before_hp, before_deck = gc.gold, gc.cur_hp, self.deck(gc)
                    self.choose(gc, index)
                    self.assertEqual(gc.gold, before_gold - cost)
                    self.assertEqual(gc.cur_hp, before_hp)
                    self.assertEqual(self.deck(gc), before_deck)
                    self.assertEqual(gc.screen_state, self.sts.ScreenState.REWARDS)
                    rewards = [a.describe(gc) for a in gc.legal_actions()]
                    self.assertEqual(sum(a.startswith("take potion ") for a in rewards), index + 1)
                    self.assertEqual(len(rewards), index + 2)  # potions plus proceed
