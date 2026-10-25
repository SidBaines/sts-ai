from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest

from sts_env._backend.glossary import _render_map
from sts_env._backend.lightspeed import LightspeedHybridEnv
from sts_env._backend.lightspeed_import import import_lightspeed
from sts_env.agents import FirstAgent
from tests.support import requires_simulator


@requires_simulator
class OutOfCombatPotionTest(unittest.TestCase):
    def setUp(self):
        self.sts = import_lightspeed()

    def map_context(self, seed=11):
        gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, seed, 0)
        for _ in range(32):
            if gc.screen_state == self.sts.ScreenState.MAP_SCREEN:
                return gc
            gc.legal_actions()[0].execute(gc)
        self.fail("bounded fixture did not reach the map")

    @staticmethod
    def scene(gc):
        return (gc.screen_state, gc.act, gc.floor_num, gc.cur_map_node_x,
                gc.cur_map_node_y, gc.current_event, gc.cur_room, gc.gold,
                tuple(str(card) for card in gc.deck),
                tuple((relic.id, relic.data) for relic in gc.relics))

    @staticmethod
    def menu(gc):
        return [(int(a.bits), a.describe(gc)) for a in gc.legal_actions()]

    @staticmethod
    def potion_actions(gc):
        return [a for a in gc.legal_actions() if int(a.bits) & 0x80000000]

    def drink(self, gc, name):
        return next(a for a in gc.legal_actions()
                    if a.describe(gc).startswith(f"drink potion {name} "))

    @staticmethod
    def potion_line(gc):
        return next(line for line in gc.describe_state().splitlines() if line.startswith("Potions"))

    def test_potion_line_lists_every_slot_after_an_earlier_slot_empties(self):
        # Regression: the line looped over the potion count, not the slots, so
        # an empty slot 0 hid the potion in slot 2 that the menu still offered.
        gc = self.map_context()
        self.assertEqual(self.potion_line(gc), "Potions (capacity 3): [0] empty, [1] empty, [2] empty")
        for name in ("BLOOD_POTION", "FIRE_POTION", "FRUIT_JUICE"):
            gc.obtain_potion(getattr(self.sts.Potion, name))
        next(a for a in gc.legal_actions()
             if a.describe(gc) == "discard potion Blood Potion (slot 0)").execute(gc)
        line = self.potion_line(gc)
        self.assertEqual(line, "Potions (capacity 3): [0] empty, [1] Fire Potion, [2] Fruit Juice")
        actions = self.potion_actions(gc)
        self.assertTrue(actions)
        for action in actions:
            name, slot = re.search(r"potion (.+) \(slot (\d+)\)", action.describe(gc)).groups()
            self.assertIn(f"[{slot}] {name}", line)
        from sts_env._backend.glossary import _potion_names
        self.assertEqual(_potion_names(gc.describe_state()), ["Fire Potion", "Fruit Juice"])

    def test_map_screen_prefix_and_slot_order_are_preserved(self):
        gc = self.map_context()
        ordinary = self.menu(gc)
        for name in ("BLOOD_POTION", "FIRE_POTION", "FRUIT_JUICE"):
            gc.obtain_potion(getattr(self.sts.Potion, name))
        menu = self.menu(gc)
        self.assertEqual(menu[:len(ordinary)], ordinary)
        self.assertEqual(menu[len(ordinary):], [
            (0x80000000, "drink potion Blood Potion (slot 0): heal 16 HP"),
            (0xC0000000, "discard potion Blood Potion (slot 0)"),
            (0xC0000001, "discard potion Fire Potion (slot 1)"),
            (0x80000002, "drink potion Fruit Juice (slot 2): gain 5 max HP and heal 5 HP"),
            (0xC0000002, "discard potion Fruit Juice (slot 2)"),
        ])
        # This binding is deliberately still the native agents' ordinary menu.
        self.assertEqual([(int(a.bits), a.describe(gc))
                          for a in self.sts.GameAction.all_in_state(gc)], ordinary)
        self.assertTrue(all(a.is_valid(gc) for a in gc.legal_actions()))

    def test_blood_healing_and_description_use_max_hp_and_sacred_bark(self):
        for bark in (False, True):
            for max_hp in (80, 83):
                with self.subTest(bark=bark, max_hp=max_hp):
                    gc = self.map_context()
                    gc.max_hp, gc.cur_hp = max_hp, 10
                    if bark:
                        gc.obtain_relic(self.sts.RelicId.SACRED_BARK)
                    gc.obtain_potion(self.sts.Potion.BLOOD_POTION)
                    amount = max_hp * (40 if bark else 20) // 100
                    action = self.drink(gc, "Blood Potion")
                    self.assertTrue(action.describe(gc).endswith(f": heal {amount} HP"))
                    before = self.scene(gc)
                    action.execute(gc)
                    self.assertEqual(gc.cur_hp, 10 + amount)
                    self.assertEqual(gc.max_hp, max_hp)
                    self.assertEqual(self.scene(gc), before)
                    self.assertEqual(self.potion_actions(gc), [])

    def test_fruit_juice_increases_max_hp_and_heals_with_sacred_bark(self):
        for bark in (False, True):
            with self.subTest(bark=bark):
                gc = self.map_context()
                gc.max_hp, gc.cur_hp = 80, 10
                if bark:
                    gc.obtain_relic(self.sts.RelicId.SACRED_BARK)
                gc.obtain_potion(self.sts.Potion.FRUIT_JUICE)
                amount = 10 if bark else 5
                action = self.drink(gc, "Fruit Juice")
                self.assertTrue(action.describe(gc).endswith(
                    f": gain {amount} max HP and heal {amount} HP"))
                before = self.scene(gc)
                action.execute(gc)
                self.assertEqual((gc.max_hp, gc.cur_hp), (80 + amount, 10 + amount))
                self.assertEqual(self.scene(gc), before)

    def test_discard_does_not_move_on_map_or_change_hp(self):
        gc = self.map_context()
        gc.obtain_potion(self.sts.Potion.FIRE_POTION)
        actions = self.potion_actions(gc)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].describe(gc), "discard potion Fire Potion (slot 0)")
        before, hp = self.scene(gc), gc.cur_hp
        actions[0].execute(gc)
        self.assertEqual(self.scene(gc), before)
        self.assertEqual(gc.cur_hp, hp)
        self.assertEqual(self.potion_actions(gc), [])

    def test_drinking_at_initialized_rest_site_does_not_also_rest(self):
        gc = self.map_context()
        # Move the public map frontier to a real rest-site parent; entering it
        # still uses the normal transition/setup path, with no private writes.
        nodes = gc.map_graph()["nodes"]
        target = next(n for n in nodes if n["room"] == "REST" and n["y"] > 0)
        parent = next(n for n in nodes if n["y"] == target["y"] - 1
                      and target["x"] in n["edges"])
        gc.cur_map_node_x, gc.cur_map_node_y = parent["x"], parent["y"]
        gc.floor_num = target["y"]
        next(a for a in gc.legal_actions()
             if a.describe(gc).startswith(f"choose map node x={target['x']} ")).execute(gc)
        self.assertEqual(gc.screen_state, self.sts.ScreenState.REST_ROOM)
        gc.max_hp, gc.cur_hp = 80, 10
        gc.obtain_potion(self.sts.Potion.BLOOD_POTION)
        before = self.scene(gc)
        self.drink(gc, "Blood Potion").execute(gc)
        self.assertEqual(gc.cur_hp, 26)
        self.assertEqual(self.scene(gc), before)
        self.assertTrue(any(a.describe(gc).startswith("rest") for a in gc.legal_actions()))

    def test_entropic_brew_fills_empty_slots_and_preserves_occupied_slot(self):
        gc = self.map_context()
        gc.obtain_potion(self.sts.Potion.ENTROPIC_BREW)
        gc.obtain_potion(self.sts.Potion.FIRE_POTION)
        action = self.drink(gc, "Entropic Brew")
        self.assertEqual(action.describe(gc),
                         "drink potion Entropic Brew (slot 0): fill empty potion slots with random potions")
        before, hp = self.scene(gc), gc.cur_hp
        action.execute(gc)
        discards = [a for a in self.potion_actions(gc) if int(a.bits) & 0x40000000]
        self.assertEqual([a.idx1() for a in discards], [0, 1, 2])
        self.assertEqual(discards[1].describe(gc), "discard potion Fire Potion (slot 1)")
        self.assertEqual(self.scene(gc), before)
        self.assertEqual(gc.cur_hp, hp)
        # Capacity accounting must agree with the visible slots after refilling.
        discards[0].execute(gc)
        gc.obtain_potion(self.sts.Potion.FRUIT_JUICE)
        self.assertEqual(self.drink(gc, "Fruit Juice").idx1(), 0)

    def test_entropic_brew_never_generates_fruit_juice(self):
        for seed in range(64):
            with self.subTest(seed=seed):
                gc = self.map_context(seed)
                gc.obtain_potion(self.sts.Potion.ENTROPIC_BREW)
                self.drink(gc, "Entropic Brew").execute(gc)
                discards = [a for a in self.potion_actions(gc) if int(a.bits) & 0x40000000]
                self.assertEqual([a.idx1() for a in discards], [0, 1, 2])
                self.assertTrue(all("Fruit Juice" not in a.describe(gc) for a in discards))

    def test_entropic_brew_fills_first_empty_slot_with_first_roll(self):
        inventories = []
        for occupied_slot in (0, 1):
            gc = self.map_context()
            names = ("FIRE_POTION", "ENTROPIC_BREW") if occupied_slot == 0 else (
                "ENTROPIC_BREW", "FIRE_POTION")
            for name in names:
                gc.obtain_potion(getattr(self.sts.Potion, name))
            self.drink(gc, "Entropic Brew").execute(gc)
            # Ignore the preserved Fire Potion and slot suffixes to compare the
            # ordered generated potions for the same world RNG state.
            inventories.append([a.describe(gc).split(" (slot")[0]
                                for a in self.potion_actions(gc)
                                if int(a.bits) & 0x40000000 and a.idx1() != occupied_slot])
        self.assertEqual(inventories[0], inventories[1])

    def test_sozu_blocks_entropic_brew_but_still_consumes_it(self):
        gc = self.map_context()
        gc.obtain_potion(self.sts.Potion.ENTROPIC_BREW)
        gc.obtain_potion(self.sts.Potion.FIRE_POTION)
        gc.obtain_relic(self.sts.RelicId.SOZU)
        before = self.scene(gc)
        rng_before = int(re.search(r"potionRng: (\d+)", repr(gc)).group(1))
        self.drink(gc, "Entropic Brew").execute(gc)
        rng_after = int(re.search(r"potionRng: (\d+)", repr(gc)).group(1))
        self.assertGreater(rng_after, rng_before)
        normal = self.map_context()
        normal.obtain_potion(self.sts.Potion.ENTROPIC_BREW)
        normal.obtain_potion(self.sts.Potion.FIRE_POTION)
        self.drink(normal, "Entropic Brew").execute(normal)
        self.assertEqual(rng_after, int(re.search(r"potionRng: (\d+)", repr(normal)).group(1)))
        self.assertEqual(self.scene(gc), before)
        self.assertEqual([a.describe(gc) for a in self.potion_actions(gc)],
                         ["discard potion Fire Potion (slot 1)"])

    def test_invalid_potion_actions_raise_without_mutation(self):
        gc = self.map_context()
        gc.obtain_potion(self.sts.Potion.FIRE_POTION)
        for bits in (0x80000000, 0x80000001, 0xC0000001, 0x800000FF, 0xC00000FF):
            with self.subTest(bits=bits):
                action = self.sts.GameAction(bits)
                before, menu = gc.describe_state(), self.menu(gc)
                self.assertFalse(action.is_valid(gc))
                with self.assertRaisesRegex(RuntimeError, "invalid out-of-combat potion action"):
                    action.execute(gc)
                self.assertEqual(gc.describe_state(), before)
                self.assertEqual(self.menu(gc), menu)
        discard = self.potion_actions(gc)[0]
        discard.execute(gc)
        before = gc.describe_state()
        with self.assertRaisesRegex(RuntimeError, "invalid out-of-combat potion action"):
            discard.execute(gc)
        self.assertEqual(gc.describe_state(), before)

    def test_potions_do_not_create_menus_on_terminal_or_battle_screens(self):
        gc = self.map_context()
        gc.obtain_potion(self.sts.Potion.BLOOD_POTION)
        gc.outcome = self.sts.GameOutcome.PLAYER_LOSS
        self.assertEqual(gc.legal_actions(), [])
        before = gc.describe_state()
        with self.assertRaises(RuntimeError):
            self.sts.GameAction(0x80000000).execute(gc)
        self.assertEqual(gc.describe_state(), before)
        gc.outcome = self.sts.GameOutcome.UNDECIDED
        for screen in (self.sts.ScreenState.BATTLE, self.sts.ScreenState.INVALID):
            with self.subTest(screen=screen):
                gc.screen_state = screen
                self.assertEqual(gc.legal_actions(), [])
                for bits in (0x80000000, 0xC0000000):
                    action = self.sts.GameAction(bits)
                    before = gc.describe_state()
                    self.assertFalse(action.is_valid(gc))
                    with self.assertRaisesRegex(RuntimeError, "invalid out-of-combat potion action"):
                        action.execute(gc)
                    self.assertEqual(gc.describe_state(), before)
        # A fresh context has no pending card selection; held potions must not
        # turn an unsupported/empty screen into a playable menu.
        gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, 12, 0)
        gc.obtain_potion(self.sts.Potion.BLOOD_POTION)
        gc.screen_state = self.sts.ScreenState.CARD_SELECT
        self.assertEqual(gc.legal_actions(), [])

    def test_potion_description_handles_empty_slots(self):
        gc = self.map_context()
        action = self.sts.GameAction(0xC0000000)
        self.assertEqual(action.describe(gc), "discard potion EMPTY_POTION_SLOT (slot 0)")

    def test_we_meet_again_excludes_drinking_and_discarding(self):
        gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, 12, 0)
        gc.obtain_potion(self.sts.Potion.BLOOD_POTION)
        # A real event initialized through a synthetic public map frontier.
        gc.act = 3
        gc.screen_state = self.sts.ScreenState.MAP_SCREEN
        gc.cur_map_node_x, gc.cur_map_node_y = 2, 0
        next(a for a in gc.legal_actions()
             if a.describe(gc).startswith("choose map node x=3 ")).execute(gc)
        self.assertEqual(gc.current_event, "WE_MEET_AGAIN")
        self.assertEqual(gc.screen_state, self.sts.ScreenState.EVENT_SCREEN)
        self.assertTrue(gc.legal_actions())
        self.assertEqual(self.potion_actions(gc), [])
        for bits in (0x80000000, 0xC0000000):
            action = self.sts.GameAction(bits)
            before = gc.describe_state()
            self.assertFalse(action.is_valid(gc))
            with self.assertRaisesRegex(RuntimeError, "invalid out-of-combat potion action"):
                action.execute(gc)
            self.assertEqual(gc.describe_state(), before)

    def test_potion_menus_and_effects_repeat_across_fresh_processes(self):
        driver = """
import json
from sts_env._backend.lightspeed_import import import_lightspeed
sts = import_lightspeed()
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 11, 0)
for _ in range(32):
    if gc.screen_state == sts.ScreenState.MAP_SCREEN:
        break
    gc.legal_actions()[0].execute(gc)
gc.cur_hp = 10
records = []
for name in ('BLOOD_POTION', 'FRUIT_JUICE', 'ENTROPIC_BREW'):
    gc.obtain_potion(getattr(sts.Potion, name))
    actions = gc.legal_actions()
    records.append([(int(a.bits), a.describe(gc)) for a in actions])
    next(a for a in actions if a.describe(gc).startswith('drink potion')).execute(gc)
    records.append(gc.describe_state())
records.append([(int(a.bits), a.describe(gc)) for a in gc.legal_actions()])
print(json.dumps(records))
"""
        results = []
        for _ in range(2):
            result = subprocess.run([sys.executable, "-c", driver],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            results.append(json.loads(result.stdout))
        self.assertEqual(results[0], results[1])

    def test_python_menu_native_preview_first_agent_and_map_rendering(self):
        env = LightspeedHybridEnv(11, battle_simulations=10, max_act=1)
        env.gc = self.map_context()
        ordinary = env.legal_actions()
        graph = env.map_graph()
        map_before = _render_map(graph, [env.action_dict(a) for a in ordinary])
        env.seed_native_ooc(17)
        preview = env.native_ooc_action()
        env.gc.obtain_potion(self.sts.Potion.FRUIT_JUICE)
        raw, display, mapping = env._action_views()
        self.assertEqual(display[:len(ordinary)], ordinary)
        self.assertEqual(mapping, list(range(len(raw))))
        self.assertEqual([a.bits for a in display], [a.bits for a in raw])
        self.assertEqual(env.native_ooc_action(), preview)
        self.assertEqual(FirstAgent().choose_action({"legal_actions": [env.action_dict(a) for a in display]}), 0)
        self.assertEqual(_render_map(graph, [env.action_dict(a) for a in display]), map_before)
        expected = env.ooc_agent.preview_out_of_combat_action(env.gc)
        self.assertEqual(display[preview].bits, expected.bits)
        # Execute through the Python menu too, including its pending preview path.
        fruit = next(a for a in display if a.description.startswith("drink potion"))
        before, max_hp = self.scene(env.gc), env.gc.max_hp
        env.step(fruit.index)
        self.assertEqual(self.scene(env.gc), before)
        self.assertEqual(env.gc.max_hp, max_hp + 5)
        self.assertEqual(env.advance_native_ooc().bits, expected.bits)

    def test_event_potion_labels_are_not_enriched_as_event_options(self):
        env = LightspeedHybridEnv(11, battle_simulations=10, max_act=1)
        self.assertEqual(env.gc.screen_state, self.sts.ScreenState.EVENT_SCREEN)
        ordinary = env.legal_actions()
        env.gc.obtain_potion(self.sts.Potion.BLOOD_POTION)
        actions = env.legal_actions()
        self.assertEqual(actions[:len(ordinary)], ordinary)
        self.assertEqual([a.description for a in actions[len(ordinary):]], [
            "drink potion Blood Potion (slot 0): heal 16 HP",
            "discard potion Blood Potion (slot 0)",
        ])
