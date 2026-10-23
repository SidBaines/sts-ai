from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
import unittest

from sts_env import SlayTheSpireEnv
from sts_env._backend.lightspeed_import import import_lightspeed
from tests.support import requires_simulator


def replay_fingerprints():
    """Replay the frozen prefixes, then choose first actions to a bounded end."""
    path = Path(__file__).parents[1] / "fixtures" / "discovery_boss_replays.json"
    results = []
    for fixture in json.loads(path.read_text()):
        digest = hashlib.sha256()
        with SlayTheSpireEnv(mode="full", max_act=1, max_steps=4096,
                            battle_simulations=200) as env:
            observation, _ = env.reset(seed=fixture["seed"])
            digest.update(json.dumps(observation, sort_keys=True).encode())
            for step in range(4096):
                index = fixture["actions"][step][0] if step < len(fixture["actions"]) else 0
                observation, reward, terminated, truncated, info = env.step(index)
                digest.update(json.dumps([observation, reward, terminated, truncated], sort_keys=True).encode())
                if terminated or truncated:
                    break
            results.append({"seed": fixture["seed"], "steps": step + 1,
                            "semantic_sha256": digest.hexdigest(), "outcome": info["outcome"],
                            "termination_reason": info["termination_reason"],
                            "fault": observation["state"]["undefined_behavior_evoked"]})
    return results


@requires_simulator
class NativeDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.sts = import_lightspeed()

    def discovery_battle(self, upgraded=False, full_hand=False, character_class=None, seed=3):
        sts = self.sts
        gc = sts.GameContext(character_class or sts.CharacterClass.IRONCLAD, seed, 0)
        for _ in range(100):
            if gc.screen_state == sts.ScreenState.BATTLE:
                break
            gc.legal_actions()[0].execute(gc)
        else:
            self.fail("fixture did not reach combat")
        for _ in range(len(gc.deck)):
            gc.remove_card(0)
        card = sts.Card(sts.CardId.DISCOVERY)
        if upgraded:
            card.upgrade()
        gc.obtain_card(card)
        if full_hand:
            for _ in range(4):
                gc.obtain_card(sts.Card(sts.CardId.BASH))
            gc.obtain_relic(sts.RelicId.SACRED_BARK)
        bc = sts.BattleContext()
        bc.init(gc)
        if full_hand:
            for potion in (sts.Potion.CUNNING_POTION, sts.Potion.DUPLICATION_POTION):
                bc.obtain_potion(potion)
                next(a for a in bc.legal_actions() if "drink" in a.describe(bc)).execute(bc)
        return bc

    def test_discovery_selects_each_distinct_offer_for_free_and_exhausts(self):
        for upgraded in (False, True):
            for index, name in enumerate(("Limit Break", "Rage", "Metallicize")):
                with self.subTest(upgraded=upgraded, index=index):
                    bc = self.discovery_battle(upgraded)
                    next(a for a in bc.legal_actions() if "play Discovery" in a.describe(bc)).execute(bc)
                    self.assertFalse(bc.undefined_behavior_evoked)
                    self.assertEqual(bc.input_state, self.sts.InputState.CARD_SELECT)
                    candidates = bc.public_cards(True)["selection"]["candidates"]
                    self.assertEqual(len(candidates), 3)
                    self.assertEqual(len(set(candidates)), 3)
                    self.assertTrue(all(expected in candidate for expected, candidate in
                                        zip(("Limit Break", "Rage", "Metallicize"), candidates)))
                    bc.legal_actions()[index].execute(bc)
                    cards = bc.public_cards(True)
                    self.assertEqual(cards["hand"][0]["name"], name)
                    self.assertEqual(cards["hand"][0]["cost_for_turn"], 0)
                    self.assertEqual(cards["hand"][0]["cost"], 1)
                    destination = cards["discard"] if upgraded else cards["exhaust"]
                    self.assertEqual(destination[0]["name"], "Discovery+" if upgraded else "Discovery")
                    self.assertFalse(bc.undefined_behavior_evoked)

    def test_discovery_cost_resets_after_the_turn(self):
        bc = self.discovery_battle()
        bc.legal_actions()[0].execute(bc)
        bc.legal_actions()[0].execute(bc)
        self.assertEqual(bc.public_cards(True)["hand"][0]["cost_for_turn"], 0)
        next(a for a in bc.legal_actions() if "end turn" in a.describe(bc)).execute(bc)
        selected = [card for pile in ("hand", "draw", "discard")
                    for card in bc.public_cards(True)[pile] if card["name"] == "Limit Break"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["cost_for_turn"], 1)
        self.assertFalse(bc.undefined_behavior_evoked)

    def test_discovery_whirlwind_keeps_x_cost_and_spends_energy_after_turn_boundary(self):
        # Game AbstractCard.setCostForTurn changes only nonnegative costs;
        # Discovery does not make X-cost cards free. Seed 112 offers Whirlwind
        # against Cultist, avoiding damage modifiers and reactive enemy blocks.
        for defer_to_next_turn in (False, True):
            with self.subTest(defer_to_next_turn=defer_to_next_turn):
                bc = self.discovery_battle(seed=112)
                bc.legal_actions()[0].execute(bc)
                choices = bc.legal_actions()
                next(a for a in choices if "Whirlwind" in a.describe(bc)).execute(bc)
                card = bc.public_cards(True)["hand"][0]
                self.assertEqual(card["name"], "Whirlwind")
                self.assertEqual(card["cost"], -1)
                self.assertEqual(card["cost_for_turn"], -1)
                self.assertFalse(card["free_to_play_once"])
                self.assertEqual(bc.player_energy, 2)
                if defer_to_next_turn:
                    next(a for a in bc.legal_actions() if "end turn" in a.describe(bc)).execute(bc)
                    card = next(card for card in bc.public_cards(True)["hand"]
                                if card["name"] == "Whirlwind")
                    self.assertEqual(card["cost_for_turn"], -1)
                    self.assertFalse(card["free_to_play_once"])
                    self.assertEqual(bc.player_energy, 3)
                before = bc.enemies()[0]
                self.assertEqual(before["name"], "CULTIST")
                self.assertEqual(before["block"], 0)
                self.assertEqual(before["weak"], 0)
                self.assertEqual(before["vulnerable"], 0)
                energy_on_use = bc.player_energy
                next(a for a in bc.legal_actions() if "play Whirlwind" in a.describe(bc)).execute(bc)
                self.assertEqual(bc.player_energy, 0)
                self.assertEqual(before["cur_hp"] - bc.enemies()[0]["cur_hp"], 5 * energy_on_use)
                self.assertFalse(bc.undefined_behavior_evoked)

    def test_duplicated_discovery_sends_overflow_to_discard_at_temporary_zero_cost(self):
        bc = self.discovery_battle(full_hand=True)
        self.assertEqual(len(bc.public_cards(True)["hand"]), 10)
        next(a for a in bc.legal_actions() if "play Discovery" in a.describe(bc)).execute(bc)
        bc.legal_actions()[0].execute(bc)
        self.assertEqual(len(bc.public_cards(True)["hand"]), 10)
        self.assertEqual(bc.input_state, self.sts.InputState.CARD_SELECT)
        bc.legal_actions()[0].execute(bc)
        cards = bc.public_cards(True)
        self.assertEqual(len(cards["hand"]), 10)
        overflow = next(card for card in cards["discard"] if card["name"] == "Seeing Red")
        self.assertEqual(overflow["count"], 1)
        self.assertEqual(overflow["cost_for_turn"], 0)
        self.assertEqual(overflow["cost"], 1)
        self.assertEqual(cards["exhaust"][0]["name"], "Discovery")
        self.assertEqual(cards["exhaust"][0]["count"], 1)
        self.assertFalse(bc.undefined_behavior_evoked)
        next(a for a in bc.legal_actions() if "end turn" in a.describe(bc)).execute(bc)
        overflow = [card for pile in ("hand", "draw", "discard")
                    for card in bc.public_cards(True)[pile] if card["name"] == "Seeing Red"]
        self.assertEqual(len(overflow), 1)
        self.assertEqual(overflow[0]["cost_for_turn"], 1)

    def test_unsupported_discovery_classes_raise_and_retain_fault_flag(self):
        for character_class in (self.sts.CharacterClass.SILENT, self.sts.CharacterClass.DEFECT,
                                self.sts.CharacterClass.WATCHER):
            with self.subTest(character_class=character_class):
                bc = self.discovery_battle(character_class=character_class)
                with self.assertRaisesRegex(RuntimeError, "Discovery class-card generation only supports Ironclad"):
                    next(a for a in bc.legal_actions() if "play Discovery" in a.describe(bc)).execute(bc)
                self.assertTrue(bc.undefined_behavior_evoked)

    def test_affected_worlds_are_deterministic_across_fresh_processes(self):
        code = ("import json; from tests.integration.test_native_discovery import replay_fingerprints; "
                "print(json.dumps(replay_fingerprints(), sort_keys=True))")
        results = [subprocess.check_output([sys.executable, "-c", code], text=True,
                                          cwd=Path(__file__).parents[2], timeout=30)
                   for _ in range(2)]
        self.assertEqual(results[0], results[1])
        for result in json.loads(results[0]):
            self.assertFalse(result["fault"])
            self.assertIn(result["termination_reason"], ("defeat", "max_act", "max_steps", "victory"))

    def test_confirmed_act1_boss_actions_open_valid_discovery_menu(self):
        path = Path(__file__).parents[1] / "fixtures" / "discovery_boss_replays.json"
        for fixture in json.loads(path.read_text()):
            with self.subTest(seed=fixture["seed"]):
                digest = hashlib.sha256()
                with SlayTheSpireEnv(mode="full", max_act=1, max_steps=4096,
                                    battle_simulations=200) as env:
                    observation, _ = env.reset(seed=fixture["seed"])
                    for step, (index, description) in enumerate(fixture["actions"]):
                        action = observation["legal_actions"][index]
                        actual_description = action["description"]
                        # The frozen trace predates multi-enemy/X-cost damage
                        # annotations. Preserve the artifact and its digest while
                        # checking the same action identity before the new suffix.
                        if (description.startswith(("play Whirlwind ", "play Cleave "))
                                and " (deal " not in description):
                            actual_description, marker, suffix = actual_description.partition(" (deal ")
                            self.assertRegex(
                                marker + suffix,
                                r"^ \(deal (per target: .+|\d+ = \d+ x\d+)\)$",
                            )
                        self.assertEqual(actual_description, description)
                        if step < fixture["original_successful_steps"]:
                            record = {"phase": observation["phase"],
                                      "action": {**action, "description": description}}
                            digest.update((json.dumps(record, sort_keys=True,
                                                      separators=(",", ":")) + "\n").encode())
                        observation, _, terminated, truncated, _ = env.step(index)
                        self.assertFalse(terminated or truncated)
                    self.assertEqual(digest.hexdigest(), fixture["original_action_trace_sha256"])
                    self.assertFalse(observation["state"]["undefined_behavior_evoked"])
                    self.assertEqual(len(observation["legal_actions"]), 3)
                    self.assertTrue(all("select card for DISCOVERY:" in action["description"]
                                        for action in observation["legal_actions"]))
                    observation, _, _, _, _ = env.step(0)
                    self.assertFalse(observation["state"]["undefined_behavior_evoked"])
