"""Unit tests for the rollout display model (pure Python, no Streamlit)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_ai.rollout_view import (
    _PLAY_RE,
    load_rollout,
    parse_state_text,
    prettify_enemy_name,
    prettify_intent,
    to_combat_view,
    to_view,
)

STATE_TEXT = (
    "Act 1, floor 6, screen REST_ROOM, room REST, boss The Guardian\n"
    "HP: 54/80, gold: 110\n"
    "Deck: (12): {Strike,Strike,Defend,Bash,Anger,Cleave,}\n"
    "Relics: {Burning Blood:0,Vajra:0,}\n"
    "Potions: Fire Potion, Block Potion\n"
)


def _record() -> dict:
    return {
        "world_seed": 3,
        "decision_index": 7,
        "state": {
            "act": 1,
            "floor": 6,
            "screen_state": "ScreenState.REST_ROOM",
            "room": "Room.REST",
            "boss": "The Guardian",
            "cur_hp": 54,
            "max_hp": 80,
            "gold": 110,
        },
        "state_text": STATE_TEXT,
        "legal_actions": [
            {"index": 0, "bits": 0, "description": "rest (heal 30% of max HP)"},
            {"index": 1, "bits": 1, "description": "smith (upgrade a card in your deck)"},
        ],
        "selected_action": {"index": 1, "description": "smith (upgrade a card in your deck)"},
        "agent": {
            "action_index": 1,
            "raw_response": "<think>weigh options</think>\n{\"reasoning\": \"upgrade\", \"action_index\": 1}",
            "reasoning": "upgrade",
            "thinking": "weigh options",
            "valid": True,
            "retries": 0,
        },
        "after_state": {"cur_hp": 54, "floor": 6},
    }


class ParseStateTextTest(unittest.TestCase):
    def test_parses_deck_relics_potions(self):
        parsed = parse_state_text(STATE_TEXT)
        self.assertEqual(parsed["deck"][:2], ["Strike", "Strike"])
        self.assertEqual(len(parsed["deck"]), 6)  # trailing comma ignored
        self.assertEqual(parsed["relics"], ["Burning Blood:0", "Vajra:0"])
        self.assertEqual(parsed["potions"], ["Fire Potion", "Block Potion"])

    def test_parses_boss_from_header(self):
        # boss is only in state_text, not the state dict
        self.assertEqual(parse_state_text(STATE_TEXT)["boss"], "The Guardian")

    def test_boss_falls_back_to_state_text_when_absent_from_state(self):
        record = _record()
        del record["state"]["boss"]
        self.assertEqual(to_view(record).boss, "The Guardian")

    def test_potions_none(self):
        parsed = parse_state_text("Potions: none\n")
        self.assertEqual(parsed["potions"], [])

    def test_missing_lines_degrade_gracefully(self):
        parsed = parse_state_text("garbage with no recognizable lines")
        self.assertEqual(parsed, {"deck": [], "relics": [], "potions": [], "boss": ""})


class ToViewTest(unittest.TestCase):
    def test_builds_decision_view(self):
        view = to_view(_record())
        self.assertEqual(view.floor, 6)
        self.assertEqual(view.screen, "REST_ROOM")
        self.assertEqual(view.room, "REST")
        self.assertEqual(view.cur_hp, 54)
        self.assertEqual(view.chosen_index, 1)
        self.assertEqual(len(view.deck), 6)
        self.assertEqual(view.thinking, "weigh options")
        # exactly one action flagged chosen, and it is the smith option
        chosen = [a for a in view.actions if a.chosen]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].index, 1)


class LoadRolloutTest(unittest.TestCase):
    def test_loads_file_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed_3.jsonl"
            path.write_text(json.dumps(_record()) + "\n")
            (Path(d) / "seed_3.error.json").write_text(json.dumps({"message": "boom"}))
            rollout = load_rollout(path)
            self.assertEqual(len(rollout.decisions), 1)
            self.assertEqual(rollout.world_seed, 3)
            self.assertEqual(rollout.boss, "The Guardian")
            self.assertEqual(rollout.error["message"], "boom")

    def test_missing_file_returns_empty(self):
        rollout = load_rollout("/nonexistent/seed_999.jsonl")
        self.assertEqual(rollout.decisions, [])
        self.assertIsNone(rollout.error)


COMBAT_STATE_TEXT = (
    "Battle turn 2\n"
    "Player HP: 40/80, block: 5, energy: 1/3\n"
    "Player powers: Strength 2, Vulnerable 1\n"
    "Enemies:\n"
    "  [1] SPIKE_SLIME_M HP 20/31, block 0, intent SPIKE_SLIME_M_LICK\n"  # dead slime omitted by serializer
    "Hand:\n"
    "  [0] Strike (cost 1)\n"
    "  [1] Bash (cost 2)\n"
    "  [2] Defend+ (cost 1)\n"
    "Piles: draw 3, discard 4, exhaust 1\n"
    "Potions: Fire Potion\n"
)


def _combat_record() -> dict:
    return {
        "world_seed": 3,
        "decision_index": 12,
        "phase": "combat",
        "state": {
            "act": 1,
            "floor": 2,
            "screen_state": "ScreenState.BATTLE",
            "room": "Room.MONSTER",
            "cur_hp": 56,  # stale map HP; combat.player_cur_hp is authoritative
            "max_hp": 80,
            "gold": 100,
            "combat": {
                "turn": 2,
                "input_state": "InputState.PLAYER_NORMAL",
                "battle_outcome": "BattleOutcome.UNDECIDED",
                "player_cur_hp": 40,
                "player_max_hp": 80,
                "player_block": 5,
                "player_energy": 1,
                "undefined_behavior_evoked": False,
                "enemies": [
                    {"index": 0, "name": "ACID_SLIME_S", "cur_hp": 0, "max_hp": 12,
                     "block": 0, "intent": "ACID_SLIME_S_LICK", "alive": False},
                    {"index": 1, "name": "SPIKE_SLIME_M", "cur_hp": 20, "max_hp": 31,
                     "block": 0, "intent": "SPIKE_SLIME_M_LICK", "alive": True},
                ],
            },
        },
        "state_text": COMBAT_STATE_TEXT,
        # Bash is absent (energy 1 can't afford cost 2) -> it must read as unplayable.
        "legal_actions": [
            {"index": 0, "bits": 0, "description": "play Strike (cost 1) -> SPIKE_SLIME_M"},
            {"index": 1, "bits": 1, "description": "play Defend+ (cost 1)"},
            {"index": 2, "bits": 2, "description": "end turn"},
        ],
        "selected_action": {"index": 0, "description": "play Strike (cost 1) -> SPIKE_SLIME_M"},
        "agent": {"action_index": 0, "reasoning": "chip the slime", "valid": True, "retries": 0},
        "after_state": {"cur_hp": 56, "floor": 2},
    }


class PrettifyTest(unittest.TestCase):
    def test_enemy_name_sizes_and_words(self):
        self.assertEqual(prettify_enemy_name("ACID_SLIME_S"), "Acid Slime (S)")
        self.assertEqual(prettify_enemy_name("SPIKE_SLIME_M"), "Spike Slime (M)")
        self.assertEqual(prettify_enemy_name("JAW_WORM"), "Jaw Worm")
        self.assertEqual(prettify_enemy_name("THE_GUARDIAN"), "The Guardian")

    def test_intent_strips_enemy_prefix(self):
        self.assertEqual(prettify_intent("ACID_SLIME_S_LICK", "ACID_SLIME_S"), "Lick")
        # falls back to the whole move id when it doesn't carry the enemy prefix
        self.assertEqual(prettify_intent("ATTACK", "JAW_WORM"), "Attack")


class ToCombatViewTest(unittest.TestCase):
    def setUp(self):
        self.cv = to_combat_view(_combat_record())

    def test_out_of_combat_record_has_no_combat_view(self):
        self.assertIsNone(to_combat_view(_record()))
        self.assertIsNone(to_view(_record()).combat)

    def test_player_stats_prefer_combat_block(self):
        self.assertEqual((self.cv.player_cur_hp, self.cv.player_max_hp), (40, 80))
        self.assertEqual(self.cv.player_block, 5)
        self.assertEqual((self.cv.player_energy, self.cv.player_energy_max), (1, 3))
        self.assertEqual(self.cv.turn, 2)
        self.assertEqual(self.cv.powers, ["Strength 2", "Vulnerable 1"])

    def test_piles_and_potions(self):
        self.assertEqual(
            (self.cv.draw_count, self.cv.discard_count, self.cv.exhaust_count), (3, 4, 1)
        )
        self.assertEqual(self.cv.potions, ["Fire Potion"])

    def test_enemies_include_dead_and_are_prettified(self):
        names = [(e.display_name, e.alive) for e in self.cv.enemies]
        self.assertEqual(names, [("Acid Slime (S)", False), ("Spike Slime (M)", True)])
        self.assertEqual(self.cv.enemies[1].intent_label, "Lick")

    def test_hand_cost_upgrade_and_playability(self):
        by_name = {(c.name, c.cost_text): c for c in self.cv.hand}
        self.assertTrue(by_name[("Strike", "1")].playable)
        self.assertTrue(by_name[("Defend+", "1")].playable)
        self.assertTrue(by_name[("Defend+", "1")].upgraded)
        # Bash isn't a legal action this turn (unaffordable) -> dimmed
        self.assertFalse(by_name[("Bash", "2")].playable)
        self.assertFalse(by_name[("Bash", "2")].upgraded)

    def test_chosen_card_and_target_mapping(self):
        self.assertEqual(self.cv.chosen_card_name, "Strike")
        self.assertFalse(self.cv.chosen_is_end_turn)
        self.assertEqual(self.cv.chosen_target_index, 1)  # the alive SPIKE_SLIME_M
        chosen = [c for c in self.cv.hand if c.chosen]
        self.assertEqual([(c.name, c.cost_text) for c in chosen], [("Strike", "1")])
        self.assertTrue(self.cv.enemies[1].targeted)
        self.assertFalse(self.cv.enemies[0].targeted)

    def test_end_turn_selection(self):
        record = _combat_record()
        record["selected_action"] = {"index": 2, "description": "end turn"}
        cv = to_combat_view(record)
        self.assertTrue(cv.chosen_is_end_turn)
        self.assertEqual(cv.chosen_card_name, "")
        self.assertIsNone(cv.chosen_target_index)
        self.assertFalse(any(c.chosen for c in cv.hand))

    def test_deal_annotation_does_not_break_target_or_chosen(self):
        # The binding appends a sim-computed "(deal N)" / "(deal T = P x H)" to attack
        # actions; _PLAY_RE must not let the target group swallow it.
        record = _combat_record()
        record["legal_actions"] = [
            {"index": 0, "bits": 0, "description": "play Strike (cost 1) -> SPIKE_SLIME_M (deal 9)"},
            {"index": 1, "bits": 1, "description": "play Defend+ (cost 1)"},
            {"index": 2, "bits": 2, "description": "play Bash (cost 2) -> SPIKE_SLIME_M (deal 12 = 6 x2)"},
            {"index": 3, "bits": 3, "description": "end turn"},
        ]
        record["selected_action"] = {"index": 0, "description": "play Strike (cost 1) -> SPIKE_SLIME_M (deal 9)"}
        cv = to_combat_view(record)
        self.assertEqual(cv.chosen_card_name, "Strike")
        self.assertEqual(cv.chosen_target_index, 1)  # the alive SPIKE_SLIME_M, not "SPIKE_SLIME_M (deal 9)"
        self.assertTrue(cv.enemies[1].targeted)
        strike = next(c for c in cv.hand if c.name == "Strike")
        self.assertTrue(strike.chosen and strike.playable)
        # Bash (multi-hit annotation) still reads as playable from its legal action
        self.assertTrue(next(c for c in cv.hand if c.name == "Bash").playable)

    def test_enemy_index_suffix_resolves_exact_target(self):
        # Two same-named living enemies -> the binding disambiguates with " [enemy i]".
        # The index is authoritative; the view must target the named slot exactly.
        record = _combat_record()
        record["state"]["combat"]["enemies"] = [
            {"index": 0, "name": "FUNGI_BEAST", "cur_hp": 12, "max_hp": 28,
             "block": 0, "intent": "FUNGI_BEAST_BITE", "alive": True},
            {"index": 1, "name": "FUNGI_BEAST", "cur_hp": 6, "max_hp": 28,
             "block": 0, "intent": "FUNGI_BEAST_BITE", "alive": True},
        ]
        record["legal_actions"] = [
            {"index": 0, "bits": 0, "description": "play Strike (cost 1) -> FUNGI_BEAST [enemy 0] (deal 6)"},
            {"index": 1, "bits": 1, "description": "play Strike (cost 1) -> FUNGI_BEAST [enemy 1] (deal 6)"},
            {"index": 2, "bits": 2, "description": "end turn"},
        ]
        record["selected_action"] = {
            "index": 1, "description": "play Strike (cost 1) -> FUNGI_BEAST [enemy 1] (deal 6)"}
        cv = to_combat_view(record)
        self.assertEqual(cv.chosen_card_name, "Strike")
        self.assertEqual(cv.chosen_target_index, 1)
        self.assertTrue(cv.enemies[1].targeted)
        self.assertFalse(cv.enemies[0].targeted)

    def test_degrades_when_state_text_pieces_missing(self):
        record = _combat_record()
        record["state_text"] = "Battle turn 0\n"  # no hand/piles/energy/potions lines
        cv = to_combat_view(record)
        self.assertEqual(cv.hand, [])
        self.assertEqual((cv.draw_count, cv.discard_count, cv.exhaust_count), (0, 0, 0))
        self.assertEqual(cv.potions, [])
        # energy max falls back to current energy when the line is absent
        self.assertEqual(cv.player_energy_max, cv.player_energy)
        # structured enemies still render
        self.assertEqual(len(cv.enemies), 2)


class PlayRegexTest(unittest.TestCase):
    def test_parses_target_and_cost_with_and_without_deal(self):
        m = _PLAY_RE.match("play Strike (cost 1) -> Jaw Worm (deal 9)")
        self.assertEqual((m.group(1), m.group(2), m.group(3)), ("Strike", "1", "Jaw Worm"))
        # multi-hit annotation
        m2 = _PLAY_RE.match("play Bash (cost 2) -> Cultist (deal 12 = 6 x2)")
        self.assertEqual((m2.group(1), m2.group(2), m2.group(3)), ("Bash", "2", "Cultist"))
        # no target, no deal
        m3 = _PLAY_RE.match("play Defend (cost 1)")
        self.assertEqual((m3.group(1), m3.group(2), m3.group(3)), ("Defend", "1", None))
        # X-cost token still parses
        m4 = _PLAY_RE.match("play Whirlwind (cost X)")
        self.assertEqual((m4.group(1), m4.group(2)), ("Whirlwind", "X"))


if __name__ == "__main__":
    unittest.main()
