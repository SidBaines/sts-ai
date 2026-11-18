"""Core difficulty rules and consecutive bosses; no learned-policy dependencies."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sts_env import SlayTheSpireEnv
from sts_env._backend.combat_public_v5 import CombatPublicV5Env
from sts_env._backend.lightspeed import LightspeedHybridEnv
from sts_env.env import ENVIRONMENT_VERSION, OBSERVATION_VERSION
from tests.support import load_search_test_fixture, requires_simulator


def backend(ascension, seed=3, *, max_act=3):
    return LightspeedHybridEnv(world_seed=seed, ascension=ascension, max_act=max_act,
                              combat_control="llm", combat_observation="combat_public_v3")


def strong_deck(native):
    # Synthetic deck/frontiers test real transitions, not player performance.
    for _ in range(len(native.gc.deck)):
        native.gc.remove_card(0)
    for _ in range(5):
        card = native.sts.Card(native.sts.CardId.SEARING_BLOW)
        for _ in range(40):
            card.upgrade()
        native.gc.obtain_card(card)


def combat_action(actions):
    return next((a.index for a in actions if a.description.startswith("play Searing Blow")),
                next(a.index for a in actions if a.description == "end turn"))


def finish_fight(native):
    combat_id = native.combat_id
    for _ in range(30):
        if native.bc is None or native.combat_id != combat_id:
            return
        native.step(combat_action(native.legal_actions()))
    raise AssertionError("Constructed fight exceeded 30 decisions")


def enter_boss(native):
    native.gc.screen_state = native.sts.ScreenState.MAP_SCREEN
    native.gc.cur_map_node_y = 14
    native.gc.floor_num = 15 + 17 * (native.gc.act - 1)
    native.step(0)


def next_act(native):
    native.step(next(a.index for a in native.legal_actions()
                     if a.description == "skip rewards / proceed"))
    native.step(next(a.index for a in native.legal_actions()
                     if a.description == "skip boss relic reward"))


def act3_frontier(ascension, *, keys=False, max_act=3):
    native = backend(ascension, max_act=max_act)
    strong_deck(native)
    native.gc.red_key = native.gc.green_key = native.gc.blue_key = keys
    for act in (1, 2):
        assert native.gc.act == act
        enter_boss(native)
        finish_fight(native)
        next_act(native)
    assert native.gc.act == 3
    native.gc.screen_state = native.sts.ScreenState.MAP_SCREEN
    native.gc.cur_map_node_y = 14
    native.gc.floor_num = 49
    return native


@requires_simulator
class AscensionRulesTests(unittest.TestCase):
    def test_starting_resources_and_curse_at_every_level(self):
        for ascension in range(21):
            with self.subTest(ascension=ascension):
                native = backend(ascension)
                self.assertEqual(native.gc.max_hp, 75 if ascension >= 14 else 80)
                self.assertEqual(native.gc.cur_hp, 68 if ascension >= 14 else 72 if ascension >= 6 else 80)
                bane = [c for c in native.gc.deck if c.id == native.sts.CardId.ASCENDERS_BANE]
                self.assertEqual(len(bane), int(ascension >= 10))
                self.assertTrue(all(not c.transformable for c in bane))
                for _ in range(30):
                    if native.gc.screen_state == native.sts.ScreenState.BATTLE:
                        break
                    native.gc.legal_actions()[0].execute(native.gc)
                else:
                    self.fail("First fight exceeded bounded OOC actions")
                native.advance_to_decision()
                potions = native.public_combat_potions()
                self.assertEqual(potions["capacity"], 2 if ascension >= 11 else 3)
                self.assertFalse(native.summary()["undefined_behavior_evoked"])

    def test_ascension1_adds_elites_and_later_levels_keep_that_map_rule(self):
        counts = []
        for ascension in (0, 1, 20):
            counts.append(sum(node["room"] == "ELITE" for seed in range(16)
                              for node in backend(ascension, seed).map_graph()["nodes"]))
        self.assertGreater(counts[1], counts[0])
        self.assertEqual(counts[1], counts[2])

    def test_ascension5_heals_only_three_quarters_of_missing_hp_between_acts(self):
        for ascension, expected in ((4, 80), (5, 70)):
            with self.subTest(ascension=ascension):
                native = backend(ascension)
                strong_deck(native)
                enter_boss(native)
                finish_fight(native)
                native.gc.cur_hp = 40
                next_act(native)
                self.assertEqual(native.gc.act, 2)
                self.assertEqual(native.gc.cur_hp, expected)

    def test_ascension13_reduces_boss_gold(self):
        rewards = []
        for ascension in (12, 13):
            native = backend(ascension)
            strong_deck(native)
            enter_boss(native)
            finish_fight(native)
            action = next(a for a in native.legal_actions() if a.description.startswith("take gold "))
            rewards.append(int(action.description.split()[2].removesuffix("g")))
        self.assertEqual(rewards[1], int(rewards[0] * .75 + .5))

    def test_every_level_reaches_the_public_api_with_difficulty_in_info(self):
        for ascension in range(21):
            with self.subTest(ascension=ascension):
                with SlayTheSpireEnv(ascension=ascension, max_steps=2) as env:
                    observation, info = env.reset(seed=3)
                    self.assertEqual(observation["state"]["ascension"], ascension)
                    self.assertEqual(info["ascension"], ascension)
                    self.assertEqual(info["observation_version"], OBSERVATION_VERSION)
                    self.assertEqual(observation["state"]["combat_id"], 0)
                    self.assertNotIn("world_seed", observation["state"])

    def test_ascension12_reduces_upgraded_card_rewards_in_later_acts(self):
        for act in (2, 3):
            upgrades = [0, 0]
            for seed in range(16):
                offers = []
                for index, ascension in enumerate((11, 12)):
                    native = backend(ascension, seed)
                    strong_deck(native)
                    # Construct a late-act reward from a normal fight. Its
                    # enemy/deck are controlled; card rewards use real RNG/rules.
                    native.gc.act = act
                    native.gc.screen_state = native.sts.ScreenState.MAP_SCREEN
                    native.gc.cur_map_node_y = -1
                    native.gc.cur_map_node_x = -1
                    native.step(0)
                    finish_fight(native)
                    cards = native.gc.get_card_reward()
                    self.assertEqual(len(cards), 3)
                    offers.append([c.id for c in cards])
                    upgrades[index] += sum(c.upgraded for c in cards)
                self.assertEqual(offers[0], offers[1])
            self.assertGreater(upgrades[0], upgrades[1])
            self.assertGreater(upgrades[1], 0)


@requires_simulator
class AscensionEnemyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_search_test_fixture()

    def battle(self, ascension, encounter, *, act=1):
        native = backend(ascension)
        native.gc.act = act
        gc = self.fixture.encounter_game_copy(native.gc, getattr(native.sts.MonsterEncounter, encounter))
        bc = native.sts.BattleContext()
        bc.init(gc)
        self.assertFalse(bc.undefined_behavior_evoked)
        return native.sts, bc

    def first_attack(self, ascension, encounter):
        sts, bc = self.battle(ascension, encounter)
        for _ in range(8):
            enemy = bc.enemies()[0]
            if enemy["intent_damage"] > 0:
                return enemy
            next(a for a in bc.legal_actions()
                 if a.action_type() == sts.BattleActionType.END_TURN).execute(bc)
            self.assertFalse(bc.undefined_behavior_evoked)
        self.fail("No attack within eight turns")

    def test_damage_thresholds_for_normal_elite_and_boss_enemies(self):
        for encounter, low, high in (("JAW_WORM", 1, 2), ("GREMLIN_NOB", 2, 3),
                                      ("SLIME_BOSS", 3, 4)):
            with self.subTest(encounter=encounter):
                before, after = (self.first_attack(a, encounter) for a in (low, high))
                self.assertEqual(before["intent"], after["intent"])
                self.assertGreater(after["intent_damage"], before["intent_damage"])

    def test_health_thresholds_for_normal_elite_and_boss_enemies(self):
        for encounter, low, high, low_range, high_range in (
                ("JAW_WORM", 6, 7, (40, 44), (42, 46)),
                ("GREMLIN_NOB", 7, 8, (82, 86), (85, 90)),
                ("SLIME_BOSS", 8, 9, (140, 140), (150, 150))):
            with self.subTest(encounter=encounter):
                for ascension, (minimum, maximum) in ((low, low_range), (high, high_range)):
                    _, bc = self.battle(ascension, encounter)
                    self.assertTrue(minimum <= bc.enemies()[0]["max_hp"] <= maximum)

    def test_late_ascensions_change_enemy_abilities(self):
        for encounter, act, low, high, target, power, values in (
                ("THREE_BYRDS", 2, 16, 17, 0, "Flight", (3, 4)),
                ("GREMLIN_NOB", 1, 17, 18, 0, "Enrage", (2, 3)),
                ("AWAKENED_ONE", 3, 18, 19, 2, "Curiosity", (1, 2))):
            with self.subTest(encounter=encounter):
                actual = []
                for ascension in (low, high):
                    sts, bc = self.battle(ascension, encounter, act=act)
                    if power == "Enrage":
                        # Nob's first move applies Enrage; it is not innate.
                        next(a for a in bc.legal_actions()
                             if a.action_type() == sts.BattleActionType.END_TURN).execute(bc)
                    actual.append(bc.enemies()[target]["powers"][power])
                self.assertEqual(tuple(actual), values)

    def test_high_ascension_text_matches_executed_debuffs_and_status_cards(self):
        cases = ((18, 1, "LAGAVULIN", "LAGAVULIN_SIPHON_SOUL", "by 2 each", None, 0),
                 (19, 1, "SLIME_BOSS", "SLIME_BOSS_GOOP_SPRAY", "adds 5 Slimed", "Slimed", 5),
                 (19, 3, "TIME_EATER", "TIME_EATER_HEAD_SLAM", "adds 2 Slimed", "Slimed", 2))
        for ascension, act, encounter, move, phrase, card_name, count in cases:
            with self.subTest(encounter=encounter):
                native = backend(ascension)
                native.gc.act = act
                native.gc = self.fixture.encounter_game_copy(
                    native.gc, getattr(native.sts.MonsterEncounter, encounter))
                native.gc.screen_state = native.sts.ScreenState.BATTLE
                native.advance_to_decision()
                native.bc = self.fixture.monster_intent_copy(native.bc, 0, move)
                with patch("sts_env.env._make_backend", return_value=CombatPublicV5Env(native)):
                    with SlayTheSpireEnv(ascension=ascension, isolated=False) as env:
                        observation, _ = env.reset(seed=3)
                        self.assertIn(phrase, observation["text"])
                        action = next(a["index"] for a in observation["legal_actions"]
                                      if a["description"] == "end turn")
                        observation, _, terminated, truncated, _ = env.step(action)
                        self.assertFalse(terminated or truncated)
                        combat = observation["state"]["combat"]
                        if card_name:
                            cards = combat["cards"]
                            actual = sum(card.get("count", 1) for pile in ("hand", "draw", "discard", "exhaust")
                                         for card in cards[pile] if card["name"] == card_name)
                            self.assertEqual(actual, count)
                        else:
                            self.assertEqual(combat["player"]["powers"]["Strength"], -2)
                            self.assertEqual(combat["player"]["powers"]["Dexterity"], -2)


@requires_simulator
class DoubleBossTests(unittest.TestCase):
    def test_automatic_combat_counts_both_bosses_before_awarding_victory(self):
        for ascension, keys in ((19, False), (20, False), (20, True)):
            with self.subTest(ascension=ascension, keys=keys):
                native = act3_frontier(ascension, keys=keys)
                native.combat_control = "search"
                native.battle_agent.simulation_count_base = 5
                before = native.combat_id
                with patch("sts_env.env._make_backend", return_value=CombatPublicV5Env(native)):
                    with SlayTheSpireEnv(mode="ooc", ascension=ascension, isolated=False) as env:
                        env.reset(seed=3)
                        observation, reward, terminated, truncated, _ = env.step(0)
                        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
                        self.assertEqual(observation["state"]["combat_id"] - before,
                                         2 if ascension == 20 else 1)

    def test_victory_requires_both_bosses_and_combat_id_marks_the_transition(self):
        for ascension, keys, max_act in ((19, False, 3), (20, False, 3),
                                         (20, True, 3), (20, True, 4)):
            with self.subTest(ascension=ascension, keys=keys, max_act=max_act):
                native = act3_frontier(ascension, keys=keys, max_act=max_act)
                with patch("sts_env.env._make_backend", return_value=CombatPublicV5Env(native)):
                    with SlayTheSpireEnv(ascension=ascension, max_act=max_act, isolated=False) as env:
                        observation, _ = env.reset(seed=3)
                        observation, *_ = env.step(0)
                        first_id = observation["state"]["combat_id"]
                        first_enemies = observation["state"]["combat"]["enemies"]
                        last_id = first_id
                        for _ in range(60):
                            action = combat_action(native.legal_actions())
                            observation, reward, terminated, truncated, info = env.step(action)
                            self.assertFalse(observation["state"]["undefined_behavior_evoked"])
                            if native.bc is None:
                                break
                            if observation["state"]["combat_id"] != last_id:
                                self.assertEqual(ascension, 20)
                                self.assertEqual((reward, terminated, truncated), (0.0, False, False))
                                self.assertEqual(observation["phase"], "combat")
                                self.assertEqual(observation["state"]["combat_id"], first_id + 1)
                                self.assertNotEqual(observation["state"]["combat"]["enemies"], first_enemies)
                                self.assertEqual(observation["state"]["combat"]["recent_actions"], [])
                                last_id += 1
                        else:
                            self.fail("Constructed final bosses exceeded 60 decisions")
                        self.assertEqual(last_id - first_id, int(ascension == 20))
                        self.assertEqual((reward, terminated, truncated),
                                         (0.0, False, False) if max_act == 4 else (1.0, True, False))
                        self.assertEqual(info["termination_reason"], "" if max_act == 4 else "victory")

    def test_losing_the_second_boss_is_a_defeat(self):
        fixture = load_search_test_fixture()
        native = act3_frontier(20)
        with patch("sts_env.env._make_backend", return_value=CombatPublicV5Env(native)):
            with SlayTheSpireEnv(ascension=20, isolated=False) as env:
                env.reset(seed=3)
                env.step(0)
                first_id = native.combat_id
                for _ in range(30):
                    observation, reward, terminated, truncated, _ = env.step(combat_action(native.legal_actions()))
                    if native.combat_id != first_id:
                        break
                else:
                    self.fail("Did not reach second boss")
                self.assertEqual((reward, terminated, truncated), (0.0, False, False))
                native.bc = fixture.player_hp_copy(native.bc, 1)
                for _ in range(8):
                    action = next(a.index for a in native.legal_actions() if a.description == "end turn")
                    observation, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                self.assertEqual((reward, terminated, truncated), (0.0, True, False))
                self.assertEqual(info["termination_reason"], "defeat")


@requires_simulator
class AscensionReplayTests(unittest.TestCase):
    def test_high_ascension_traces_match_fresh_processes_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            for ascension, mode in ((1, "full"), (10, "full"), (15, "full"),
                                     (20, "full"), (20, "ooc"), (20, "combat")):
                with self.subTest(ascension=ascension, mode=mode):
                    traces = []
                    for repetition in range(2):
                        path = Path(directory) / f"a{ascension}-{mode}-{repetition}.jsonl"
                        command = [sys.executable, "-m", "sts_env", "run", "--seed", "3",
                                   "--ascension", str(ascension), "--mode", mode, "--agent", "random",
                                   "--max-steps", "12", "--max-act", "1", "--battle-simulations", "20",
                                   "--step-timeout", "30", "--output", str(path)]
                        run = subprocess.run(command, capture_output=True, text=True, timeout=90)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        trace = [json.loads(line) for line in path.read_text().splitlines()]
                        self.assertEqual(trace[0]["env_config"]["ascension"], ascension)
                        self.assertEqual(trace[0]["environment_version"], ENVIRONMENT_VERSION)
                        trace[-1].pop("elapsed_seconds")
                        traces.append(trace)
                    self.assertEqual(traces[0], traces[1])
                    replay = subprocess.run([sys.executable, "-m", "sts_env", "replay", str(path)],
                                            capture_output=True, text=True, timeout=90)
                    self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                    self.assertTrue(json.loads(replay.stdout)["verified_prefix"])
