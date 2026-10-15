"""Public map/key metadata and the existing Emerald elite mechanics."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from sts_env import SlayTheSpireEnv
from sts_env._backend.combat_public_v5 import CombatPublicV5Env
from sts_env._backend.glossary import BURNING_ELITE_MARKER, _render_map
from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.support import requires_simulator


MARKER = "(burning: Emerald Key)"
assert BURNING_ELITE_MARKER == MARKER  # Python constant matches the native suffix pinned below.
NO_KEYS = {"ruby": False, "emerald": False, "sapphire": False}


@requires_simulator
class BurningEliteTest(unittest.TestCase):
    def backend(self, seed: int = 3, *, max_act: int = 3) -> LightspeedHybridEnv:
        return LightspeedHybridEnv(
            world_seed=seed, max_act=max_act, combat_control="llm",
            combat_observation="combat_public_v3",
        )

    def frontier(self, env: LightspeedHybridEnv, target: dict) -> None:
        # Construct only the frontier; native map transitions initialize the
        # encounter and rewards. This is not a claim about a played trajectory.
        parent = next(node for node in env.map_graph()["nodes"]
                      if node["y"] == target["y"] - 1 and target["x"] in node["edges"])
        env.gc.screen_state = env.sts.ScreenState.MAP_SCREEN
        env.gc.cur_map_node_x = parent["x"]
        env.gc.cur_map_node_y = parent["y"]
        env.gc.floor_num = target["y"]

    def enter_burning_elite(self, env: LightspeedHybridEnv) -> None:
        self.frontier(env, env.map_graph()["burning_elite"])
        env.step(next(action.index for action in env.legal_actions()
                      if MARKER in action.description))
        self.assertIsNotNone(env.bc)
        self.assertEqual(env.gc.cur_room, env.sts.Room.ELITE)

    def test_map_nodes_actions_and_branch_summaries_agree(self) -> None:
        env = self.backend()
        graph = env.map_graph()
        burning = graph["burning_elite"]
        self.assertEqual(env.gc.keys(), NO_KEYS)
        self.assertEqual(env.summary()["keys"], NO_KEYS)
        self.assertEqual(set(graph), {"cur_y", "burning_elite", "nodes"})
        flagged = []
        elites = [node for node in graph["nodes"] if node["room"] == "ELITE"]
        self.assertGreater(len(elites), 1)
        for node in graph["nodes"]:
            self.assertIs(type(node["burning_elite"]), bool)
            self.assertEqual(set(node), {"x", "y", "room", "edges", "burning_elite"})
            expected = (node["x"], node["y"]) == (burning["x"], burning["y"])
            self.assertEqual(node["burning_elite"], expected)
            if expected:
                flagged.append(node)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["room"], "ELITE")

        # Every ordinary elite retains its original action description. Exercise
        # the Python x-coordinate parser with the suffixed burning action too.
        for target in elites:
            with self.subTest(x=target["x"], y=target["y"]):
                self.frontier(env, target)
                action = next(action for action in env.legal_actions()
                              if action.description.startswith(f"choose map node x={target['x']} "))
                expected = f"choose map node x={target['x']} room=ELITE"
                if target["burning_elite"]:
                    expected += " " + MARKER
                self.assertEqual(action.description, expected)
                rendered = _render_map(env.map_graph(), [{"description": action.description}])
                self.assertIn(f"x={target['x']}: next room Elite", rendered)
                immediate = rendered.split("next room ", 1)[1].split(".", 1)[0]
                self.assertEqual(MARKER in immediate, target["burning_elite"])

    def test_public_observation_carries_keys_and_visible_burning_route(self) -> None:
        with SlayTheSpireEnv(mode="full", max_act=1) as env:
            observation, _ = env.reset(seed=3)
        self.assertEqual(observation["state"]["keys"], NO_KEYS)
        graph = observation["state"]["map"]
        burning = graph["burning_elite"]
        self.assertIn(f"({burning['x']},{burning['y']}) ELITE {MARKER} ->", observation["text"])
        self.assertEqual(sum(node["burning_elite"] for node in graph["nodes"]), 1)

    def test_key_names_are_independent_and_reflect_native_ownership(self) -> None:
        env = self.backend()
        for name, attribute in (("ruby", "red_key"), ("emerald", "green_key"),
                                ("sapphire", "blue_key")):
            with self.subTest(key=name):
                setattr(env.gc, attribute, True)
                expected = {**NO_KEYS, name: True}
                self.assertEqual(env.gc.keys(), expected)
                self.assertEqual(env.summary()["keys"], expected)
                setattr(env.gc, attribute, False)

    def test_entering_burning_elite_applies_exactly_one_act_scaled_buff(self) -> None:
        observed_buffs = set()
        # Fixed seeds cover all four buffs and a multi-enemy encounter.
        for seed in (1, 3, 5, 6, 9):
            with self.subTest(seed=seed):
                env = self.backend(seed)
                self.enter_burning_elite(env)
                burning = env.bc.enemies()
                # Initialize the identical encounter/RNG with only the map
                # position moved off the burning node, then restore it.
                x = env.gc.cur_map_node_x
                env.gc.cur_map_node_x = -1
                try:
                    normal = env.sts.BattleContext()
                    normal.init(env.gc)
                finally:
                    env.gc.cur_map_node_x = x
                self.assertFalse(normal.undefined_behavior_evoked)
                self.assertFalse(env.bc.undefined_behavior_evoked)
                candidates = {}
                for buff in ("strength", "max_hp", "Metallicize", "Regen"):
                    expected = deepcopy(normal.enemies())
                    for enemy in expected:
                        if buff == "strength":
                            enemy["strength"] += env.gc.act
                            if enemy["intent_hits"] > 0:
                                enemy["intent_damage"] += env.gc.act
                        elif buff == "max_hp":
                            increase = int(enemy["max_hp"] * 0.25 + 0.5)
                            enemy["max_hp"] += increase
                            enemy["cur_hp"] += increase
                        else:
                            amount = 2 * env.gc.act + (2 if buff == "Metallicize" else 1)
                            enemy["powers"][buff] = enemy["powers"].get(buff, 0) + amount
                    candidates[buff] = expected
                matches = [buff for buff, expected in candidates.items() if burning == expected]
                self.assertEqual(len(matches), 1, (burning, candidates))
                observed_buffs.update(matches)
                self.assertEqual(env.summary()["keys"], NO_KEYS)
        self.assertEqual(observed_buffs, {"strength", "max_hp", "Metallicize", "Regen"})

    def win_constructed_combat(self, env: LightspeedHybridEnv) -> None:
        for _ in range(20):
            if env.bc is None:
                break
            self.assertFalse(env.bc.undefined_behavior_evoked)
            actions = env.legal_actions()
            action = next((a for a in actions if a.description.startswith("play Searing Blow")),
                          next(a for a in actions if a.description == "end turn"))
            env.step(action.index)
        self.assertIsNone(env.bc, "constructed combat exceeded 20 decisions")
        self.assertEqual(env.gc.outcome, env.sts.GameOutcome.UNDECIDED)
        self.assertEqual(env.gc.screen_state, env.sts.ScreenState.REWARDS)
        self.assertFalse(env.summary()["undefined_behavior_evoked"])

    def give_test_deck(self, env: LightspeedHybridEnv, *, upgrades: int = 20) -> None:
        for _ in range(len(env.gc.deck)):
            env.gc.remove_card(0)
        for _ in range(5):
            card = env.sts.Card(env.sts.CardId.SEARING_BLOW)
            for _ in range(upgrades):
                card.upgrade()
            env.gc.obtain_card(card)

    def advance_after_boss_rewards(self, env: LightspeedHybridEnv) -> None:
        env.step(next(a.index for a in env.legal_actions()
                      if a.description == "skip rewards / proceed"))
        self.assertEqual(env.gc.screen_state, env.sts.ScreenState.BOSS_RELIC_REWARDS)
        env.step(next(a.index for a in env.legal_actions()
                      if a.description == "skip boss relic reward"))

    def test_emerald_reward_and_native_next_act_map(self) -> None:
        for take_key in (True, False):
            with self.subTest(take_key=take_key):
                env = self.backend()
                # Construct a deterministic deck and map frontiers, then win the
                # elite and boss through ordinary combat and reward actions.
                self.give_test_deck(env)
                self.enter_burning_elite(env)
                self.win_constructed_combat(env)
                if take_key:
                    env.step(next(a.index for a in env.legal_actions()
                                  if a.description == "take key"))
                expected_keys = {**NO_KEYS, "emerald": take_key}
                self.assertEqual(env.summary()["keys"], expected_keys)
                env.step(next(a.index for a in env.legal_actions()
                              if a.description == "skip rewards / proceed"))

                # Synthesize the final map frontier, then use the actual boss,
                # reward and boss-relic transitions to generate the next act map.
                self.assertEqual(env.gc.screen_state, env.sts.ScreenState.MAP_SCREEN)
                env.gc.cur_map_node_y = 14
                env.gc.floor_num = 15
                env.step(0)
                self.assertIsNotNone(env.bc)
                self.win_constructed_combat(env)
                self.advance_after_boss_rewards(env)
                self.assertEqual(env.gc.act, 2)
                self.assertEqual(env.summary()["keys"], expected_keys)
                graph = env.map_graph()
                self.assertEqual(graph["burning_elite"] is None, take_key)
                self.assertTrue(graph["nodes"])
                self.assertEqual(sum(node["burning_elite"] for node in graph["nodes"]),
                                 0 if take_key else 1)
                self.assertTrue(all(MARKER not in a.description for a in env.legal_actions()))

    def test_act3_boss_victory_with_and_without_keys(self) -> None:
        for all_keys, max_act in ((False, 3), (True, 3), (True, 4)):
            with self.subTest(all_keys=all_keys, max_act=max_act):
                native = self.backend(max_act=max_act)
                self.give_test_deck(native, upgrades=40)
                # Construct deck/map frontiers and key ownership; native boss
                # fights and act transitions remain real, including the Act 3 win.
                native.gc.red_key = all_keys
                native.gc.green_key = all_keys
                native.gc.blue_key = all_keys
                for act in (1, 2):
                    self.assertEqual(native.gc.act, act)
                    native.gc.screen_state = native.sts.ScreenState.MAP_SCREEN
                    native.gc.cur_map_node_y = 14
                    native.gc.floor_num = 15 + 17 * (act - 1)
                    native.step(0)
                    self.win_constructed_combat(native)
                    self.advance_after_boss_rewards(native)
                self.assertEqual(native.gc.act, 3)
                native.gc.screen_state = native.sts.ScreenState.MAP_SCREEN
                native.gc.cur_map_node_y = 14
                native.gc.floor_num = 49
                with patch("sts_env.env._make_backend", return_value=CombatPublicV5Env(native)):
                    with SlayTheSpireEnv(mode="full", max_act=max_act, isolated=False) as env:
                        env.reset(seed=3)
                        observation, *_ = env.step(0)  # Enter the Act 3 boss.
                        self.assertIsNotNone(native.bc)
                        for _ in range(20):
                            actions = observation["legal_actions"]
                            action = next((a for a in actions
                                           if a["description"].startswith("play Searing Blow")),
                                          next(a for a in actions if a["description"] == "end turn"))
                            observation, reward, terminated, truncated, info = env.step(action["index"])
                            if native.bc is None:
                                break
                        self.assertIsNone(native.bc, "constructed boss exceeded 20 decisions")
                        self.assertFalse(observation["state"]["undefined_behavior_evoked"])
                        self.assertEqual(observation["state"]["keys"],
                                         {key: all_keys for key in NO_KEYS})
                        self.assertEqual(native.gc.act, 4 if all_keys else 3)
                        if max_act == 3:
                            self.assertEqual((reward, terminated, truncated), (1.0, True, False))
                            self.assertEqual(info["termination_reason"], "victory")
                            self.assertEqual(info["outcome"], "GameOutcome.PLAYER_VICTORY")
                            self.assertEqual(observation["legal_actions"], [])
                        else:
                            self.assertEqual((reward, terminated, truncated), (0.0, False, False))
                            self.assertEqual(info["termination_reason"], "")
                            self.assertEqual(info["outcome"], "GameOutcome.UNDECIDED")
                            self.assertTrue(observation["legal_actions"])


if __name__ == "__main__":
    unittest.main()
