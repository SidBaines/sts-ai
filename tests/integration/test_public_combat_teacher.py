"""Public combat observation and non-mutating search-teacher regressions."""
from __future__ import annotations

import re
import unittest

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.local_tasks import base, get_task
from sts_ai.local_tasks.runner import replay_task_start, resolve_task_replay_action

from tests.support import requires_simulator


def _advance_to_first_combat(env: LightspeedHybridEnv) -> None:
    """Take deterministic first actions until the first combat choice."""
    for _ in range(100):
        env.advance_to_decision()
        if env.bc is not None:
            return
        actions = env.legal_actions()
        if not actions:
            break
        env.step(0)
    raise AssertionError("did not reach a combat decision")


@requires_simulator
class PublicCombatObservationTest(unittest.TestCase):
    def test_public_v2_exposes_card_type_without_changing_v1(self):
        v2 = LightspeedHybridEnv(
            world_seed=3,
            battle_simulations=10,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v2",
        )
        _advance_to_first_combat(v2)

        cards = v2.public_combat_cards()
        all_cards = list(cards["hand"])
        for pile in ("draw", "discard", "exhaust"):
            all_cards.extend(cards[pile])
        by_base_name = {
            re.sub(r"\+\d*$", "", str(card["name"])): card
            for card in all_cards
        }
        self.assertEqual(by_base_name["Strike"]["type"], "Attack")
        self.assertEqual(by_base_name["Defend"]["type"], "Skill")
        self.assertIn("[Attack]", by_base_name["Strike"]["label"])
        self.assertIn("[Skill]", by_base_name["Defend"]["label"])
        self.assertIn("[Attack]", v2.describe_state())
        self.assertIn("[Skill]", v2.describe_state())

        play_actions = [
            action.description
            for action in v2.legal_actions()
            if action.description.startswith("play ")
        ]
        self.assertTrue(play_actions)
        self.assertTrue(
            all(re.search(r"\[(?:Attack|Skill|Power|Curse|Status)\] \(cost", desc)
                for desc in play_actions)
        )
        search = v2.search_best_combat_action(8, search_seed=123)
        search_plays = [
            str(edge["description"])
            for edge in search["root_edges"]
            if str(edge["description"]).startswith("play ")
        ]
        self.assertTrue(search_plays)
        self.assertTrue(all("[" in desc.split(" (cost", 1)[0] for desc in search_plays))

        v1 = LightspeedHybridEnv(
            world_seed=3,
            battle_simulations=10,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        _advance_to_first_combat(v1)
        self.assertNotIn("[Attack]", v1.describe_state())
        self.assertNotIn("[Skill]", v1.describe_state())
        self.assertTrue(
            all("type" not in card for card in v1.public_combat_cards()["hand"])
        )

    def test_public_v1_is_explicit_and_legacy_remains_available(self):
        env = LightspeedHybridEnv(
            world_seed=3,
            battle_simulations=10,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        _advance_to_first_combat(env)

        legacy = str(env.bc.describe_state())
        self.assertEqual(legacy, str(env.bc.describe_state(public_state=False)))
        self.assertNotIn("Draw pile contents (unordered):", legacy)
        self.assertNotIn("Turn counters:", legacy)
        self.assertIn("\nPotions: none", legacy)

        state = env.describe_state()
        self.assertIn("Turn counters: cards played 0, attacks 0, skills 0, discarded 0", state)
        self.assertRegex(state, r"Piles: draw \d+, discard \d+, exhaust \d+")
        self.assertIn("Draw pile contents (unordered):", state)
        self.assertIn("Discard pile contents (unordered):", state)
        self.assertIn("Exhaust pile contents (unordered):", state)
        self.assertIn("Relics: Burning Blood", state)
        self.assertRegex(
            state,
            r"Potions \(capacity \d+\): \[0\] empty(?:, \[\d+\] empty)*",
        )
        self.assertIn("Recent actions this turn: none", state)
        self.assertEqual(env.summary()["combat_observation"], "combat_public_v1")
        self.assertEqual(
            env.summary()["combat"]["player_energy_per_turn"],
            int(env.bc.player_energy_per_turn),
        )
        self.assertEqual(env.summary()["combat"]["recent_actions"], [])
        self.assertIn("cards", env.summary()["combat"])
        self.assertIn("relics", env.summary()["combat"])
        self.assertIn("player", env.summary()["combat"])
        self.assertIn("potions", env.summary()["combat"])

        player = env.public_combat_player()
        self.assertEqual(player["energy_per_turn"], int(env.bc.player_energy_per_turn))
        self.assertIsInstance(player["powers"], dict)

        potions = env.public_combat_potions()
        self.assertEqual(len(potions["slots"]), potions["capacity"])
        self.assertEqual(
            [slot["index"] for slot in potions["slots"]],
            list(range(potions["capacity"])),
        )
        self.assertTrue(all(slot["empty"] for slot in potions["slots"]))

        cards = env.public_combat_cards()
        self.assertIn("unordered aggregates", cards["order_semantics"])
        for pile in ("draw", "discard", "exhaust"):
            labels = [entry["label"] for entry in cards[pile]]
            self.assertEqual(labels, sorted(labels))
            self.assertTrue(all(entry["count"] >= 1 for entry in cards[pile]))
            self.assertTrue(all("card_id" not in entry for entry in cards[pile]))
        hand_text = state.split("\nHand:", 1)[1].split("\nPiles:", 1)[0]
        self.assertEqual(
            len(cards["hand"]),
            len(re.findall(r"^  \[\d+\]", hand_text, re.MULTILINE)),
        )

        relics = env.public_combat_relics()
        self.assertEqual([relic["name"] for relic in relics], ["Burning Blood"])
        self.assertTrue(all("relic_id" not in relic for relic in relics))

    def test_upgrade_markers_are_present_in_cards_and_actions(self):
        env = LightspeedHybridEnv(
            world_seed=3,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        # Make upgraded cards dominate the deterministic first draw. This uses the
        # public GameContext card API, rather than mutating BattleContext internals.
        for _ in range(30):
            card = env.sts.Card(env.sts.CardId.STRIKE_RED)
            card.upgrade()
            env.gc.obtain_card(card)
            env.gc.obtain_card(env.sts.Card(env.sts.CardId.DUAL_WIELD))
        _advance_to_first_combat(env)

        all_cards = env.public_combat_cards()
        names = [card["name"] for card in all_cards["hand"]]
        names.extend(card["name"] for card in all_cards["draw"])
        self.assertIn("Strike+", names)
        self.assertTrue(
            any(action.description.startswith("play Strike+ ") for action in env.legal_actions()),
            "upgraded card action was rendered with its base name",
        )

        dual_wield_idx = next(
            action.index
            for action in env.legal_actions()
            if action.description.startswith("play Dual Wield")
        )
        env.step(dual_wield_idx)
        self.assertIn("Selection type: DUAL_WIELD", env.describe_state())
        self.assertIn("select card for DUAL_WIELD: Strike+", env.describe_state())
        self.assertTrue(
            all(action.description.endswith("Strike+") for action in env.legal_actions()),
            "card-select candidates lost their upgrade marker",
        )

    def test_recent_actions_are_scoped_to_current_turn(self):
        env = LightspeedHybridEnv(
            world_seed=3,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        _advance_to_first_combat(env)
        play_idx = next(
            action.index for action in env.legal_actions() if action.description.startswith("play ")
        )
        selected = env.legal_actions()[play_idx]
        start_turn = int(env.bc.turn)
        env.step(play_idx)
        self.assertEqual(int(env.bc.turn), start_turn)
        self.assertIn(f"Recent actions this turn: {selected.description}", env.describe_state())

        end_idx = next(action.index for action in env.legal_actions() if action.description == "end turn")
        env.step(end_idx)
        self.assertGreater(int(env.bc.turn), start_turn)
        self.assertIn("Recent actions this turn: none", env.describe_state())

    def test_bitfield_only_corruption_power_is_public(self):
        manifest = base.load_manifest(
            "data/local_curricula/gremlin_nob/manifests/source.validated_v1.json"
        )
        window = next(
            item for item in manifest["windows"] if int(item["world_seed"]) == 74
        )
        task = get_task("gremlin_nob")
        env = LightspeedHybridEnv(
            world_seed=74,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        replay_task_start(env, window, task)

        found = False
        for record in base.source_records_for_window(manifest, window):
            env.advance_to_decision()
            selected = record["selected_action"]
            index = resolve_task_replay_action(env, selected)
            live_description = env.legal_actions()[index].description
            env.step(index)
            if live_description.startswith("play Corruption"):
                found = True
                self.assertIn("Player powers: Corruption 1", env.describe_state())
                self.assertEqual(env.public_combat_player()["powers"]["Corruption"], 1)
                break
        self.assertTrue(found, "seed 74 source window never played Corruption")

    def test_targeted_potion_keeps_same_name_enemy_slots_distinct(self):
        manifest = base.load_manifest(
            "data/local_curricula/sentries/manifests/source.validated_v1.json"
        )
        window = next(
            item for item in manifest["windows"] if int(item["world_seed"]) == 118
        )
        env = LightspeedHybridEnv(
            world_seed=118,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        replay_task_start(env, window, get_task("sentries"))

        weak_targets = [
            action.description
            for action in env.legal_actions()
            if action.description.startswith("drink potion Weak Potion -> SENTRY")
        ]
        self.assertEqual(
            weak_targets,
            [
                "drink potion Weak Potion -> SENTRY [enemy 0]",
                "drink potion Weak Potion -> SENTRY [enemy 1]",
                "drink potion Weak Potion -> SENTRY [enemy 2]",
            ],
        )

    def test_secret_technique_candidates_are_public_sorted_and_structured(self):
        env = LightspeedHybridEnv(
            world_seed=1,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v2",
        )
        # Make Secret Technique reliably appear while leaving distinct skills in
        # the hidden draw pile. The menu must be a function of the public multiset,
        # not of their underlying draw-vector positions.
        for _ in range(8):
            env.gc.obtain_card(env.sts.Card(env.sts.CardId.SECRET_TECHNIQUE))
        for card_id in (
            env.sts.CardId.CORRUPTION,
            env.sts.CardId.TRUE_GRIT,
            env.sts.CardId.SHRUG_IT_OFF,
            env.sts.CardId.ARMAMENTS,
        ):
            env.gc.obtain_card(env.sts.Card(card_id))
        _advance_to_first_combat(env)

        play_idx = next(
            action.index
            for action in env.legal_actions()
            if action.description.startswith("play Secret Technique")
        )
        env.step(play_idx)
        displayed = [action.description for action in env.legal_actions()]
        self.assertGreaterEqual(len(displayed), 2)
        self.assertEqual(displayed, sorted(displayed))
        self.assertTrue(all("[Skill] (cost" in action for action in displayed))
        self.assertIn("Selection type: SECRET_TECHNIQUE", env.describe_state())

        structured = env.public_combat_cards()["selection"]
        self.assertEqual(structured["type"], "SECRET_TECHNIQUE")
        self.assertEqual(structured["candidates"], sorted(structured["candidates"]))
        self.assertEqual(sorted(set(structured["candidates"])), displayed)
        with self.assertRaisesRegex(ValueError, "draw-index card selection"):
            env.search_best_combat_action(
                8,
                search_seed=1,
                draw_order_seed=100,
            )


@requires_simulator
class SearchTeacherTest(unittest.TestCase):
    def test_query_is_legal_deterministic_and_non_mutating(self):
        env = LightspeedHybridEnv(
            world_seed=3,
            max_act=1,
            combat_control="llm",
            combat_observation="combat_public_v1",
        )
        _advance_to_first_combat(env)

        legacy_before = str(env.bc.describe_state())
        public_before = env.describe_state()
        cards_before = env.public_combat_cards()
        relics_before = env.public_combat_relics()
        legal_before = [int(action.bits) for action in env.bc.legal_actions()]
        summary_before = env.summary()

        first = env.search_best_combat_action(8, search_seed=123, draw_order_seed=456)
        second = env.search_best_combat_action(8, search_seed=123, draw_order_seed=456)

        self.assertIn(int(first["bits"]), legal_before)
        self.assertTrue(first["action"].is_valid(env.bc))
        self.assertEqual(first["simulations_requested"], 8)
        self.assertEqual(first["root_simulations"], 8)
        self.assertEqual(len(first["root_edges"]), len(legal_before))
        self.assertEqual(first["selection_method"], second["selection_method"])
        self.assertEqual(int(first["bits"]), int(second["bits"]))
        self.assertEqual(
            [(int(e["bits"]), int(e["visits"]), e["evaluation_sum"]) for e in first["root_edges"]],
            [(int(e["bits"]), int(e["visits"]), e["evaluation_sum"]) for e in second["root_edges"]],
        )

        provenance = first["provenance"]
        self.assertEqual(provenance["searcher"], "BattleScumSearcher2")
        self.assertEqual(int(provenance["search_seed"]), 123)
        self.assertTrue(provenance["draw_order_perturbed"])
        self.assertEqual(int(provenance["draw_order_seed"]), 456)
        self.assertTrue(provenance["state_copied"])
        self.assertFalse(provenance["live_state_mutated"])
        self.assertTrue(provenance["teacher_uses_full_simulator_state"])
        self.assertFalse(provenance["student_safe_observation"])

        # Strong mutation audit: every policy-facing snapshot and legal identity is
        # exactly unchanged after both normal search and hidden-order intervention.
        self.assertEqual(str(env.bc.describe_state()), legacy_before)
        self.assertEqual(env.describe_state(), public_before)
        self.assertEqual(env.public_combat_cards(), cards_before)
        self.assertEqual(env.public_combat_relics(), relics_before)
        self.assertEqual([int(action.bits) for action in env.bc.legal_actions()], legal_before)
        self.assertEqual(env.summary(), summary_before)

    def test_query_rejects_invalid_budget(self):
        env = LightspeedHybridEnv(world_seed=3, max_act=1, combat_control="llm")
        _advance_to_first_combat(env)
        with self.assertRaises(ValueError):
            env.search_best_combat_action(0)


if __name__ == "__main__":
    unittest.main()
