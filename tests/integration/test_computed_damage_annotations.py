"""Oracle tests for opt-in computed damage on special and untargeted attacks."""
from __future__ import annotations

import re
import unittest
from typing import Any

from sts_env._backend import combat_public_v4, glossary, lightspeed
from sts_env._backend.lightspeed import LightspeedHybridEnv

from tests.support import load_search_test_fixture, requires_simulator


_DEAL_RE = re.compile(r" \(deal (?P<total>\d+)(?: = \d+ x\d+)?\)$")


def _reach_uninitialized_battle(world_seed: int) -> LightspeedHybridEnv:
    """Reach the first battle without letting the environment construct its context."""
    env = LightspeedHybridEnv(
        world_seed=world_seed,
        max_act=1,
        combat_control="llm",
    )
    for _ in range(20):
        if env.gc.screen_state == env.sts.ScreenState.BATTLE:
            return env
        actions = list(env.gc.legal_actions())
        if not actions:
            break
        actions[0].execute(env.gc)
    raise AssertionError("did not reach an uninitialized battle")


def _make_card(
    env: LightspeedHybridEnv,
    name: str,
    upgrades: int = 0,
    special_data: int | None = None,
) -> Any:
    card = env.sts.Card(getattr(env.sts.CardId, name))
    for _ in range(upgrades):
        card.upgrade()
    if special_data is not None:
        card.misc = special_data
    return card


def _battle_with_deck(
    cards: list[tuple[str, int] | tuple[str, int, int]],
    *,
    world_seed: int = 4,
    relics: tuple[str, ...] = (),
) -> LightspeedHybridEnv:
    env = _reach_uninitialized_battle(world_seed)
    while len(env.gc.deck):
        env.gc.remove_card(0)
    for card_spec in cards:
        name, upgrades, *special_data = card_spec
        env.gc.obtain_card(
            _make_card(env, name, upgrades, special_data[0] if special_data else None)
        )
    for relic in relics:
        relic_id = getattr(env.sts.RelicId, relic)
        env.gc.obtain_relic(relic_id)
        if not any(item.id == relic_id for item in env.gc.relics):
            raise AssertionError(f"could not install relic {relic}")
    env.advance_to_decision()
    if env.bc is None:
        raise AssertionError("battle context was not initialized")
    return env


def _find_action(env: LightspeedHybridEnv, card_name: str) -> Any:
    prefix = f"play {card_name}"
    matches = [
        action
        for action in env.bc.legal_actions()
        if action.describe(env.bc).startswith(prefix)
    ]
    if not matches:
        raise AssertionError(f"no legal action found for {card_name}")
    return matches[0]


def _play_setup_card(env: LightspeedHybridEnv, card_name: str) -> None:
    _find_action(env, card_name).execute(env.bc)
    if env.bc.outcome != env.sts.BattleOutcome.UNDECIDED:
        raise AssertionError(f"setup card {card_name} unexpectedly ended the battle")


@requires_simulator
class ComputedDamageAnnotationsTest(unittest.TestCase):
    def test_damage_fixture_sets_strength_and_rejects_negative_statuses(self) -> None:
        fixture = load_search_test_fixture()
        env = _battle_with_deck([("INFLAME", 0)] + [("STRIKE_RED", 0)] * 4)
        _play_setup_card(env, "Inflame")
        self.assertEqual(env.bc.public_player()["powers"]["Strength"], 2)
        for strength in (0, 3):
            with self.subTest(strength=strength):
                scratch = fixture.damage_annotations_copy(env.bc, 3, strength, 0, [0], False)
                self.assertEqual(scratch.public_player()["powers"].get("Strength", 0), strength)
                self.assertEqual(env.bc.public_player()["powers"]["Strength"], 2)
        for strength, weak in ((-1, 0), (0, -1)):
            with self.subTest(strength=strength, weak=weak):
                with self.assertRaisesRegex(RuntimeError, "invalid damage annotation fixture parameters"):
                    fixture.damage_annotations_copy(env.bc, 3, strength, weak, [0], False)

    def test_same_name_labels_match_for_cards_potions_and_damage_lists(self) -> None:
        fixture = load_search_test_fixture()
        env = _battle_with_deck([("STRIKE_RED", 0)] * 4 + [("CLEAVE", 0)], world_seed=3)
        count = len(env.bc.enemies())
        self.assertGreater(count, 1)
        env.bc = fixture.damage_annotations_copy(env.bc, 3, 0, 0, [0] * count, True)
        env.bc.obtain_potion(env.sts.Potion.FIRE_POTION)
        labels = [f"{enemy['name']} [enemy {enemy['index']}]" for enemy in env.bc.enemies()]
        for computed in (False, True):
            with self.subTest(computed=computed):
                descriptions = [action.describe(env.bc, include_computed_damage=computed)
                                for action in env.bc.legal_actions()]
                for label in labels:
                    self.assertIn(f"play Strike (cost 1) -> {label} (deal 6)", descriptions)
                    self.assertIn(f"drink potion Fire Potion -> {label}", descriptions)
                cleave = next(desc for desc in descriptions if desc.startswith("play Cleave "))
                expected = "play Cleave (cost 1)"
                if computed:
                    expected += " (deal per target: " + "; ".join(f"{label}: 8" for label in labels) + ")"
                self.assertEqual(cleave, expected)

    def test_annotations_match_executed_damage(self) -> None:
        cases = (
            # name, deck, optional non-damaging setup card, category
            (
                "Body Slam",
                [("DEFEND_RED", 0), ("BODY_SLAM", 0)] * 2 + [("BODY_SLAM", 0)],
                "Defend",
                "special",
            ),
            (
                "Heavy Blade",
                [("INFLAME", 0), ("HEAVY_BLADE", 0)] * 2 + [("HEAVY_BLADE", 0)],
                "Inflame",
                "special",
            ),
            (
                "Perfected Strike",
                [("PERFECTED_STRIKE", 0)] + [("STRIKE_RED", 0)] * 4,
                None,
                "special",
            ),
            (
                "Rampage=10",
                [("RAMPAGE", 0, 10)] * 5,
                None,
                "special",
            ),
            (
                "Searing Blow+3",
                [("SEARING_BLOW", 3)] * 5,
                None,
                "special",
            ),
            (
                "Cleave",
                [("CLEAVE", 0)] * 5,
                None,
                "untargeted",
            ),
            (
                "Dramatic Entrance",
                [("DRAMATIC_ENTRANCE", 0)] * 5,
                None,
                "untargeted",
            ),
            (
                "Thunderclap",
                [("THUNDERCLAP", 0)] * 5,
                None,
                "untargeted",
            ),
            (
                "Immolate",
                [("IMMOLATE", 0)] * 5,
                None,
                "untargeted",
            ),
            (
                "Sword Boomerang",
                [("SWORD_BOOMERANG", 0)] * 5,
                None,
                "untargeted",
            ),
        )
        exercised: set[str] = set()

        for card_name, deck, setup_card, category in cases:
            with self.subTest(card=card_name):
                env = _battle_with_deck(deck)
                self.assertEqual(
                    sum(bool(enemy["alive"]) for enemy in env.bc.enemies()),
                    1,
                )
                if setup_card is not None:
                    _play_setup_card(env, setup_card)

                action = _find_action(env, card_name)
                omitted = action.describe(env.bc)
                flag_off = action.describe(env.bc, include_computed_damage=False)
                annotated = action.describe(env.bc, include_computed_damage=True)
                self.assertEqual(omitted, flag_off)
                self.assertNotIn("(deal ", omitted)
                match = _DEAL_RE.search(annotated)
                self.assertIsNotNone(match, annotated)
                total = int(match.group("total"))

                if card_name == "Sword Boomerang":
                    self.assertRegex(annotated, r"\(deal \d+ = \d+ x3\)$")

                enemy_before = next(enemy for enemy in env.bc.enemies() if enemy["alive"])
                self.assertEqual(enemy_before["block"], 0)
                hp_before = int(enemy_before["cur_hp"])
                target_idx = int(enemy_before["index"])
                action.execute(env.bc)
                enemy_after = list(env.bc.enemies())[target_idx]
                hp_after = int(enemy_after["cur_hp"])
                if hp_after == 0:
                    self.assertGreaterEqual(total, hp_before)
                else:
                    self.assertEqual(hp_before - hp_after, total)
                exercised.add(category)

        self.assertIn("special", exercised, "no special-formula card was exercised")
        self.assertIn("untargeted", exercised, "no untargeted attack was exercised")

    def test_vigor_annotations_match_executed_damage(self) -> None:
        for card_name, card_id, expected_damage in (
            ("Cleave", "CLEAVE", 16),
            ("Strike", "STRIKE_RED", 14),
        ):
            with self.subTest(card=card_name):
                env = _battle_with_deck(
                    [(card_id, 0)] * 5,
                    relics=("AKABEKO",),
                )
                self.assertEqual(env.bc.public_player()["powers"]["Vigor"], 8)
                action = _find_action(env, card_name)
                annotated = action.describe(env.bc, include_computed_damage=True)
                match = _DEAL_RE.search(annotated)
                self.assertIsNotNone(match, annotated)
                total = int(match.group("total"))
                self.assertEqual(total, expected_damage)

                enemy_before = next(enemy for enemy in env.bc.enemies() if enemy["alive"])
                self.assertEqual(enemy_before["block"], 0)
                hp_before = int(enemy_before["cur_hp"])
                target_idx = int(enemy_before["index"])
                action.execute(env.bc)
                hp_after = int(list(env.bc.enemies())[target_idx]["cur_hp"])
                self.assertEqual(hp_before - hp_after, total)

    def test_v3_mind_blast_override_preserves_flag_off_defect(self) -> None:
        env = _battle_with_deck(
            [("MIND_BLAST", 0)] + [("STRIKE_RED", 0)] * 11,
        )
        action = _find_action(env, "Mind Blast")
        omitted = action.describe(env.bc)
        flag_off = action.describe(env.bc, include_computed_damage=False)
        annotated = action.describe(env.bc, include_computed_damage=True)
        self.assertEqual(omitted, flag_off)
        self.assertRegex(flag_off, r"\(deal 0\)$")
        match = _DEAL_RE.search(annotated)
        self.assertIsNotNone(match, annotated)
        total = int(match.group("total"))
        self.assertEqual(total, 7)

        enemy_before = next(enemy for enemy in env.bc.enemies() if enemy["alive"])
        self.assertEqual(enemy_before["block"], 0)
        hp_before = int(enemy_before["cur_hp"])
        target_idx = int(enemy_before["index"])
        action.execute(env.bc)
        hp_after = int(list(env.bc.enemies())[target_idx]["cur_hp"])
        self.assertEqual(hp_before - hp_after, total)

    def test_v3_ritual_dagger_override_preserves_flag_off_defect(self) -> None:
        env = _battle_with_deck(
            [("RITUAL_DAGGER", 0, 23)] + [("STRIKE_RED", 0)] * 4,
        )
        action = _find_action(env, "Ritual Dagger")
        omitted = action.describe(env.bc)
        flag_off = action.describe(env.bc, include_computed_damage=False)
        annotated = action.describe(env.bc, include_computed_damage=True)
        self.assertEqual(omitted, flag_off)
        self.assertRegex(flag_off, r"\(deal 15\)$")
        match = _DEAL_RE.search(annotated)
        self.assertIsNotNone(match, annotated)
        total = int(match.group("total"))
        self.assertEqual(total, 23)

        enemy_before = next(enemy for enemy in env.bc.enemies() if enemy["alive"])
        self.assertEqual(enemy_before["block"], 0)
        hp_before = int(enemy_before["cur_hp"])
        target_idx = int(enemy_before["index"])
        action.execute(env.bc)
        hp_after = int(list(env.bc.enemies())[target_idx]["cur_hp"])
        self.assertEqual(hp_before - hp_after, total)

    def test_multi_enemy_annotations_match_executed_damage(self) -> None:
        fixture = load_search_test_fixture()
        for card_name, card_id in (
            ("Cleave", "CLEAVE"),
            ("Dramatic Entrance", "DRAMATIC_ENTRANCE"),
            ("Immolate", "IMMOLATE"),
            ("Reaper", "REAPER"),
            ("Thunderclap", "THUNDERCLAP"),
            ("Whirlwind", "WHIRLWIND"),
            ("Sword Boomerang", "SWORD_BOOMERANG"),
        ):
            for same_name in (False, True):
                with self.subTest(card=card_name, same_name=same_name):
                    env = _battle_with_deck([(card_id, 0)] * 5, world_seed=3)
                    count = len(env.bc.enemies())
                    self.assertGreater(count, 1)
                    # Include different Vulnerable values, Strength and Weak;
                    # the fixture removes reactive powers and prevents overkill.
                    env.bc = fixture.damage_annotations_copy(
                        env.bc, 3, 3, 1, [0] + [2] * (count - 1), same_name,
                    )
                    action = _find_action(env, card_name)
                    flag_off = action.describe(env.bc)
                    cost = {"WHIRLWIND": "X", "DRAMATIC_ENTRANCE": "0",
                            "IMMOLATE": "2", "REAPER": "2"}.get(card_id, "1")
                    self.assertEqual(flag_off, f"play {card_name} (cost {cost})")
                    self.assertEqual(flag_off, action.describe(
                        env.bc, include_computed_damage=False,
                    ))
                    self.assertNotIn("(deal ", flag_off)
                    annotated = action.describe(env.bc, include_computed_damage=True)
                    # All current scalar consumers must skip a per-target list.
                    for parser in (lightspeed._DEAL_RE, combat_public_v4._DEAL_RE,
                                   glossary._DEAL_RE):
                        self.assertIsNone(parser.search(annotated), annotated)
                    prefix = (" (deal per random hit, 3 hits total: "
                              if card_id == "SWORD_BOOMERANG" else " (deal per target: ")
                    entries = annotated.split(prefix, 1)[1][:-1].split("; ")
                    before = list(env.bc.enemies())
                    self.assertEqual(len(entries), len(before))
                    scratch = fixture.copy(env.bc)
                    action.execute(scratch)
                    self.assertFalse(scratch.undefined_behavior_evoked)
                    random_hits = 0
                    for entry, enemy, after in zip(entries, before, scratch.enemies()):
                        label, damage = entry.rsplit(": ", 1)
                        expected_label = enemy["name"]
                        if same_name:
                            expected_label += f" [enemy {enemy['index']}]"
                        self.assertEqual(label, expected_label)
                        total = int(damage.split(" = ", 1)[0])
                        dealt = enemy["cur_hp"] - after["cur_hp"]
                        if card_id == "SWORD_BOOMERANG":
                            per = int((3 + 3) * .75 * (1.5 if enemy["vulnerable"] else 1))
                            self.assertEqual(total, per)
                            self.assertEqual(dealt % per, 0)
                            random_hits += dealt // per
                        else:
                            self.assertEqual(dealt, total)
                        if card_id == "WHIRLWIND":
                            self.assertRegex(damage, r"^\d+ = \d+ x3$")
                    if card_id == "SWORD_BOOMERANG":
                        self.assertEqual(random_hits, 3)
                    self.assertNotEqual(entries[0].rsplit(": ", 1)[1],
                                        entries[1].rsplit(": ", 1)[1])

    def test_whirlwind_energy_and_chemical_x(self) -> None:
        fixture = load_search_test_fixture()
        for energy in (0, 1, 2, 3):
            for chemical_x in (False, True):
                for upgraded in (False, True):
                    with self.subTest(energy=energy, chemical_x=chemical_x, upgraded=upgraded):
                        env = _battle_with_deck(
                            [("WHIRLWIND", int(upgraded))] * 5,
                            relics=("CHEMICAL_X", "AKABEKO") if chemical_x else ("AKABEKO",),
                        )
                        env.bc = fixture.damage_annotations_copy(env.bc, energy, 2, 1, [2], False)
                        action = _find_action(env, "Whirlwind")
                        name = "Whirlwind+" if upgraded else "Whirlwind"
                        self.assertEqual(action.describe(env.bc), f"play {name} (cost X)")
                        self.assertEqual(action.describe(env.bc, include_computed_damage=False),
                                         f"play {name} (cost X)")
                        # Damage arithmetic floors only after all modifiers.
                        per = int(((8 if upgraded else 5) + 8 + 2) * .75 * 1.5)
                        hits = energy + (2 if chemical_x else 0)
                        self.assertTrue(action.describe(env.bc, include_computed_damage=True).endswith(
                            f"(deal {per * hits} = {per} x{hits})",
                        ))
                        for free_once in (False, True):
                            scratch = fixture.hand_flags_copy(env.bc, action.source_idx(), free_once, False)
                            annotated = action.describe(scratch, include_computed_damage=True)
                            self.assertTrue(annotated.endswith(f"(deal {per * hits} = {per} x{hits})"))
                            before = scratch.enemies()[0]["cur_hp"]
                            action.execute(scratch)
                            self.assertFalse(scratch.undefined_behavior_evoked)
                            self.assertEqual(before - scratch.enemies()[0]["cur_hp"], per * hits)
                            self.assertEqual(scratch.public_player()["energy"], energy if free_once else 0)

    def test_fiend_fire_counts_other_cards_in_hand(self) -> None:
        fixture = load_search_test_fixture()
        for other_cards in (0, 1, 4):
            for upgraded in (False, True):
                with self.subTest(other_cards=other_cards, upgraded=upgraded):
                    env = _battle_with_deck([("FIEND_FIRE", int(upgraded))] * (other_cards + 1))
                    env.bc = fixture.damage_annotations_copy(env.bc, 3, 2, 1, [2], False)
                    self.assertEqual(len(env.bc.public_cards()["hand"]), other_cards + 1)
                    action = _find_action(env, "Fiend Fire")
                    name = "Fiend Fire+" if upgraded else "Fiend Fire"
                    expected_off = f"play {name} (cost 2) -> {env.bc.enemies()[0]['name']}"
                    self.assertEqual(action.describe(env.bc), expected_off)
                    self.assertEqual(action.describe(env.bc, include_computed_damage=False), expected_off)
                    per = int(((10 if upgraded else 7) + 2) * .75 * 1.5)
                    self.assertEqual(action.describe(env.bc, include_computed_damage=True),
                                     f"{expected_off} (deal {per * other_cards} = {per} x{other_cards})")
                    scratch = fixture.copy(env.bc)
                    before = scratch.enemies()[0]["cur_hp"]
                    action.execute(scratch)
                    self.assertFalse(scratch.undefined_behavior_evoked)
                    self.assertEqual(before - scratch.enemies()[0]["cur_hp"], per * other_cards)
                    self.assertEqual(len(scratch.public_cards()["hand"]), 0)
                    self.assertEqual(sum(card["count"] for card in scratch.public_cards()["exhaust"]),
                                     other_cards + 1)

    def test_existing_targeted_annotation_is_flag_independent(self) -> None:
        env = _battle_with_deck([("STRIKE_RED", 0)] * 5)
        action = _find_action(env, "Strike")
        omitted = action.describe(env.bc)
        self.assertIn("(deal ", omitted)
        self.assertEqual(
            omitted,
            action.describe(env.bc, include_computed_damage=False),
        )
        self.assertEqual(
            omitted,
            action.describe(env.bc, include_computed_damage=True),
        )

    def test_turn_math_excludes_fiend_fire_hand_dependent_damage(self) -> None:
        fixture = load_search_test_fixture()
        for upgraded in (False, True):
            with self.subTest(upgraded=upgraded):
                env = _battle_with_deck(
                    [("FIEND_FIRE", int(upgraded))] + [("STRIKE_RED", 0)] * 4,
                )
                env.combat_observation = "combat_public_v3"
                env.bc = fixture.damage_annotations_copy(env.bc, 3, 0, 0, [0], False)
                self.assertEqual(env.bc.enemies()[0]["name"], "JAW_WORM")
                self.assertEqual(env.bc.enemies()[0]["cur_hp"], 500)
                self.assertEqual(len(env.bc.public_cards()["hand"]), 5)
                name = "Fiend Fire+" if upgraded else "Fiend Fire"
                per = 10 if upgraded else 7
                annotated = _find_action(env, name).describe(
                    env.bc, include_computed_damage=True,
                )
                self.assertTrue(annotated.endswith(f"(deal {per * 4} = {per} x4)"))

                v3_lines = env._turn_math_lines()
                self.assertEqual(
                    v3_lines[1],
                    "Max attack damage playable this turn (using shown deal values, "
                    f"current modifiers only): 18. Excluded from this total: {name}.",
                )
                self.assertEqual(
                    v3_lines[2],
                    "Lethal check vs JAW_WORM (HP 500, block 0): not lethal this turn.",
                )
                v4_lines = combat_public_v4.CombatPublicV4Env(env)._turn_math_lines()
                self.assertTrue(v4_lines[1].startswith(
                    "Fixed-cost attack arithmetic: largest sum of shown Attack damage "
                    "within current energy 3, holding current costs and shown damage fixed: 18. ",
                ))
                self.assertTrue(v4_lines[1].endswith(f"Attacks excluded from this sum: {name}."))


if __name__ == "__main__":
    unittest.main()
