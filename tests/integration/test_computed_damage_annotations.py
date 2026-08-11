"""Oracle tests for opt-in computed damage on special and untargeted attacks."""
from __future__ import annotations

import re
import unittest
from typing import Any

from sts_ai.lightspeed import LightspeedHybridEnv

from tests.support import requires_simulator


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

    def test_single_enemy_gate_and_unsupported_formulas(self) -> None:
        multi_enemy = _battle_with_deck([("CLEAVE", 0)] * 5, world_seed=3)
        self.assertGreater(
            sum(bool(enemy["alive"]) for enemy in multi_enemy.bc.enemies()),
            1,
        )
        cleave = _find_action(multi_enemy, "Cleave")
        self.assertNotIn(
            "(deal ",
            cleave.describe(multi_enemy.bc, include_computed_damage=True),
        )

        for card_name, card_id in (
            ("Fiend Fire", "FIEND_FIRE"),
            ("Whirlwind", "WHIRLWIND"),
        ):
            with self.subTest(card=card_name):
                env = _battle_with_deck([(card_id, 0)] * 5)
                action = _find_action(env, card_name)
                self.assertNotIn(
                    "(deal ",
                    action.describe(env.bc, include_computed_damage=True),
                )

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


if __name__ == "__main__":
    unittest.main()
