from __future__ import annotations

import unittest

from sts_env._backend.ooc import describe_event_action


class EventDescriptionsTests(unittest.TestCase):
    def describe(self, name, index, ascension=0, **values):
        return describe_event_action({"event": name, **values}, index,
                                     ascension=ascension, max_hp=80, act=1)

    def test_living_wall_options_distinguish_effects(self):
        self.assertIn("remove", self.describe("LIVING_WALL", 0))
        self.assertIn("transform", self.describe("LIVING_WALL", 1))
        self.assertIn("upgrade", self.describe("LIVING_WALL", 2))

    def test_dynamic_hp_and_ascension_effects_are_public(self):
        self.assertIn("lose 40 maximum HP", self.describe("GHOSTS", 0, hp_amount0=40))
        self.assertIn("gain 5 Apparitions", self.describe("GHOSTS", 0, hp_amount0=40))
        self.assertIn("gain 3 Apparitions", self.describe("GHOSTS", 0, 15, hp_amount0=40))

    def test_falling_identifies_public_offered_card(self):
        self.assertIn("lose Inflame", self.describe("FALLING", 1, falling_power_card="Inflame"))

    def test_dead_adventurer_uses_visible_progress_not_hidden_rolls(self):
        self.assertIn("60%", self.describe("DEAD_ADVENTURER", 0, 15, phase=1))

    def test_implemented_mushroom_effect_not_assumed_vanilla(self):
        self.assertIn("gain 99 gold", self.describe("HYPNOTIZING_COLORED_MUSHROOMS", 1))
        self.assertNotIn("heal", self.describe("HYPNOTIZING_COLORED_MUSHROOMS", 1))

    def test_unknown_event_or_option_is_explicitly_incomplete(self):
        self.assertIsNone(self.describe("UNIMPLEMENTED", 0))
        self.assertIsNone(self.describe("LIVING_WALL", 99))

    def test_scrap_ooze_matches_upstream_inclusive_comparison(self):
        self.assertIn("26%", self.describe("SCRAP_OOZE", 0, event_data=0))

    def test_designer_transformation_and_full_service_have_distinct_costs(self):
        transform = self.describe("DESIGNER_IN_SPIRE", 3)
        self.assertIn("pay 60 gold", transform)
        self.assertIn("transform two", transform)
        self.assertNotIn("remove", transform)
        self.assertNotIn("lose", transform)
        full_service = self.describe("DESIGNER_IN_SPIRE", 4, 15)
        self.assertIn("pay 110 gold", full_service)
        self.assertIn("remove a card", full_service)
        self.assertIn("upgrade", full_service)
        self.assertNotIn("lose", full_service)

    def test_designer_leave_penalty_is_explicit_and_ascension_dependent(self):
        self.assertIn("lose 3 HP", self.describe("DESIGNER_IN_SPIRE", 5))
        self.assertIn("lose 5 HP", self.describe("DESIGNER_IN_SPIRE", 5, 15))
