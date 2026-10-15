from __future__ import annotations

import unittest

from sts_env._backend.ooc import describe_event_action


class EventDescriptionsTests(unittest.TestCase):
    def describe(self, name, index, ascension=0, *, max_hp=80, **values):
        return describe_event_action({"event": name, **values}, index,
                                     ascension=ascension, max_hp=max_hp, act=1)

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

    def test_mushrooms_eat_describes_healing_and_parasite(self):
        for ascension in (0, 15):
            description = self.describe("HYPNOTIZING_COLORED_MUSHROOMS", 1, ascension)
            self.assertIn("heal 20 HP", description)
            self.assertIn("Parasite curse", description)
            self.assertNotIn("gold", description)
        description = self.describe("HYPNOTIZING_COLORED_MUSHROOMS", 1, max_hp=83)
        self.assertIn("heal 20 HP", description)

    def test_sphere_leave_has_no_gold_reward(self):
        for ascension in (0, 15):
            self.assertEqual(self.describe("MYSTERIOUS_SPHERE", 1, ascension),
                             "Mysterious Sphere: leave")

    def test_mausoleum_describes_relic_and_public_curse_probability(self):
        for ascension in (0, 15):
            description = self.describe("THE_MAUSOLEUM", 0, ascension)
            self.assertIn("gain a random relic", description)
            if ascension >= 15:
                self.assertIn("gain a random relic and the Writhe curse", description)
                self.assertNotIn("chance", description)
            else:
                self.assertIn("50% chance of the Writhe curse", description)
            self.assertNotIn("upstream", description)

    def test_woman_in_blue_describes_potion_costs(self):
        for ascension in (0, 15):
            for index, (cost, count) in enumerate(((20, "one"), (30, "two"), (40, "three"))):
                description = self.describe("THE_WOMAN_IN_BLUE", index, ascension)
                self.assertIn(f"pay {cost} gold", description)
                self.assertIn(f"{count} random potion", description)
                self.assertNotIn("upstream", description)

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

    def test_designer_adjustments_variants_describe_the_same_cost(self):
        for ascension, cost in ((0, 40), (15, 50)):
            with self.subTest(ascension=ascension):
                self.assertEqual(self.describe("DESIGNER_IN_SPIRE", 0, ascension),
                                 f"Designer In Spire: pay {cost} gold to upgrade a card")
                self.assertEqual(self.describe("DESIGNER_IN_SPIRE", 1, ascension),
                                 f"Designer In Spire: pay {cost} gold to upgrade two random cards")

    def test_designer_leave_penalty_is_explicit_and_ascension_dependent(self):
        self.assertIn("lose 3 HP", self.describe("DESIGNER_IN_SPIRE", 5))
        self.assertIn("lose 5 HP", self.describe("DESIGNER_IN_SPIRE", 5, 15))
