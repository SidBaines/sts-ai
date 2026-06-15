"""Unit tests for risk proxy classification and aggregation.

Pure Python: operates on recorded decision dicts, no simulator required.
"""
from __future__ import annotations

import unittest

from sts_ai.risk_proxies import (
    classify_decision,
    hp_bucket,
    risk_events,
    summarize_risk,
)


def _record(
    screen: str,
    desc: str,
    cur_hp: int = 80,
    max_hp: int = 80,
    floor: int = 5,
    world_seed: int = 1,
    idx: int = 0,
) -> dict:
    return {
        "world_seed": world_seed,
        "decision_index": idx,
        "state": {
            "screen_state": f"ScreenState.{screen}",
            "cur_hp": cur_hp,
            "max_hp": max_hp,
            "floor": floor,
        },
        "selected_action": {"description": desc},
    }


class HpBucketTest(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(hp_bucket(0.1), "low")
        self.assertEqual(hp_bucket(0.5), "medium")
        self.assertEqual(hp_bucket(0.9), "high")
        self.assertEqual(hp_bucket(1.0 / 3.0), "medium")  # boundary -> not low


class ClassifyCampfireTest(unittest.TestCase):
    def test_smith_at_low_hp_is_risk_seeking(self):
        ev = classify_decision(_record("REST_ROOM", "smith", cur_hp=10, max_hp=80, world_seed=12))
        self.assertEqual(ev.world_seed, 12)
        self.assertEqual(ev.category, "campfire")
        self.assertEqual(ev.choice, "smith")
        self.assertEqual(ev.hp_bucket, "low")
        self.assertTrue(ev.risk_seeking)

    def test_smith_at_high_hp_is_ambiguous(self):
        ev = classify_decision(_record("REST_ROOM", "smith", cur_hp=78, max_hp=80))
        self.assertIsNone(ev.risk_seeking)

    def test_rest_is_not_risk_seeking(self):
        ev = classify_decision(_record("REST_ROOM", "rest", cur_hp=10, max_hp=80))
        self.assertFalse(ev.risk_seeking)
        self.assertEqual(ev.choice, "rest")

    def test_skip_campfire(self):
        ev = classify_decision(_record("REST_ROOM", "skip campfire"))
        self.assertEqual(ev.choice, "skip")

    def test_enriched_campfire_labels_still_parse(self):
        # The serializer spells out campfire effects; the leading keyword must
        # still drive classification.
        self.assertEqual(
            classify_decision(_record("REST_ROOM", "smith (upgrade a card in your deck)", cur_hp=10, max_hp=80)).choice,
            "smith",
        )
        self.assertEqual(
            classify_decision(_record("REST_ROOM", "rest (heal 30% of max HP)")).choice,
            "rest",
        )
        self.assertEqual(
            classify_decision(_record("REST_ROOM", "skip campfire (proceed without resting)")).choice,
            "skip",
        )


class ClassifyMapTest(unittest.TestCase):
    def test_elite_node_is_risk_seeking(self):
        ev = classify_decision(_record("MAP_SCREEN", "choose map node x=2 room=ELITE"))
        self.assertEqual(ev.category, "map_node")
        self.assertEqual(ev.choice, "ELITE")
        self.assertTrue(ev.risk_seeking)

    def test_monster_node_ambiguous(self):
        ev = classify_decision(_record("MAP_SCREEN", "choose map node x=1 room=MONSTER"))
        self.assertEqual(ev.choice, "MONSTER")
        self.assertIsNone(ev.risk_seeking)

    def test_boss_advance(self):
        ev = classify_decision(_record("MAP_SCREEN", "choose map node x=0 advance to boss"))
        self.assertEqual(ev.choice, "BOSS")


class ClassifyNeowTest(unittest.TestCase):
    def test_drawback_option_is_risk_seeking(self):
        ev = classify_decision(
            _record("EVENT_SCREEN", "event option 2: Remove two cards. / Take 30% Hp damage.", floor=0)
        )
        self.assertEqual(ev.category, "neow")
        self.assertTrue(ev.risk_seeking)
        self.assertEqual(ev.choice, "drawback")

    def test_no_drawback_option(self):
        ev = classify_decision(_record("EVENT_SCREEN", "event option 0: Upgrade a card.", floor=0))
        self.assertFalse(ev.risk_seeking)

    def test_non_neow_event_ignored(self):
        # floor > 0 events are not classified as neow (and currently not risk-tagged)
        self.assertIsNone(classify_decision(_record("EVENT_SCREEN", "event option 0: whatever", floor=7)))


class ClassifyRewardsTest(unittest.TestCase):
    def test_take_self_damage_card_is_risk_seeking(self):
        ev = classify_decision(_record("REWARDS", "take card Offering"))
        self.assertEqual(ev.category, "card_reward")
        self.assertEqual(ev.choice, "take")
        self.assertTrue(ev.risk_seeking)

    def test_take_normal_card_ambiguous(self):
        ev = classify_decision(_record("REWARDS", "take card Iron Wave"))
        self.assertEqual(ev.choice, "take")
        self.assertIsNone(ev.risk_seeking)

    def test_take_card_with_type_rarity_tag_still_flags_self_damage(self):
        # The serializer now appends a " [Type, Rarity]" tag to card choices; the
        # card-name match against SELF_DAMAGE_CARDS must see through it.
        ev = classify_decision(_record("REWARDS", "take card Offering [Skill, Uncommon]"))
        self.assertEqual(ev.choice, "take")
        self.assertTrue(ev.risk_seeking)
        ev2 = classify_decision(_record("REWARDS", "take card Iron Wave [Attack, Common]"))
        self.assertEqual(ev2.choice, "take")
        self.assertIsNone(ev2.risk_seeking)

    def test_skip_rewards(self):
        ev = classify_decision(_record("REWARDS", "skip rewards / proceed"))
        self.assertEqual(ev.choice, "skip")

    def test_potion_take(self):
        ev = classify_decision(_record("REWARDS", "take potion Fire Potion"))
        self.assertEqual(ev.category, "potion")

    def test_take_gold_is_not_risk_relevant(self):
        self.assertIsNone(classify_decision(_record("REWARDS", "take gold 25g")))


class LegacyBitsPrefixTest(unittest.TestCase):
    def test_legacy_bits_prefix_is_stripped(self):
        # traces recorded before 2026-06-14 carry a "bits=N " prefix
        ev = classify_decision(_record("REWARDS", "bits=256 take card Offering"))
        self.assertEqual(ev.category, "card_reward")
        self.assertEqual(ev.choice, "take")
        self.assertTrue(ev.risk_seeking)

    def test_legacy_prefix_on_campfire(self):
        ev = classify_decision(_record("REST_ROOM", "bits=0 rest", cur_hp=10, max_hp=80))
        self.assertEqual(ev.choice, "rest")

    def test_legacy_prefix_on_leave_shop(self):
        ev = classify_decision(_record("SHOP_ROOM", "bits=5 leave shop"))
        self.assertEqual(ev.choice, "leave")


class ClassifyShopTest(unittest.TestCase):
    def test_buy_potion_records_spend(self):
        ev = classify_decision(_record("SHOP_ROOM", "buy potion Fire Potion for 50g"))
        self.assertEqual(ev.category, "shop")
        self.assertEqual(ev.choice, "buy_potion")
        self.assertEqual(ev.value, 50)

    def test_leave_shop(self):
        ev = classify_decision(_record("SHOP_ROOM", "leave shop"))
        self.assertEqual(ev.choice, "leave")
        self.assertIsNone(ev.value)

    def test_buy_card_remove(self):
        ev = classify_decision(_record("SHOP_ROOM", "buy card remove for 75g"))
        self.assertEqual(ev.choice, "buy_remove")
        self.assertEqual(ev.value, 75)


class SummarizeTest(unittest.TestCase):
    def test_aggregate_metrics(self):
        records = [
            _record("REST_ROOM", "rest", cur_hp=10, max_hp=80),
            _record("REST_ROOM", "smith", cur_hp=10, max_hp=80),
            _record("MAP_SCREEN", "choose map node x=2 room=ELITE"),
            _record("MAP_SCREEN", "choose map node x=1 room=MONSTER"),
            _record("EVENT_SCREEN", "event option 2: x. / Take damage.", floor=0),
            _record("REWARDS", "take card Offering"),
            _record("REWARDS", "skip rewards / proceed"),
            _record("SHOP_ROOM", "buy potion P for 50g"),
            _record("SHOP_ROOM", "leave shop"),
            _record("REWARDS", "take gold 25g"),  # ignored
        ]
        events = risk_events(records)
        self.assertEqual(len(events), 9)  # gold ignored
        summary = summarize_risk(events)
        self.assertEqual(summary["campfire_rest_rate_by_hp"]["low"]["n"], 2)
        self.assertEqual(summary["campfire_rest_rate_by_hp"]["low"]["rest_rate"], 0.5)
        self.assertEqual(summary["map_elite_rate"]["elite_rate"], 0.5)
        self.assertEqual(summary["neow_drawback_rate"]["drawback_rate"], 1.0)
        self.assertEqual(summary["card_take_rate"]["self_damage_takes"], 1)
        self.assertEqual(summary["shop"]["total_spend"], 50)
        self.assertEqual(summary["potion_acquire_count"], 0)


if __name__ == "__main__":
    unittest.main()
