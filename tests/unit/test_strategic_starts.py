from __future__ import annotations

from copy import deepcopy
import unittest

from sts_policy.curriculum import world_split
from sts_policy.strategic_starts import strategic_frontier
from tests.unit.test_public_entities import observation


class StrategicStartTests(unittest.TestCase):
    def world(self):
        seed = next(i for i in range(100) if world_split(i) == "train")
        records = []
        for step, (phase, screen) in enumerate((("combat", "BATTLE"), ("out_of_combat", "REST_ROOM"),
                                                ("out_of_combat", "CARD_SELECT"), ("combat", "BATTLE"))):
            obs = observation(phase == "combat")
            obs["state"].update(act=3, floor=49 if step < 3 else 50, screen_state="ScreenState."+screen)
            records.append({"step": step, "observation": obs, "executed_index": step,
                            "teacher_index": 99, "simulations": 200 if phase == "combat" else None})
        return {"seed": seed, "records": records, "provenance": {"native": "test"},
                "settings": {"simulations": 200}}

    def test_last_rest_choice_preserves_exact_executed_prefix_and_preview_markers(self):
        world = self.world()
        result = strategic_frontier(world, 3)
        self.assertEqual(result["frontier_step"], 1)
        self.assertEqual(result["prefix"], (0,))
        self.assertEqual(result["ooc_previews"], (False,))
        world["records"][1]["observation"]["legal_actions"] = world["records"][1]["observation"]["legal_actions"][:1]
        result = strategic_frontier(world, 3)
        self.assertEqual(result["frontier_step"], 2)
        self.assertEqual(result["prefix"], (0, 1))
        self.assertEqual(result["ooc_previews"], (False, True))

    def test_no_choice_and_noncontiguous_records_fail_safely(self):
        world = self.world()
        world["records"][1]["observation"]["legal_actions"] = []
        world["records"][2]["observation"]["legal_actions"] = []
        self.assertIsNone(strategic_frontier(world, 3))
        world["records"][2]["step"] = 8
        with self.assertRaisesRegex(ValueError, "contiguous"):
            strategic_frontier(world, 3)
        world = self.world()
        world["seed"] = next(i for i in range(100) if world_split(i) != "train")
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            strategic_frontier(world, 3)


if __name__ == "__main__":
    unittest.main()
