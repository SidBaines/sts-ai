from __future__ import annotations

from pathlib import Path
import unittest

from sts_policy.curriculum import world_split
from sts_policy.ooc_controls import ooc_evaluation_command, ooc_screen_plan, ooc_screen_recipes, ooc_training_command


def values(command):
    return {token: command[index+1] for index, token in enumerate(command[:-1]) if token.startswith("--")}


def generated_worlds(start, count, split):
    result = []
    while len(result) < count:
        if world_split(start) == split: result.append(start)
        start += 1
    return result


class OOCControlTests(unittest.TestCase):
    def test_five_controls_have_frozen_native_and_training_settings(self):
        recipes = ooc_screen_recipes()
        self.assertEqual(len(recipes),5)
        self.assertEqual([recipe["learning_rate"] for recipe in recipes[:3]], [1e-4,3e-4,1e-3])
        self.assertEqual(recipes[3]["reward_mode"],"sparse")
        self.assertEqual(recipes[4]["progress_floor_weight"],.02)
        for recipe in recipes:
            command = ooc_training_command(Path("same.pt"),Path("new.pt"),recipe)
            opts = values(command)
            for flag, expected in {"--checkpoint":"same.pt","--task-kind":"ooc","--split":"train",
                                   "--world-start":"22000","--episodes":"64","--seed":"11","--horizon":"512",
                                   "--simulations":"200","--gamma":"1","--clip-ratio":".1","--seconds":"90.0"}.items():
                self.assertEqual(opts[flag],expected)
            self.assertEqual(float(opts["--learning-rate"]),recipe["learning_rate"])
            self.assertEqual(opts["--reward-mode"],recipe["reward_mode"])
            self.assertEqual(float(opts["--progress-floor-weight"]),recipe["progress_floor_weight"])
            for flag in ("--goal-act-clear","--fast-observations","--fast-replay","--deferred-bootstrap"):
                self.assertIn(flag,command)

    def test_every_policy_uses_the_same_32_dev_worlds_and_native_budget(self):
        worlds = []
        for index in range(5):
            opts = values(ooc_evaluation_command(f"policy{index}.pt",f"metrics{index}.json"))
            worlds.append(generated_worlds(int(opts["--world-start"]),int(opts["--episodes"]),opts["--split"]))
            self.assertEqual(opts["--horizon"],"1024")
            self.assertEqual(opts["--max-act"],"1")
            self.assertEqual(opts["--simulations"],"200")
            self.assertEqual(opts["--seed"],"777")
        self.assertEqual(len(worlds[0]),32)
        self.assertTrue(all(cohort == worlds[0] for cohort in worlds))
        self.assertTrue(all(world_split(seed) == "dev" for seed in worlds[0]))

    def test_plan_is_commands_only_with_consistent_metric_paths(self):
        plan = ooc_screen_plan("same.pt","screen")
        self.assertEqual(len(plan),5)
        self.assertEqual(sum(row["training_seconds"] for row in plan),450)
        for row in plan:
            train, evaluate = values(row["training_command"]), values(row["evaluation_command"])
            self.assertEqual(train["--output"],evaluate["--checkpoint"])
            self.assertIn("cohort_completed",row["selection_metrics"])
