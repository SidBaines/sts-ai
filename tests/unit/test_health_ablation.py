from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sts_policy.curriculum import TaskSpec, world_split
try:
    import torch
except ImportError:
    torch = None

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_health_ablation.py"
if torch is not None:
    from sts_policy.rollout_training import task_identity
    spec = importlib.util.spec_from_file_location("health_ablation_test", SCRIPT)
    health = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(health)


@unittest.skipIf(torch is None, "optional torch training extra unavailable")
class HealthAblationTests(unittest.TestCase):
    def test_training_conditions_only_change_hp_weight_and_output(self):
        first = health.training_command(Path("source.pt"), Path("a.pt"), Path("train.jsonl"), .25)
        second = health.training_command(Path("source.pt"), Path("b.pt"), Path("train.jsonl"), 1.)
        for command in (first, second):
            self.assertIn("--deferred-bootstrap", command)
            self.assertIn("--warm-optimizer", command)
            self.assertEqual(command[command.index("--seconds")+1], "180")
            self.assertEqual(command[command.index("--seed")+1], "11")
            self.assertEqual(command[command.index("--episodes")+1], "128")
        for flag in ("--output", "--metrics", "--terminal-hp-weight"):
            second[second.index(flag)+1] = first[first.index(flag)+1]
        self.assertEqual(first, second)

    def test_balanced_bank_never_contains_dev_or_duplicate_world_within_act(self):
        seeds = [seed for seed in range(100) if world_split(seed) == "train"][:5]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for act in (1,2,3):
                rows = [{"seed": seed, "act": act} for seed in seeds for _ in range(2)]
                rows.append({"seed": next(seed for seed in range(100) if world_split(seed) == "dev"), "act": act})
                (root/f"train_act{act}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            first = health.balanced_train_rows(root, per_act=3)
            self.assertEqual(first, health.balanced_train_rows(root, per_act=3))
            self.assertEqual(len(first),9)
            for act in (1,2,3):
                selected = [row for row in first if row["act"] == act]
                self.assertEqual(len({row["seed"] for row in selected}),3)
                self.assertTrue(all(world_split(row["seed"]) == "train" for row in selected))

    def test_three_eval_cohorts_match_across_models_and_reject_censoring(self):
        # Public task definitions, not machine-local ignored experiment banks.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = [seed for seed in range(200000, 202000) if world_split(seed) == "dev"][:48]
            act1 = root / "act1.jsonl"
            act1.write_text("".join(json.dumps(TaskSpec(seed=seed, prefix=(0,), max_act=1).to_dict()) + "\n"
                                   for seed in seeds))
            panels = root / "panels"; panels.mkdir()
            (panels / "dev_hard.jsonl").write_text("".join(
                json.dumps(TaskSpec(seed=seed, prefix=(0, 1), max_act=3).to_dict()) + "\n"
                for seed in seeds[:32]))
            with patch.object(health, "ACT1", act1), patch.object(health, "PANELS", panels):
                for cohort, count in (("act1_48",48),("expanded_hard32",32),("composite_act1_24",24)):
                    first = health.evaluation_command(Path("a.pt"),Path("ooc.pt"),Path("a.json"),cohort)
                    second = health.evaluation_command(Path("b.pt"),Path("ooc.pt"),Path("b.json"),cohort)
                    one = [task_identity(task) for task in health.command_tasks(first)]
                    two = [task_identity(task) for task in health.command_tasks(second)]
                    self.assertEqual(one,two)
                    self.assertEqual(len(one),count)
                    row = {"task_ids":one,"episodes":[{"task_id":identity} for identity in one],"faults":[],"cohort_completed":True}
                    metrics={"rounds":[row]}
                    self.assertTrue(health.completed_matching(metrics,one))
                    row["episodes"][0]["termination_reason"]="collector_budget"
                    self.assertFalse(health.completed_matching(metrics,one))

    def test_selection_allows_bounded_battle_tradeoff_but_prioritizes_composite(self):
        def condition(a,b,full,hp):
            return {"act1_48":{"wins":a,"episodes":48,"faults":0,"mean_hp_cost":hp,"mean_floor":1},
                    "expanded_hard32":{"wins":b,"episodes":32,"faults":0,"mean_hp_cost":hp,"mean_floor":25},
                    "composite_act1_24":{"wins":full,"episodes":24,"faults":0,"mean_hp_cost":hp,"mean_floor":17}}
        results={"reference":condition(45,29,5,20),"hp0.25":condition(44,28,6,15),"hp1.0":condition(45,29,5,10)}
        selected=health.select_health(results)
        self.assertEqual(selected["selected_condition"],"hp0.25")
        self.assertEqual(selected["tradeoffs"]["hp0.25"]["extra_battle_losses_of_80"],2)
        results["hp0.25"]=condition(40,28,6,15)
        self.assertEqual(health.select_health(results)["selected_condition"],"hp1.0")
