from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.early_frontiers import PROVENANCE_KEYS, EarlyFrontierWorker, completed_frontiers, coverage, frontier_before_action, menu_stage
from tests.unit.test_public_entities import observation


class EarlyFrontierTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskSpec(seed=next(seed for seed in range(100) if world_split(seed)=="train"),kind="run",max_act=3)
        self.observation = observation(False)
        self.observation["state"].update(act=2,screen_state="ScreenState.REWARDS")
        self.info = {"world_seed":self.task.seed,"task_steps":3,"action_prefix":[2,0,1],"ooc_previews":[False]*3,
                     **{key:{"test":True} if key=="native_provenance" else "test" for key in PROVENANCE_KEYS}}

    def test_exact_prior_history_excludes_current_choice_and_is_immutable(self):
        row = frontier_before_action(self.task,self.observation,self.info,4)
        self.assertEqual(row["prefix"],(2,0,1))
        self.assertEqual(row["source_executed_index"],4)
        self.assertEqual(row["frontier_step"],3)
        self.observation["state"]["gold"]=900
        self.info["native_provenance"]["test"]=False
        self.assertEqual(row["source_observation"]["state"]["gold"],100)
        self.assertTrue(row["expected_provenance"]["native_provenance"]["test"])

    def test_forced_combat_and_act1_decisions_do_not_create_frontiers(self):
        for phase,act,choices in (("combat",2,3),("out_of_combat",1,3),("out_of_combat",2,1)):
            obs=deepcopy(self.observation);obs["phase"]=phase;obs["state"]["act"]=act
            obs["legal_actions"]=obs["legal_actions"][:choices]
            self.assertIsNone(frontier_before_action(self.task,obs,self.info,7))

    def test_broken_history_or_nontrain_source_fails_closed(self):
        for field,value in (("task_steps",4),("world_seed",self.task.seed+1),("ooc_previews",[])):
            info={**self.info,field:value}
            with self.assertRaises(ValueError):
                frontier_before_action(self.task,self.observation,info,4)
        other=TaskSpec(seed=next(seed for seed in range(100) if world_split(seed)!="train"),kind="run",max_act=3)
        with self.assertRaises(ValueError):
            frontier_before_action(other,self.observation,self.info,4)

    def test_later_fault_removes_every_frontier_from_source_world(self):
        records=[{"seed":1,"start_act":2,"start_screen":"MAP"},{"seed":1,"start_act":3,"start_screen":"REST"},
                 {"seed":2,"start_act":2,"start_screen":"MAP"},{"seed":3,"start_act":2,"start_screen":"MAP"}]
        collection={"episodes":[{"world_seed":1,"terminated":True,"truncated":False},
                    {"world_seed":2,"terminated":True,"truncated":False},
                    {"world_seed":3,"terminated":False,"truncated":True}],"faults":[{"world_seed":1}]}
        usable,quarantine=completed_frontiers(records,collection)
        self.assertEqual([row["seed"] for row in usable],[2])
        self.assertEqual(len(quarantine),3)
        summary=coverage(records)
        self.assertEqual(summary["independent_source_worlds"],3)
        self.assertEqual(summary["frontiers"],4)
        self.assertEqual(summary["source_worlds_by_act"]["3"],1)

    def test_reward_stages_record_actual_card_choice_after_gold(self):
        mixed=deepcopy(self.observation)
        mixed["legal_actions"]=[{"index":0,"public":{"kind":"take_gold"}},
                                {"index":1,"public":{"kind":"take_card"}},
                                {"index":2,"public":{"kind":"skip_card"}}]
        cards=deepcopy(mixed);cards["legal_actions"]=cards["legal_actions"][1:]
        self.assertNotEqual(menu_stage(mixed),menu_stage(cards))
        with EarlyFrontierWorker() as worker, patch("sts_policy.workers.CurriculumWorker.step",return_value=None):
            worker.source_task=self.task;worker.info=self.info;worker.observation=mixed
            worker.step(0);worker.step(0)
            worker.observation=cards;worker.step(1)
            self.assertEqual(len(worker.frontiers),2)
            self.assertEqual(worker.frontiers[-1]["decision_family"],"card_only_rewards")

    def test_late_act1_boss_relic_is_eligible_and_select_purposes_are_distinct(self):
        obs=deepcopy(self.observation);obs["state"].update(act=1,floor=17,screen_state="ScreenState.BOSS_RELIC_REWARDS")
        self.assertIsNotNone(frontier_before_action(self.task,obs,self.info,4))
        a=deepcopy(self.observation);b=deepcopy(a)
        a["legal_actions"][0]["public"]["selection_type"]=3
        b["legal_actions"][0]["public"]["selection_type"]=7
        self.assertNotEqual(menu_stage(a),menu_stage(b))


if __name__ == "__main__":
    unittest.main()
