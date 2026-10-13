from __future__ import annotations

from copy import deepcopy
import unittest

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.early_frontiers import PROVENANCE_KEYS, completed_frontiers, coverage, frontier_before_action
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


if __name__ == "__main__":
    unittest.main()
