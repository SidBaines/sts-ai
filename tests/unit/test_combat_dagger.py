from __future__ import annotations

from copy import deepcopy
import unittest

from sts_env.env import SimulatorError
from sts_policy.combat_dagger import LabelingWorker
from sts_policy.curriculum import TaskSpec, world_split
from tests.unit.test_public_entities import observation


class FakeWorker:
    def __init__(self, fail=False):
        self.fail = fail; self.actions = []; self.queries = []
    def reset(self, task):
        return observation(False), {}
    def teacher_action(self, budget):
        self.queries.append(budget)
        if self.fail and budget == 1000:
            raise SimulatorError("injected search failure")
        return 7 if budget == 200 else 4
    def step(self, action):
        self.actions.append(action)
        terminal = len(self.actions) == 3
        return deepcopy(observation()),0.,terminal,False,{"task_outcome":"victory" if terminal else ""}


class CombatDaggerTests(unittest.TestCase):
    def task(self):
        seed = next(s for s in range(10,100) if world_split(s)=="train")
        return TaskSpec(seed=seed,kind="run",max_act=3)

    def test_queries_do_not_replace_policy_actions_or_label_ooc(self):
        raw, journals = FakeWorker(), []
        with LabelingWorker(raw,(200,1000),sink=journals.append) as worker:
            worker.submit("reset",self.task()).result()
            for action in (7,20,7):
                worker.submit("step",action).result()
        self.assertEqual(raw.actions,[7,20,7])
        self.assertEqual(raw.queries,[200,1000,200,1000])
        self.assertEqual(journals[0]["status"],"completed")
        self.assertEqual([r["executed_index"] for r in journals[0]["records"]],[20,7])
        self.assertTrue(all(r["teacher_index"]==4 for r in journals[0]["records"]))

    def test_search_fault_discards_whole_world(self):
        raw, journals = FakeWorker(), []
        with LabelingWorker(raw,(200,1000),sink=journals.append) as worker:
            worker.submit("reset",self.task()).result()
            worker.submit("step",7).result()
            worker.submit("step",20).result()
            raw.fail = True
            with self.assertRaises(SimulatorError):
                worker.submit("step",7).result()
        self.assertEqual(journals[0]["status"],"simulator_fault")
        self.assertEqual(journals[0]["discarded_records"],1)
        self.assertEqual(journals[0]["records"],[])
        self.assertEqual(raw.actions,[7,20])

    def test_incomplete_world_labels_never_publish_as_training_rows(self):
        journals = []
        with LabelingWorker(FakeWorker(),(200,1000),sink=journals.append) as worker:
            worker.submit("reset",self.task()).result()
            worker.submit("step",7).result()
            worker.submit("step",20).result()
        self.assertEqual(journals[0]["status"],"incomplete")
        self.assertEqual(journals[0]["records"],[])
        self.assertEqual(journals[0]["discarded_records"],1)


if __name__ == "__main__":
    unittest.main()
