from __future__ import annotations

import unittest

from sts_policy.curriculum import TaskSpec, public_hash
from sts_policy.teacher_panel import PanelWorker
from tests.support import requires_simulator


@requires_simulator
class TeacherPanelTests(unittest.TestCase):
    def test_panel_labels_are_legal_and_leave_live_state_unchanged(self):
        with PanelWorker({"fast_observations":True},timeout=10.) as worker:
            obs,_=worker.reset(TaskSpec(seed=3,kind="battle",max_act=1,battle_simulations=10))
            before=public_hash(obs)
            panel=worker.teacher_panel(16)
            legal={a["index"] for a in obs["legal_actions"]}
            self.assertEqual(len(panel["results"]),6)
            self.assertTrue(all(r["index"] in legal for r in panel["results"] if r["status"]=="ok"))
            self.assertEqual(public_hash(worker.observation),before)
            action=panel["primary_index"]
            candidate=worker.step(action)[0]
        with PanelWorker({"fast_observations":True},timeout=10.) as worker:
            obs,_=worker.reset(TaskSpec(seed=3,kind="battle",max_act=1,battle_simulations=10))
            self.assertEqual(public_hash(obs),before)
            reference=worker.step(action)[0]
        self.assertEqual(public_hash(candidate),public_hash(reference))


if __name__=="__main__":
    unittest.main()
