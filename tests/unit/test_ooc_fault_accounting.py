from __future__ import annotations
from concurrent.futures import Future
from dataclasses import dataclass
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from sts_policy.rollout_training import collect_rollouts


@dataclass
class Task:
    seed: int
    kind: str = "ooc"


def obs():
    return {"phase":"out_of_combat","state":{"cur_hp":80,"max_hp":80},
            "legal_actions":[{"index":0,"description":"Leave"},{"index":1,"description":"Rest"}]}


@unittest.skipIf(torch is None,"optional torch unavailable")
class FaultAccountingTests(unittest.TestCase):
    def test_failed_native_wait_is_counted_but_episode_and_labels_quarantined(self):
        clock={'seconds':0.}
        class DelayedFailure(Future):
            def result(self,timeout=None):
                clock['seconds']+=2.
                raise TimeoutError('native step timed out')
        class Worker:
            def submit(self,method,*args):
                if method=='reset':self.steps=0
                if method=='step':
                    self.steps+=1
                    if self.steps==2:return DelayedFailure()
                future=Future()
                value=(obs(),{}) if method=='reset' else 0 if method=='teacher_action' else (obs(),1.,False,False,{})
                future.set_result(value)
                return future
        class Model(torch.nn.Module):
            def forward(self,count):return torch.zeros(count,2),torch.zeros(count)
        def forward(model,batch):
            logits,values=model(batch)
            return logits,values,torch.ones_like(logits,dtype=torch.bool)
        with patch('sts_policy.rollout_training.time.monotonic',side_effect=lambda:clock['seconds']):
            result=collect_rollouts(Model(),[Worker()],[Task(1)],collate=lambda batch,device:len(batch),
                forward=forward,teacher_simulations=1,deferred_bootstrap=True,max_seconds=10.)
        self.assertEqual(result['timings']['fault_wait_seconds'],2.)
        self.assertEqual(result['timings']['step_wait_seconds'],0.)
        self.assertEqual(result['faults'][0]['fault_stage'],'step')
        self.assertEqual(result['faults'][0]['wait_seconds'],2.)
        self.assertEqual(result['excluded_transitions'],1)
        self.assertEqual(result['transitions'],[])
        self.assertEqual(result['teacher_records'],[])
        self.assertEqual(len(result['faults'][0]['action_trace_sha256']),64)
        self.assertIn(result['faults'][0]['attempted_action_index'],(0,1))
