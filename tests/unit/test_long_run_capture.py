from __future__ import annotations

from concurrent.futures import Future
import unittest
import torch

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.long_run_tasks import NATIVE_SHA, INTERFACE_SHA
from sts_policy.rollout_training import collect_rollouts


def observation(act, floor, legal=True):
    return {'state': {'act': act, 'floor': floor}, 'phase': 'out_of_combat',
            'legal_actions': [{'index': 0, 'description': 'map'}] if legal else []}


def info(prefix, outcome='UNDECIDED'):
    return {'environment_version':'0.1.1','observation_version':'sts_env_public_v1',
            'python_interface_sha256':INTERFACE_SHA,'runtime_python_version':'3.12.13',
            'native_provenance':{'module_sha256':NATIVE_SHA},
            'action_prefix': [0] * prefix, 'ooc_previews':[False] * prefix,
            'outcome':f'GameOutcome.{outcome}', 'task_outcome':'defeat' if outcome=='PLAYER_LOSS' else '',
            'termination_reason':'defeat' if outcome=='PLAYER_LOSS' else '', 'split':'train'}


class Worker:
    def __init__(self, fault=False, truncate=False):
        self.steps=0;self.fault=fault;self.truncate=truncate
    def submit(self, name, argument):
        future=Future()
        if name=='reset':future.set_result((observation(1,16),info(0)))
        else:
            self.steps+=1
            if self.steps==1:future.set_result((observation(2,17),0.,False,False,info(1)))
            elif self.fault:future.set_exception(RuntimeError('native fault after captured frontier'))
            elif self.truncate:future.set_result((observation(2,20),0.,False,True,info(2)))
            else:future.set_result((observation(2,20,False),-1.,True,False,info(2,'PLAYER_LOSS')))
        return future


def forward(model, observations):
    size=len(observations)
    return torch.zeros(size,1),torch.zeros(size),torch.ones(size,1,dtype=torch.bool)


class LongRunCaptureTests(unittest.TestCase):
    def collect(self, worker, capture=True):
        seed=next(seed for seed in range(1000000,1000100) if world_split(seed)=='train')
        return collect_rollouts(torch.nn.Linear(1,1),[worker],[TaskSpec(seed=seed,kind='run',max_act=3)],
                                collate=lambda obs,device:obs,forward=forward,
                                deterministic=True,capture_continuations=capture,max_seconds=5.)
    def test_complete_fault_free_episode_keeps_public_frontier(self):
        result=self.collect(Worker())
        self.assertEqual(len(result['continuations']),1)
        row=result['continuations'][0]
        self.assertEqual(row['long_metadata']['start_act'],2)
        self.assertEqual(row['prefix'],(0,))
        self.assertEqual(len(result['episodes']),1)
    def test_later_fault_quarantines_earlier_frontier_and_transitions(self):
        result=self.collect(Worker(fault=True))
        self.assertEqual(result['continuations'],[])
        self.assertEqual(result['transitions'],[])
        self.assertEqual(len(result['faults']),1)
    def test_horizon_cutoff_does_not_publish_partial_source_episode(self):
        result=self.collect(Worker(truncate=True))
        self.assertEqual(result['continuations'],[])
        self.assertEqual(len(result['episodes']),1)
    def test_capture_disabled_keeps_existing_result_contract(self):
        result=self.collect(Worker(),capture=False)
        self.assertNotIn('continuations',result)
