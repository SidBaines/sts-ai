import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import torch

from sts_policy.entity_model import EntityActorCritic
from sts_policy.entity_encoding import collate_observations, encode_observation
from sts_rl.fresh_study import StudyPolicy, policy_observation
from sts_policy.training import load_checkpoint, save_checkpoint
from sts_rl.fresh_training import TrainingControl
from tests.unit.test_fresh_training import fixture, synthetic_source
from tests.unit.test_entity_encoding import combat_observation, ooc_observation

spec=importlib.util.spec_from_file_location('fresh_benchmark',Path(__file__).resolve().parents[2]/'scripts/rl/fresh_benchmark.py')
benchmark=importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class FreshBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_bounded_cpu_microbenchmark_and_fixed_workload(self):
        observation=combat_observation()
        results=benchmark.microbenchmark([observation]*4,width=32,layers=1,heads=4,steps=10)
        self.assertTrue(results['cpu']['finite_gradients'])
        self.assertTrue(results['cpu']['loss_decreased'])
        self.assertTrue(results['cpu']['tiny_overfit_target_met'])
        with tempfile.TemporaryDirectory() as tmp:
            source,_=synthetic_source(Path(tmp))
            policy=StudyPolicy(source['policy']['champion']['path'])
            measured=benchmark.fixed_workload(policy,[observation]*4,seconds=.1,max_passes=2)
            self.assertEqual(measured['decisions'],8)
            self.assertGreater(measured['inference_seconds'],0.)
            self.assertGreater(measured['encode_collate_seconds'],0.)
        print('D1 bounded synthetic benchmark '+json.dumps({'microbenchmark':results,'fixed_workload':measured},allow_nan=False))

    def test_benchmark_cli_records_requested_old_architecture_on_same_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data,source=fixture(root)
            with patch.object(benchmark,'TrainingControl',side_effect=lambda out,d:
                TrainingControl(out,d,ac=lambda:True,caffeinate=False)),redirect_stdout(io.StringIO()):
                benchmark.main(['--data',str(data),'--checkpoint',source['policy']['champion']['path'],
                    '--out',str(root/'benchmark'),'--deadline-utc','2099-01-01T00:00:00Z',
                    '--old-size','medium','--old-context','mean','--width','16','--layers','1',
                    '--observations','2','--seconds','1'])
            report=json.loads((root/'benchmark'/'benchmark.json').read_text())
            self.assertEqual(set(report['fixed_workload']),{'champion','random_old','random_entity'})
            self.assertEqual({m['decisions'] for m in report['fixed_workload'].values()},{20})
            for identity in report['old_models'].values():
                payload=load_checkpoint(identity['path'])
                self.assertEqual((payload['model_config']['size'],payload['model_config']['context']),('medium','mean'))
                self.assertEqual((payload['metadata']['old_size'],payload['metadata']['old_context']),('medium','mean'))
                self.assertIsNone(payload['metadata']['parent_checkpoint'])
            self.assertEqual(report['old_control']['old_size'],'medium')

    def test_vectorized_cpu_fixture_measurement_no_speed_claim(self):
        torch.manual_seed(1919)
        model=EntityActorCritic().eval()
        batch=collate_observations([encode_observation(combat_observation()) for _ in range(16)])
        with torch.inference_mode():
            model.training_logits(batch)
            model.actor_logits(batch)
            start=time.perf_counter()
            for _ in range(10):
                model.training_logits(batch)
            batched=(time.perf_counter()-start)/10
            start=time.perf_counter()
            for _ in range(10):
                model.actor_logits(batch)
            singleton=(time.perf_counter()-start)/10
            logits=model.training_logits(batch)
        self.assertTrue(torch.isfinite(logits).all())
        print('D1 synthetic CPU width128 layers2 heads4 batch16 '+json.dumps({
            'vectorized_batch_ms':batched*1000,'singleton_loop_batch_ms':singleton*1000,
            'vectorized_rows_per_s':16/batched,'singleton_rows_per_s':16/singleton,
            'mps_available':torch.backends.mps.is_available()},allow_nan=False))

    def test_real_format_slot_gap_single_preprocessing_and_filtered_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,_=synthetic_source(root)
            model=EntityActorCritic(width=16,layers=1,heads=4,phase='ooc').eval()
            path=root/'ooc.pt'
            save_checkpoint(path,model,model_config=model.metadata(),metadata=model.metadata())
            policy=StudyPolicy(source['policy']['champion']['path'],ooc=str(path))
            raw=ooc_observation()
            raw['text']=raw['text'].split('Potions:')[0]+'Potions: EMPTY_POTION_SLOT\n'
            raw['legal_actions']=[{'index':3,'description':'leave'},
                {'index':17,'description':'select card index 1: Strike+ [Attack, Basic]'},
                {'index':64,'description':'skip'},
                {'index':81,'description':'discard potion Fire Potion (slot 1)'},
                {'index':99,'description':'drink potion Fire Potion (slot 1): deal 20 damage'}]
            original=deepcopy(raw)
            prepared=policy_observation(raw)
            expected=collate_observations([encode_observation(prepared)])
            self.assertEqual(prepared['adapter_public_ooc_potions'],[{'slot':1,'name':'Fire Potion'}])
            forward=policy.replacements['ooc'].actor_logits
            def checked(batch):
                for key in expected:
                    self.assertTrue(torch.equal(expected[key],batch[key]),key)
                return forward(batch)
            with patch.object(policy.replacements['ooc'],'actor_logits',side_effect=checked) as actor:
                action=policy.choose_action(raw)
                timed=benchmark.TimedCollectorPolicy(policy)
                self.assertEqual(timed.choose_action(raw),action)
                self.assertEqual(actor.call_count,2)
                self.assertIn(action,[3,17,64])
                self.assertEqual(timed.choice,1)
            self.assertEqual(raw,original)

    def test_force_champion_keeps_action_trajectories_with_replacement_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,_=synthetic_source(root)
            model=EntityActorCritic(width=16,layers=1,heads=4,phase='ooc').eval()
            path=root/'ooc.pt'
            save_checkpoint(path,model,model_config=model.metadata(),metadata=model.metadata())
            policy=StudyPolicy(source['policy']['champion']['path'],ooc=str(path))
            champion=StudyPolicy(source['policy']['champion']['path'])
            replacement=policy.replacements
            timed=benchmark.TimedCollectorPolicy(policy,force_champion=True)
            # A bounded branching public trajectory: the next menu depends on past actions.
            def trajectory(actor):
                actions=[]
                for _ in range(4):
                    raw=ooc_observation()
                    raw['text']=raw['text'].replace('Potions: Blood Potion, EMPTY_POTION_SLOT','Potions: EMPTY_POTION_SLOT')
                    raw['state']['gold']=200+sum(actions)
                    raw['legal_actions']=[{'index':3,'description':'leave'},
                        {'index':17+sum(actions),'description':'skip'},
                        {'index':81+sum(actions),'description':'discard potion Fire Potion (slot 1)'}]
                    actions.append(actor.choose_action(raw))
                return actions
            with patch.object(policy.replacements['ooc'],'forward',side_effect=AssertionError('full model called')):
                self.assertEqual(trajectory(timed),trajectory(champion))
            self.assertIs(policy.replacements,replacement)
            self.assertEqual(timed.decisions,4)

if __name__=='__main__':
    unittest.main()
