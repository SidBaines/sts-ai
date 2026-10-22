"""Synthetic contracts for architecture fairness and blind deferred TEST scoring."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from sts_policy import encoding, entity_encoding
from sts_policy.model import CandidateActorCritic
from sts_policy.training import load_checkpoint
from sts_rl import fresh_training, fresh_scoring
from sts_rl.fresh_study import atomic_json
from sts_rl.fresh_training import TrainingControl, make_model, run_training, validate_training_config
from sts_rl.fresh_scoring import SELECTION_CRITERION, score_deferred
from sts_rl.recording import file_identity
from tests.unit.test_fresh_training import fixture, synthetic_begin_run


class FreshScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def runtime(self, stack, source):
        native=json.loads(Path(source['launch']['path']).read_text())['native']
        stack.enter_context(patch('sts_rl.fresh_training.validate_training_config',
            side_effect=lambda c:validate_training_config(c,synthetic=True)))
        for module in (fresh_training, fresh_scoring):
            stack.enter_context(patch.object(module,'TrainingControl',side_effect=lambda out,d:
                TrainingControl(out,d,ac=lambda:True,caffeinate=False)))
            stack.enter_context(patch.object(module,'begin_run',side_effect=synthetic_begin_run(native)))

    def train(self, root, data, architecture='new', **extra):
        config=dict(data=[str(p) for p in data] if isinstance(data,list) else [str(data)],
            out=str(root/'run'),architecture=architecture,phase='combat',
            fit_size='full',width=16,layers=1,heads=4,steps=2,min_steps=2,batch=2,
            eval_every=1,patience=6,lr=.001,seed=1919,device='cpu',deadline_utc=None,
            defer_test=True,**extra)
        return run_training(config)

    def selection(self, root):
        selected={'selected_checkpoint':file_identity(root/'run'/'best.pt'),
                  'manifest':file_identity(root/'run'/'manifest.json'),
                  'selection_criterion':dict(SELECTION_CRITERION)}
        path=root/'selection.json'
        atomic_json(path,selected)
        return path,selected

    def test_old_default_large_none_independent_weights_and_trainability(self):
        torch.manual_seed(101)
        champion=CandidateActorCritic(size='large',context='none')
        torch.manual_seed(202)
        with patch('sts_rl.fresh_training.load_checkpoint',side_effect=AssertionError('checkpoint input')), \
             patch('sts_policy.training.load_checkpoint',side_effect=AssertionError('teacher weights')):
            model=make_model('old','combat')
        self.assertNotIn('checkpoint',inspect.signature(make_model).parameters)
        self.assertEqual(model.config,champion.config)
        self.assertEqual((model.config.size,model.config.context),('large','none'))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(model.parameters(),champion.parameters())))
        for (name,p),q in zip(model.named_parameters(),champion.parameters()):
            self.assertNotEqual(p.data_ptr(),q.data_ptr())
            self.assertEqual(p.requires_grad,not name.startswith('critic.'))
        small=make_model('old','ooc',old_size='small',old_context='mean')
        self.assertEqual((small.config.size,small.config.context),('small','mean'))

    def test_condition_uses_saved_entity_shape_and_rejects_config_disagreement(self):
        model=make_model('new','combat',width=16,layers=1,heads=4)
        config=dict(architecture='new',phase='combat',lr=.001,seed=11,steps=2,min_steps=2,
                    eval_every=1,patience=6,device='cpu',batch=2)
        payload={'metadata':{**model.metadata(),'config':config}}
        condition=fresh_training.checkpoint_training_condition(payload)
        self.assertEqual([condition[k] for k in ('width','layers','heads')],[16,1,4])
        payload['metadata']['config']={**config,'width':128}
        with self.assertRaisesRegex(ValueError,'architecture/config mismatch'):
            fresh_training.checkpoint_training_condition(payload)

    def test_cli_defaults_and_explicit_old_controls(self):
        for script in ('fresh_train','fresh_benchmark'):
            spec=importlib.util.spec_from_file_location(script,Path(__file__).resolve().parents[2]/'scripts/rl'/f'{script}.py')
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            args=['--data','synthetic','--out','out','--deadline-utc','2099-01-01T00:00:00Z']
            args+=['--architecture','old','--phase','combat'] if script=='fresh_train' else ['--checkpoint','teacher.pt']
            defaults=module.parser().parse_args(args)
            self.assertEqual((defaults.old_size,defaults.old_context),('large','none'))
            self.assertEqual((defaults.width,defaults.layers,defaults.heads),(128,2,4))
            configured=module.parser().parse_args(args+['--old-size','medium','--old-context','mean'])
            self.assertEqual((configured.old_size,configured.old_context),('medium','mean'))
            if script=='fresh_train':
                self.assertFalse(defaults.defer_test)
                self.assertTrue(module.parser().parse_args(args+['--defer-test']).defer_test)

    def test_defer_has_no_test_encoding_evaluation_or_checkpoint_input(self):
        for architecture in ('old','new'):
            with self.subTest(architecture=architecture),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp)
                data,source=fixture(root)
                self.runtime(stack,source)
                split=[None]
                original_iter=fresh_training.iter_rows
                def tracked_rows(*args,**kwargs):
                    for row,meta in original_iter(*args,**kwargs):
                        split[0]=meta['split']
                        yield row,meta
                module=encoding if architecture=='old' else entity_encoding
                encode=module.encode_observation
                seen=[]
                def guarded_encode(obs):
                    self.assertNotEqual(split[0],'TEST','TEST passed to encoder')
                    seen.append(split[0])
                    return encode(obs)
                evaluate=fresh_training.evaluate
                def guarded_eval(model,examples,*args,**kwargs):
                    self.assertTrue(all(x.meta['split']!='TEST' for x in examples))
                    return evaluate(model,examples,*args,**kwargs)
                stack.enter_context(patch.object(fresh_training,'iter_rows',side_effect=tracked_rows))
                stack.enter_context(patch.object(module,'encode_observation',side_effect=guarded_encode))
                stack.enter_context(patch.object(fresh_training,'evaluate',side_effect=guarded_eval))
                stack.enter_context(patch.object(fresh_training,'load_checkpoint',side_effect=AssertionError('checkpoint input')))
                summary=self.train(root,data,architecture)
                self.assertEqual(set(seen),{'FIT','VALIDATION'})
                self.assertFalse(summary['test_evaluated'])
                self.assertTrue(summary['test_deferred'])
                self.assertNotIn('test',summary)
                self.assertFalse((root/'run'/'test_rows.jsonl').exists())
                self.assertEqual(set(json.loads((root/'run'/'coverage.json').read_text())),{'FIT','VALIDATION'})
                payload=load_checkpoint(root/'run'/'best.pt')
                receipt=json.loads((root/'run'/'receipt.json').read_text())
                launch=json.loads(Path(receipt['launch']['path']).read_text())
                self.assertIsNone(launch['parent_checkpoint'])
                self.assertIsNone(payload['metadata']['parent_checkpoint'])
                self.assertTrue(payload['metadata']['config']['defer_test'])
                self.assertEqual(summary['best_validation_loss'],payload['metadata']['training_state']['best_loss'])
                if architecture=='old':
                    self.assertEqual((payload['model_config']['size'],payload['model_config']['context']),('large','none'))
                    self.assertEqual((receipt['old_size'],receipt['old_context']),('large','none'))
                    self.assertEqual(payload['metadata']['old_size'],'large')

    def test_selection_mismatches_reject_before_any_raw_test_access(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp)
            data,source=fixture(root)
            self.runtime(stack,source)
            self.train(root,data)
            path,selection=self.selection(root)
            with patch.object(fresh_scoring,'freeze_manifest',side_effect=AssertionError('raw data access')), \
                 patch.object(fresh_scoring,'preencode',side_effect=AssertionError('TEST access')):
                for key in ('selected_checkpoint','manifest','selection_criterion'):
                    invalid=deepcopy(selection)
                    if key=='selection_criterion':
                        invalid[key]['metric']='test_soft_ce'
                    else:
                        invalid[key]['sha256']='0'*64
                    atomic_json(path,invalid)
                    with self.subTest(key=key),self.assertRaises(ValueError):
                        score_deferred(path,root/'score')
                # Even a self-consistent receipt cannot pair a checkpoint with another manifest.
                manifest=json.loads((root/'run'/'manifest.json').read_text())
                manifest['counts']['raw_rows']+=1
                manifest['data_identity']=fresh_training.digest({k:v for k,v in manifest.items() if k!='data_identity'})
                atomic_json(root/'other-manifest.json',manifest)
                invalid=deepcopy(selection)
                invalid['manifest']=file_identity(root/'other-manifest.json')
                atomic_json(path,invalid)
                with self.assertRaisesRegex(ValueError,'dataset identity'):
                    score_deferred(path,root/'score')
            self.assertFalse((root/'score').exists())

    def test_deferred_scoring_reopens_mixed_manifest(self):
        for mixed_sampling in (False,True):
            with self.subTest(mixed_sampling=mixed_sampling),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp)
                first,source=fixture(root/'first',config={'teacher_probabilities':{'BOSS2':1.0,'MONSTER1':.04}})
                second,_=fixture(root/'second',interface_suffix=b'# faster interface\n',
                    config={'world_start':92_000_001_000,
                            'teacher_probabilities':{'BOSS2':1.0,'MONSTER1':0.0 if mixed_sampling else .04}})
                self.runtime(stack,source)
                self.train(root,[first,second],mixed_sources=True,mixed_sampling=mixed_sampling)
                path,_=self.selection(root)
                summary=score_deferred(path,root/'score')
                self.assertEqual(summary['status'],'completed')
                self.assertEqual(summary['test_rows'],6)

    def test_finalizer_cpu_singletons_actor_only_no_updates_and_fresh_output(self):
        for architecture in ('new','old'):
            with self.subTest(architecture=architecture),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp)
                data,source=fixture(root)
                self.runtime(stack,source)
                self.train(root,data,architecture,old_size='small')
                path,selection=self.selection(root)
                before=load_checkpoint(root/'run'/'best.pt')
                evaluate=fresh_training.evaluate
                seen=[]
                def checked(model,examples,arch,**kwargs):
                    self.assertEqual(arch,architecture)
                    self.assertTrue(kwargs['singleton'])
                    self.assertEqual(kwargs.get('device','cpu'),'cpu')
                    self.assertTrue(all(x.meta['split']=='TEST' for x in examples))
                    self.assertFalse(any(p.requires_grad for p in model.parameters()))
                    state={k:v.clone() for k,v in model.state_dict().items()}
                    # Singleton actor calls may never execute the critic.
                    with patch.object(model.critic,'forward',side_effect=AssertionError('critic evaluated')):
                        result=evaluate(model,examples,arch,**kwargs)
                    self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    seen.append(True)
                    return result
                with patch.object(fresh_training,'evaluate',side_effect=checked), \
                     patch('torch.optim.AdamW',side_effect=AssertionError('scorer optimizer')), \
                     patch('torch.Tensor.backward',side_effect=AssertionError('scorer backward')), \
                     patch.object(fresh_training,'save_checkpoint',side_effect=AssertionError('scorer weight save')):
                    result=score_deferred(path,root/'score')
                self.assertEqual(len(seen),1)
                self.assertTrue(result['test_evaluated'])
                self.assertEqual(result['weight_updates'],0)
                self.assertEqual(result['test_rows'],3)
                self.assertEqual(file_identity(root/'run'/'best.pt'),selection['selected_checkpoint'])
                after=load_checkpoint(root/'run'/'best.pt')
                self.assertTrue(all(torch.equal(v,after['model_state'][k]) for k,v in before['model_state'].items()))
                rows=[json.loads(line) for line in (root/'score'/'test_rows.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),3)
                self.assertTrue(all(r['selected_checkpoint']==selection['selected_checkpoint']['sha256'] for r in rows))
                self.assertTrue(all(r['architecture']==architecture and
                                    r['training_condition']==fresh_training.checkpoint_training_condition(before) for r in rows))
                manifest=json.loads((root/'run'/'manifest.json').read_text())
                by_id={r['source_row_id']:r for r in manifest['rows']}
                self.assertTrue(all(r['data_identity']==manifest['data_identity'] and r['split']=='TEST' for r in rows))
                self.assertTrue(all(r['source_identity']==by_id[r['source_row_id']]['source_identity'] for r in rows))
                receipt=json.loads((root/'score'/'receipt.json').read_text())
                self.assertEqual(receipt['selection'],file_identity(path))
                self.assertEqual(receipt['selected_checkpoint'],selection['selected_checkpoint'])
                with self.assertRaises(FileExistsError):
                    score_deferred(path,root/'score')


if __name__=='__main__':
    unittest.main()
