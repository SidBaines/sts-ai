"""Bounded public-format fixtures; probes never supply supervised targets."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch

from sts_policy import entity_encoding
from sts_rl.fresh_study import PROTOCOL, atomic_json, policy_observation, read_json
from sts_rl.fresh_training import (TrainingControl, freeze_manifest, iter_rows, make_model,
    minibatch_order, preencode, fit, run_training, validate_training_config)
from sts_rl.fresh_probes import (ProbeMonitor, encode_probes, iter_probe_rows, select_probes)
from sts_rl.recording import file_identity
from tests.unit.test_fresh_training import (fixture, row, synthetic_source, synthetic_begin_run)
from tests.unit.test_entity_encoding import combat_observation, ooc_observation


def probe_fixture(root, count=4, **source_options):
    root.mkdir()
    source, prov = synthetic_source(root, **source_options)
    launch_path = Path(source['launch']['path'])
    launch = json.loads(launch_path.read_text())
    source['config'].update(teacher_labels=False, world_start=93_000_000_000, n=3)
    launch['effective_config'] = source['config']
    # Older B2 encoder bytes are allowed; public environment/context are identical.
    with zipfile.ZipFile(launch['source_archive']['path'], 'a') as z:
        z.writestr('src/sts_policy/entity_encoding.py', '# archived older B2 encoder\n')
    launch['source_archive'] = file_identity(launch['source_archive']['path'])
    atomic_json(launch_path, launch)
    source['launch'] = file_identity(launch_path)
    data = root/'data'
    for act in (1, 2, 3):
        world = 93_000_000_000+act-1
        prefix, rows = [], []
        for position in range(count+1):
            obs = ooc_observation() if position==count else combat_observation()
            obs['state']['act'] = act
            if position==count:
                obs['legal_actions'].append({'index':42, 'description':'leave'})
            r = row(obs,world,source,prov,prefix)
            for key in ('label','votes','search','choice'):
                r.pop(key,None)
            r['label_kind'] = 'probe_unlabelled'
            r['source_action'] = policy_observation(obs)['legal_actions'][-1]['index']
            rows.append(r)
            prefix.append(r['source_action'])
        atomic_json(data/'worlds'/f'{world}.json.gz', {'protocol':PROTOCOL,'status':'completed',
            'world_seed':world,'source':source,'expected_provenance':prov,
            'labels':rows,'action_prefix':prefix},compressed=True)
    return data


class FreshProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fixed_shared_ids_64_per_act_archived_encoder_may_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data,_ = fixture(root)
            manifest = freeze_manifest([data])
            probes = probe_fixture(root/'probe',count=70)
            selection, raws = select_probes([probes],'combat',expected_surface=manifest['public_surface'])
            self.assertEqual(selection['selected_count'],192)
            self.assertEqual(selection,select_probes([probes],'combat',expected_surface=manifest['public_surface'])[0])
            self.assertEqual(selection,select_probes(list(reversed(sorted((probes/'worlds').glob('*')))),
                                                   'combat',expected_surface=manifest['public_surface'])[0])
            all_meta = [m for _,m in iter_probe_rows([probes])]
            for act in (1,2,3):
                expected = sorted(m['source_row_id'] for m in all_meta if m['act']==act and m['phase']=='combat')[:64]
                self.assertEqual(expected,[m['source_row_id'] for m in selection['rows'] if m['act']==act])
            new = encode_probes(raws,selection,'new')
            old = encode_probes(raws,selection,'old')
            self.assertEqual([x.meta for x in new],[x.meta for x in old])
            self.assertTrue(all(not hasattr(x,'target') for x in new+old))
            ignored=deepcopy(raws)
            for raw in ignored:
                raw.update(source_action=-999,label='not a target',world_seed=-1,hidden_state={'seed':1234})
            unchanged=encode_probes(ignored,selection,'new')
            for a,b in zip(new,unchanged):
                for key in ('entity_ids','entity_numeric','action_ids','action_numeric','action_indices'):
                    np.testing.assert_array_equal(getattr(a.encoded,key),getattr(b.encoded,key))
            self.assertEqual(selection['counts']['raw_rows'],213)
            self.assertEqual(len(selection['source_worlds']),3)
            self.assertIn('b3_coverage',selection)
            self.assertNotEqual(next(iter(selection['sources'].values()))['archive']['sha256'],
                json.loads(Path(manifest['collection_sources'][next(iter(manifest['collection_sources']))]['launch']['path']).read_text())['source_archive']['sha256'])
            with self.assertRaisesRegex(ValueError,'TRAIN'):
                list(iter_rows([probes]))
            with self.assertRaises(ValueError):
                list(iter_probe_rows([data]))

    def test_probe_shard_validation_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            probes = probe_fixture(Path(tmp)/'probe')
            path = next((probes/'worlds').glob('*'))
            with gzip.open(path,'rt') as f:
                original = json.load(f)
            def change_status(s): s['status']='censored'
            def change_hash(s): s['labels'][0]['public_hash']='bad'
            def change_menu(s): s['labels'][0]['source_action']=12345
            def change_group(s): s['labels'][0]['group']='BOSS9'
            def add_label(s): s['labels'][0]['label']=17
            def change_context(s): s['source']['policy']['public_context_id']='old'
            def change_prefix(s): s['labels'][1]['prefix']=[]
            def truncate(s): s['labels'].pop()
            def change_world(s): s['world_seed']=92_000_000_000
            for mutation in (change_status,change_hash,change_menu,change_group,add_label,
                             change_context,change_prefix,truncate,change_world):
                broken = deepcopy(original)
                mutation(broken)
                atomic_json(path,broken,compressed=True)
                with self.subTest(mutation=mutation.__name__), self.assertRaises(ValueError):
                    list(iter_probe_rows([probes]))
            atomic_json(path,original,compressed=True)
            surface = deepcopy(next(iter_probe_rows([probes]))[1]['public_surface'])
            surface['provenance']['native_provenance']['module_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'identity'):
                select_probes([probes],'combat',expected_surface=surface)
            # A self-consistent receipt with incompatible archived adapter code also fails.
            source = original['source']
            launch_path = Path(source['launch']['path'])
            launch = json.loads(launch_path.read_text())
            archive_path = Path(launch['source_archive']['path'])
            with zipfile.ZipFile(archive_path) as z:
                contents = {name:z.read(name) for name in z.namelist()}
            contents['src/sts_rl/fresh_study.py'] = contents['src/sts_rl/fresh_study.py'].replace(
                b'ooc_potion_public_menu_inventory_v1',b'ooc_potion_public_menu_inventory_old')
            with zipfile.ZipFile(archive_path,'w') as z:
                for name,value in contents.items(): z.writestr(name,value)
            launch['source_archive'] = file_identity(archive_path)
            atomic_json(launch_path,launch)
            source['launch'] = file_identity(launch_path)
            atomic_json(path,original,compressed=True)
            with self.assertRaisesRegex(ValueError,'Archived public'):
                list(iter_probe_rows([path]))

    def test_mixed_manifest_probes_compare_versions_and_native_but_not_interface_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first,_=fixture(root/'first')
            second,_=fixture(root/'second',interface_suffix=b'# faster interface\n',
                             config={'world_start':92_000_001_000})
            strict=freeze_manifest([first])
            mixed=freeze_manifest([first,second],mixed_sources=True)
            surface=deepcopy(mixed['public_surface'])
            # Even an unseen source hash is compatible when explicit versions match.
            for i,suffix in enumerate((b'',b'# faster interface\n',b'# another speed-only edit\n')):
                probes=probe_fixture(root/f'probe{i}',count=1,interface_suffix=suffix)
                rows=list(iter_probe_rows([probes],expected_surface=surface))
                self.assertEqual(len(rows),6)
                if suffix:
                    with self.assertRaisesRegex(ValueError,'identity mismatch'):
                        list(iter_probe_rows([probes],expected_surface=strict['public_surface']))
            for i,options in enumerate(({'native_bytes':b'different native module'},
                                       {'environment_version':'0.1.6'},
                                       {'observation_version':'sts_env_public_v3'},
                                       {'runtime_python_version':'3.13'})):
                probes=probe_fixture(root/f'incompatible{i}',count=1,**options)
                with self.subTest(options=options),self.assertRaisesRegex(ValueError,'identity mismatch'):
                    list(iter_probe_rows([probes],expected_surface=surface))
            self.assertEqual(surface,mixed['public_surface'])

    def test_probe_cpu_actor_only_singleton_rng_and_mode_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            probes = probe_fixture(Path(tmp)/'probe',count=2)
            selection,raws = select_probes([probes],'combat')
            examples = encode_probes(raws,selection,'new')
            model = make_model('new','combat',width=16,layers=1,heads=4).train()
            state = torch.get_rng_state().clone()
            parameters = {n:p.clone() for n,p in model.named_parameters()}
            with patch.object(model.critic,'forward',side_effect=AssertionError('critic used')), \
                 patch.object(model,'training_logits',side_effect=AssertionError('batched path used')), \
                 patch.object(model,'actor_logits',wraps=model.actor_logits) as actor:
                monitor = ProbeMonitor(model,examples,'new')
                metric = monitor.measure(model)
                self.assertEqual(actor.call_count,2*len(examples))
                self.assertTrue(all(call.args[0]['entity_ids'].shape[0]==1 for call in actor.call_args_list))
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(parameters[n],p) for n,p in model.named_parameters()))
            self.assertEqual(metric['label'],'fixed93e9 probes')
            self.assertEqual(metric['pooled']['kl_start_to_current'],0.)
            self.assertEqual(metric['pooled']['count'],6)
            with torch.no_grad():
                for p in model.actor.parameters(): p.add_(.01)
            self.assertGreater(monitor.measure(model)['pooled']['kl_start_to_current'],0.)

    def test_probe_argmax_uses_logits_even_when_probabilities_round_to_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            probes=probe_fixture(Path(tmp)/'probe',count=1)
            selection,raws=select_probes([probes],'combat')
            examples=encode_probes(raws,selection,'new')
            model=make_model('new','combat',width=16,layers=1,heads=4)
            logits=torch.zeros((1,8));logits[0,1]=1e-8
            with patch.object(model,'actor_logits',side_effect=lambda _:logits.clone()):
                monitor=ProbeMonitor(model,examples,'new')
                logits[0,1]=0;logits[0,0]=1e-8
                event=monitor.measure(model)
            self.assertEqual(event['pooled']['argmax_change_start'],1.)

    def test_training_receipt_probes_inputs_and_validation_only_selection(self):
        for mixed in (False,True):
            with self.subTest(mixed=mixed),tempfile.TemporaryDirectory() as tmp:
                self.check_training_receipt(Path(tmp),mixed=mixed)

    def check_training_receipt(self, root, *, mixed):
        data,source=fixture(root)
        probes=probe_fixture(root/'probe',count=2)
        cfg=dict(data=[str(data)],probe=[str(probes)],out=str(root/'run'),architecture='new',phase='combat',
            fit_size='full',width=16,layers=1,heads=4,steps=2,min_steps=2,batch=2,
            eval_every=1,patience=6,lr=.001,seed=1919,device='cpu',deadline_utc=None)
        teachers=[source['policy']['champion']['path']]
        if mixed:
            second,source2=fixture(root/'second',interface_suffix=b'# faster interface\n',
                                  config={'world_start':92_000_001_000})
            cfg.update(data=[str(data),str(second)],mixed_sources=True)
            teachers.append(source2['policy']['champion']['path'])
        native=json.loads(Path(source['launch']['path']).read_text())['native']
        with patch('sts_rl.fresh_training.validate_training_config',side_effect=lambda c:validate_training_config(c,synthetic=True)), \
             patch('sts_rl.fresh_training.TrainingControl',side_effect=lambda out,d:TrainingControl(out,d,ac=lambda:True,caffeinate=False)), \
             patch('sts_rl.fresh_training.begin_run',side_effect=synthetic_begin_run(native)):
            summary=run_training(cfg)
        out=root/'run'
        receipt=json.loads((out/'receipt.json').read_text())
        selection=json.loads((out/'probe_selection.json').read_text())
        manifest=read_json(out/'manifest.json.gz')
        launch=json.loads(Path(receipt['launch']['path']).read_text())
        self.assertEqual(manifest['collected_total'],36 if mixed else 18)
        self.assertEqual(receipt['teacher_checkpoints'],sorted(teachers))
        self.assertEqual(launch['effective_config']['teacher_checkpoints'],sorted(teachers))
        self.assertEqual(len(manifest['collection_sources']),2 if mixed else 1)
        self.assertTrue(all(r['world_seed']<93_000_000_000 for r in manifest['rows']))
        self.assertEqual(receipt['probe_selection']['selection_identity'],selection['selection_identity'])
        inputs=[i['path'] for i in launch['data_inputs']]
        self.assertTrue(all(teacher in inputs for teacher in teachers))
        self.assertIn(str((out/'probe_selection.json').resolve()),inputs)
        self.assertTrue(all(i['path'] in inputs for i in selection['source_files']))
        events=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        self.assertTrue(all(e['fixed93e9_probes']['pooled']['count']==6 for e in events))
        self.assertEqual(summary['best_step'],min(events,key=lambda e:e['validation']['loss'])['step'])
        from sts_policy.training import load_checkpoint
        checkpoint=load_checkpoint(out/'latest.pt')
        self.assertEqual(checkpoint['metadata']['probe_selection'],receipt['probe_selection'])
        self.assertIsNone(checkpoint['metadata']['parent_checkpoint'])

    def test_probes_leave_exact_optimizer_updates_and_selection_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data,_=fixture(root)
            manifest=freeze_manifest([data])
            probes=probe_fixture(root/'probe',count=1)
            for architecture,phase in (('new','combat'),('old','ooc')):
                selection,raws=select_probes([probes],phase)
                monitor_rows=encode_probes(raws,selection,architecture)
                dataset=preencode([data],manifest,phase,architecture)
                states=[];summaries=[]
                for enabled in (False,True):
                    out=root/f'{architecture}-{enabled}';out.mkdir()
                    torch.manual_seed(71)
                    model=make_model(architecture,phase,width=16,layers=1,heads=4,old_size='small')
                    cfg=dict(architecture=architecture,phase=phase,device='cpu',seed=11,batch=2,
                        lr=.001,steps=2,min_steps=2,eval_every=1,patience=6,old_size='small')
                    control=TrainingControl(out,ac=lambda:True,caffeinate=False)
                    try:
                        summaries.append(fit(model,dataset,cfg,out,manifest,
                            {'probe_selection':{'selection_identity':selection['selection_identity']}},control,
                            probes=monitor_rows if enabled else None))
                    finally:
                        control.close()
                    from sts_policy.training import load_checkpoint
                    states.append(load_checkpoint(out/'latest.pt'))
                self.assertEqual({k:v for k,v in summaries[0].items() if k!='selected_checkpoint'},
                                 {k:v for k,v in summaries[1].items() if k!='selected_checkpoint'})
                for key in states[0]['model_state']:
                    self.assertTrue(torch.equal(states[0]['model_state'][key],states[1]['model_state'][key]),key)
                for key,value in states[0]['optimizer_state']['state'].items():
                    for field,tensor in value.items():
                        self.assertTrue(torch.equal(tensor,states[1]['optimizer_state']['state'][key][field]))

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_cpu_shadow_is_detached_preserves_live_device_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            probes=probe_fixture(Path(tmp)/'probe',count=1)
            selection,raws=select_probes([probes],'combat')
            examples=encode_probes(raws,selection,'new')
            model=make_model('new','combat',width=16,layers=1,heads=4).to('mps').train()
            cpu_rng=torch.get_rng_state().clone()
            mps_rng=torch.mps.get_rng_state().clone()
            before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            monitor=ProbeMonitor(model,examples,'new')
            self.assertIsNotNone(monitor.shadow)
            monitor.measure(model)
            self.assertTrue(model.training)
            self.assertTrue(all(p.device.type=='mps' for p in model.parameters()))
            self.assertTrue(all(p.device.type=='cpu' for p in monitor.shadow.parameters()))
            self.assertTrue(all(torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters()))
            self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(mps_rng,torch.mps.get_rng_state()))
            with torch.no_grad():
                next(monitor.shadow.parameters()).add_(1)
            self.assertTrue(torch.equal(before[next(iter(before))],next(model.parameters()).detach().cpu()))

    def test_two_axis_bucketing_reduces_real_padding_and_keeps_exact_exposures(self):
        from sts_rl.fresh_training import Example,collate
        base=combat_observation()
        examples=[]
        for count in [2]*10+[1024]*2:
            obs=deepcopy(base)
            obs['legal_actions']=[{'index':i,'description':'end turn'} for i in range(count)]
            encoded=entity_encoding.encode_observation(obs)
            examples.append(Example(encoded,np.ones(count,dtype=np.float32)/count,{}))
        self.assertEqual(len({len(x.encoded.entity_ids) for x in examples}),1)
        rng=random.Random(12)
        batches=minibatch_order(examples,'new',4,rng)
        self.assertEqual(batches,minibatch_order(examples,'new',4,random.Random(12)))
        self.assertEqual(sorted(i for b in batches for i in b),list(range(12)))
        slots=0
        for indices in batches:
            chosen=[examples[i] for i in indices]
            batch,_,mask=collate(chosen,'new','cpu')
            slots+=mask.numel()
            self.assertEqual(int(mask.sum()),sum(len(x.target) for x in chosen))
            self.assertEqual(len({len(x.target) for x in chosen}),1)
        # Same entity-only grouping would allocate 4096 slots to a rare row's batch.
        self.assertEqual(slots,2068)
        self.assertLess(slots,10*2+4*1024)
        # A restored local RNG plus remaining order reproduces the next epoch.
        state=rng.getstate()
        expected=minibatch_order(examples,'new',4,rng)
        restored=random.Random();restored.setstate(state)
        self.assertEqual(expected,minibatch_order(examples,'new',4,restored))


if __name__=='__main__':
    unittest.main()
