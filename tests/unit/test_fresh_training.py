"""Bounded pure/synthetic warm-up contracts; no native simulator or real worlds."""
from copy import deepcopy
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch

from sts_policy import entity_encoding
from sts_policy.entity_model import EntityActorCritic, policy_from_payload
from sts_policy.model import CandidateActorCritic
from sts_policy.training import load_checkpoint, save_checkpoint
from sts_rl.fresh_study import (INFERENCE, PROTOCOL, PUBLIC_CONTEXT_ID, StudyPolicy, Interrupted,
    atomic_json, fit_manifest_path, read_json, policy_observation)
from sts_rl.fresh_training import (FILTERS, MIN_DISK, TrainingControl, collate, digest, fit,
    freeze_manifest, make_model, preencode, soft_cross_entropy, target_distribution, validate_source,
    validate_training_config, world_split, run_training, interleave_fit, iter_rows)
from sts_rl.recording import file_identity
from sts_rl.teacher_votes import RULES
from tests.unit.test_entity_encoding import combat_observation, ooc_observation


def synthetic_source(root, *, interface_suffix=b'', environment_version='0.1.5',
                     observation_version='sts_env_public_v2', runtime_python_version='3.12',
                     native_bytes=b'synthetic-no-native-execution', config=None):
    teacher=root/'teacher.pt'
    save_checkpoint(teacher,CandidateActorCritic(),model_config=CandidateActorCritic().metadata(),
                    metadata={'encoder_version':'public_candidate_v2'})
    native=root/'slaythespire.synthetic'
    native.write_bytes(native_bytes)
    archive=root/'source.zip'
    env = f'ENVIRONMENT_VERSION = "{environment_version}"\nOBSERVATION_VERSION = "{observation_version}"\n'.encode()
    interface_source = b'# synthetic only\n'+interface_suffix
    with zipfile.ZipFile(archive,'w') as output:
        output.writestr('src/sts_env/synthetic.py',interface_source)
        output.writestr('src/sts_env/env.py',env)
        output.write(Path(__file__).resolve().parents[2]/'src/sts_rl/fresh_study.py','src/sts_rl/fresh_study.py')
    interface=hashlib.sha256(b'env.py\0'+env+b'\0synthetic.py\0'+interface_source+b'\0').hexdigest()
    config={'mode':'collect','teacher_labels':True,'out':'synthetic','n':50,'world_start':92_000_000_000,
            'checkpoint':str(teacher),**(config or {})}
    launch={'source_archive':file_identity(archive),'native':{'module_file':native.name,
            'module_sha256':file_identity(native)['sha256']},'parent_checkpoint':file_identity(teacher),
            'effective_config':config,'runtime':{'python':runtime_python_version+' synthetic'}}
    path=root/'launch.json'
    atomic_json(path,launch)
    source={'protocol':PROTOCOL,'policy':{'protocol':PROTOCOL,'filter_ids':FILTERS,'inference':INFERENCE,
            'public_context_id':PUBLIC_CONTEXT_ID,
            'champion':file_identity(teacher),'combat':None,'ooc':None},'config':config,'launch':file_identity(path)}
    provenance={'python_interface_sha256':interface,'runtime_python_version':runtime_python_version,
                'environment_version':environment_version,'observation_version':observation_version,
                'native_provenance':launch['native']}
    return source,provenance


def row(observation,seed,source,provenance,prefix=()):
    legal=policy_observation(observation)['legal_actions']
    result={'protocol':PROTOCOL,'observation':observation,'public_hash':digest_public(observation),
            'group':str(observation['state']['room']).rsplit('.',1)[-1]+str(observation['state']['act']),
            'world_seed':seed,'prefix':list(prefix),'source':source,'expected_provenance':provenance,
            'source_action':legal[0]['index'],'label':legal[0]['index']}
    if observation['phase']=='combat':
        result.update(label_kind='canonical_rng_resampled_native_v2',votes={str(legal[0]['index']):3,str(legal[-1]['index']):1},
                      search={'samples':[1,2,3,4],'draw_order_seeds':[1,2,3,4], 'search_seed':5,
                              'effective_rng_seeds':[5046,5047,5048,5049],'effective_search_seeds':[6,7,8,9]})
    else:
        result.update(label_kind='champion_hard',choice=len(legal)>1)
    return result


def digest_public(o):
    from sts_policy.curriculum import public_hash
    return public_hash(o)


def fixture(root, *, record_search=False, **source_options):
    root.mkdir(parents=True,exist_ok=True)
    if record_search:
        source_options['config'] = {**source_options.get('config',{}), 'record_search':True}
    source,provenance=synthetic_source(root, **source_options)
    data=root/'shards'
    (data/'worlds').mkdir(parents=True)
    # Several worlds per split, both phases per world, real C1 raw shard format.
    seeds={s:[] for s in ('FIT','VALIDATION','TEST')}
    start=source['config']['world_start']
    for seed in range(start,start+1000):
        split=world_split(seed)
        if len(seeds[split])<3:
            seeds[split].append(seed)
        if all(len(v)==3 for v in seeds.values()):
            break
    for seed in sum(seeds.values(),[]):
        c=combat_observation()
        o=ooc_observation()
        o['legal_actions'].append({'index':9,'description':'select card index 0: Strike [Attack, Basic]'})
        a=row(c,seed,source,provenance)
        if record_search:
            first,last=(c['legal_actions'][i]['index'] for i in (0,-1))
            a['search_details']=[{'chosen':index,'edges':[
                {'index':first,'visits':100,'evaluation_sum':100.,'mean_evaluation':1.},
                {'index':last,'visits':20,'evaluation_sum':40.,'mean_evaluation':2.}]}
                for index in (first,first,first,last)]
        b=row(o,seed,source,provenance,[a['source_action']])
        shard={'protocol':PROTOCOL,'world_seed':seed,'status':'completed','source':source,
               'expected_provenance':provenance,'action_prefix':[a['source_action'],b['source_action']],
               'labels':[a,b]}
        with gzip.open(data/'worlds'/f'{seed}.json.gz','wt') as stream:
            stream.write(json.dumps(shard)+'\n')
    return data,source


def synthetic_begin_run(native):
    """Only synthetic fixture tests replace the strict native recording boundary."""
    from sts_rl.recording import begin_run
    def wrapper(*args, **kwargs):
        if 'preparation' in kwargs:
            raise AssertionError('Production caller must use strict begin_run')
        result = begin_run(*args, **kwargs, preparation=True)
        path = Path(result['path'])
        launch = json.loads(path.read_text())
        launch['native'] = native
        atomic_json(path, launch)
        return file_identity(path)
    return wrapper


class FreshTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_relabel_recomputes_votes_labels_and_encoded_targets_without_changing_sources(self):
        expected={'best_line':({10:3,17:1},10), 'most_visited':({10:4},10),
                  'mean_value':({17:4},17), 'pooled_mean':({17:4},17)}
        with tempfile.TemporaryDirectory() as tmp:
            data,_=fixture(Path(tmp),record_search=True)
            stored=list(iter_rows([data]))
            original_files=[file_identity(p) for p in sorted((data/'worlds').glob('*'))]
            baseline=freeze_manifest([data])
            identities={baseline['data_identity']}
            self.assertNotIn('relabel_vote_rule',baseline)
            for rule,(votes,label) in expected.items():
                with self.subTest(rule=rule):
                    relabelled=list(iter_rows([data],relabel_vote_rule=rule))
                    for (raw,meta),(original,original_meta) in zip(relabelled,stored):
                        if meta['phase']=='combat':
                            self.assertEqual(raw['votes'],votes)
                            self.assertEqual(raw['label'],label)
                            self.assertEqual(raw['vote_rule'],rule)
                            self.assertEqual(raw['source'],original['source'])
                            self.assertEqual(meta['source_identity'],original_meta['source_identity'])
                        else:
                            self.assertEqual((raw,meta),(original,original_meta))
                    manifest=freeze_manifest([data],relabel_vote_rule=rule)
                    self.assertEqual(manifest['relabel_vote_rule'],rule)
                    self.assertEqual(manifest['collection_sources'],baseline['collection_sources'])
                    self.assertEqual(manifest['public_surface'],baseline['public_surface'])
                    self.assertEqual(manifest['data_identity'],digest({k:v for k,v in manifest.items() if k!='data_identity'}))
                    identities.add(manifest['data_identity'])
                    for architecture in ('old','new','controlled'):
                        dataset=preencode([data],manifest,'combat',architecture)
                        for examples in dataset.values():
                            for example in examples:
                                indices=example.encoded.action_indices.tolist()
                                np.testing.assert_array_equal(example.target,
                                    np.array([votes.get(i,0)/4 for i in indices],dtype=np.float32))
            self.assertEqual(len(identities),len(RULES)+1)
            self.assertEqual(list(iter_rows([data])),stored)
            self.assertEqual(freeze_manifest([data]),baseline)
            self.assertEqual([file_identity(p) for p in sorted((data/'worlds').glob('*'))],original_files)

    def test_relabel_uses_collector_plurality_tie_break_in_current_menu_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,prov=synthetic_source(root)
            observation=combat_observation()
            observation['legal_actions'].reverse()
            raw=row(observation,92_000_000_000,source,prov)
            raw['search_details']=[{'chosen':i} for i in (10,17,10,17)]
            export=root/'rows.jsonl'
            export.write_text(json.dumps(raw)+'\n')
            from sts_rl.fresh_collection import plurality
            with patch('sts_rl.fresh_training.plurality',wraps=plurality) as choose:
                relabelled,_=next(iter_rows([export],relabel_vote_rule='best_line'))
            choose.assert_called_once()
            self.assertEqual(relabelled['votes'],{10:2,17:2})
            self.assertEqual(relabelled['label'],17)

    def test_relabel_requires_details_even_for_recorded_abstentions(self):
        for abstention in (False,True):
            with self.subTest(abstention=abstention),tempfile.TemporaryDirectory() as tmp:
                data,_=fixture(Path(tmp))
                if abstention:
                    for path in (data/'worlds').glob('*'):
                        shard=read_json(path)
                        shard['labels'][0].update(label=None,votes={})
                        atomic_json(path,shard,compressed=True)
                freeze_manifest([data])  # Stored votes still permit legacy shards.
                for load in (freeze_manifest,lambda s,**kw:list(iter_rows(s,**kw))):
                    with self.assertRaisesRegex(ValueError,'requires search_details.*--record-search'):
                        load([data],relabel_vote_rule='pooled_mean')

    def test_relabel_preserves_empty_details_abstentions_and_rejects_empty_labelled_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,_=fixture(Path(tmp),record_search=True)
            seen=set()
            for path in (data/'worlds').glob('*'):
                shard=read_json(path)
                split=world_split(shard['world_seed'])
                if split not in seen:
                    seen.add(split)
                    shard['labels'][0].update(label=None,votes={},search_details=[])
                shard['sampling']={shard['labels'][0]['group']+(
                    ':abstain' if shard['labels'][0]['label'] is None else ':labelled'):1}
                atomic_json(path,shard,compressed=True)
            baseline=freeze_manifest([data])
            manifest=freeze_manifest([data],relabel_vote_rule='pooled_mean')
            self.assertEqual(manifest['counts'],baseline['counts'])
            self.assertEqual(manifest['counts']['combat_abstentions'],3)
            for raw,meta in iter_rows([data],relabel_vote_rule='pooled_mean'):
                if meta['abstention']:
                    self.assertEqual(raw['search_details'],[])
                    self.assertEqual(raw['votes'],{})
                    self.assertIsNone(raw['label'])
            dataset=preencode([data],manifest,'combat','old')
            self.assertEqual({s:len(rows) for s,rows in dataset.items()},
                             {s:2 for s in ('FIT','VALIDATION','TEST')})
            path=next((data/'worlds').glob('*'))
            shard=read_json(path)
            shard['labels'][0].update(label=10,votes={'10':4},search_details=[])
            shard['sampling']={shard['labels'][0]['group']+':labelled':1}
            for details,message in (([],'non-empty search_details'),(None,'search_details to be a list')):
                shard['labels'][0]['search_details']=details
                atomic_json(path,shard,compressed=True)
                with self.subTest(details=details),self.assertRaisesRegex(ValueError,message):
                    freeze_manifest([data],relabel_vote_rule='pooled_mean')

    def test_mixed_sources_cannot_relax_differing_relabel_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first,_=fixture(root/'first',record_search=True,config={'relabel_vote_rule':'best_line'})
            for rule in (None,'pooled_mean'):
                second,_=fixture(root/str(rule),record_search=True,config={
                    'world_start':92_000_001_000,**({'relabel_vote_rule':rule} if rule else {})})
                for relabel in (None,'pooled_mean'):
                    for mixed_sampling in (False,True):
                        with self.subTest(rule=rule,relabel=relabel,mixed_sampling=mixed_sampling):
                            with self.assertRaisesRegex(ValueError,'Incompatible mixed source field: config.relabel_vote_rule'):
                                freeze_manifest([first,second],mixed_sources=True,
                                    mixed_sampling=mixed_sampling,relabel_vote_rule=relabel)

    def test_relabel_cli_and_config_are_combat_only_and_optin(self):
        script=Path(__file__).resolve().parents[2]/'scripts/rl/fresh_train.py'
        spec=importlib.util.spec_from_file_location('relabel_cli',script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args=['--data','synthetic','--out','out','--architecture','old','--phase','combat',
              '--deadline-utc','2099-01-01T00:00:00Z']
        parser=module.parser()
        self.assertNotIn('relabel_vote_rule',vars(parser.parse_args(args)))
        for rule in RULES:
            config=vars(parser.parse_args(args+['--relabel-vote-rule',rule]))
            self.assertEqual(config['relabel_vote_rule'],rule)
            validate_training_config(config,synthetic=True)
            with self.assertRaisesRegex(ValueError,'only valid for combat fits'):
                validate_training_config({**config,'phase':'ooc'},synthetic=True)
        self.assertIn('--relabel-vote-rule',parser.format_help())
        with self.assertRaisesRegex(ValueError,'relabel_vote_rule must be one of'):
            validate_training_config({'relabel_vote_rule':'unknown'},synthetic=True)
        with self.assertRaisesRegex(ValueError,'relabel_vote_rule must be one of'):
            list(iter_rows([],relabel_vote_rule='unknown'))
        with self.assertRaisesRegex(ValueError,'config/manifest relabel_vote_rule mismatch'):
            fit(None,{}, {'relabel_vote_rule':'pooled_mean'},None,
                {'relabel_vote_rule':'best_line'},{},None)

    def test_teacher_play_corpora_are_refused_even_without_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,source=fixture(Path(tmp),config={'teacher_plays':True})
            with self.assertRaisesRegex(ValueError,'teacher_plays true cannot be used for training'):
                validate_source(source)
            for load in (freeze_manifest,lambda s:list(iter_rows(s))):
                with self.assertRaisesRegex(ValueError,'teacher_plays true cannot be used for training'):
                    load([data])
            for path in (data/'worlds').glob('*'):
                with gzip.open(path,'rt') as stream:
                    shard=json.load(stream)
                shard['labels']=[]
                atomic_json(path,shard,compressed=True)
            with self.assertRaisesRegex(ValueError,'teacher_plays true cannot be used for training'):
                freeze_manifest([data])

    def test_nontraining_samples_refused_in_world_shards_and_row_exports(self):
        for phase in ('combat','out_of_combat'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                data,_=fixture(root)
                path=next((data/'worlds').glob('*'))
                with gzip.open(path,'rt') as stream:
                    shard=json.load(stream)
                sample=next(r for r in shard['labels'] if r['observation']['phase']==phase)
                sample['use_for_training']=False
                atomic_json(path,shard,compressed=True)
                export=root/'rows.jsonl'
                export.write_text(json.dumps(sample)+'\n')
                for sources in ([data],[export]):
                    for load in (freeze_manifest,lambda s:list(iter_rows(s))):
                        with self.assertRaisesRegex(ValueError,'use_for_training false cannot be used for training'):
                            load(sources)

    def test_mixed_sources_optin_preserves_all_sources_and_individual_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first,source1=fixture(root/'first')
            second,source2=fixture(root/'second',interface_suffix=b'# faster interface\n',
                config={'world_start':92_000_001_000,'workers':2,'deadline_utc':None,'max_labels':50})
            sources=[first,second]
            for load in (freeze_manifest,lambda s:list(iter_rows(s))):
                with self.assertRaisesRegex(ValueError,'^Incompatible scientific source/provenance identities$'):
                    load(sources)
            manifest=freeze_manifest(sources,mixed_sources=True)
            self.assertIs(manifest['mixed_sources'],True)
            self.assertEqual(manifest['collection_sources'],{digest(s):s for s in (source1,source2)})
            identities={validate_source(s)[0] for s in (source1,source2)}
            interfaces=sorted({validate_source(s)[1]['verified_interface_sha256'] for s in (source1,source2)})
            self.assertEqual(len(identities),2)
            self.assertEqual(len(interfaces),2)
            self.assertEqual({m['source_identity'] for m in manifest['rows']},identities)
            self.assertEqual(manifest['public_surface']['provenance']['python_interface_sha256'],interfaces)
            self.assertEqual(manifest['collected_total'],36)
            self.assertEqual(manifest['collected_worlds'],18)
            self.assertEqual(manifest['data_identity'],digest({k:v for k,v in manifest.items() if k!='data_identity'}))
            dataset=preencode(sources,manifest,'combat','old')
            self.assertEqual({x.meta['source_identity'] for rows in dataset.values() for x in rows},identities)

    def test_mixed_sampling_optin_accepts_and_records_distinct_profiles(self):
        standard={'BOSS2':1.0,'MONSTER1':.04}
        bosses={'MONSTER1':0.0,'BOSS2':1.0}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first,source1=fixture(root/'first',config={'teacher_probabilities':standard,'samples':4,'simulations':50})
            second,source2=fixture(root/'second',config={'world_start':92_000_001_000,
                'teacher_probabilities':bosses,'samples':4,'simulations':50})
            sources=[first,second]
            for load in (freeze_manifest,lambda s,**kw:list(iter_rows(s,**kw))):
                with self.assertRaisesRegex(ValueError,'Incompatible mixed source field: config.teacher_probabilities'):
                    load(sources,mixed_sources=True)
            manifest=freeze_manifest(sources,mixed_sources=True,mixed_sampling=True)
            self.assertIs(manifest['mixed_sampling'],True)
            self.assertEqual(manifest['teacher_probability_profiles'],sorted(
                json.dumps(p,sort_keys=True,separators=(',',':')) for p in (standard,bosses)))
            self.assertEqual(manifest['collection_sources'],{digest(s):s for s in (source1,source2)})
            self.assertEqual(manifest['data_identity'],digest({k:v for k,v in manifest.items() if k!='data_identity'}))
            single=freeze_manifest([first],mixed_sources=True,mixed_sampling=True)
            self.assertEqual(len(single['teacher_probability_profiles']),1)
            dataset=preencode(sources,manifest,'combat','old')
            self.assertEqual({s:len(rows) for s,rows in dataset.items()},{s:6 for s in ('FIT','VALIDATION','TEST')})

    def test_mixed_sources_reject_other_scientific_differences_by_field(self):
        config={'teacher_probabilities':{'BOSS1':.5},'samples':4,'simulations':50}
        changes=(
            ({'native_bytes':b'different native module'},'native.module_sha256'),
            ({'config':{'teacher_probabilities':{'BOSS1':.75}}},'config.teacher_probabilities.BOSS1'),
            ({'config':{'simulations':51}},'config.simulations'),
            ({'config':{'samples':5}},'config.samples'),
            ({'config':{'samples':4.0}},'config.samples'),
            ({'config':{'self_consistency_every':7}},'config.self_consistency_every'),
            ({'config':{'teacher_probabilities':{'BOSS2':0.5},'samples':4.0}},'config.samples'),
            ({'config':{'new_scientific_setting':True}},'config.new_scientific_setting'),
            ({'config':{'vote_rule':'pooled_mean'}},'config.vote_rule'),
            ({'config':{'relabel_vote_rule':'pooled_mean'}},'config.relabel_vote_rule'),
            ({'environment_version':'0.1.6'},'provenance.environment_version'),
            ({'observation_version':'sts_env_public_v3'},'provenance.observation_version'),
            ({'runtime_python_version':'3.13'},'provenance.runtime_python_version'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first,_=fixture(root/'first',config=config)
            for i,(options,field) in enumerate(changes):
                second,_=fixture(root/str(i),**{**options,'config':{
                    **config,'world_start':92_000_001_000,**options.get('config',{})}})
                for mixed_sampling in (False,True):
                    if mixed_sampling and field.startswith('config.teacher_probabilities'):
                        continue
                    with self.subTest(field=field,mixed_sampling=mixed_sampling),self.assertRaisesRegex(
                            ValueError,'Incompatible mixed source field: '+field):
                        freeze_manifest([first,second],mixed_sources=True,mixed_sampling=mixed_sampling)

    def test_mixed_sources_still_verify_each_source_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,source=fixture(Path(tmp))
            path=next((data/'worlds').glob('*'))
            with gzip.open(path,'rt') as stream:
                shard=json.load(stream)
            shard['labels'][0]['expected_provenance']['python_interface_sha256']='unverified'
            atomic_json(path,shard,compressed=True)
            with self.assertRaisesRegex(ValueError,'Missing/mismatched native/public provenance'):
                freeze_manifest([data],mixed_sources=True)
            Path(source['policy']['champion']['path']).write_bytes(b'corrupt checkpoint')
            with self.assertRaisesRegex(ValueError,'Source file identity mismatch'):
                freeze_manifest([data],mixed_sources=True)

    def test_mixed_sources_off_keeps_single_source_manifest_and_scientific_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,source=fixture(Path(tmp))
            manifest=freeze_manifest([data])
            self.assertEqual(manifest,freeze_manifest([data],mixed_sources=False))
            self.assertEqual(manifest,freeze_manifest([data],mixed_sampling=False))
            self.assertNotIn('mixed_sources',manifest)
            self.assertNotIn('mixed_sampling',manifest)
            self.assertNotIn('teacher_probability_profiles',manifest)
            launch=json.loads(Path(source['launch']['path']).read_text())
            legacy_config={k:v for k,v in source['config'].items() if k not in {
                'out','n','world_start','workers','deadline_utc','max_labels'}}
            legacy_identity=digest({'policy':source['policy'],'native':launch['native'],'config':legacy_config,
                                    'source_archive_sha256':launch['source_archive']['sha256']})
            self.assertEqual(validate_source(source)[0],legacy_identity)
            self.assertEqual({m['source_identity'] for m in manifest['rows']},{legacy_identity})
            self.assertEqual(manifest['data_identity'],digest({k:v for k,v in manifest.items() if k!='data_identity'}))

    def test_mixed_sampling_off_keeps_legacy_manifest_snapshots(self):
        # Fixed raw-reader output makes complete pre-option manifest digests reproducible.
        source={'policy':{'protocol':PROTOCOL,'filter_ids':FILTERS,
                         'public_context_id':PUBLIC_CONTEXT_ID,'inference':INFERENCE},
                'config':{'teacher_probabilities':{'BOSS2':1.0,'MONSTER1':.04}}}
        raw={'source':source,'expected_provenance':{'python_interface_sha256':'interface'}}
        meta=dict(source_row_id='row',world_seed=92_000_000_000,group='BOSS2',phase='combat',
                  has_target=True,abstention=False,choice=True,split='FIT',source_identity='source')
        identity=dict(path='synthetic',sha256='shard',bytes=1)
        # Captured from freeze_manifest before mixed_sampling was added.
        for mixed_sources,expected in (
                (False,'306ba46541b788feac972a2fcc79aa039ffa413effe02f55ea5e4bea2707fcb7'),
                (True,'343cd6fe02cc7afd0082f6068678f1093ada62b77bb756eb7730a284927b126f')):
            with self.subTest(mixed_sources=mixed_sources), \
                 patch('sts_rl.fresh_training.shard_files',return_value=[Path('synthetic')]), \
                 patch('sts_rl.fresh_training.file_identity',return_value=identity), \
                 patch('sts_rl.fresh_training.iter_rows',side_effect=lambda *a,**kw:iter([(raw,meta)])):
                manifest=freeze_manifest(['synthetic'],mixed_sources=mixed_sources)
                self.assertEqual(manifest,freeze_manifest(['synthetic'],mixed_sources=mixed_sources,mixed_sampling=False))
                self.assertEqual(manifest,freeze_manifest(['synthetic'],mixed_sources=mixed_sources,relabel_vote_rule=None))
                self.assertNotIn('relabel_vote_rule',manifest)
                self.assertNotIn('mixed_sampling',manifest)
                self.assertNotIn('teacher_probability_profiles',manifest)
                self.assertEqual(manifest['data_identity'],expected)
                self.assertEqual(digest({k:v for k,v in manifest.items() if k!='data_identity'}),expected)

    def test_mixed_sources_cli_is_optin(self):
        script=Path(__file__).resolve().parents[2]/'scripts/rl/fresh_train.py'
        spec=importlib.util.spec_from_file_location('mixed_sources_cli',script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args=['--data','synthetic','--out','out','--architecture','old','--phase','combat',
              '--deadline-utc','2099-01-01T00:00:00Z']
        self.assertNotIn('mixed_sources',vars(module.parser().parse_args(args)))
        self.assertNotIn('mixed_sampling',vars(module.parser().parse_args(args)))
        self.assertIs(module.parser().parse_args(args+['--mixed-sources']).mixed_sources,True)
        config=vars(module.parser().parse_args(args+['--mixed-sources','--mixed-sampling']))
        self.assertIs(config['mixed_sampling'],True)
        validate_training_config(config,synthetic=True)
        with self.assertRaisesRegex(ValueError,'mixed_sources must be boolean'):
            validate_training_config({'mixed_sources':'yes'},synthetic=True)
        with self.assertRaisesRegex(ValueError,'only valid for combat fits'):
            validate_training_config({'mixed_sources':True,'phase':'ooc'},synthetic=True)
        with self.assertRaisesRegex(ValueError,'mixed_sampling must be boolean'):
            validate_training_config({'mixed_sampling':'yes'},synthetic=True)
        for mixed_sources in (False,None):
            config={'mixed_sampling':True}
            if mixed_sources is not None:
                config['mixed_sources']=mixed_sources
            with self.subTest(config=config),self.assertRaisesRegex(ValueError,'mixed_sampling requires mixed_sources'):
                validate_training_config(config,synthetic=True)
        with self.assertRaisesRegex(ValueError,'mixed_sampling requires mixed_sources'):
            list(iter_rows([],mixed_sampling=True))

    def test_collection_checkpoint_must_match_recorded_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,_=synthetic_source(root,config={'checkpoint':str(root/'elsewhere.pt')})
            with self.assertRaisesRegex(ValueError,'checkpoint does not match the recorded champion'):
                validate_source(source)

    def test_soft_ce_gapped_mapping_and_padding(self):
        r=row(combat_observation(),92_000_000_000,{}, {})
        target,indices=target_distribution(r)
        self.assertEqual(indices,[10,11,12,13,14,15,16,17])
        np.testing.assert_array_equal(target,[.75,0,0,0,0,0,0,.25])
        logits=torch.tensor([[np.log(3),0.,-torch.inf]],dtype=torch.float64,requires_grad=True)
        loss=soft_cross_entropy(logits,torch.tensor([[.75,.25,0.]]),torch.tensor([[True,True,False]]))
        self.assertAlmostEqual(float(loss.detach()),-.75*np.log(.75)-.25*np.log(.25))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ooc_only_choice_rows_reach_optimizer_for_both_encoders(self):
        from sts_rl.fresh_training import minibatch_order
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data,_=fixture(root)
            # Every split keeps three real choices plus three forced targets.
            for path in (data/'worlds').glob('*.json.gz'):
                with gzip.open(path,'rt') as stream:
                    shard=json.load(stream)
                forced=row(ooc_observation(),shard['world_seed'],shard['source'],
                           shard['expected_provenance'],shard['action_prefix'])
                shard['labels'].append(forced)
                shard['action_prefix'].append(forced['source_action'])
                atomic_json(path,shard,compressed=True)
            manifest=freeze_manifest([data])
            frozen=digest(manifest)
            fit_ids=set(manifest['subsets']['ooc']['full'])
            forced_ids={r['source_row_id'] for r in manifest['rows'] if r['phase']=='ooc' and not r['choice']}
            self.assertEqual(len(fit_ids & forced_ids),3)
            datasets={a:preencode([data],manifest,'ooc',a) for a in ('new','old')}
            self.assertEqual([[x.meta for x in datasets[a]['FIT']] for a in ('new','old')][0],
                             [[x.meta for x in datasets[a]['FIT']] for a in ('new','old')][1])
            for architecture,dataset in datasets.items():
                out=root/architecture
                out.mkdir()
                counts=dict(manifest_fit_rows=6,optimization_fit_choice_rows=3,forced_fit_rows_excluded=3)
                seen=[]
                def choice_batches(examples,arch,batch_size,rng):
                    self.assertEqual({x.meta['source_row_id'] for x in examples},fit_ids-forced_ids)
                    self.assertTrue(all(len(x.target)>1 for x in examples))
                    batches=minibatch_order(examples,arch,batch_size,rng)
                    self.assertTrue(all(any(len(examples[i].target)>1 for i in b) for b in batches))
                    seen.extend(batches)
                    return batches
                config=dict(architecture=architecture,phase='ooc',fit_size='full',device='cpu',seed=12,batch=2,
                            lr=.01,steps=6,min_steps=6,eval_every=2,patience=6,width=16,layers=1,heads=4,old_size='small')
                model=make_model(architecture,'ooc',width=16,layers=1,heads=4,old_size='small')
                control=TrainingControl(out,ac=lambda:True,caffeinate=False)
                try:
                    with patch('sts_rl.fresh_training.minibatch_order',side_effect=choice_batches):
                        summary=fit(model,dataset,config,out,manifest,{},control)
                finally:
                    control.close()
                self.assertEqual(len(seen),6)
                self.assertEqual(summary['updates'],6)
                self.assertEqual(summary['examples'],9)  # Three complete choice-only epochs.
                self.assertEqual(summary['actual_fit_rows'],6)
                self.assertEqual({k:summary[k] for k in counts},counts)
                self.assertEqual(summary['test']['groups']['all_rows']['rows'],6)
                self.assertEqual(summary['test']['groups']['choice']['rows'],3)
                payload=load_checkpoint(out/'latest.pt')
                self.assertEqual(payload['metadata']['optimization_rows'],counts)
                self.assertEqual(payload['metadata']['training_state']['examples'],9)
                self.assertTrue(all(int(s['step'])==6 for s in payload['optimizer_state']['state'].values()))
                events=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
                self.assertTrue(all({k:e[k] for k in counts}==counts for e in events))
                self.assertTrue(all(e['validation']['groups']['all_rows']['rows']==6 and
                                    e['validation']['groups']['choice']['rows']==3 for e in events))
                rows=[json.loads(line) for line in (out/'test_rows.jsonl').read_text().splitlines()]
                self.assertEqual(len({r['source_row_id'] for r in rows}&forced_ids),3)
                self.assertTrue(all(r['target']==[1.] and r['correct'] and r['ce']==0 for r in rows if not r['choice']))
                self.assertEqual(digest(manifest),frozen)
            self.assertEqual(freeze_manifest([data]),manifest)

    def test_forced_only_fit_fails_before_optimizer_or_weight_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data,_=fixture(root)
            manifest=freeze_manifest([data])
            for architecture in ('new','old'):
                dataset=preencode([data],manifest,'ooc',architecture)
                # Use a real encoded single-action public observation.
                from sts_policy import encoding
                encoder=entity_encoding if architecture=='new' else encoding
                forced=encoder.encode_observation(policy_observation(ooc_observation()))
                for x in dataset['FIT']:
                    x.encoded=forced
                    x.target=np.ones(1,dtype=np.float32)
                    x.meta={**x.meta,'choice':False}
                model=make_model(architecture,'ooc',width=16,layers=1,heads=4,old_size='small')
                before={k:v.clone() for k,v in model.state_dict().items()}
                config=dict(architecture=architecture,phase='ooc',device='cpu',seed=12,batch=2,
                            lr=.01,steps=4,min_steps=2,eval_every=2,patience=6,old_size='small')
                control=TrainingControl(root,ac=lambda:True,caffeinate=False)
                try:
                    with patch('torch.optim.AdamW',side_effect=AssertionError('optimizer created')):
                        with self.assertRaisesRegex(ValueError,'No optimization FIT choice rows'):
                            fit(model,dataset,config,root,manifest,{},control)
                finally:
                    control.close()
                self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
                self.assertFalse((root/'metrics.jsonl').exists())
                self.assertFalse((root/'latest.pt').exists())

    def test_ooc_fit_size_is_full_only(self):
        for size in ('5000','20000'):
            with self.assertRaisesRegex(ValueError,'OOC warm-up is full-only'):
                validate_training_config({'phase':'ooc','fit_size':size},synthetic=True)

    def test_invalid_votes_indices_hashes_and_provenance(self):
        r=row(combat_observation(),92_000_000_000,{}, {})
        for votes in ({},{'10':-1},{'10':float('nan')},{'10':float('inf')},{'99':4},{'10':0}):
            with self.subTest(votes=votes),self.assertRaises(ValueError):
                target_distribution({**r,'votes':votes})
        with tempfile.TemporaryDirectory() as tmp:
            data,source=fixture(Path(tmp))
            invalid=deepcopy(source)
            invalid['policy']['filter_ids']=['stale']
            with self.assertRaisesRegex(ValueError,'filter'):
                validate_source(invalid)
            manifest=freeze_manifest([data])
            path=Path(manifest['rows'][0]['shard'])
            with gzip.open(path,'rt') as stream:
                shard=json.load(stream)
            shard['labels'][0]['observation']['state']['gold']=999
            with gzip.open(path,'wt') as stream:
                stream.write(json.dumps(shard)+'\n')
            with self.assertRaisesRegex(ValueError,'hash'):
                freeze_manifest([data])

    def test_world_split_nested_selection_and_same_raw_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,_=fixture(Path(tmp))
            m=freeze_manifest([data])
            self.assertEqual(m,freeze_manifest([data]))
            worlds=[set(m['splits'][s]['worlds']) for s in ('FIT','VALIDATION','TEST')]
            self.assertFalse(worlds[0]&worlds[1] or worlds[0]&worlds[2] or worlds[1]&worlds[2])
            for phase in ('combat','ooc'):
                subsets=m['subsets'][phase]
                self.assertEqual(subsets['5000'],subsets['full'][:5000])
                self.assertEqual(subsets['20000'],subsets['full'][:20000])
                by_id={r['source_row_id']:r for r in m['rows']}
                for size,ids in subsets.items():
                    self.assertEqual(m['fit_subset_group_counts'][phase][size],
                                     dict(Counter(by_id[i]['group'] for i in ids)))
                    self.assertTrue(all(by_id[i]['split']=='FIT' for i in ids))
                    new=preencode([data],m,phase,'new',size)
                    old=preencode([data],m,phase,'old',size)
                    self.assertEqual({x.meta['source_row_id'] for x in new['FIT']},set(ids))
                    for split in ('FIT','VALIDATION','TEST'):
                        self.assertEqual([x.meta for x in new[split]],[x.meta for x in old[split]])
                        self.assertTrue(all(np.array_equal(a.target,b.target) for a,b in zip(new[split],old[split])))

    def test_vectorized_logits_gradients_padding_and_owners(self):
        torch.manual_seed(11)
        model=EntityActorCritic(width=16,layers=2,heads=4)
        c=combat_observation()
        o=ooc_observation()
        o['legal_actions']=o['legal_actions'][:1]
        c['state']['combat']['recent_actions']*=2
        batch=entity_encoding.collate_observations([entity_encoding.encode_observation(c),entity_encoding.encode_observation(o)])
        singleton=model.actor_logits(batch)
        batched=model.training_logits(batch)
        mask=batch['action_mask']
        torch.testing.assert_close(singleton[mask],batched[mask],atol=5e-7,rtol=1e-5)
        self.assertTrue(torch.isneginf(batched[~mask]).all())
        singleton[mask].square().sum().backward()
        expected={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.training_logits(batch)[mask].square().sum().backward()
        for n,p in model.named_parameters():
            if n in expected:
                torch.testing.assert_close(p.grad,expected[n],atol=3e-6,rtol=3e-4)
        # Training path never calls the per-row validator or .item().
        with patch.object(model,'_rows',side_effect=AssertionError('per-row loop')):
            model.training_logits(batch)

    def test_tiny_two_phase_training_actual_updates_frozen_critic_and_loaders(self):
        started=time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data,source=fixture(root)
            manifest=freeze_manifest([data])
            for architecture in ('new','old'):
                for phase in ('combat','ooc'):
                    with self.subTest(architecture=architecture,phase=phase):
                        out=root/(architecture+'-'+phase)
                        out.mkdir()
                        dataset=preencode([data],manifest,phase,architecture)
                        model=make_model(architecture,phase,width=16,layers=1,heads=4,old_size='small')
                        before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('critic.')}
                        initial_features={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('state_encoder.')}
                        config=dict(architecture=architecture,phase=phase,device='cpu',seed=12,batch=2,
                                    lr=.01,steps=4,min_steps=2,eval_every=2,patience=6,width=16,layers=1,heads=4,
                                    old_size='small')
                        control=TrainingControl(out,ac=lambda:True,caffeinate=False)
                        summary=fit(model,dataset,config,out,manifest,{'teacher':source['policy']['champion']},control)
                        control.close()
                        self.assertEqual(summary['updates'],4)
                        self.assertGreater(summary['examples'],4)
                        self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before))
                        payload=load_checkpoint(out/'latest.pt')
                        self.assertEqual(payload['step'],4)
                        self.assertTrue(payload['optimizer_state']['state'])
                        self.assertTrue(all(int(s['step'])==4 for s in payload['optimizer_state']['state'].values()))
                        status=('head_frozen_shared_actor_features_trained' if architecture=='old' else 'frozen_untrained')
                        self.assertEqual(payload['metadata']['critic_status'],status)
                        self.assertEqual(payload['metadata']['explained_variance'],'not_applicable')
                        events=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
                        self.assertTrue(all(e['critic_status']==status and e['explained_variance']=='not_applicable' for e in events))
                        if architecture=='old':
                            self.assertTrue(any(not torch.equal(p,initial_features[n]) for n,p in model.named_parameters()
                                                if n in initial_features))
                        self.assertNotIn('parent_policy_checkpoint',payload['metadata'])
                        replacement=StudyPolicy(str(root/'teacher.pt'),**{phase:str(out/'best.pt')})
                        obs=combat_observation() if phase=='combat' else ooc_observation()
                        if phase=='ooc':
                            obs['legal_actions'].append({'index':9,'description':'select card index 0: Strike [Attack, Basic]'})
                        self.assertIn(replacement.choose_action(obs),[a['index'] for a in policy_observation(obs)['legal_actions']])
                        lines=[json.loads(line) for line in (out/'test_rows.jsonl').read_text().splitlines()]
                        self.assertEqual(len(lines),3)
                        self.assertTrue(all(r['inference']==INFERENCE for r in lines))
                        self.assertTrue(all(r['architecture']==architecture and
                                            r['training_condition']['phase']==phase and
                                            r['training_condition']['lr']==config['lr'] and
                                            r['training_condition']['seed']==config['seed'] for r in lines))
                        self.assertTrue(all(r['data_identity']==manifest['data_identity'] and r['split']=='TEST' for r in lines))
                        by_id={r['source_row_id']:r for r in manifest['rows']}
                        self.assertTrue(all(r['source_identity']==by_id[r['source_row_id']]['source_identity'] for r in lines))
                        if architecture=='new':
                            policy_from_payload(load_checkpoint(out/'best.pt'))
            # Also verify B1's bare payload through StudyPolicy, no format_version.
            bare=root/'bare.pt'
            model=EntityActorCritic(width=16,layers=1,heads=4)
            torch.save({'model_state':model.state_dict(),'metadata':model.metadata()},bare)
            StudyPolicy(str(root/'teacher.pt'),combat=str(bare)).choose_action(combat_observation())
            with self.assertRaisesRegex(ValueError,'phase'):
                StudyPolicy(str(root/'teacher.pt'),ooc=str(bare))
        print(f'D1 synthetic 2-phase new+old smoke seconds={time.monotonic()-started:.3f}')
        self.assertLess(time.monotonic()-started,30)

    def test_stop_deadline_disk_and_cancellable_battery_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            disk=lambda _:SimpleNamespace(free=MIN_DISK+1)
            tick=[0.]
            powered=[True]
            control=TrainingControl(root,ac=lambda:powered[0],now=lambda:tick[0],monotonic=lambda:tick[0],disk=disk,caffeinate=False)
            control.check()
            tick[0]=3
            self.assertEqual(control.stats()['active_ac_seconds'],3)
            powered[0]=False
            tick[0]=6
            control.stats()
            def cancel(seconds):
                tick[0]+=seconds
                (root/'STOP').touch()
            control.sleep=cancel
            with self.assertRaisesRegex(Interrupted,'STOP'):
                control.check()
            self.assertEqual(control.stats()['active_ac_seconds'],3)
            (root/'STOP').unlink()
            control.deadline=0
            with self.assertRaisesRegex(Interrupted,'deadline'):
                control.check()
            control.deadline=None
            control.disk=lambda _:SimpleNamespace(free=1)
            with self.assertRaisesRegex(Interrupted,'disk'):
                control.check()
            control.close()
        with self.assertRaisesRegex(ValueError,'UTC'):
            TrainingControl('/tmp',deadline_utc='2026-10-15T12:00:00')

    def test_production_minimum(self):
        config=dict(steps=20,min_steps=2,eval_every=2,patience=6,width=16,layers=1,heads=4,lr=.001,device='cpu')
        with self.assertRaisesRegex(ValueError,'2000'):
            validate_training_config(config)
        validate_training_config(config,synthetic=True)

    def test_production_requires_probes_before_any_optimizer_or_output(self):
        config=dict(steps=2000,min_steps=2000,eval_every=100,patience=6,width=128,layers=2,
                    heads=4,lr=.0003,device='cpu')
        for probe in (None,[]):
            with self.subTest(probe=probe),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'run'
                actual={**config,'probe':probe,'out':str(out)}
                with patch('torch.optim.AdamW',side_effect=AssertionError('optimizer created')), \
                     patch('sts_rl.fresh_training.freeze_manifest',side_effect=AssertionError('data opened')):
                    with self.assertRaisesRegex(ValueError,'dedicated 93e9 probe corpus'):
                        run_training(actual)
                self.assertFalse(out.exists())
                validate_training_config(actual,synthetic=True)
        validate_training_config({**config,'probe':['synthetic-probe-path-validation-only']})

    def test_real_run_receipt_records_teacher_as_data_never_weight_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data,source=fixture(root)
            config=dict(data=[str(data)],out=str(root/'run'),architecture='new',phase='combat',
                fit_size='full',width=16,layers=1,heads=4,steps=2,min_steps=2,batch=2,
                eval_every=1,patience=6,lr=.001,seed=1919,device='cpu',deadline_utc=None)
            frozen_path = root/'legacy-manifest.json'
            atomic_json(frozen_path, freeze_manifest([str(data)]))
            config['manifest'] = str(frozen_path)
            def control(out,deadline):
                return TrainingControl(out,deadline,ac=lambda:True,caffeinate=False)
            with patch('sts_rl.fresh_training.validate_training_config',
                       side_effect=lambda c:validate_training_config(c,synthetic=True)), \
                 patch('sts_rl.fresh_training.TrainingControl',side_effect=control):
                native=json.loads(Path(source['launch']['path']).read_text())['native']
                with patch('sts_rl.fresh_training.begin_run',side_effect=synthetic_begin_run(native)):
                    summary=run_training(config)
            self.assertEqual(summary['status'],'completed')
            manifest_path = fit_manifest_path(root/'run')
            self.assertEqual(manifest_path.name, 'manifest.json.gz')
            self.assertFalse((root/'run'/'manifest.json').exists())
            self.assertEqual(read_json(manifest_path), freeze_manifest([str(data)]))
            receipt=json.loads((root/'run'/'receipt.json').read_text())
            launch=json.loads(Path(receipt['launch']['path']).read_text())
            self.assertIsNone(launch['parent_checkpoint'])
            self.assertEqual(launch['effective_config']['teacher_checkpoints'],[source['policy']['champion']['path']])
            inputs=[r['path'] for r in launch['data_inputs']]
            self.assertIn(source['policy']['champion']['path'],inputs)
            self.assertTrue(any(p.endswith('entity_vocabulary.json') for p in inputs))
            self.assertEqual(launch['effective_config']['filters'],FILTERS)

    def test_large_nested_interleave_keeps_exact_unique_prefixes(self):
        counts=dict(MONSTER1=12701,ELITE1=6307,BOSS1=3037,MONSTER2=1019,BOSS2=700,BOSS3=237)
        rows=[dict(group=g,world_seed=i%7,source_row_id=f'{g}:{i}') for g,n in counts.items() for i in range(n)]
        ordered=interleave_fit(rows)
        self.assertEqual(ordered,interleave_fit(list(reversed(rows))))
        self.assertEqual(set(ordered),{r['source_row_id'] for r in rows})
        self.assertEqual(len(set(ordered)),24001)
        self.assertTrue(set(ordered[:5000])<set(ordered[:20000])<set(ordered))
        by_id={r['source_row_id']:r for r in rows}
        used=Counter()
        for k,identity in enumerate(ordered,1):
            used[by_id[identity]['group']]+=1
            for g,n in counts.items():
                self.assertLess(abs(used[g]-k*n/len(rows)),2,(k,g,used[g]))

    def test_group_world_round_robin_and_hashed_rows_until_exhaustion(self):
        rows=[dict(group=g,world_seed=w,source_row_id=f'{g}:{w}:{i}')
              for g in ('MONSTER1','BOSS3') for w,n in ((1,1),(2,4),(3,2)) for i in range(n)]
        by_id={r['source_row_id']:r for r in rows}
        ordered=interleave_fit(rows)
        self.assertEqual(by_id[ordered[0]]['group'],'BOSS3')  # Equal-quota tie: group name.
        for g in ('MONSTER1','BOSS3'):
            ids=[i for i in ordered if by_id[i]['group']==g]
            worlds=sorted((1,2,3),key=lambda w:digest(['world-order-v1',w]))
            hashed={w:sorted([r['source_row_id'] for r in rows if r['group']==g and r['world_seed']==w],
                            key=lambda i:digest(['fit-order-v1',i])) for w in worlds}
            # Each round visits every surviving source world exactly once,
            # including unequal world lengths and depletion between rounds.
            offset=0
            while worlds:
                current=ids[offset:offset+len(worlds)]
                self.assertEqual([by_id[i]['world_seed'] for i in current],worlds)
                for w,i in zip(worlds,current):
                    self.assertEqual(i,hashed[w].pop(0))
                offset+=len(worlds)
                worlds=[w for w in worlds if hashed[w]]

    def test_abstentions_preserved_counted_excluded_from_all_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,_=fixture(Path(tmp))
            seen=set()
            for path in sorted((data/'worlds').glob('*')):
                with gzip.open(path,'rt') as stream:
                    shard=json.load(stream)
                split=world_split(shard['world_seed'])
                if split not in seen:
                    seen.add(split)
                    shard['labels'][0].update(label=None,votes={})
                    self.assertIsNone(target_distribution(shard['labels'][0])[0])
                    atomic_json(path,shard,compressed=True)
            manifest=freeze_manifest([data])
            self.assertEqual(manifest['collected_total'],18)
            self.assertEqual(manifest['counts']['combat_abstentions'],3)
            self.assertEqual(manifest['counts']['combat_informative_labels'],6)
            self.assertEqual(manifest['counts']['target_rows'],15)
            self.assertEqual(manifest['counts']['ooc_raw_rows'],9)
            abstentions={r['source_row_id'] for r in manifest['rows'] if r['abstention']}
            for split in ('FIT','VALIDATION','TEST'):
                self.assertFalse(abstentions & set(manifest['splits'][split]['row_ids']))
            for architecture in ('new','old'):
                examples=preencode([data],manifest,'combat',architecture)
                self.assertEqual({s:len(v) for s,v in examples.items()},{'FIT':2,'VALIDATION':2,'TEST':2})
                self.assertFalse(abstentions & {x.meta['source_row_id'] for v in examples.values() for x in v})
            path=next((data/'worlds').glob('*'))
            with gzip.open(path,'rt') as stream:
                shard=json.load(stream)
            shard['sampling']={'BOSS1:labelled':50,'BOSS1:abstain':0}
            atomic_json(path,shard,compressed=True)
            with self.assertRaisesRegex(ValueError,'counts mismatch'):
                freeze_manifest([data])
        original=row(combat_observation(),92_000_000_000,{}, {})
        for invalid in ({'label':None,'votes':{'10':4}}, {'label':None,'votes':{'10':-1}},
                        {'label':None,'votes':{'10':3}}, {'label':10,'votes':{}},
                        {'label':None,'votes':[]}, {'label':10,'votes':{'10':True}}):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                target_distribution({**original,**invalid})

    def test_real_training_requires_strict_matching_native_receipt(self):
        for native in ({'preparation_only':True}, {'module_file':'synthetic','module_sha256':'wrong'}):
            with self.subTest(native=native),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                data,_=fixture(root)
                config=dict(data=[str(data)],out=str(root/'run'),architecture='new',phase='combat',
                    fit_size='full',width=16,layers=1,heads=4,steps=2,min_steps=2,batch=2,
                    eval_every=1,patience=6,lr=.001,seed=1919,device='cpu',deadline_utc=None)
                with patch('sts_rl.fresh_training.validate_training_config',side_effect=lambda c:validate_training_config(c,synthetic=True)), \
                     patch('sts_rl.fresh_training.TrainingControl',side_effect=lambda out,d:TrainingControl(out,d,ac=lambda:True,caffeinate=False)), \
                     patch('sts_rl.fresh_training.begin_run',side_effect=synthetic_begin_run(native)) as begin:
                    with self.assertRaisesRegex(ValueError,'validated native identity'):
                        run_training(config)
                self.assertNotIn('preparation',begin.call_args.kwargs)
                self.assertFalse((root/'run'/'best.pt').exists())

if __name__=='__main__':
    unittest.main()
