from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import importlib.util
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
from sts_rl.fresh_statistics import (clustered_mean, compare_games, compare_labels,
    compare_panel, exact_discordance, gate_c, teacher_consistency, validate_expectations)
from tests.unit.test_fresh_training import row
from sts_rl.fresh_training import digest
from sts_rl.fresh_study import INFERENCE, PROTOCOL, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID, atomic_json
from sts_rl.fresh_panel import paired_summary
from tests.unit.test_entity_encoding import combat_observation
from sts_rl.recording import file_identity


def compare_cli():
    spec=importlib.util.spec_from_file_location('compare_cli',Path(__file__).resolve().parents[2]/'scripts/rl/fresh_compare.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_identity(name):
    return {'path':'/synthetic/'+name+'.pt','bytes':123,'sha256':digest(name)}


CHAMPION_POLICY = dict(protocol=PROTOCOL,filter_ids=[FILTER_ID,DEAD_FILTER_ID],
    public_context_id=PUBLIC_CONTEXT_ID,inference=INFERENCE,
    champion=synthetic_identity('champion'),combat=None,ooc=None)
STUDENT_POLICY = {**CHAMPION_POLICY,'combat':synthetic_identity('selected-combat')}
GAME_EXPECTATIONS = dict(schema='fresh_comparison_expectations_v1',version=1,mode='games',
    student_policy=STUDENT_POLICY,champion_policy=CHAMPION_POLICY)
COMMON_CONDITION = dict(phase='combat',fit_size='full',lr=3e-4,seed=1919,
    steps=12000,min_steps=2000,eval_every=100,patience=6,device='cpu',batch=128,defer_test=True)
LABEL_EXPECTATIONS = dict(schema='fresh_comparison_expectations_v1',version=1,mode='labels',
    student_checkpoint=synthetic_identity('selected-new'),control_checkpoint=synthetic_identity('selected-old'),
    student_architecture='new',control_architecture='old',
    student_condition={**COMMON_CONDITION,'width':128,'layers':2,'heads':4},
    control_condition={**COMMON_CONDITION,'old_size':'large','old_context':'none'})


def control_rows(rows):
    return [{**r,'selected_checkpoint':LABEL_EXPECTATIONS['control_checkpoint']['sha256'],
             'architecture':'old','training_condition':deepcopy(LABEL_EXPECTATIONS['control_condition'])} for r in rows]


def champion_rows(rows):
    return [{**r,'source':{'policy':deepcopy(CHAMPION_POLICY)}} for r in rows]


def game(seed,win,floor=10,bosses=None,*,arm='student'):
    return dict(world_seed=seed,status='completed',win=win,floor=floor,bosses=bosses or {},
                source={'policy':deepcopy(STUDENT_POLICY if arm=='student' else CHAMPION_POLICY)},
                expected_provenance={'fixture':'synthetic-no-native-execution',
                                     'scientific_source':digest('synthetic-game-source')})


def label(i,world,correct):
    return dict(source_row_id=str(i),world_seed=world,group='BOSS2',target=[1.,0.],
                target_sha256=digest([1.,0.]),public_hash=digest(['synthetic-public',i]),
                choice=True,confident=True,correct=correct,ce=.7,split='TEST',inference=INFERENCE,
                selected_checkpoint=LABEL_EXPECTATIONS['student_checkpoint']['sha256'],architecture='new',
                training_condition=deepcopy(LABEL_EXPECTATIONS['student_condition']),
                data_identity=digest('synthetic-label-manifest'),source_identity=digest('synthetic-label-source'))


class FreshStatisticsTests(unittest.TestCase):
    def test_final_helpers_require_explicit_frozen_bindings(self):
        with self.assertRaises(TypeError):
            compare_games([],[],[])
        with self.assertRaises(TypeError):
            compare_labels([],[])
        for mode in ('games','labels'):
            for expectations in (None,{},dict(schema='wrong',version=1,mode=mode),
                                 dict(schema='fresh_comparison_expectations_v1',version=2,mode=mode),
                                 dict(schema='fresh_comparison_expectations_v1',version=1,mode='panel')):
                with self.subTest(mode=mode),self.assertRaises(ValueError):
                    validate_expectations(expectations,mode)

    def test_game_policy_arm_errors_reject_before_statistics(self):
        a=[game(1,False),game(2,True)]
        b=champion_rows(a)
        cases=[(b,a,GAME_EXPECTATIONS), (b,b,GAME_EXPECTATIONS), (a,a,GAME_EXPECTATIONS)]
        cases.append((a,[b[0],a[1]],GAME_EXPECTATIONS))  # Mixed champion arm.
        for field, value in (('champion',synthetic_identity('wrong-champion')),
                             ('combat',synthetic_identity('wrong-replacement')),
                             ('ooc',synthetic_identity('unexpected-ooc')),
                             ('inference','wrong'), ('public_context_id','wrong'), ('filter_ids',[])):
            bad=deepcopy(a)
            bad[-1]['source']['policy'][field]=value
            cases.append((bad,b,GAME_EXPECTATIONS))
            wrong_expected=deepcopy(GAME_EXPECTATIONS)
            wrong_expected['student_policy'][field]=value
            cases.append((a,b,wrong_expected))
        # A single wrong identity across the whole arm also fails the frozen binding.
        bad=deepcopy(a)
        for r in bad:
            r['source']['policy']['combat']=synthetic_identity('another-selected-combat')
        cases.append((bad,b,GAME_EXPECTATIONS))
        bad=deepcopy(a)
        bad[-1].pop('source')
        cases.append((bad,b,GAME_EXPECTATIONS))
        for update in ({'student_policy':CHAMPION_POLICY},
                       {'champion_policy':STUDENT_POLICY},
                       {'student_policy':CHAMPION_POLICY,'champion_policy':STUDENT_POLICY},
                       {'student_policy':{**STUDENT_POLICY,'combat':CHAMPION_POLICY['champion']}}):
            cases.append((a,b,{**GAME_EXPECTATIONS,**update}))
        with patch('sts_rl.fresh_statistics.clustered_mean',side_effect=AssertionError('CI started')):
            for student,champion,expected in cases:
                with self.subTest(student=student,expected=expected),self.assertRaises(ValueError):
                    compare_games(student,champion,[1,2],expectations=expected,resamples=100)

    def test_label_arm_checkpoint_architecture_condition_errors_before_statistics(self):
        a=[label(0,1,True),label(1,2,False)]
        b=control_rows(a)
        cases=[(b,a,LABEL_EXPECTATIONS),(a,a,LABEL_EXPECTATIONS),(b,b,LABEL_EXPECTATIONS)]
        for field,value in (('selected_checkpoint',digest('wrong-checkpoint')),('architecture','old'),
                            ('training_condition',{**LABEL_EXPECTATIONS['student_condition'],'lr':1e-3}),
                            ('training_condition',{**LABEL_EXPECTATIONS['student_condition'],'seed':7}),
                            ('training_condition',{**LABEL_EXPECTATIONS['student_condition'],'phase':'ooc'})):
            for count in (1,2):
                bad=deepcopy(a)
                for r in bad[:count]:
                    r[field]=value
                cases.append((bad,b,LABEL_EXPECTATIONS))
        for arm,rows in (('student',a),('control',b)):
            for field in ('selected_checkpoint','architecture','training_condition'):
                bad=deepcopy(rows)
                bad[-1].pop(field)
                cases.append((bad,b,LABEL_EXPECTATIONS) if arm=='student' else (a,bad,LABEL_EXPECTATIONS))
        for field,value in (('selected_checkpoint',digest('wrong-control')),('architecture','new'),
                            ('training_condition',{**LABEL_EXPECTATIONS['control_condition'],'fit_size':'5000'})):
            bad=deepcopy(b)
            bad[-1][field]=value
            cases.append((a,bad,LABEL_EXPECTATIONS))
        for update in ({'student_architecture':'old','control_architecture':'new'},
                       {'control_checkpoint':LABEL_EXPECTATIONS['student_checkpoint']},
                       {'control_checkpoint':{**LABEL_EXPECTATIONS['student_checkpoint'],'path':'/synthetic/copy.pt'}},
                       {'student_checkpoint':synthetic_identity('other-frozen')},
                       {'student_condition':{**LABEL_EXPECTATIONS['student_condition'],'lr':1e-3}},
                       {'control_condition':{**LABEL_EXPECTATIONS['control_condition'],'phase':'ooc'}}):
            cases.append((a,b,{**LABEL_EXPECTATIONS,**update}))
        with patch('sts_rl.fresh_statistics.clustered_mean',side_effect=AssertionError('CI started')):
            for student,control,expected in cases:
                with self.subTest(student=student,expected=expected),self.assertRaises(ValueError):
                    compare_labels(student,control,expectations=expected,resamples=100)
        # Different selected checkpoints with identical observed outcomes are valid.
        self.assertEqual(compare_labels(a,b,expectations=LABEL_EXPECTATIONS,resamples=100)['groups']['choice']['point'],0)

    def test_games_cli_uses_only_validated_cohort_policy_and_records_expectations(self):
        module=compare_cli()
        fake=ModuleType('sts_rl.fresh_cohort')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected=deepcopy(GAME_EXPECTATIONS)
            for name in ('champion','replacement'):
                (root/(name+'.pt')).write_bytes(('synthetic-frozen-'+name).encode())
            expected['champion_policy']['champion']=file_identity(root/'champion.pt')
            expected['student_policy']['champion']=file_identity(root/'champion.pt')
            expected['student_policy']['combat']=file_identity(root/'replacement.pt')
            expectation=root/'expectations.json'
            atomic_json(expectation,expected)
            seeds=[92000000000,92000000001]
            atomic_json(root/'cohort.json',seeds)
            cohorts={}
            for arm in ('student','champion'):
                directory=root/arm
                directory.mkdir()
                # Like C4's summary, records contain no source field.
                records=[{k:v for k,v in game(s,False).items() if k!='source'} for s in seeds]
                atomic_json(directory/'summary.json',{'records':records})
                cohorts[directory.resolve()]={'records':records,'science_identity':{'policy':expected[arm+'_policy']}}
            seen=[]
            def validate(paths, *, allow_teacher_plays=False):
                self.assertFalse(allow_teacher_plays)
                seen.append(paths)
                return cohorts[paths[0]]
            fake.validate_cohorts=validate
            args=['games','--student',str(root/'student'),'--champion',str(root/'champion'/'summary.json'),
                  '--cohort',str(root/'cohort.json'),'--expectations',str(expectation),'--out',str(root/'result.json')]
            with patch.dict('sys.modules',{'sts_rl.fresh_cohort':fake}),redirect_stdout(StringIO()),redirect_stderr(StringIO()) as report:
                module.main(args)
                self.assertIn(str(expectation),report.getvalue())
                self.assertIn(file_identity(expectation)['sha256'],report.getvalue())
                self.assertEqual(seen,[[(root/'student').resolve()],[(root/'champion').resolve()]])
                result=json.loads((root/'result.json').read_text())
                self.assertEqual(result['provenance']['expectations_file'],file_identity(expectation))
                self.assertEqual(result['expected_bindings'],expected)
                self.assertEqual(result['win']['point'],0)
                self.assertEqual(module.read_rows(root/'student'/'summary.json','games')[0]['source']['policy'],expected['student_policy'])
                # Even a plausible unbound source field is not accepted as a cohort.
                atomic_json(root/'unbound.json',game(seeds[0],False))
                with self.assertRaisesRegex(ValueError,'receipted'):
                    module.read_rows(root/'unbound.json','games')
                for arm in ('student','champion'):
                    with self.subTest(arm=arm),patch.object(module,'compare_games',side_effect=AssertionError('CI started')):
                        fake.validate_cohorts=lambda paths, **kwargs: (_ for _ in ()).throw(ValueError('Invalid completed cohort receipt'))
                        with self.assertRaisesRegex(ValueError,'receipt'):
                            module.main(args)
                fake.validate_cohorts=validate
                # Checkpoint mismatch is checked before cohort reads/statistics.
                (root/'replacement.pt').write_bytes(b'changed')
                with patch.object(module,'read_rows',side_effect=AssertionError('input read')):
                    with self.assertRaisesRegex(ValueError,'checkpoint file identity'):
                        module.main(args)

    def test_final_cli_requires_expectations_before_input_access(self):
        module=compare_cli()
        with patch.object(module,'read_rows',side_effect=AssertionError('input read')),redirect_stderr(StringIO()):
            for mode in ('games','labels'):
                with self.subTest(mode=mode),self.assertRaises(SystemExit):
                    module.main([mode,'--student','s','--champion','c','--cohort','ids','--out','unused'])

    @unittest.skipUnless(importlib.util.find_spec('sts_rl.fresh_cohort'), 'C4 dependency not staged')
    def test_games_reader_real_c4_receipts_bind_source_omitted_by_summary(self):
        from tests.unit.fresh_cohort_fixture import write_cohort
        from sts_policy.curriculum import public_hash
        module=compare_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected=deepcopy(GAME_EXPECTATIONS)
            for name in ('champion','replacement'):
                (root/(name+'.pt')).write_bytes(('synthetic-no-native-'+name).encode())
            champion=file_identity(root/'champion.pt')
            expected['champion_policy']['champion']=champion
            expected['student_policy']['champion']=champion
            expected['student_policy']['combat']=file_identity(root/'replacement.pt')
            seeds=[92000000000,92000000001]
            provenance={'environment_version':'0.1.5','native_provenance':{'module_sha256':digest('synthetic-native')}}
            worlds=[{**game(s,False),'protocol':PROTOCOL,'expected_provenance':provenance,
                     'task':{'seed':s,'expected_provenance':provenance},'action_prefix':[0],
                     'action_hash':public_hash({'actions':[0]}),'labels':[],'bank':[],'sampling':{},'behavior':{}}
                    for s in seeds]
            for arm in ('student','champion'):
                policy=expected[arm+'_policy']
                write_cohort(root/arm,worlds,policy_changes=policy,
                             config={'checkpoint':champion['path'],'combat':policy['combat']['path'] if policy['combat'] else None})
                raw=json.loads((root/arm/'summary.json').read_text())
                self.assertTrue(all('source' not in r for r in raw['records']))
            a=module.read_rows(root/'student','games')
            b=module.read_rows(root/'champion'/'summary.json','games')
            self.assertEqual(a[0]['source']['policy'],expected['student_policy'])
            self.assertEqual(compare_games(a,b,seeds,expectations=expected,resamples=100)['win']['point'],0)
            with patch('sts_rl.fresh_statistics.clustered_mean',side_effect=AssertionError('CI started')):
                with self.assertRaises(ValueError):
                    compare_games(b,a,seeds,expectations=expected,resamples=100)
            receipt=json.loads((root/'student'/'receipt.json').read_text())
            receipt['policy']['combat']=None
            atomic_json(root/'student'/'receipt.json',receipt)
            with self.assertRaises(ValueError):
                module.read_rows(root/'student'/'summary.json','games')

    def test_labels_cli_checks_real_checkpoint_identities_and_selected_configs(self):
        from sts_rl.fresh_training import make_model, checkpoint_training_condition
        from sts_policy.training import save_checkpoint, load_checkpoint
        module=compare_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected=deepcopy(LABEL_EXPECTATIONS)
            for arm,architecture in (('student','new'),('control','old')):
                config=dict(COMMON_CONDITION,architecture=architecture,width=16,layers=1,heads=4,
                            old_size='small',old_context='none')
                model=make_model(architecture,'combat',width=16,layers=1,heads=4,old_size='small')
                metadata={**model.metadata(),'config':config,
                          'encoder_version':'public_entities_fresh_v1' if architecture=='new' else 'public_candidate_v2'}
                path=root/(arm+'.pt')
                save_checkpoint(path,model,model_config=model.metadata(),metadata=metadata)
                expected[arm+'_checkpoint']=file_identity(path)
                expected[arm+'_condition']=checkpoint_training_condition(load_checkpoint(path))
            for arm in ('student','control'):
                rows=[{**label(i,92000000000+i,True),
                       'selected_checkpoint':expected[arm+'_checkpoint']['sha256'],
                       'architecture':expected[arm+'_architecture'],'training_condition':expected[arm+'_condition']}
                      for i in range(2)]
                (root/(arm+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            atomic_json(root/'cohort.json',['0','1'])
            atomic_json(root/'expectations.json',expected)
            args=['labels','--student',str(root/'student.jsonl'),'--champion',str(root/'control.jsonl'),
                  '--cohort',str(root/'cohort.json'),'--expectations',str(root/'expectations.json'),
                  '--out',str(root/'result.json')]
            with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
                module.main(args)
                result=json.loads((root/'result.json').read_text())
                self.assertEqual(result['provenance']['expectations_file'],file_identity(root/'expectations.json'))
                self.assertEqual(result['groups']['choice']['point'],0)
                for arm in ('student','control'):
                    wrong=deepcopy(expected)
                    wrong[arm+'_condition']['lr']=1e-3
                    atomic_json(root/'expectations.json',wrong)
                    with patch.object(module,'read_rows',side_effect=AssertionError('input read')):
                        with self.assertRaisesRegex(ValueError,'selected checkpoint config/architecture'):
                            module.main(args)
                wrong=deepcopy(expected)
                wrong['student_checkpoint']['sha256']=digest('wrong-sha')
                atomic_json(root/'expectations.json',wrong)
                with patch.object(module,'read_rows',side_effect=AssertionError('input read')):
                    with self.assertRaisesRegex(ValueError,'checkpoint file identity'):
                        module.main(args)

    def test_exact_discordance_and_gate_precedence(self):
        self.assertEqual(exact_discordance(0,0),1)
        self.assertAlmostEqual(exact_discordance(3,0),.25)
        self.assertAlmostEqual(exact_discordance(1,4),.375)
        self.assertEqual(exact_discordance(10000,0),0.)
        self.assertEqual(gate_c([-.009,-.001]),{'gate':'PASS','evidence_of_decline':True,'ni_margin':.01})
        self.assertEqual(gate_c([-.02,-.001])['gate'],'FAIL')
        self.assertEqual(gate_c([-.01,.03])['gate'],'INCONCLUSIVE')
        self.assertEqual(gate_c([None,None])['gate'],'INCONCLUSIVE')

    def test_world_block_weighting_keeps_rows(self):
        # Blocks [0,0] and [1]: resampling world blocks gives only 0,1/3,1.
        result=clustered_mean({10:[0.,0.],11:[1.]},resamples=1000)
        self.assertEqual(result['point'],1/3)
        self.assertEqual(result['rows'],3)
        self.assertEqual(result['worlds'],2)
        self.assertEqual(result['ci95'],[0.,1.])
        equal=clustered_mean({10:[0.,0.],11:[1.]},resamples=1000,equal_world=True)
        self.assertEqual(equal['point'],.5)

    def test_label_alignment_choice_metrics(self):
        a=[label(i,i//2,i%2==0) for i in range(4)]
        b=control_rows([{**r,'correct':False} for r in a])
        result=compare_labels(a,b,expectations=LABEL_EXPECTATIONS,resamples=100)
        self.assertEqual(result['groups']['choice']['point'],.5)
        self.assertEqual(result['groups']['choice']['worlds'],2)
        for invalid in (b[:-1],b+[b[0]],[{**r,'target_sha256':'bad'} for r in b]):
            with self.assertRaises(ValueError):
                compare_labels(a,invalid,expectations=LABEL_EXPECTATIONS,resamples=100)

    def test_label_primary_stays_label_weighted_world_clustered(self):
        a=[label(0,1,False),label(1,1,False),label(2,2,True)]
        b=control_rows([{**r,'correct':False} for r in a])
        metric=compare_labels(a,b,expectations=LABEL_EXPECTATIONS,resamples=100)['groups']['choice']
        self.assertEqual(metric['point'],1/3)
        self.assertEqual(metric['weighting'],'label_weighted')
        self.assertEqual(metric['cluster_unit'],'source_world')

    def test_label_split_inference_and_manifest_source_reject_before_ci(self):
        a=[label(i,i//2,True) for i in range(4)]
        for field,invalid in (('split','FIT'),('split','VALIDATION'),('inference','batched_cpu'),
                              ('data_identity',digest('other-synthetic-manifest')),
                              ('source_identity',digest('other-synthetic-scientific-source'))):
            b=control_rows(deepcopy(a))
            b[-1][field]=invalid
            with self.subTest(field=field),patch('sts_rl.fresh_statistics.clustered_mean',
                                                side_effect=AssertionError('CI started')):
                with self.assertRaises(ValueError):
                    compare_labels(a,b,expectations=LABEL_EXPECTATIONS,resamples=100)
        for field in ('split','inference','data_identity','source_identity'):
            for absent in (True,False):
                b=control_rows(deepcopy(a))
                if absent:
                    b[-1].pop(field)
                else:
                    b[-1][field]=''
                with self.subTest(field=field,absent=absent),self.assertRaises(ValueError):
                    compare_labels(a,b,expectations=LABEL_EXPECTATIONS,resamples=100)
        # Matching pairs from mixed manifests/sources are also invalid.
        for field in ('data_identity','source_identity'):
            b=deepcopy(a)
            b[-1][field]=digest('other-synthetic-identity')
            with self.subTest(mixed=field),self.assertRaises(ValueError):
                compare_labels(b,control_rows(b),expectations=LABEL_EXPECTATIONS,resamples=100)
        for field,value in (('split','VALIDATION'),('inference','batched_cpu')):
            b=[{**r,field:value} for r in a]
            with self.subTest(both=field),self.assertRaises(ValueError):
                compare_labels(b,control_rows(b),expectations=LABEL_EXPECTATIONS,resamples=100)

    def test_exact_games_no_discordance_missing_duplicate_fault_reject(self):
        seeds=[92000000000,92000000001,92000000002]
        a=[game(seed,False) for seed in seeds]
        b=champion_rows(a)
        result=compare_games(a,b,seeds,expectations=GAME_EXPECTATIONS,resamples=100)
        self.assertEqual(result['win']['point'],0.)
        self.assertEqual(result['win']['exact_mcnemar_p'],1.)
        self.assertTrue(result['win']['zero_discordance'])
        self.assertTrue(result['win']['bootstrap_degenerate'])
        self.assertEqual(result['win']['gate'],'PASS')
        for invalid in (b[:-1],b+[b[0]],[{**r,'status':'fault'} for r in b],b+[game(92000000003,True,arm='champion')]):
            with self.assertRaises(ValueError):
                compare_games(a,invalid,seeds,expectations=GAME_EXPECTATIONS,resamples=100)
        with self.assertRaises(ValueError):
            compare_games(a,b,seeds+[seeds[0]],expectations=GAME_EXPECTATIONS,resamples=100)

    def test_game_boss_conditional_cohorts_and_explicit_denominators(self):
        boss=lambda hp,p: {'2':dict(reach=True,entry_hp=hp,**{'pass':p})}
        a=[game(1,True,30,boss(50,True)),game(2,False,20),game(3,False,25,boss(30,False))]
        b=champion_rows([game(1,False,20),game(2,False,30,boss(20,False)),game(3,False,20,boss(40,True))])
        result=compare_games(a,b,[1,2,3],expectations=GAME_EXPECTATIONS,resamples=100)
        self.assertAlmostEqual(result['floor']['point'],5/3)
        self.assertEqual(result['win']['student_only'],1)
        boss2=result['bosses']['2']
        self.assertEqual(boss2['student_reached'],2)
        self.assertEqual(boss2['champion_reached'],2)
        self.assertEqual(boss2['entry_hp_conditional']['point'],10)
        self.assertEqual(boss2['cohort_denominator'],3)
        json.dumps(result,allow_nan=False)

    def test_games_provenance_and_explicit_valid_bosses_reject_before_ci(self):
        a=[game(1,False),game(2,False)]
        invalid=[]
        for field in ('expected_provenance','bosses'):
            b=champion_rows(deepcopy(a))
            b[-1].pop(field)
            invalid.append(b)
        for provenance in ({},None,[],{'fixture':'different-synthetic-source'}):
            invalid.append(champion_rows([a[0],{**a[1],'expected_provenance':provenance}]))
        boss=dict(reach=True,entry_hp=30,**{'pass':False})
        for bosses in (None,[],{'4':boss},{'1':None},{'1':[]},{'1':{}},
                       *({'1':{**boss,field:value}} for field,value in (
                           ('reach',1),('reach','yes'),('reach',False),('pass',1),('pass',None),
                           ('entry_hp',None),('entry_hp',float('nan')),('entry_hp',float('inf')),
                           ('entry_hp',-1),('entry_hp','30'),('entry_hp',True))),
                       {'1':{'reach':True,'pass':False}}):
            invalid.append(champion_rows([a[0],{**a[1],'bosses':bosses}]))
        with patch('sts_rl.fresh_statistics.bootstrap_components',side_effect=AssertionError('CI started')):
            for b in invalid:
                with self.subTest(b=b[-1]),self.assertRaises(ValueError):
                    compare_games(a,b,[1,2],expectations=GAME_EXPECTATIONS,resamples=100)
        self.assertEqual(compare_games(a,champion_rows(a),[1,2],expectations=GAME_EXPECTATIONS,resamples=100)['bosses']['1']['student_reached'],0)

    def test_panel_cluster_source_world_full_grid_and_faults(self):
        grid=[dict(world_seed=w,start_id=s,salt=t,group='ELITE2') for w in (1,2) for s in (0,1) for t in (0,1)]
        records=[{**item,'policy':p,'win':p=='replacement','hp_lost':10 if p=='replacement' else 20,
                  'first_combat_hash':f"{item['world_seed']}:{item['start_id']}"}
                 for item in grid for p in ('champion','replacement')]
        frozen={'grid':grid,'coverage_sufficient':True}
        result=compare_panel(records,frozen,resamples=100)
        metric=result['groups']['ELITE2:resampled']['hp_lost']
        self.assertEqual(metric['point'],-10)
        self.assertEqual(metric['worlds'],2)
        self.assertEqual(metric['rows'],8)
        self.assertTrue(result['grid_complete'])
        self.assertFalse(compare_panel(records[:-1],frozen,resamples=100)['grid_complete'])
        faulty=deepcopy(records)
        faulty[0].update(fault='RPC',win=None,hp_lost=None)
        result=compare_panel(faulty,frozen,resamples=100)
        self.assertFalse(result['confidence_claims_allowed'])
        self.assertEqual(result['paired_fights'],7)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            compare_panel(records+[records[0]],frozen,resamples=100)
        with self.assertRaisesRegex(ValueError,'lacks'):
            compare_panel(records,{'grid':grid[:-1],'salts':[0,1]},resamples=100)

    def test_salt_replication_does_not_inflate_effective_worlds(self):
        grid=[dict(world_seed=w,start_id=0,salt=t,group='BOSS3') for w in (1,2) for t in range(20)]
        records=[{**item,'policy':p,'win':(p=='replacement')==(item['world_seed']==1),
                  'hp_lost':0,'first_combat_hash':str(item['world_seed'])}
                 for item in grid for p in ('champion','replacement')]
        result=compare_panel(records,{'grid':grid,'salts':list(range(20))},resamples=1000)
        win=result['groups']['BOSS3:resampled']['win']
        self.assertEqual(win['worlds'],2)
        self.assertEqual(win['rows'],40)
        self.assertEqual(win['point'],0)
        self.assertGreater(win['se'],.65)

    def test_panel_equal_world_point_matches_collection_with_unequal_starts(self):
        grid=[dict(world_seed=w,start_id=s,salt=t,group='ELITE2')
              for w,starts in ((1,range(3)),(2,range(1))) for s in starts for t in (-1,0,1,2)]
        records=[{**item,'policy':p,'win':(p=='replacement')==(item['world_seed']==1),
                  'hp_lost':(10 if item['world_seed']==1 else 22) if p=='replacement' else 20,
                  'first_combat_hash':f"synthetic:{item['world_seed']}:{item['start_id']}"}
                 for item in grid for p in ('champion','replacement')]
        frozen={'grid':grid,'salts':[-1,0,1,2],'coverage_sufficient':True}
        result=compare_panel(records,frozen,resamples=1000)
        collection=paired_summary(records,frozen)
        self.assertEqual(result['weighting'],'equal_world')
        for suffix in (':unsalted',':resampled'):
            for metric,point in (('win',0.),('hp_lost',-4.)):
                for group in ('ELITE2','all'):
                    actual=result['groups'][group+suffix][metric]
                    self.assertEqual(actual['point'],point)
                    self.assertEqual(actual['point'],collection['groups']['ELITE2'+suffix][metric]['paired_difference'])
                    self.assertEqual(actual['weighting'],'equal_world')
                    self.assertEqual(actual['worlds'],2)
                    self.assertEqual(actual['rows'],4 if suffix==':unsalted' else 12)
        self.assertGreater(result['groups']['ELITE2:resampled']['win']['se'],.65)

    def test_consistency_cli_rejects_empty_sharded_group_root_without_output(self):
        spec=importlib.util.spec_from_file_location('compare_cli',Path(__file__).resolve().parents[2]/'scripts/rl/fresh_compare.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shard=root/'group'/'shard-0'/'worlds'
            shard.mkdir(parents=True)
            seed=92000000000
            atomic_json(shard/f'{seed}.json.gz',dict(world_seed=seed,labels=[
                {'self_consistency':{'fixture':'synthetic'}}]),compressed=True)
            with patch.object(module,'teacher_consistency',side_effect=AssertionError('empty success')):
                with self.assertRaisesRegex(ValueError,'pass actual shard directories'):
                    module.main(['consistency','--student',str(root/'group'),'--out',str(root/'result.json')])
            self.assertFalse((root/'result.json').exists())
            self.assertEqual(len(module.read_rows(shard.parent,'consistency')),1)
            empty=root/'empty.json'
            empty.write_text('[]')
            error=StringIO()
            with redirect_stderr(error),self.assertRaises(SystemExit):
                module.main(['consistency','--student',str(empty),'--out',str(root/'result.json')])
            self.assertIn('no repeat-query rows',error.getvalue())
            self.assertFalse((root/'result.json').exists())

    def test_repeat_query_requires_disjoint_draw_and_rng_seeds(self):
        rows=[]
        for seed in (1,2):
            r=row(combat_observation(),seed,{}, {})
            r['self_consistency']={'search':{'search_seed':77,'samples':[1025,1026,1027,1028],
                'draw_order_seeds':[1025,1026,1027,1028], 'effective_search_seeds':[1102,1103,1104,1105],
                'effective_rng_seeds':[78718,78719,78720,78721]},'votes':{'10':4},'use_for_training':False}
            rows.append(r)
        self.assertEqual(teacher_consistency(rows,resamples=100)['groups']['all']['point'],1)
        bad=deepcopy(rows)
        bad[0]['self_consistency']['search']['samples']=[1,2,3,4]
        with self.assertRaisesRegex(ValueError,'independent'):
            teacher_consistency(bad,resamples=100)
        bad=deepcopy(rows)
        bad[0]['self_consistency']['search']['effective_rng_seeds']=[5046]
        with self.assertRaisesRegex(ValueError,'independent'):
            teacher_consistency(bad,resamples=100)

    def test_compare_reader_pretty_json_releases_ordinary_histories(self):
        spec=importlib.util.spec_from_file_location('compare_reader',Path(__file__).resolve().parents[2]/'scripts/rl/fresh_compare.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.json'
            value={'records':[dict(world_seed=92000000000,labels=[{'observation':'ordinary'},
                     {'observation':'repeat','self_consistency':{'votes':{'0':1}}}])]}
            path.write_text(json.dumps(value,indent=2))
            with self.assertRaisesRegex(ValueError,'receipted'):
                module.read_rows(path,'games')
            self.assertEqual([r['observation'] for r in module.read_rows(path,'consistency')],['repeat'])

    def test_repeat_query_counts_recorded_abstentions_without_inventing_targets(self):
        original=row(combat_observation(),1,{}, {})
        original.update(label=None,votes={})
        original['self_consistency']={'search':{'search_seed':77,'samples':[1025,1026,1027,1028],
            'draw_order_seeds':[1025,1026,1027,1028], 'effective_search_seeds':[1102,1103,1104,1105],
            'effective_rng_seeds':[78718,78719,78720,78721]},'votes':{},'use_for_training':False}
        result=teacher_consistency([original],resamples=100)
        self.assertEqual(result['rows'],0)
        self.assertEqual(result['abstained_pairs'],1)
        self.assertEqual(result['groups'],{})

if __name__=='__main__':
    unittest.main()
