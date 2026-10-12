from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.report_nonllm_campaign import classify_panel,compare_panels,collect_campaign


class CampaignReportReadinessTests(unittest.TestCase):
    def panel(self):
        return {'command':'evaluate','rounds':[{'task_ids':['a','b','c'],
                'episodes':[{'task_id':'a','world_seed':1,'task_kind':'run','terminated':True,'outcome':'act_clear'},
                            {'task_id':'b','world_seed':2,'task_kind':'run','truncated':True,'termination_reason':'task_horizon'}],
                'faults':[{'task_id':'c','world_seed':3,'task_kind':'run','error':'SimulatorTimeout'}]}]}

    def test_cohort_complete_can_have_fault_and_predeclared_horizon_censor(self):
        p=classify_panel(self.panel());s=p['summary']['run']
        self.assertTrue(p['cohort_complete']);self.assertEqual(p['requested'],3)
        self.assertEqual(s['successes'],1);self.assertEqual(s['losses'],0)
        self.assertEqual(s['faults'],1);self.assertEqual(s['truncated'],1)

    def test_duplicate_missing_and_budget_cases_are_incomplete(self):
        m=self.panel();m['rounds'][0]['episodes'][1]['task_id']='a'
        self.assertFalse(classify_panel(m)['cohort_complete'])
        m=self.panel();m['rounds'][0]['episodes'][1]['termination_reason']='collector_budget'
        self.assertFalse(classify_panel(m)['cohort_complete'])

    def test_requested_mismatch_does_not_generate_spurious_paired_interval(self):
        a=classify_panel(self.panel());m=self.panel();m['rounds'][0]['task_ids'][1]='d';m['rounds'][0]['episodes'][1]['task_id']='d'
        result=compare_panels(a,classify_panel(m))
        self.assertFalse(result['eligible']);self.assertNotIn('paired_success_difference',result)

    def test_default_development_never_reads_final_metric_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'campaign.json').write_text(json.dumps({'selected':{},'commands':[]}))
            (root/'final_run_act3.metrics.json').write_text('deliberately invalid JSON; must not read')
            self.assertEqual(collect_campaign(root)['panels'],[])
            with self.assertRaises(ValueError):collect_campaign(root,stage='final')

    def test_final_scope_cannot_be_invented_from_available_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'campaign.json').write_text(json.dumps({'selected':{},'commands':[]}))
            with self.assertRaisesRegex(ValueError,'final_required_panels'):collect_campaign(root,stage='final',allow_final=True)


class FinalEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        from sts_policy.final_freeze import capture_final_freeze
        from sts_policy.campaign_identity import condition_signature
        core=root/'env.py';core.write_text('public interface only\n')
        checkpoint=root/'model.pt';checkpoint.write_bytes(b'synthetic model bytes; never loaded')
        native=root/'native';native.mkdir();module=native/'slaythespire.synthetic.so';module.write_bytes(b'native placeholder')
        manifest={'module_file':module.name,'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest()}
        (native/'build_manifest.json').write_text(json.dumps(manifest))
        definitions={'world_start':10000,'episodes':1,'horizon':2048,'simulations':200,
                     'panels':{'run_act1':{'task_ids':['a']}}}
        freeze=capture_final_freeze({'shared':checkpoint},{'src/sts_env/env.py':core},native,generated_definitions=definitions)
        config={'split':'final','seed':777,'task_kind':'run','max_act':1,'horizon':2048,'simulations':200,'checkpoint':str(checkpoint),'gamma':1.0,'reward_mode':'potential','terminal_hp_weight':0.0,'progress_floor_weight':0.05,'goal_act_clear':True}
        command=['python','train.py','evaluate','--split','final','--seed','777','--task-kind','run',
                 '--max-act','1','--horizon','2048','--simulations','200','--world-start','10000','--episodes','1','--checkpoint',str(checkpoint)]
        environment={'native_module_sha256':manifest['module_sha256'],'sourced_env_manifest':manifest,
                     'sourced_env_manifest_sha256':freeze['backend']['manifest']['sha256'],
                     'python_interface_sha256':hashlib.sha256(b'env.py\0'+core.read_bytes()+b'\0').hexdigest()}
        metrics={'command':'evaluate','config':config,'training_environment':environment,
                 'rounds':[{'task_ids':['a'],'episodes':[{'task_id':'a','world_seed':1,'task_kind':'run','outcome':'act_clear','terminated':True}],'faults':[]}]}
        path=root/'final_run_act1.metrics.json';path.write_text(json.dumps(metrics))
        record={'name':'final_run_act1','status':'complete','command':command,'condition_signature':condition_signature(command),
                'metrics_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'finished':1}
        state={'native_build':str(native),'frozen_identities':freeze,'final_required_panels':['final_run_act1'],'commands':[record]}
        return state,metrics

    def write(self,root,state,metrics,refresh=True):
        path=root/'final_run_act1.metrics.json';path.write_text(json.dumps(metrics))
        if refresh:state['commands'][0]['metrics_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'campaign.json').write_text(json.dumps(state))
        return collect_campaign(root,stage='final',allow_final=True)

    def test_verified_complete_synthetic_final_is_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state,metrics=self.fixture(root)
            self.assertTrue(self.write(root,state,metrics)['final_report_complete'])

    def test_partial_command_or_missing_exact_journal_sha_never_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state,metrics=self.fixture(root)
            for status in ('partial','running',None):
                state['commands'][0]['status']=status
                self.assertFalse(self.write(root,state,metrics)['final_report_complete'])
            state['commands'][0]['status']='complete'
            state['commands'][0].pop('metrics_sha256')
            self.assertFalse(self.write(root,state,metrics,refresh=False)['final_report_complete'])
            state['commands'][0]['metrics_sha256']='stale'
            self.assertFalse(self.write(root,state,metrics,refresh=False)['final_report_complete'])

    def test_self_consistent_but_different_requested_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state,metrics=self.fixture(root)
            metrics['rounds'][0]['task_ids']=['b'];metrics['rounds'][0]['episodes'][0]['task_id']='b'
            report=self.write(root,state,metrics)
            self.assertTrue(report['panels'][0]['cohort_complete']);self.assertFalse(report['final_report_complete'])
            self.assertIn('requested task identities differ from frozen cohort',report['panels'][0]['readiness_errors'])

    def test_other_policy_and_stale_input_signature_rejected(self):
        from sts_policy.campaign_identity import condition_signature
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state,metrics=self.fixture(root)
            state['commands'][0]['condition_signature']['digest']='stale'
            self.assertFalse(self.write(root,state,metrics)['final_report_complete'])
            other=root/'other.pt';other.write_bytes(b'other model')
            command=state['commands'][0]['command'];command[-1]=str(other)
            state['commands'][0]['condition_signature']=condition_signature(command)
            metrics['config']['checkpoint']=str(other)
            self.assertFalse(self.write(root,state,metrics)['final_report_complete'])

    def test_missing_wrong_backend_interface_or_changed_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state,original=self.fixture(root)
            for key in ('native_module_sha256','sourced_env_manifest_sha256','python_interface_sha256'):
                metrics=deepcopy(original);metrics['training_environment'].pop(key)
                self.assertFalse(self.write(root,state,metrics)['final_report_complete'])
                metrics=deepcopy(original);metrics['training_environment'][key]='wrong'
                self.assertFalse(self.write(root,state,metrics)['final_report_complete'])
            (root/'env.py').write_text('mutated interface')
            report=self.write(root,state,original)
            self.assertFalse(report['freeze']['verified']);self.assertFalse(report['final_report_complete'])

    def test_baseline_bank_cohort_reconstructed_from_frozen_task_bytes(self):
        from sts_policy.curriculum import TaskSpec,world_split
        from sts_policy.final_freeze import capture_final_freeze
        from sts_policy.campaign_identity import condition_signature
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state,metrics=self.fixture(root)
            bank=root/'battles.jsonl'
            world=next(seed for seed in range(1000,1100) if world_split(seed)=='final')
            task=TaskSpec(seed=world,horizon=256,max_act=1,battle_simulations=200)
            bank.write_text(json.dumps(task.to_dict())+'\n')
            frozen=state['frozen_identities']
            state['frozen_identities']=capture_final_freeze({'shared':root/'model.pt'},
                {'src/sts_env/env.py':root/'env.py','act1_battles':bank},root/'native',generated_definitions=frozen['generated_definitions'])
            record=state['commands'][0];record['name']='final_baseline_random_battle'
            record['command']=['python','baseline.py','--baseline','random','--kind','battle','--split','final','--seed','777',
                '--max-act','1','--horizon','256','--simulations','200','--episodes','160','--tasks',str(bank)]
            record['condition_signature']=condition_signature(record['command'])
            metrics['config']={'baseline':'random','kind':'battle','split':'final','seed':777,'max_act':1,'horizon':256,'simulations':200}
            task_id=hashlib.sha256(json.dumps(task.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            metrics['rounds'][0]['task_ids']=[task_id];metrics['rounds'][0]['episodes'][0].update(task_id=task_id,world_seed=world,task_kind='battle',outcome='battle_victory')
            state['final_required_panels']=[record['name']]
            path=root/(record['name']+'.metrics.json');path.write_text(json.dumps(metrics))
            record['metrics_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'campaign.json').write_text(json.dumps(state))
            self.assertTrue(collect_campaign(root,stage='final',allow_final=True)['final_report_complete'])
            # A self-consistent result from another prefix still violates the frozen bank.
            metrics['rounds'][0]['task_ids']=['other'];metrics['rounds'][0]['episodes'][0]['task_id']='other'
            path.write_text(json.dumps(metrics));record['metrics_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'campaign.json').write_text(json.dumps(state))
            self.assertFalse(collect_campaign(root,stage='final',allow_final=True)['final_report_complete'])

    def test_actual_campaign_final_act1_commands_use_frozen_default_max_act(self):
        import os
        from dataclasses import replace
        from unittest.mock import patch
        from scripts import run_nonllm_campaign as scheduler
        from scripts.report_nonllm_campaign import final_evidence
        from sts_policy.campaign_identity import condition_signature
        from sts_policy.curriculum import TaskSpec,world_split
        from tests.unit.test_final_campaign_core_freeze import FinalCampaignCoreFreezeTests
        class CapturedInitialFinals(Exception):pass
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            checkpoints=FinalCampaignCoreFreezeTests().fixture(root)
            world=next(seed for seed in range(1000,1100) if world_split(seed)=='final')
            bank_task=TaskSpec(seed=world)
            (root/'bank/battles.jsonl').write_text(json.dumps(bank_task.to_dict())+'\n')
            captured={}
            def capture(name,command,*args,**kwargs):
                if name=='final_run_act3':raise CapturedInitialFinals
                captured[name]=list(map(str,command))
            with patch.object(scheduler,'ROOT',root),patch.object(scheduler,'BANK',root/'bank'),\
                 patch.object(scheduler,'__file__',str(root/'scripts/run_nonllm_campaign.py')),\
                 patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':str(root/'backend')}):
                runner=scheduler.Campaign(root/'output')
                runner.state['active_panel_directory']=str(root/'panels')
                with patch.object(runner,'run',side_effect=capture):
                    with self.assertRaises(CapturedInitialFinals):runner.final(**checkpoints)
            freeze=runner.state['frozen_identities']
            interface=hashlib.sha256()
            for role,row in sorted(freeze['panel_files'].items()):
                if role.startswith('src/sts_env/') and role.endswith('.py'):
                    interface.update(role.removeprefix('src/sts_env/').encode());interface.update(b'\0')
                    interface.update(Path(row['path']).read_bytes());interface.update(b'\0')
            environment={'native_module_sha256':freeze['backend']['module']['sha256'],
                         'sourced_env_manifest':freeze['backend']['provenance'],
                         'sourced_env_manifest_sha256':freeze['backend']['manifest']['sha256'],
                         'python_interface_sha256':interface.hexdigest()}
            self.assertEqual(set(captured),{'final_battle_act1','final_ooc_act1','final_run_act1'})
            for name,command in captured.items():
                self.assertNotIn('--max-act',command)
                kind=command[command.index('--task-kind')+1]
                horizon=int(command[command.index('--horizon')+1])
                if kind=='battle':
                    task=replace(bank_task,horizon=horizon,max_act=1,battle_simulations=200)
                    ids=[hashlib.sha256(json.dumps(task.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()]
                else:ids=freeze['generated_definitions']['panels'][kind+'_act1']['task_ids']
                config={'split':'final','seed':777,'task_kind':kind,'max_act':1,'horizon':horizon,'simulations':200,
                        'checkpoint':command[command.index('--checkpoint')+1],'gamma':1.0,'reward_mode':'potential',
                        'terminal_hp_weight':0.0,'progress_floor_weight':0.05,'goal_act_clear':True}
                metrics={'command':'evaluate','config':config,'training_environment':environment,
                         'rounds':[{'task_ids':ids,'episodes':[{'task_id':task_id,'world_seed':index,'task_kind':kind,
                                    'terminated':True,'outcome':'battle_victory' if kind=='battle' else 'act_clear'}
                                    for index,task_id in enumerate(ids)],'faults':[]}]}
                record={'status':'complete','command':command,'condition_signature':condition_signature(command)}
                evidence=final_evidence(name,metrics,record,classify_panel(metrics),freeze)
                self.assertTrue(evidence['final_evidence_verified'],evidence['readiness_errors'])
                metrics['config']['max_act']=3
                self.assertFalse(final_evidence(name,metrics,record,classify_panel(metrics),freeze)['final_evidence_verified'])
                metrics['config']['max_act']=1
                wrong=command+['--max-act','3'];record.update(command=wrong,condition_signature=condition_signature(wrong))
                self.assertFalse(final_evidence(name,metrics,record,classify_panel(metrics),freeze)['final_evidence_verified'])
