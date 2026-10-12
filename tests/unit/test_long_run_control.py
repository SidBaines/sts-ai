from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sts_policy import long_run_control as control
from sts_policy.campaign_identity import condition_signature
from sts_policy.curriculum import TaskSpec
from sts_policy.rollout_training import task_identity
from sts_env.provenance import interface_identity


class LongRunControlTests(unittest.TestCase):
    def fixture(self, directory, **kwargs):
        root=Path(directory)/'repo';root.mkdir();(root/'docs').mkdir();(root/'scripts').mkdir()
        (root/'scripts/train_nonllm.py').write_text('# test-source')
        (root/'scripts/run_long_curriculum.py').write_text('# test-controller')
        build=root/'build';build.mkdir();module=build/'slaythespire.fake.so';module.write_bytes(b'fake native')
        native={'module_file':module.name,'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest()}
        (build/'build_manifest.json').write_text(json.dumps(native))
        checkpoint=root/'initial.pt';checkpoint.write_bytes(b'initial actor')
        cfg=control.LongRunConfig(checkpoint=str(checkpoint),backend_build=str(build),pool_bank=str(root/'pool'),pool_panels=str(root/'panels'),deadline=time.time()+3600,device='cpu',prevent_sleep=False,**kwargs)
        controller=control.LongRunController(root,root/'output',cfg);self.addCleanup(controller.close)
        return controller

    def quality(self, clears=0, floor=28., *, fault=False):
        return {'complete':True,'cohort':['a','b'],'clears':clears,'requested':2,'fault_ids':['a'] if fault else [],'faults':int(fault),'truncated':0,'floor_mean':floor}

    def test_zero_win_floor_regression_rolls_back_and_incoherent_faults_do_not(self):
        anchor=self.quality(floor=29.65);ret=self.quality(2)
        self.assertEqual(control.selection_decision(self.quality(floor=24.92),anchor,ret,ret)['action'],'rollback')
        self.assertEqual(control.selection_decision(self.quality(floor=24.92,fault=True),anchor,ret,ret)['action'],'hold')
        candidate=self.quality(1,40);candidate['truncated']=1
        self.assertEqual(control.selection_decision(candidate,anchor,ret,ret)['action'],'hold')

    def test_retention_reference_threshold_and_cohort_identity(self):
        anchor=self.quality(0,28);ret=self.quality(10,17)
        self.assertEqual(control.selection_decision(self.quality(1,30),anchor,self.quality(7),ret)['action'],'rollback')
        self.assertEqual(control.selection_decision(self.quality(1,30),anchor,self.quality(8),ret)['action'],'promote')
        altered=self.quality(2,50);altered['cohort']=['a','c']
        self.assertEqual(control.selection_decision(altered,anchor,ret,ret)['action'],'hold')

    def test_goal_change_and_mixture_change_reset_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d)
            self.assertFalse(control.optimizer_compatible(control.goal_signature(c.config,3),c.config,2))
            first=control.STAGES[0];second=control.STAGES[1]
            prior=control.goal_signature(c.config,2,first)
            self.assertTrue(control.optimizer_compatible(prior,c.config,2,first))
            self.assertFalse(control.optimizer_compatible(prior,c.config,2,second))

    def test_weak_ac_discharging_battery_pauses_and_only_safe_charge_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d)
            p={'on_ac':True,'battery_percent':25.,'discharging':True}
            self.assertEqual(control.resource_pause(c.config,c.output,power=p,free_bytes=20*1024**3),'battery_below_threshold')
            self.assertEqual(control.resource_pause(c.config,c.output,power=p,free_bytes=5*1024**3),'disk_below_minimum')
            charged={**p,'discharging':False}
            self.assertIsNone(control.resource_pause(c.config,c.output,power=charged,free_bytes=20*1024**3,previously_battery_paused=True))

    def test_duplicate_controller_cannot_mutate_running_journal(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);before=c.journal.read_bytes()
            with self.assertRaises(RuntimeError):control.LongRunController(c.root,c.output,c.config)
            self.assertEqual(c.journal.read_bytes(),before)

    def test_restart_preserves_deadline_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);deadline=c.config.deadline;c.close()
            new=control.LongRunController(c.root,c.output,c.config);self.addCleanup(new.close)
            self.assertEqual(new.state['allocation']['authorized_seconds'],c.state['allocation']['authorized_seconds']);self.assertEqual(new.state['allocation']['used_seconds'],0);new.close()
            resumed=control.LongRunController(c.root,c.output,replace(c.config,deadline=deadline+1000));self.addCleanup(resumed.close)
            self.assertEqual(resumed.state['allocation']['authorized_seconds'],new.state['allocation']['authorized_seconds'])

    def test_retention_preserves_protected_recent_and_foreign_files(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);owned=c.output/'checkpoints';owned.mkdir()
            paths=[]
            for i in range(8):
                path=owned/f'p{i}.pt';path.write_bytes(str(i).encode());os.utime(path,(i+1,i+1));paths.append(path)
            foreign=c.root/'foreign.pt';foreign.write_bytes(b'not owned')
            c.state.update(best_checkpoint=str(paths[0]),current_checkpoint=str(paths[1]),last_checkpoint=str(paths[2]),hourly_anchors={'hour':str(paths[3])},stages={'phase':{'best_checkpoint':str(paths[4])}})
            deleted=control.retain_checkpoints(c.output,c.state)
            self.assertEqual([row['path'] for row in deleted],[str(paths[5].resolve())])
            self.assertTrue(all(path.exists() for path in paths if path!=paths[5]));self.assertTrue(foreign.exists())
            ledger=json.loads(c.journal.read_text())['deletions'][0]
            self.assertEqual(ledger['sha256'],hashlib.sha256(b'5').hexdigest());self.assertEqual(ledger['status'],'deleted')

    def test_symlinked_checkpoint_directory_never_deletes_external_files(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);other=c.root/'other';other.mkdir();(other/'valuable.pt').write_bytes(b'x');(c.output/'checkpoints').symlink_to(other,target_is_directory=True)
            with self.assertRaises(RuntimeError):control.retain_checkpoints(c.output,c.state)
            self.assertTrue((other/'valuable.pt').exists())

    def test_deadline_prevents_child_launch(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);c.state['allocation']['used_seconds']=c.state['allocation']['authorized_seconds']
            command=['python','unused','--metrics',str(c.output/'never.json')]
            with patch.object(control.subprocess,'Popen') as popen:
                metrics,record=c.run_job(command,kind='eval',budget=10,stage_end=time.time()+100)
            popen.assert_not_called();self.assertIsNone(metrics);self.assertEqual(record['status'],'insufficient_budget')

    def test_cached_completed_job_verifies_current_inputs_and_metrics_without_popen(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);path=c.output/'m.json';path.write_text('{"done":true}')
            command=['python','unused','--checkpoint',c.config.checkpoint,'--metrics',str(path)]
            c.state['jobs'].append({'status':'complete','signature':c._signature(command),'artifacts':{str(path):control.sha_file(path)}})
            with patch.object(control.subprocess,'Popen') as popen:
                actual,_=c.run_job(command,kind='eval',budget=10,stage_end=time.time()+100)
            self.assertEqual(actual,{'done':True});popen.assert_not_called()
            path.write_text('{"done":false}')
            with self.assertRaises(RuntimeError):c.run_job(command,kind='eval',budget=10,stage_end=time.time()+100)

    def test_published_checkpoint_modification_is_not_a_new_valid_condition(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);path=c.output/'own.pt';path.write_bytes(b'published');c.state['current_checkpoint']=str(path)
            c.state['jobs'].append({'artifacts':{str(path):control.sha_file(path)}})
            c.verify_pins();path.write_bytes(b'modified')
            with self.assertRaisesRegex(RuntimeError,'checkpoint identity'):c.verify_pins()

    def test_command_preserves_approved_defaults_and_fresh_round_capture(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);command=list(map(str,c.command('ppo',Path(c.config.checkpoint),c.root/'tasks',c.output/'cp',max_act=2,seconds=180,seed=11)))
            def flag(key):return command[command.index(key)+1]
            self.assertEqual(flag('--updates'),'1');self.assertEqual(flag('--batch-size'),'1024');self.assertEqual(flag('--learning-rate'),'0.0001');self.assertEqual(flag('--gae-lambda'),'0.95');self.assertEqual(flag('--workers'),'8');self.assertEqual(flag('--threads'),'1')
            self.assertNotIn('--warm-optimizer',command);self.assertIn('--preserve-task-prefixes',command)

    def test_actor_identity_includes_state_encoder_but_ignores_critic_only(self):
        import torch
        payload={'step':1,'model_state':{'state_encoder.weight':torch.zeros(1),'action_encoder.weight':torch.zeros(1),'scorer.weight':torch.zeros(1),'critic.weight':torch.zeros(1)}}
        with patch('sts_policy.training.load_checkpoint',return_value=payload):
            original=control.actor_weight_identity(Path('unused'))
            payload['model_state']['critic.weight'].fill_(1)
            self.assertEqual(control.actor_weight_identity(Path('unused')),original)
            payload['model_state']['state_encoder.weight'].fill_(1)
            self.assertNotEqual(control.actor_weight_identity(Path('unused'))['actor_sha256'],original['actor_sha256'])

    def test_process_birth_whitespace_matches_group_snapshot(self):
        from types import SimpleNamespace
        with patch.object(control.subprocess,'run',return_value=SimpleNamespace(stdout='Mon Nov  1 09:00:00 2026\n')):
            self.assertEqual(control.process_birth(123),'Mon Nov 1 09:00:00 2026')

    def test_resource_resume_status_and_reason_are_real(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);c.state.update(status='paused_resource',pause_reason='battery_below_threshold',battery_paused=True)
            with patch.object(control,'resource_pause',return_value=None):
                self.assertTrue(c.pause_until_safe())
            self.assertEqual(c.state['status'],'running');self.assertNotIn('pause_reason',c.state)
            self.assertEqual(c.state['events'][-1]['type'],'resource_resume')

    def test_protocol_anchors_validate_exact_tasks_signatures_and_native_identity(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d,dev_episodes=1);proto={'checkpoint':{'path':str(Path(c.config.checkpoint).resolve()),'sha256':control.sha_file(c.config.checkpoint)},'panels':{},'anchors':{}}
            for act in (1,2,3):
                task=TaskSpec(seed=200043,kind='run',max_act=act,horizon=2048 if act==1 else 4096)
                tasks=c.root/f'dev{act}.jsonl';tasks.write_text(json.dumps(task.to_dict())+'\n');identity=task_identity(task)
                metric=c.root/f'anchor{act}.json';data={'config':{'checkpoint':c.config.checkpoint,'split':'dev','task_kind':'run','seed':777,'max_act':act,'horizon':task.horizon},'training_environment':{'native_module_sha256':c.backend_identity['provenance']['module_sha256'],'python_interface_sha256':interface_identity()['python_interface_sha256']},'rounds':[{'task_ids':[identity],'cohort_completed':True,'episodes':[{'task_id':identity,'outcome':'defeat','floor':2,'truncated':False}],'faults':[]}]};metric.write_text(json.dumps(data))
                command=['python','train.py','evaluate','--checkpoint',c.config.checkpoint,'--tasks',str(tasks)]
                proto['panels'][f'dev_act{act}']={'path':str(tasks),'sha256':control.sha_file(tasks),'task_ids':[identity],'max_act':act,'horizon':task.horizon}
                proto['anchors'][f'dev_act{act}']={'metrics':str(metric),'metrics_sha256':control.sha_file(metric),'checkpoint_sha256':control.sha_file(c.config.checkpoint),'command':command,'condition_signature':condition_signature(command)}
            path=c.root/'protocol.json';path.write_text(json.dumps(proto));c.config=replace(c.config,protocol=str(path));panels,anchors=c.protocol_inputs()
            self.assertEqual(set(anchors),{'dev_act1','dev_act2','dev_act3'});self.assertEqual(anchors['dev_act2']['requested'],1)
            metric=Path(proto['anchors']['dev_act2']['metrics']);metric.write_text('{}')
            with self.assertRaises((ValueError,KeyError)):c.protocol_inputs()


if __name__=='__main__':unittest.main()
