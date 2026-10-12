from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sts_policy import long_run_v2 as v2
from sts_policy import long_run_control as ops
from sts_policy.training import CHECKPOINT_VERSION


def quality(n=192, wins=2, floor=28., faults=()):
    ids=[f't{i}' for i in range(n)]
    healthy=[i for i in ids if i not in faults]
    outcomes={identity:{'outcome':'victory' if i<wins else 'defeat','floor':floor} for i,identity in enumerate(healthy)}
    return {'complete':True,'cohort':ids,'requested':n,'clears':wins,'faults':len(faults),'fault_ids':list(faults),
            'truncated':0,'unstarted':0,'floor_mean':floor,'outcomes':outcomes}


def config(**kwargs):
    args=dict(checkpoint='/initial',backend_build='/build',pool_bank='/pool',pool_panels='/metadata',deadline=9999999999)
    args.update(kwargs);return v2.V2Config(**args)


class V2DecisionTests(unittest.TestCase):
    def test_champion_pair_fault_tolerance_and_unproven_victory_not_gain(self):
        anchor=quality(faults=('t190','t191'));candidate=quality(wins=3,faults=('t189',))
        decision=v2.confirmed_decision(candidate,anchor,quality(),quality())
        self.assertEqual(decision['action'],'promote');self.assertEqual(decision['paired']['common_valid'],189)
        candidate=quality(wins=2,faults=('t189',));candidate['outcomes']['t190']['outcome']='victory';candidate['clears']=3
        self.assertEqual(v2.confirmed_decision(candidate,anchor,quality(),quality())['action'],'keep_champion')
        candidate=quality(faults=('t0','t1','t2'));anchor=quality(faults=('t3','t4','t5'))
        self.assertFalse(v2.paired_evidence(candidate,anchor)['valid'])
        self.assertEqual(v2.confirmed_decision(candidate,anchor,quality(),quality())['action'],'keep_champion')

    def test_exact_cohort_no_cuts_and_more_than_fault_budget_rejected(self):
        anchor=quality()
        for candidate in (quality(faults=('t0','t1','t2','t3')),dict(quality(),truncated=1),dict(quality(),unstarted=1),dict(quality(),cohort=['wrong'])):
            self.assertFalse(v2.paired_evidence(candidate,anchor)['valid'])
        self.assertEqual(v2.paired_evidence(quality(48,faults=('t0',)),quality(48))['reason'],'fault_rate_above_prespecified_two_percent')

    def test_severe_guard_uses_prechampion_and_fixed_initial_paired_retention(self):
        cfg=config();anchor=quality(wins=2,floor=30);retention=quality(wins=151)
        self.assertIsNone(v2.severe_regression(quality(wins=3,floor=10),anchor,quality(wins=10),retention,cfg))
        self.assertEqual(v2.severe_regression(quality(wins=2,floor=30),anchor,quality(wins=122),retention,cfg),'large_act1_retention_loss_without_full3_win_gain')
        self.assertEqual(v2.severe_regression(quality(wins=2,floor=22),anchor,quality(wins=151),retention,cfg),'large_full3_floor_regression_without_win_gain')
        self.assertIsNone(v2.severe_regression(quality(wins=2,floor=23),anchor,retention,retention,cfg))

    def test_config_fixed_goal_reward_namespace_and_intervals(self):
        cfg=config();self.assertEqual(cfg.train_world_start,100000000);self.assertEqual(cfg.dev_world_start,400000)
        self.assertEqual(v2.V2_STAGE.max_act,3)
        for values in ({'gamma':.99},{'progress_floor_weight':.1},{'train_world_start':1000000},{'severe_cycles':1}):
            with self.assertRaises(ValueError):config(**values)

    def test_same_sha_does_not_count_as_second_severe_confirmation(self):
        controller=v2.V2Controller.__new__(v2.V2Controller);controller.config=config();controller.save=lambda:None
        controller.state={'champion':{'quality':quality(wins=2,floor=30),'screen_quality':quality(48), 'retention':quality(wins=151), 'promotions':0},
            'initial_panels':{'retention_act1':quality(wins=151),'confirm_full3':quality(wins=2)},
            'learning_health':{'severe_streak':0,'last_severe_sha':None},'monitoring':{},'events':[],
            'hourly_anchors':{},'allocation':{'used_seconds':100},'current_checkpoint':'continuous'}
        def pending(sha):return {'source':'/file','sha256':sha,'round':3,'screen':quality(48), 'quality':quality(wins=2,floor=20),'retention':quality(wins=151)}
        with patch.object(v2,'sha_file',return_value='A'):
            controller.state['pending_confirmation']=pending('A');controller.finish_confirmation({})
            self.assertEqual(controller.state['learning_health']['severe_streak'],1)
            controller.state['pending_confirmation']=pending('A');controller.finish_confirmation({})
            self.assertEqual(controller.state['learning_health']['severe_streak'],1)
        with patch.object(v2,'sha_file',return_value='B'):
            controller.state['pending_confirmation']=pending('B')
            with self.assertRaisesRegex(RuntimeError,'Repeated strong regression'):controller.finish_confirmation({})
        self.assertEqual(controller.state['current_checkpoint'],'continuous')

    def test_plateau_warns_once_without_changing_parent_or_reward(self):
        c=v2.V2Controller.__new__(v2.V2Controller);c.config=config();c.save=lambda:None
        c.state={'learning_health':{'plateau_warning':False,'inspection_needed':False},'allocation':{'used_seconds':3601},
                 'champion':{'active_at':3000},'events':[],'current_checkpoint':'warm','best_checkpoint':'champion'}
        c.update_plateau();c.update_plateau()
        self.assertEqual(len(c.state['events']),2);self.assertEqual(c.state['current_checkpoint'],'warm')
        self.assertTrue(c.state['learning_health']['inspection_needed'])

    def test_safe_pause_permission_race_does_not_mask_user_pause_when_group_gone(self):
        c=v2.V2Controller.__new__(v2.V2Controller);c.pause_requested=True
        c.state={'jobs':[{'status':'running','pid':123,'pgid':123,'process_birth':'birth','owned_members':{'123':'birth'},'metrics':'/absent','checkpoint':None}], 'active_job':0}
        with patch.object(v2,'stop_owned_group',side_effect=PermissionError),patch.object(v2,'group_members',return_value={}):
            c.safe_owned_cleanup()
        self.assertEqual(c.state['jobs'][0]['status'],'operator_interrupted');self.assertNotIn('active_job',c.state)
        c.state['jobs'][0]['status']='running'
        with patch.object(v2,'stop_owned_group',side_effect=PermissionError),patch.object(v2,'group_members',return_value={'123':'birth'}):
            with self.assertRaisesRegex(RuntimeError,'remains alive'):c.safe_owned_cleanup()

    def test_implicit_attention_resume_rejected_before_any_jobs(self):
        c=v2.V2Controller.__new__(v2.V2Controller);c.state={'status':'attention_required'}
        with self.assertRaisesRegex(RuntimeError,'acknowledge-attention'):c.execute()

    def test_bad_paired_floors_and_unresolved_outcome_cannot_promote(self):
        for bad in (float('nan'),float('inf'),True):
            candidate=quality(wins=3);candidate['outcomes']['t0']['floor']=bad
            self.assertFalse(v2.paired_evidence(candidate,quality())['valid'])
        candidate=quality();candidate['outcomes']['t0']['outcome']='unfinished'
        self.assertEqual(v2.paired_evidence(candidate,quality())['reason'],'unresolved_terminal_outcome')

    def test_single_severe_retention_flag_blocks_floor_promotion_without_learner_reset(self):
        c=v2.V2Controller.__new__(v2.V2Controller);c.config=config();c.save=lambda:None
        c.state={'champion':{'checkpoint':'trusted','quality':quality(wins=2,floor=28),'screen_quality':quality(48),
                            'retention':quality(wins=151),'promotions':0},
                 'initial_panels':{'retention_act1':quality(wins=151),'confirm_full3':quality()},
                 'learning_health':{'severe_streak':0,'last_severe_sha':None},'monitoring':{},'events':[],
                 'hourly_anchors':{},'allocation':{'used_seconds':100},'current_checkpoint':'warm', 'best_checkpoint':'trusted',
                 'pending_confirmation':{'source':'/candidate','sha256':'A','round':4,'screen':quality(48),
                    'quality':quality(wins=2,floor=32),'retention':quality(wins=122)}}
        with patch.object(v2,'sha_file',return_value='A'): c.finish_confirmation({})
        self.assertEqual(c.state['best_checkpoint'],'trusted');self.assertEqual(c.state['current_checkpoint'],'warm')
        self.assertEqual(c.state['champion']['promotions'],0);self.assertEqual(c.state['learning_health']['severe_streak'],1)
        self.assertEqual(c.state['events'][-1]['decision']['reason'],'promotion_blocked_by_classified_severe_regression')

    def test_eval_retries_expected_pauses_do_not_consume_failure_budget(self):
        rows=[{'status':'operator_interrupted'}, {'status':'interrupted','stop_reason':'resource:battery_below_threshold'},
              {'status':'interrupted','stop_reason':'active_budget_or_session_limit'}]
        self.assertEqual(v2.evaluation_attempt_policy(rows,1),3)
        rows.append({'status':'failed'});self.assertEqual(v2.evaluation_attempt_policy(rows,1),4)
        rows.append({'status':'interrupted','stop_reason':'child_stall'})
        with self.assertRaisesRegex(RuntimeError,'genuine evaluation failures'):v2.evaluation_attempt_policy(rows,1)

    def test_two_paused_confirmation_cycles_resume_promote_without_parent_or_budget_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'learner.pt';source.write_bytes(b'completed learner')
            c=v2.V2Controller.__new__(v2.V2Controller);c.config=config();c.save=lambda:None
            c.state={'champion':{'checkpoint':'original','quality':quality(wins=2),'screen_quality':quality(48),'retention':quality(wins=151),'promotions':0},
                'initial_panels':{'retention_act1':quality(wins=151),'confirm_full3':quality(wins=2)},
                'learning_health':{'severe_streak':0,'last_severe_sha':None},'monitoring':{},'events':[],
                'hourly_anchors':{'pending_confirmation':str(source)},'allocation':{'used_seconds':900},
                'current_checkpoint':str(source),'best_checkpoint':'original',
                'pending_confirmation':{'source':str(source),'sha256':v2.sha_file(source),'round':8,'screen':quality(48)}}
            interruptions=[None,None]
            def evaluate(label,cp,panels):
                if interruptions:return interruptions.pop()
                return quality(wins=3) if label=='confirm_full3' else quality(wins=151)
            c.evaluate_panel=evaluate
            for _ in range(2):
                c.finish_confirmation({});self.assertIn('pending_confirmation',c.state)
                self.assertEqual(c.state['best_checkpoint'],'original')
            c.finish_confirmation({})
            self.assertEqual(c.state['best_checkpoint'],str(source));self.assertEqual(c.state['current_checkpoint'],str(source))
            self.assertEqual(c.state['allocation']['used_seconds'],900);self.assertNotIn('pending_confirmation',c.state)

    def test_controller_eval_command_pins_exact_panel_size_as_string(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=v2.V2Controller.__new__(v2.V2Controller);c.config=config();c.root=root
            for count in (48,192):
                tasks=root/f'panel{count}.jsonl';tasks.write_text(('{}\n')*count)
                argv=c.command('evaluate',root/'source.pt',tasks,root/'destination',max_act=3,seconds=300,seed=777,horizon=4096)
                self.assertEqual(argv[argv.index('--episodes')+1],str(count));self.assertTrue(all(isinstance(arg,str) for arg in argv))


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.state_encoder=torch.nn.Linear(2,2);self.action_encoder=torch.nn.Linear(2,2);self.scorer=torch.nn.Linear(2,1)
    def forward(self,x):return self.scorer(self.state_encoder(x)+self.action_encoder(x))


class V2ContinuityTests(unittest.TestCase):
    def test_three_genuine_cpu_updates_all_retained_and_champion_rejection_never_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);torch.manual_seed(11);model=TinyActor();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            def write(path,step):
                torch.save({'format_version':CHECKPOINT_VERSION,'model_state':model.state_dict(),'step':step,'metadata':{},'optimizer_state':optimizer.state_dict()},path)
            initial=root/'initial.pt';write(initial,0)
            c=v2.V2Controller.__new__(v2.V2Controller);c.config=config(episodes=1,device='cpu',workers=1);c.save=lambda:None
            c.state={'learner':{'lineage_chunks':0,'updates':0,'transitions':0,'optimizer_restore_count':0,'cold_optimizer_count':0},
                     'current_checkpoint':str(initial),'best_checkpoint':str(initial),'last_checkpoint':str(initial),'events':[],'stages':{}}
            c.validate_training_result=lambda *args:None
            hashes=[]
            for step in range(1,4):
                source=c.state['current_checkpoint'];optimizer.zero_grad();loss=(model(torch.ones(4,2))-3).square().mean();loss.backward();optimizer.step()
                target=root/f'chunk{step}.pt';write(target,step)
                pending={'source':source,'candidate':str(target),'index':step,'warm':step>1,'metadata':{'tasks':[{'task_id':'t','start_source':'fresh_scratch','start_act':1,'goal_act':3,'horizon':4096}]}}
                metrics={'optimizer_carry':{'restored':step>1},'rounds':[{'transitions':40,'task_ids':['t'],'episodes':[{'task_id':'t','outcome':'defeat'}],'faults':[],'update':{'steps':1,'policy_loss':.1}}]}
                record={};c.advance_learner(pending,metrics,record);hashes.append(record['checkpoint_sha256'])
                self.assertEqual(record['parent_sha256'],v2.sha_file(source));self.assertEqual(c.state['best_checkpoint'],str(initial))
                decision=v2.confirmed_decision(quality(wins=0,floor=10),quality(wins=2,floor=30),quality(wins=151),quality(wins=151))
                self.assertEqual(decision['action'],'keep_champion');self.assertEqual(c.state['current_checkpoint'],str(target))
            self.assertEqual(len(set(hashes)),3)
            self.assertEqual((c.state['learner']['lineage_chunks'],c.state['learner']['updates'],c.state['learner']['transitions']),(3,3,120))
            self.assertEqual((c.state['learner']['cold_optimizer_count'],c.state['learner']['optimizer_restore_count']),(1,2))

    def test_nonfinite_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'nan.pt';torch.save({'format_version':CHECKPOINT_VERSION,'model_state':{'state_encoder.weight':torch.tensor([float('nan')])}},path)
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):v2.finite_policy_checkpoint(path)

    def test_bad_task_coverage_rejected_before_current_parent_mutation(self):
        c=v2.V2Controller.__new__(v2.V2Controller);c.config=config(episodes=1,workers=1,device='cpu');c.backend_identity={'provenance':{'module_sha256':'native'}}
        c.state={'current_checkpoint':'source'}
        pending={'source':'source','candidate':'candidate','tasks':'tasks','metadata':{'tasks':[{'task_id':'t','goal_act':3,'horizon':4096}]}}
        cfg={'task_kind':'run','split':'train','max_act':3,'horizon':4096,'gamma':1.,'gae_lambda':.95,'progress_floor_weight':.05,'terminal_hp_weight':0.,'goal_act_clear':True,'reward_mode':'progress','learning_rate':1e-4,'clip_ratio':.1,'entropy_coef':.01,'workers':1,'threads':1,'episodes':1,'device':'cpu','inference_device':'cpu','updates':1,'checkpoint':'source','tasks':'tasks'}
        metrics={'config':cfg,'training_environment':{'native_module_sha256':'native'},'rounds':[{'task_ids':['t'],'episodes':[{'task_id':'foreign'}],'faults':[],'unstarted_tasks':0}]}
        with self.assertRaisesRegex(RuntimeError,'TaskID coverage'):c.validate_training_result(pending,metrics,{'checkpoint':'candidate'})
        self.assertEqual(c.state['current_checkpoint'],'source')


if __name__=='__main__':unittest.main()
