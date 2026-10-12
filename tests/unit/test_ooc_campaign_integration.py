from __future__ import annotations

from pathlib import Path
import os
import json
import time
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_nonllm_campaign as campaign


def complete_panel(wins):
    return {'command':'evaluate','rounds':[{'task_ids':['a','b'], 'cohort_completed':True,
        'episodes':[{'task_id':name,'outcome':'act_clear' if index<wins else 'defeat','terminated':True,
                     'termination_reason':'victory' if index<wins else 'defeat','floor':17,'hp_cost':10.}
                    for index,name in enumerate(('a','b'))], 'faults':[]}]}


class OOCCampaignIntegrationTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(campaign.time, "time", return_value=campaign.INITIAL + 3600)
        clock.start()
        self.addCleanup(clock.stop)

    def make_runner(self, directory):
        with patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':'test'}):
            return campaign.Campaign(Path(directory))

    def test_screen_uses_fixed_source_and_rejects_incomplete_higher_score(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            observed=[]
            def run(name, command, seconds, end):
                observed.append((name,command,seconds,end))
                if command[2]=='ppo':
                    Path(command[command.index('--output')+1]).touch()
                    return {'training':{'steps':1}}
                if 'lr0.0003' in name:
                    panel=complete_panel(2)
                    panel['rounds'][0]['cohort_completed']=False
                    return panel
                return complete_panel(2 if 'lr0.001' in name else 1)
            with patch.object(runner,'run',side_effect=run),patch.object(runner,'save'):
                selected,settings=runner.ooc_screen(Path('fixed.pt'),12345)
            self.assertIn('lr0.001',str(selected))
            self.assertEqual(settings['learning_rate'],1e-3)
            self.assertEqual(settings['gamma'],1.)
            self.assertEqual(settings['clip_ratio'],.1)
            self.assertEqual(len(observed),11)
            for _,command,seconds,end in observed:
                self.assertEqual(end,12345)
                if command[2]=='ppo':
                    self.assertEqual(command[command.index('--checkpoint')+1],'fixed.pt')
                    self.assertEqual(seconds,90)
                else:
                    self.assertEqual(seconds,120)
                    self.assertEqual(command[command.index('--episodes')+1],'32')
                    self.assertEqual(command[command.index('--horizon')+1],'1024')

    def test_existing_ooc_phase_skips_pilots_and_returns_saved_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            settings={'learning_rate':1e-4,'reward_mode':'sparse','gamma':1.,'clip_ratio':.1,
                      'entropy_coef':.01,'progress_floor_weight':.05,'terminal_hp_weight':0.}
            runner.state['phase_progress']={'ooc':{'best':'saved.pt','next_round':7}}
            runner.state['ooc_effective_settings']=settings
            with patch.object(runner,'run') as run:
                checkpoint,actual=runner.ooc_screen(Path('new-source.pt'),12345)
            run.assert_not_called()
            self.assertEqual(checkpoint,Path('saved.pt'))
            self.assertEqual(actual,settings)
            self.assertEqual(runner.state['phase_progress']['ooc']['next_round'],7)

    def test_initial_and_chunk_ooc_dev_evaluation_share_fixed_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            with patch.object(runner,'run',return_value=None) as run:
                runner.evaluate('ooc_initial_dev',Path('x.pt'),'ooc',end=12345)
            command=run.call_args.args[1]
            def last(flag):return command[len(command)-1-command[::-1].index(flag)+1]
            self.assertEqual(last('--episodes'),32)
            self.assertEqual(str(last('--horizon')),'1024')
            self.assertEqual(str(last('--world-start')),'10000')
            self.assertEqual(run.call_args.args[3],12345)

    def test_reference_retained_when_all_pilots_regress(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            def run(name,command,seconds,end):
                if command[2]=='ppo':
                    Path(command[command.index('--output')+1]).touch()
                    return {'training':{'steps':1}}
                return complete_panel(2 if 'warmstart_reference' in name else 1)
            with patch.object(runner,'run',side_effect=run),patch.object(runner,'save'):
                checkpoint,settings=runner.ooc_screen(Path('source.pt'),12345)
            self.assertEqual(checkpoint,Path('source.pt'))
            self.assertEqual(settings['learning_rate'],1e-4)
            self.assertEqual(runner.state['ooc_pilot_panel'][0]['role'],'reference')

    def test_score_rejects_incomplete_and_keeps_faults_in_requested_coverage(self):
        full=complete_panel(2)
        self.assertAlmostEqual(campaign.Campaign.score(full),101.6)
        full['rounds'][0]['episodes'].pop()
        self.assertEqual(campaign.Campaign.score(full),-1e9)
        full['rounds'][0]['faults']=[{'task_id':'b'}]
        self.assertAlmostEqual(campaign.Campaign.score(full),51.6)
        self.assertEqual(full['rounds'][0]['faults'],[{'task_id':'b'}])

    def test_combat_panel_cannot_drop_missing_hard_states(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            panels=Path(directory)/'panels';panels.mkdir();(panels/'dev_hard.jsonl').touch()
            runner.state['active_panel_directory']=str(panels)
            with patch.object(runner,'evaluate',side_effect=[complete_panel(2),None]):
                result=runner.evaluate_combat('dev',Path('policy.pt'))
            self.assertEqual(result['selection_score'],-1e9)

    def test_legacy_cache_accepts_only_verified_performance_flags_and_pins_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            root=Path(directory)
            source=root/'input.jsonl';source.write_text('original')
            started=time.time()+1
            # Match the explicitly frozen fixture clock for legacy mtime checks.
            os.utime(source, (started - 1, started - 1))
            metrics=root/'old.metrics.json';metrics.write_text(json.dumps({'completed':True}))
            command=['python','train','--tasks',str(source),'--simulations','200','--seed','11']
            record={'name':'old','status':'complete','command':command,'started':started,'finished':time.time()+2,
                    'metrics_sha256':hashlib.sha256(metrics.read_bytes()).hexdigest()}
            runner.state['commands'].append(record)
            new=command+['--fast-replay','--fast-observations','--deferred-bootstrap']
            self.assertEqual(runner.run('old',new,1,phase_end=0),{'completed':True})
            self.assertIn('condition_signature',record)
            changed=list(new);changed[changed.index('--seed')+1]='12'
            self.assertIsNone(runner.run('old',changed,1,phase_end=0))
            source.write_text('changed')
            self.assertIsNone(runner.run('old',new,1,phase_end=0))

    def test_finished_architecture_selection_resumes_without_new_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            saved=Path(directory)/'saved.pt';saved.touch()
            runner.state['architecture_panel']=[{'checkpoint':str(saved),'score':95.}]
            runner.state['selected']['shared_bc']=str(saved)
            runner.state['phase']='combat'
            with patch.object(runner,'train') as train,patch.object(runner,'evaluate') as evaluate:
                self.assertEqual(runner.screen(),saved)
            train.assert_not_called();evaluate.assert_not_called()
            self.assertEqual(runner.state['phase'],'combat')

    def test_recovered_frontier_uses_original_budget_instead_of_late_remaining_time(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            now=time.time();output=Path(directory)/'ooc_0000.pt';output.write_text('checkpoint')
            os.utime(output, (now, now))
            metrics=Path(directory)/'ooc_0000.metrics.json';metrics.write_text('{}')
            runner.state['commands'].append({'name':'ooc_0000','status':'complete','command':['train','--seconds','180'],
                'started':now-200,'finished':now,'metrics_sha256':hashlib.sha256(metrics.read_bytes()).hexdigest()})
            runner.state['phase_progress']={'ooc':{'next_round':0,'started':now-300,'current':'source.pt',
                                                  'best':'source.pt','best_score':0.}}
            clock={'now':now}
            def train(*args,**kwargs):
                self.assertEqual(kwargs['seconds'],180.)
                self.assertEqual(str(kwargs['seconds']),'180')
                clock['now']=now+1000
                return output,{}
            with patch.object(campaign.time,'time',side_effect=lambda:clock['now']),patch.object(runner,'train',side_effect=train),\
                 patch.object(runner,'evaluate',return_value=complete_panel(2)),patch.object(runner,'save'):
                self.assertEqual(runner.curriculum('ooc','source.pt',now+250,'ooc'),output)
            self.assertEqual(runner.state['phase_progress']['ooc']['next_round'],1)
            metrics.write_text('changed')
            self.assertIsNone(runner.completed_frontier_seconds('ooc_0000'))

    def test_new_condition_archives_completed_checkpoint_before_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory)
            root=Path(directory);checkpoint=root/'same.pt';checkpoint.write_text('original')
            metrics=root/'same.metrics.json';metrics.write_text('{"original":true}')
            code="from pathlib import Path;import sys;Path(sys.argv[1]).write_text('replacement');raise SystemExit(2)"
            command=[sys.executable,'-c',code,str(checkpoint),'--seed','11']
            old={'name':'same','status':'complete','command':command,'started':time.time()-10,'finished':time.time(),
                 'metrics_sha256':hashlib.sha256(metrics.read_bytes()).hexdigest()}
            runner.state['commands'].append(old)
            changed=command[:-1]+['12']
            real_sleep=time.sleep
            with patch.object(campaign.time,'sleep',side_effect=lambda _:real_sleep(.01)):
                self.assertIsNone(runner.run('same',changed,1))
            self.assertEqual(checkpoint.read_text(),'replacement')
            saved=Path(old['preserved_artifacts'][str(checkpoint)]['path'])
            self.assertEqual(saved.read_text(),'original')
            saved_metrics=Path(old['preserved_artifacts'][str(metrics)]['path'])
            self.assertEqual(saved_metrics.read_text(),'{"original":true}')

    def test_selected_completed_checkpoint_cannot_be_replaced_by_new_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.make_runner(directory);root=Path(directory)
            checkpoint=root/'same.pt';checkpoint.write_text('selected')
            metrics=root/'same.metrics.json';metrics.write_text('{}')
            old={'name':'same','status':'complete','command':['python','train','--seed','11'],
                 'started':time.time()-10,'finished':time.time(),'metrics_sha256':hashlib.sha256(metrics.read_bytes()).hexdigest()}
            runner.state['commands'].append(old)
            runner.state['selected']['combat']=str(checkpoint)
            with patch.object(campaign.subprocess,'Popen') as process:
                with self.assertRaisesRegex(RuntimeError,'rename the condition artifact'):
                    runner.run('same',['python','train','--seed','12'],1)
            process.assert_not_called()
            self.assertEqual(checkpoint.read_text(),'selected')
            self.assertEqual(runner.state['commands'][-1]['status'],'blocked_protected_artifact')
