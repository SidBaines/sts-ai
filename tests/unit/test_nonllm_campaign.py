"""Regression checks for failed-job recovery and honest progress logging."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from scripts import run_nonllm_campaign as campaign
from scripts.update_nonllm_progress import render


class CampaignRecoveryTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(campaign.time, "time", return_value=campaign.INITIAL + 3600)
        clock.start()
        self.addCleanup(clock.stop)

    def test_failed_command_does_not_return_stale_success_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            old=root/'failed.metrics.json'
            old.write_text(json.dumps({'elapsed_seconds':1,'training':{'steps':100}}))
            os.utime(old,(100,100))
            original_sleep=time.sleep
            with patch.object(campaign,'ROOT',root),patch.object(campaign,'DEADLINE',time.time()+3600),\
                 patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':'test-build'}),\
                 patch.object(campaign.time,'sleep',side_effect=lambda _: original_sleep(.01)):
                runner=campaign.Campaign(root)
                result=runner.run('failed',[sys.executable,'-c','raise SystemExit(2)'],seconds=1)
            self.assertIsNone(result)
            self.assertEqual(runner.state['commands'][-1]['status'],'failed')

    def test_scratch_training_omits_checkpoint_only_optimizer_flag(self):
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':'test'}):
            runner=campaign.Campaign(Path(temporary))
            with patch.object(runner,'run',return_value=None) as run:
                runner.train('scratch','ppo',deferred_bootstrap=True)
            command=run.call_args.args[1]
            self.assertNotIn('--warm-optimizer',command)
            self.assertIn('--deferred-bootstrap',command)
            self.assertNotIn('True',command)

    def test_cached_metrics_do_not_cross_changed_evaluation_cohorts(self):
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':'test'}):
            root=Path(temporary)
            metrics=root/'same.metrics.json';metrics.write_text('{}')
            runner=campaign.Campaign(root)
            runner.state['commands'].append({'name':'same','status':'complete','finished':time.time(),
                'command':[sys.executable,'-c','pass','--tasks','old.jsonl']})
            with patch.object(campaign,'DEADLINE',time.time()+3600),patch.object(campaign,'ROOT',root):
                result=runner.run('same',[sys.executable,'-c','raise SystemExit(2)','--tasks','new.jsonl'],seconds=1)
            self.assertIsNone(result)
            self.assertEqual(runner.state['commands'][-1]['status'],'failed')

    def test_final_panel_requires_full_requested_cohort(self):
        complete={'command':'evaluate','rounds':[{'task_ids':['a','b'], 'episodes':[{'task_id':'a','termination_reason':'battle_finished'}],
                                                'faults':[{'task_id':'b'}], 'cohort_completed':True}]}
        self.assertTrue(campaign.panel_complete(complete))
        complete['rounds'][0]['episodes'][0]['termination_reason']='collector_budget'
        self.assertFalse(campaign.panel_complete(complete))
        complete['rounds'][0]['episodes'][0]['termination_reason']='battle_finished'
        complete['rounds'][0]['task_ids'].append('unstarted')
        self.assertFalse(campaign.panel_complete(complete))

    def test_failed_log_does_not_attribute_later_rerun_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            metrics=root/'same.metrics.json'
            metrics.write_text(json.dumps({'training':{'steps':123,'examples':456,'elapsed_seconds':1}}))
            journal=root/'campaign.json'
            journal.write_text(json.dumps({'status':'running','phase':'screens','started_at':1,
                'deadline':time.time()+3600,'heartbeat':time.time(),'selected':{},'native_build':'test',
                'commands':[{'name':'first_failed_invocation','started':1,'finished':2,
                             'status':'failed','exit_code':2,'command':['test'], 'metrics':str(metrics)}]}))
            output=root/'progress.md'
            render(journal,output,{})
            log=output.read_text()
            self.assertIn('belong to a later rerun',log)
            self.assertNotIn('123 updates',log)


if __name__=='__main__':unittest.main()
