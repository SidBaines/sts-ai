from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_nonllm_campaign as campaign
from sts_policy.final_freeze import FreezeChangedError
from sts_policy.rollout_training import task_identity


class CampaignRevisionTests(unittest.TestCase):
    def fixture(self, root, stack):
        # The historical production deadline must not make this fixture expire.
        stack.enter_context(patch.object(campaign.time, "time", return_value=campaign.INITIAL + 3600))
        scripts=root/'scripts';scripts.mkdir()
        for name in ('train_nonllm.py','evaluate_nonllm_baselines.py','run_nonllm_campaign.py'):
            (scripts/name).write_text('# synthetic source\n')
        policy=root/'src/sts_policy';policy.mkdir(parents=True);(policy/'model.py').write_text('# synthetic model\n')
        bank=root/'bank';bank.mkdir();(bank/'battles.jsonl').write_text('{}\n')
        panels=root/'panels';panels.mkdir();(panels/'final_hard.jsonl').write_text('{}\n');(panels/'panels.json').write_text('{}')
        build=root/'build';build.mkdir();module=build/'slaythespire.fake.so';module.write_bytes(b'synthetic binary')
        provenance={'module_file':module.name,'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
                    'upstream_commit':'fake','patch_sha256':'fake'}
        (build/'build_manifest.json').write_text(json.dumps(provenance))
        (bank/'manifest.json').write_text(json.dumps({'provenance':{'native_provenance':provenance}}))
        stack.enter_context(patch.object(campaign,'ROOT',root))
        stack.enter_context(patch.object(campaign,'BANK',bank))
        stack.enter_context(patch.object(campaign,'PANELS',panels))
        stack.enter_context(patch.object(campaign,'__file__',str(scripts/'run_nonllm_campaign.py')))
        stack.enter_context(patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':str(build)}))
        runner=campaign.Campaign(root/'output')
        models={label:root/(label+'.pt') for label in ('shared','combat','ooc')}
        for label,path in models.items():path.write_bytes(label.encode())
        return runner,models

    def test_unchanged_refreeze_preserves_time_and_changed_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            runner,models=self.fixture(Path(directory),stack)
            first=runner.freeze_final_selection(**models)
            second=runner.freeze_final_selection(**models)
            self.assertEqual(first['frozen_at'],second['frozen_at'])
            self.assertEqual(first['digest'],second['digest'])
            changed=Path(directory)/'new_shared.pt';changed.write_bytes(b'new')
            with self.assertRaises(FreezeChangedError):
                runner.freeze_final_selection(changed,models['combat'],models['ooc'])
            self.assertEqual(runner.state['frozen_selection']['shared'],str(models['shared']))

    def test_mutated_checkpoint_prevents_any_final_child_launch(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            runner,models=self.fixture(Path(directory),stack)
            runner.freeze_final_selection(**models)
            models['combat'].write_bytes(b'changed weights')
            with patch.object(campaign.subprocess,'Popen') as process:
                with self.assertRaises(FreezeChangedError):
                    runner.run('final_check',['python','unused'],1)
            process.assert_not_called()

    def test_generated_final_definitions_match_real_cli_tasks(self):
        from scripts.train_nonllm import tasks_for
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            runner,models=self.fixture(Path(directory),stack)
            freeze=runner.freeze_final_selection(**models)
            panels=freeze['generated_definitions']['panels']
            for label,kind,max_act in (('ooc_act1','ooc',1),('run_act1','run',1),('run_act3','run',3)):
                args=SimpleNamespace(tasks=None,task_kind=kind,horizon=2048,max_act=max_act,simulations=200,
                    world_start=10000,episodes=21,split='final',seed=777)
                actual=[task_identity(task) for task in tasks_for(args)]
                self.assertEqual(len(actual),21)
                self.assertEqual(Counter(actual),Counter(panels[label]['task_ids']))

    def test_revision_retains_combat_and_runs_only_ooc_and_joint_curricula(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            runner,models=self.fixture(Path(directory),stack)
            runner.state['retained_combat']={'checkpoint':str(models['combat'])}
            runner.state['quarantine_lookahead']=True
            settings={'learning_rate':1e-4,'reward_mode':'progress','gamma':1.,'clip_ratio':.1,
                      'entropy_coef':.01,'terminal_hp_weight':0.,'progress_floor_weight':.05}
            def curriculum(name,source,end,kind,**kwargs):return Path(directory)/(name+'.pt')
            with patch.object(runner,'screen',return_value=models['shared']), \
                 patch.object(runner,'train',return_value=(models['ooc'],{})) as train, \
                 patch.object(runner,'ppo_screen') as combat_screen, \
                 patch.object(runner,'evaluate_combat') as combat_eval, \
                 patch.object(runner,'ooc_screen',return_value=(models['ooc'],settings)) as ooc_screen, \
                 patch.object(runner,'joint_screen',return_value=(models['shared'],settings)), \
                 patch.object(runner,'joint_composition',return_value=(models['shared'],{})), \
                 patch.object(runner,'curriculum',side_effect=curriculum) as stages, \
                 patch.object(runner,'evaluate',return_value={'selection_score':1.}), \
                 patch.object(runner,'evaluate_composite',return_value={'selection_score':1.}), \
                 patch.object(runner,'final') as final, patch.object(runner,'save'):
                runner.main()
            combat_screen.assert_not_called();combat_eval.assert_not_called()
            self.assertEqual([call.args[0] for call in train.call_args_list],['ooc_specialist_bc'])
            self.assertEqual([call.args[0] for call in stages.call_args_list],['ooc','joint_act1','joint_act3'])
            self.assertEqual(runner.state['selected']['combat'],str(models['combat']))
            ooc_screen.assert_called_once();final.assert_called_once()
