from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_nonllm_campaign as campaign
from sts_policy.final_freeze import FreezeChangedError


class FinalCampaignCoreFreezeTests(unittest.TestCase):
    def fixture(self, root):
        for directory in ('src/sts_env/_backend','src/sts_policy','scripts','bank','panels','backend'):
            (root/directory).mkdir(parents=True,exist_ok=True)
        for path in ('src/sts_env/__init__.py','src/sts_env/env.py','src/sts_env/_backend/public.py',
                     'src/sts_policy/policy.py','scripts/train_nonllm.py',
                     'scripts/evaluate_nonllm_baselines.py','scripts/run_nonllm_campaign.py'):
            (root/path).write_text('# synthetic source; never imported or executed\n')
        native=root/'backend/slaythespire.test.so';native.write_bytes(b'synthetic native bytes; never loaded')
        provenance={'module_file':native.name,'module_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}
        (root/'backend/build_manifest.json').write_text(json.dumps(provenance))
        (root/'bank/manifest.json').write_text(json.dumps({'provenance':{'native_provenance':provenance}}))
        (root/'bank/battles.jsonl').write_text('{"synthetic_panel":true}\n')
        (root/'panels/final_hard.jsonl').write_text('{"synthetic_panel":true}\n')
        (root/'panels/panels.json').write_text('{}')
        checkpoints={}
        for name in ('shared','combat','ooc'):
            path=root/(name+'.pt');path.write_bytes(('synthetic '+name).encode());checkpoints[name]=path
        return checkpoints

    def test_core_files_recursively_frozen_and_mutation_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoints=self.fixture(root)
            with patch.object(campaign,'ROOT',root),patch.object(campaign,'BANK',root/'bank'),\
                 patch.object(campaign,'__file__',str(root/'scripts/run_nonllm_campaign.py')),\
                 patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':str(root/'backend')}):
                runner=campaign.Campaign(root/'output')
                runner.state['active_panel_directory']=str(root/'panels')
                snapshot=runner.freeze_final_selection(**checkpoints)
                for name in ('src/sts_env/__init__.py','src/sts_env/env.py','src/sts_env/_backend/public.py'):
                    self.assertIn(name,snapshot['panel_files'])
                self.assertEqual(snapshot['generated_definitions']['shuffle_seed'],777)
                self.assertEqual(snapshot['generated_definitions']['gamma'],1.)
                self.assertEqual(snapshot['generated_definitions']['reward_mode'],'potential')
                # Merely changing the public Python serializer is enough to
                # invalidate the final freeze, with an unchanged native binary.
                (root/'src/sts_env/_backend/public.py').write_text('# changed public serializer\n')
                with patch.object(campaign.subprocess,'Popen') as launch:
                    with self.assertRaises(FreezeChangedError):
                        runner.run('final_synthetic_probe',['never-execute'],seconds=1)
                    launch.assert_not_called()

    def test_final_composite_command_explicitly_matches_seed_and_reward_scope(self):
        class StopAfterComposite(Exception):pass
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'STS_LIGHTSPEED_BUILD':'never-loaded'}):
            runner=campaign.Campaign(Path(temporary))
            captured=[]
            def capture(name,command,*args,**kwargs):
                if name=='final_composite':
                    captured.extend(map(str,command));raise StopAfterComposite
            with patch.object(runner,'freeze_final_selection',return_value={}),\
                 patch.object(runner,'save'),patch.object(runner,'evaluate'),\
                 patch.object(runner,'run',side_effect=capture):
                with self.assertRaises(StopAfterComposite):runner.final('shared.pt','combat.pt','ooc.pt')
            values={token:captured[i+1] for i,token in enumerate(captured[:-1]) if token in
                    {'--seed','--gamma','--reward-mode','--terminal-hp-weight','--progress-floor-weight','--horizon','--max-act'}}
            self.assertEqual(values['--seed'],'777')
            self.assertEqual(values['--gamma'],'1.0')
            self.assertEqual(values['--reward-mode'],'potential')
            self.assertEqual(values['--terminal-hp-weight'],'0.0')
            self.assertEqual(values['--progress-floor-weight'],'0.05')
            self.assertEqual(values['--horizon'],'2048')
            self.assertEqual(values['--max-act'],'3')
            self.assertIn('--goal-act-clear',captured)
