from __future__ import annotations
import os
from pathlib import Path
import tempfile
import unittest
from sts_policy.campaign_identity import condition_signature, legacy_cache_valid


class CampaignIdentityTests(unittest.TestCase):
    def test_outputs_ignored_scientific_knobs_and_input_content_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.jsonl';data.write_text('original')
            cmd=['python','train.py','ppo','--tasks',str(data),'--simulations','200','--gamma','1',
                 '--terminal-hp-weight','0','--learning-rate','0.001','--output','out.pt','--metrics','m.json']
            base=condition_signature(cmd)
            output_changed=list(cmd);output_changed[-3]='different.pt';output_changed[-1]='other.json'
            self.assertEqual(base,condition_signature(output_changed))
            for flag in ['--simulations','--gamma','--terminal-hp-weight','--learning-rate']:
                changed=list(cmd);changed[changed.index(flag)+1]='different'
                self.assertNotEqual(base['digest'],condition_signature(changed)['digest'])
            data.write_text('changed')
            self.assertNotEqual(base['digest'],condition_signature(cmd)['digest'])

    def test_legacy_input_mtime_gate_and_effective_last_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actual=root/'actual';actual.write_text('x');os.utime(actual,(10,10))
            cmd=['python','train.py','--teacher',str(root/'unused_missing'), '--teacher',str(actual),'--output','out']
            self.assertTrue(legacy_cache_valid({'command':cmd,'started':20},cmd))
            signature=condition_signature(cmd)
            self.assertEqual(signature['inputs']['--teacher']['path'],str(actual.resolve()))
            os.utime(actual,(30,30))
            self.assertFalse(legacy_cache_valid({'command':cmd,'started':20},cmd))
