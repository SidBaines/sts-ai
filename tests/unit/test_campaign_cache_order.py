from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_nonllm_campaign as campaign
from sts_policy.campaign_identity import CACHE_VERSION, condition_signature, semantic_arguments


class CampaignCacheOrderTests(unittest.TestCase):
    def command(self, checkpoint, **settings):
        return ['python', 'train.py', 'ppo', '--checkpoint', str(checkpoint),
                *[token for key, value in settings.items() for token in ['--' + key.replace('_', '-'), str(value)]]]

    def setup_cached(self, directory, *, legacy):
        root = Path(directory)
        input_file = root / 'source.pt'
        input_file.write_text('frozen-source')
        command = self.command(input_file, gamma=1., progress_floor_weight=.05, learning_rate=.0003)
        with patch.dict(os.environ, {'STS_LIGHTSPEED_BUILD': 'test'}):
            runner = campaign.Campaign(root)
        metrics = root / 'job.metrics.json'
        result = {'cached': 'completed'}
        metrics.write_text(json.dumps(result))
        checkpoint = root / 'job.pt'
        checkpoint.touch()
        runner.state['selected']['best'] = str(checkpoint)
        signature = condition_signature(command)
        if legacy:
            signature['version'] = 'campaign_condition_v1'
            signature['arguments'] = command
            signature['digest'] = 'original-v1-digest'
        record = {'name': 'job', 'command': command, 'status': 'complete', 'started': time.time() - 1,
                  'finished': time.time(), 'metrics_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest(),
                  'condition_signature': signature}
        runner.state['commands'].append(record)
        return runner, input_file, command, result, record

    def test_json_sorted_settings_reuse_current_cache_without_child(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, source, original, expected, record = self.setup_cached(directory, legacy=False)
            settings = json.loads(json.dumps({'gamma': 1., 'progress_floor_weight': .05,
                                             'learning_rate': .0003}, sort_keys=True))
            reordered = self.command(source, **settings)
            self.assertNotEqual(original, reordered)
            with patch.object(campaign.subprocess, 'Popen') as popen:
                result = runner.run('job', reordered, 180)
            popen.assert_not_called()
            self.assertEqual(result, expected)
            self.assertEqual(record['condition_signature']['version'], CACHE_VERSION)

    def test_v1_cache_migrates_only_identical_effective_values_and_full_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, source, _, expected, record = self.setup_cached(directory, legacy=True)
            reordered = self.command(source, learning_rate=.0003, progress_floor_weight=.05, gamma=1.)
            with patch.object(campaign.subprocess, 'Popen') as popen:
                result = runner.run('job', reordered, 180)
            popen.assert_not_called()
            self.assertEqual(result, expected)
            self.assertEqual(record['condition_signature']['version'], 'campaign_condition_v2')
            persisted = json.loads(runner.journal.read_text())
            self.assertEqual(persisted['commands'][0]['condition_signature'], condition_signature(reordered))

    def test_v1_changed_scientific_knob_rejected_before_protected_artifact_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, source, _, _, _ = self.setup_cached(directory, legacy=True)
            changed = self.command(source, learning_rate=.0003, progress_floor_weight=.02, gamma=1.)
            with patch.object(campaign.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'rename'):
                    runner.run('job', changed, 180)
            popen.assert_not_called()

    def test_v1_changed_input_hash_rejected_before_protected_artifact_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, source, _, _, _ = self.setup_cached(directory, legacy=True)
            source.write_text('different-source')
            reordered = self.command(source, learning_rate=.0003, progress_floor_weight=.05, gamma=1.)
            with patch.object(campaign.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'rename'):
                    runner.run('job', reordered, 180)
            popen.assert_not_called()

    def test_effective_last_option_wins_and_positional_code_order_is_preserved(self):
        first = ['python', '-c', '--this-is-code', 'arg1', 'arg2', '--seed', '3', '--gamma', '-1', '--seed', '11']
        equivalent = ['python', '-c', '--this-is-code', 'arg1', 'arg2', '--seed=11', '--gamma=-1']
        self.assertEqual(semantic_arguments(first), semantic_arguments(equivalent))
        changed = [*equivalent, '--seed', '12']
        self.assertNotEqual(semantic_arguments(first), semantic_arguments(changed))
        self.assertEqual(semantic_arguments(first)[:5], ['python', '-c', '--this-is-code', 'arg1', 'arg2'])

    def test_boolean_optional_alias_order_retains_last_effective_choice(self):
        enabled = ['python', 'train.py', 'ppo', '--no-goal-act-clear', '--goal-act-clear']
        disabled = ['python', 'train.py', 'ppo', '--goal-act-clear', '--no-goal-act-clear']
        self.assertNotEqual(semantic_arguments(enabled), semantic_arguments(disabled))
        self.assertEqual(semantic_arguments(enabled),
                         semantic_arguments(['python', 'train.py', 'ppo', '--goal-act-clear']))
        self.assertEqual(semantic_arguments(disabled),
                         semantic_arguments(['python', 'train.py', 'ppo', '--no-goal-act-clear']))
