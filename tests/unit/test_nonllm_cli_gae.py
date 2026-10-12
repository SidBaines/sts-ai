from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_nonllm as cli
from sts_policy.losses import PPOConfig
from sts_policy.rollout_training import collect_rollouts
from tests.unit.test_nonllm_training import FakeTask, FakeWorker, TinyModel, tiny_collate, tiny_forward


class NonllmCLIGAETests(unittest.TestCase):
    def run_cli(self, arguments):
        collected, logged = [], []
        class Model(TinyModel):
            def metadata(self): return {'test_model': True}
        class Worker(FakeWorker):
            def __init__(self, config, timeout): pass
            def close(self): pass
        def collect(model, workers, tasks, **kwargs):
            result = collect_rollouts(model, workers, tasks, collate=tiny_collate,
                forward=tiny_forward, config=kwargs['config'], deterministic=True)
            collected.append((kwargs['config'], result['transitions']))
            return result
        def update(model, transitions, **kwargs):
            self.assertEqual(kwargs['config'].gae_lambda, collected[-1][0].gae_lambda)
            return {'steps': 1, 'optimizer': kwargs['optimizer']}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            argv = ['train_nonllm', 'ppo', '--output', str(Path(directory)/'test.pt'),
                    '--updates', '1', '--workers', '1', '--seconds', '1', *arguments]
            with patch.object(sys, 'argv', argv), \
                 patch.object(cli, 'source_model', return_value=(Model(), None)), \
                 patch.object(cli, 'tasks_for', return_value=[FakeTask(1)]), \
                 patch.object(cli, 'CurriculumWorker', Worker), \
                 patch.object(cli, 'collect_rollouts', side_effect=collect), \
                 patch.object(cli, 'update_ppo', side_effect=update), \
                 patch.object(cli, 'write_metrics', side_effect=lambda path, metrics: logged.append(metrics)), \
                 patch.object(cli, 'save_checkpoint') as saved, \
                 patch('sts_policy.composition.training_environment_record', return_value={'test': True}):
                cli.main()
            checkpoint_config = saved.call_args.kwargs['metadata']['config']
        return collected[0], logged[-1], checkpoint_config

    def test_cli_lambda_reaches_collector_update_and_saved_config_and_changes_gae(self):
        (default, baseline), metrics, saved = self.run_cli([])
        (one, monte_carlo), changed_metrics, changed_saved = self.run_cli(['--gae-lambda', '1'])
        self.assertEqual(default.gae_lambda, .95)
        self.assertEqual(one.gae_lambda, 1.)
        self.assertAlmostEqual(baseline[0].advantage, 1.95)
        self.assertAlmostEqual(monte_carlo[0].advantage, 2.)
        self.assertEqual(baseline[-1].advantage, monte_carlo[-1].advantage)
        self.assertEqual(metrics['config']['gae_lambda'], .95)
        self.assertEqual(changed_metrics['config']['gae_lambda'], 1.)
        self.assertEqual(saved['gae_lambda'], .95)
        self.assertEqual(changed_saved['gae_lambda'], 1.)

    def test_default_and_explicit_point95_preserve_original_targets(self):
        (default, baseline), _, _ = self.run_cli([])
        (explicit, replay), _, _ = self.run_cli(['--gae-lambda', '.95'])
        self.assertEqual(default, PPOConfig(learning_rate=.0003))
        self.assertEqual(default, explicit)
        self.assertEqual(baseline, replay)

    def test_out_of_range_nonfinite_lambda_rejected_before_model_or_worker(self):
        for value in ('-.1', '1.1', 'nan', 'inf'):
            with self.subTest(value=value), patch.object(sys, 'argv',
                    ['train_nonllm', 'ppo', '--output', '/private/tmp/unused-gae.pt', '--gae-lambda', value]), \
                 patch.object(cli, 'source_model') as model, patch.object(cli, 'CurriculumWorker') as worker, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): cli.main()
                model.assert_not_called()
                worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
