from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_nonllm_campaign as campaign


class CurriculumRunScopeTests(unittest.TestCase):
    def evaluate_commands(self, kind, **settings):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'STS_LIGHTSPEED_BUILD': 'test'}):
                runner = campaign.Campaign(Path(directory))
            now = [1000.]
            end = 2000.
            observed = []

            def run(name, command, seconds, deadline):
                observed.append((name, command, seconds, deadline))
                return {'selection_score': 1.}

            def train(*args, **kwargs):
                now[0] = end + 1
                return Path(directory) / 'updated.pt', {}

            with patch.object(campaign.time, 'time', side_effect=lambda: now[0]), \
                 patch.object(runner, 'run', side_effect=run), \
                 patch.object(runner, 'train', side_effect=train), \
                 patch.object(runner, 'completed_frontier_seconds', return_value=None), \
                 patch.object(runner, 'save'):
                runner.curriculum('joint', Path('source.pt'), end, kind, **settings)
            self.assertEqual([entry[0] for entry in observed], ['joint_initial_dev', 'joint_0000_dev'])
            return observed

    def assert_cohort(self, observed, *, kind, max_act, horizon, episodes, seconds):
        for _, command, allowance, end in observed:
            values = {str(command[i]): str(command[i + 1]) for i in range(len(command) - 1)
                      if str(command[i]).startswith('--')}
            self.assertEqual(values['--task-kind'], kind)
            self.assertEqual(values.get('--max-act', '1'), str(max_act))
            self.assertEqual(values['--horizon'], str(horizon))
            self.assertEqual(values['--episodes'], str(episodes))
            self.assertEqual(values['--world-start'], '10000')
            self.assertEqual(values['--seed'], '777')
            self.assertEqual(values['--simulations'], '200')
            self.assertEqual(values['--seconds'], str(seconds))
            self.assertEqual(allowance, seconds)
            self.assertEqual(end, 2000.)

    def test_act3_initial_and_every_chunk_evaluate_full_configured_run(self):
        observed = self.evaluate_commands('run', max_act=3, horizon=4096)
        self.assert_cohort(observed, kind='run', max_act=3, horizon=4096, episodes=48, seconds=240)

    def test_unspecified_run_scope_preserves_act1_defaults(self):
        observed = self.evaluate_commands('run')
        self.assert_cohort(observed, kind='run', max_act=1, horizon=2048, episodes=48, seconds=90)

    def test_ooc_training_scope_does_not_change_fixed_dev_panel(self):
        observed = self.evaluate_commands('ooc', max_act=3, horizon=4096)
        self.assert_cohort(observed, kind='ooc', max_act=1, horizon=1024, episodes=32, seconds=90)
