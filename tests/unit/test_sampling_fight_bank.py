from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from scripts import sample_fight_bank as sample
from sts_ai import fight_lab as lab
from sts_ai.agents import MlxQwenJsonAgent
from sts_ai.agent_factory import build_agent
from tests.unit.test_fight_lab import record


class SamplingTest(unittest.TestCase):
    def fake_mlx(self):
        module = ModuleType("mlx_lm")
        module.load = Mock(return_value=(object(), object()))
        module.generate = Mock()
        utils = ModuleType("mlx_lm.sample_utils")
        utils.make_sampler = Mock(return_value=object())
        return {"mlx_lm": module, "mlx_lm.sample_utils": utils}, utils

    def test_mlx_receives_top_p_and_top_k_and_records_them(self):
        modules, utils = self.fake_mlx()
        with patch.dict(sys.modules, modules):
            agent = build_agent("mlx", temperature=1.0, top_p=.95, top_k=64)
        utils.make_sampler.assert_called_once_with(temp=1.0, top_p=.95, top_k=64)
        self.assertEqual((agent.config['temperature'], agent.config['top_p'], agent.config['top_k']),
                         (1.0, .95, 64))

    def test_default_mlx_sampler_call_preserves_previous_behavior(self):
        modules, utils = self.fake_mlx()
        with patch.dict(sys.modules, modules):
            MlxQwenJsonAgent(temperature=.2)
        utils.make_sampler.assert_called_once_with(temp=.2)

    def test_bad_sampling_filters_rejected_before_loading_model(self):
        for kwargs in ({'top_p': 0}, {'top_p': 1.1}, {'top_k': -2}):
            with self.assertRaises(ValueError):
                MlxQwenJsonAgent(**kwargs)

    def test_partial_archive_retains_data_without_merging_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / 'state_r0.jsonl'
            trace.write_text('partial\n')
            sample.archive_partial(root, 'state')
            self.assertFalse(trace.exists())
            self.assertEqual(next((root/'partials').glob('*/*.jsonl')).read_text(), 'partial\n')

    def test_resume_rejects_modified_settings_or_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, out = root/'bank', root/'run'
            rows = lab.index_records([record()], 'seed_1004_r0.jsonl')
            lab.write_new(bank/'collection.json', {'states': rows})
            sample.prepare(bank, out)
            manifest, _, _ = sample.validate(out)
            self.assertEqual(manifest['sampling_seeds'][rows[0]['state_id']], sample.sampling_seed(rows[0]['state_id']))
            settings = lab.read_json(out/'settings.json')
            settings['temperature'] = 0
            sample.atomic(out/'settings.json', settings)
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                sample.validate(out)

    def test_resume_checks_completed_trace_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, out = root/'bank', root/'run'
            rows = lab.index_records([record()], 'seed_1004_r0.jsonl')
            lab.write_new(bank/'collection.json', {'states': rows})
            sample.prepare(bank, out)
            state_id = rows[0]['state_id']
            (out/f'{state_id}_r0.jsonl').write_text('changed')
            sample.atomic(out/f'{state_id}.result.json', {'state_id':state_id,'status':'ok',
                'trace_sha256':'wrong','meta_sha256':'wrong'})
            with self.assertRaisesRegex(ValueError, 'finished trajectory changed'):
                sample.validate(out)

    def test_progress_does_not_count_invalid_as_completed_fight(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            sample.atomic(out/'a.result.json', {'state_id': 'a', 'status': 'ok',
                'stopped_reason': 'agent_invalid', 'n_invalid': 1, 'metrics': {'completed': False, 'won': False}})
            p = sample.progress(out, {'order': ['a', 'b']}, 'paused')
            self.assertEqual((p['finished'],p['completed'],p['incomplete'],p['expected']), (1,0,1,2))
