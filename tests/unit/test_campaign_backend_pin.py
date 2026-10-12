from __future__ import annotations

from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import evaluate_nonllm_baselines as baselines
from scripts import run_nonllm_campaign as campaign
from sts_env.provenance import interface_identity
from tests.unit import test_campaign_revision as revision


class CampaignBackendPinTests(unittest.TestCase):
    def test_frozen_and_nonfinal_children_override_different_inherited_backend(self):
        for final in (False, True):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                runner, models = revision.CampaignRevisionTests().fixture(Path(directory), stack)
                expected = str(Path(runner.state['native_build']).resolve())
                if final:
                    runner.freeze_final_selection(**models)
                inherited = Path(directory) / 'other-compatible-build'
                inherited.mkdir()
                inherited.joinpath('slaythespire.fake.so').write_bytes(b'synthetic binary')
                inherited.joinpath('build_manifest.json').write_bytes(Path(expected, 'build_manifest.json').read_bytes())
                process = MagicMock(pid=123, returncode=0)
                process.poll.return_value = 0
                with patch.dict(os.environ, {'STS_LIGHTSPEED_BUILD': str(inherited)}), \
                     patch.object(campaign.subprocess, 'Popen', return_value=process) as popen, \
                     patch('sys.stdout', io.StringIO()):
                    runner.run('final_probe' if final else 'ordinary_probe', ['python', 'unused.py'], 1)
                    self.assertEqual(os.environ['STS_LIGHTSPEED_BUILD'], str(inherited))
                child_env = popen.call_args.kwargs['env']
                self.assertEqual(child_env['STS_LIGHTSPEED_BUILD'], expected)
                record = runner.state['commands'][-1]
                self.assertEqual(record['effective_environment'], {'STS_LIGHTSPEED_BUILD': expected})
                identity = record['effective_backend_identity']
                self.assertEqual(identity['build_dir'], expected)
                self.assertEqual(identity['provenance_status'], 'verified')
                self.assertEqual(identity['module_sha256'], hashlib.sha256(b'synthetic binary').hexdigest())
                self.assertEqual(identity['native_provenance']['module_sha256'], identity['module_sha256'])

    def test_nonfinal_manifest_binary_mismatch_cannot_launch_child(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            runner, _ = revision.CampaignRevisionTests().fixture(Path(directory), stack)
            Path(runner.state['native_build'], 'slaythespire.fake.so').write_bytes(b'changed native binary')
            with patch.object(campaign.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'binary differs'):
                    runner.run('ordinary_probe', ['python', 'unused.py'], 1)
            popen.assert_not_called()

    def baseline_fixture(self, directory):
        build = Path(directory) / 'build'
        build.mkdir()
        module = build / 'slaythespire.fake.so'
        module.write_bytes(b'baseline-native-binary')
        provenance = {'module_file': module.name, 'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
                      'patch_sha256': 'fake-patch'}
        (build / 'build_manifest.json').write_text(json.dumps(provenance))
        return build, provenance

    def test_baseline_metrics_stamp_verified_native_and_python_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            build, provenance = self.baseline_fixture(directory)
            output = Path(directory) / 'dev.metrics.json'
            class Worker:
                def __init__(self, config, timeout):
                    self.config = config
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def reset(self, task):
                    return {'legal_actions': [], 'state': {'floor': 1, 'act': 1}}, \
                           {'terminated': True, 'outcome': 'battle_victory', 'task_outcome': 'battle_victory',
                            'termination_reason': 'battle_finished', 'hp_cost': 0., 'native_steps': 0}
            argv = ['baseline', '--output', str(output), '--episodes', '1', '--workers', '1',
                    '--world-start', '10000', '--split', 'dev', '--seconds', '1']
            with patch.dict(os.environ, {'STS_LIGHTSPEED_BUILD': str(build)}), \
                 patch.object(sys, 'argv', argv), patch.object(baselines, 'CurriculumWorker', Worker), \
                 patch('sys.stdout', io.StringIO()):
                baselines.main()
            metrics = json.loads(output.read_text())
            self.assertEqual(metrics['native_provenance'], provenance)
            self.assertEqual(metrics['training_environment']['sourced_env_manifest'], provenance)
            self.assertEqual(metrics['training_environment']['native_module_sha256'], provenance['module_sha256'])
            self.assertEqual(metrics['python_interface_sha256'], interface_identity()['python_interface_sha256'])
            self.assertEqual(metrics['runtime_python_version'], interface_identity()['runtime_python_version'])
            self.assertEqual(metrics['requested_tasks'], 1)
            self.assertEqual(metrics['episodes'][0]['world_seed'], 10011)

    def test_baseline_provenance_rejects_wrong_binary_before_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            build, _ = self.baseline_fixture(directory)
            (build / 'slaythespire.fake.so').write_bytes(b'not the stamped binary')
            argv = ['baseline', '--output', str(Path(directory) / 'dev.metrics.json'), '--episodes', '1']
            with patch.dict(os.environ, {'STS_LIGHTSPEED_BUILD': str(build)}), \
                 patch.object(sys, 'argv', argv), patch.object(baselines, 'CurriculumWorker') as worker:
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    baselines.main()
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
