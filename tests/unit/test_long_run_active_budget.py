"""Active allocation and control CLI tests; no learner/native/GPU child is launched."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import run_long_curriculum as cli
from sts_policy import long_run_control as control


class ActiveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.directory)
        self.clock = {'wall': 1800000000., 'mono': 100.}
        self.stack.enter_context(patch.object(control.time, 'time', side_effect=lambda: self.clock['wall']))
        self.stack.enter_context(patch.object(control.time, 'monotonic', side_effect=lambda: self.clock['mono']))
        self.stack.enter_context(patch.object(control, 'process_birth', return_value='fixture controller birth'))
        build = self.root / 'build'; build.mkdir()
        module = build / 'slaythespire.fake.so'; module.write_bytes(b'fake native identity')
        (build / 'build_manifest.json').write_text(json.dumps({'module_file': module.name,
            'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest()}))
        model = self.root / 'initial.pt'; model.write_bytes(b'initial frozen weights')
        self.config = control.LongRunConfig(checkpoint=str(model), backend_build=str(build),
            pool_bank=str(self.root / 'bank'), pool_panels=str(self.root / 'panels'),
            deadline=self.clock['wall'] + 36000, duration_hours=10.,
            device='cpu', prevent_sleep=False, workers=8, duty_cycle=1.)
        self.output = self.root / 'run'
        self.source_root = Path(__file__).resolve().parents[2]

    def controller(self, config=None):
        runner = control.LongRunController(self.source_root, self.output, config or self.config)
        # Avoid a stale closed fixture object saving over a later resumed object.
        self.stack.callback(lambda: runner.close() if runner._run_lock is not None else None)
        return runner

    def advance(self, seconds):
        self.clock['wall'] += seconds; self.clock['mono'] += seconds

    def test_only_active_intervals_charge_and_close_preserves_partial_interval(self):
        runner = self.controller()
        self.advance(7200)
        self.assertEqual(runner.remaining_seconds(), 36000.)
        runner.set_accounting(True); self.advance(120); runner.account_time()
        self.assertEqual(runner.state['allocation']['used_seconds'], 120.)
        runner.set_accounting(False); self.advance(18000)
        self.assertEqual(runner.remaining_seconds(), 35880.)
        runner.set_accounting(True); self.advance(37); runner.close()
        saved = json.loads(runner.journal.read_text())['allocation']
        self.assertEqual(saved['used_seconds'], 157.)
        self.assertEqual(saved['initial_seconds'], 36000.)

    def test_pilot_then_main_resume_excludes_offline_gap_without_resetting_budget(self):
        pilot = self.controller(); pilot.session_kind = 'pilot'
        pilot.session_end = self.clock['wall'] + 900
        pilot.set_accounting(True); self.advance(600); pilot.close()
        self.advance(86400)
        resumed = self.controller(replace(self.config, deadline=self.clock['wall'] + 35400))
        self.assertEqual(resumed.state['allocation']['used_seconds'], 600.)
        self.assertEqual(resumed.remaining_seconds(), 35400.)
        resumed.set_accounting(True); self.advance(300); resumed.close()
        self.assertEqual(json.loads(resumed.journal.read_text())['allocation']['used_seconds'], 900.)

    def test_extension_adds_only_final_stage_and_keeps_reward_weights_constant(self):
        runner = self.controller()
        initial = dict(runner.state['allocation']['stage_boundaries'])
        reward = control.goal_signature(runner.config, 3, control.STAGES[-1])
        runner.extend_hours(2.)
        allocation = runner.state['allocation']
        self.assertEqual(allocation['initial_seconds'], 36000.)
        self.assertEqual(allocation['authorized_seconds'], 43200.)
        self.assertEqual(len(allocation['extensions']), 1)
        self.assertEqual(allocation['extensions'][0]['authority'], 'explicit_operator_extension')
        for stage in control.STAGES[:-1]:
            self.assertEqual(allocation['stage_boundaries'][stage.name], initial[stage.name])
        self.assertEqual(allocation['stage_boundaries']['mixed_full3'], initial['mixed_full3'] + 7200)
        self.assertEqual(control.goal_signature(runner.config, 3, control.STAGES[-1]), reward)
        self.assertEqual(runner.config.progress_floor_weight, .05)
        self.assertEqual(runner.config.terminal_hp_weight, 0.)
        runner.close(); self.advance(86400)
        resumed = self.controller(replace(self.config, deadline=self.clock['wall'] + 43200))
        self.assertEqual(resumed.remaining_seconds(), 43200.)
        self.assertEqual(len(resumed.state['allocation']['extensions']), 1)

    def test_stage_deadline_tracks_remaining_active_time_and_invocation_cap(self):
        runner = self.controller()
        stage = control.STAGES[0]
        expected = self.clock['wall'] + 10800
        self.assertAlmostEqual(runner.stage_deadline(stage), expected)
        self.advance(86400)  # Inactive wall time must not expire the stage.
        self.assertAlmostEqual(runner.stage_deadline(stage.name), self.clock['wall'] + 10800)
        runner.set_accounting(True); self.advance(100)
        self.assertAlmostEqual(runner.stage_deadline(stage), self.clock['wall'] + 10700)
        runner.session_end = self.clock['wall'] + 50
        self.assertEqual(runner.stage_deadline(stage), runner.session_end)

    def test_resource_pause_recomputes_stage_job_deadline_and_launches_only_fake_child(self):
        runner = self.controller()
        runner.current_stage = control.STAGES[0].name
        stale_stage_end = runner.stage_deadline(runner.current_stage)
        runner.set_accounting(True); self.advance(300)
        metrics = self.output / 'fake.metrics.json'
        command = ['python', '-c', 'not executed', '--metrics', str(metrics)]
        process = MagicMock(pid=123456, returncode=0)
        process.poll.return_value = 0
        def fake_child(*args, **kwargs):
            metrics.write_text(json.dumps({'probe': 'completed fake child'}))
            return process
        def resource_wait(seconds):
            self.assertFalse(runner._clock_active)
            self.advance(86400)  # Resource pause crosses the old UTC stage end.
        with patch.object(control, 'resource_pause', side_effect=['battery_below_threshold', None]), \
             patch.object(control.time, 'sleep', side_effect=resource_wait), \
             patch.object(control.subprocess, 'Popen', side_effect=fake_child) as launch, \
             patch.object(control, 'track_owned_members'), patch.object(control, 'stop_owned_group', return_value=True):
            result, record = runner.run_job(command, kind='evaluate', budget=180,
                                            stage_end=stale_stage_end)
        launch.assert_called_once()
        self.assertEqual(result, {'probe': 'completed fake child'})
        self.assertEqual(record['status'], 'complete')
        self.assertGreater(record['started'], stale_stage_end)
        self.assertEqual(runner.state['allocation']['used_seconds'], 300.)
        self.assertEqual(runner.remaining_seconds(), 35700.)
        self.assertEqual([e['type'] for e in runner.state['events']], ['resource_pause', 'resource_resume'])

    def test_exhaustion_invalid_extensions_and_changed_science_fail_closed(self):
        runner = self.controller(); runner.set_accounting(True); self.advance(36001)
        self.assertEqual(runner.remaining_seconds(), 0.)
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError): runner.extend_hours(value)
        runner.close()
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.controller(replace(self.config, progress_floor_weight=.04,
                                    deadline=self.clock['wall'] + 36000))


class ActiveBudgetCliTests(unittest.TestCase):
    def journal(self, root):
        output = root / 'run'; output.mkdir()
        state = {'status': 'operator_paused', 'allocation': {'initial_seconds': 36000.,
            'authorized_seconds': 36000., 'used_seconds': 600., 'extensions': []},
            'controller': {'pid': 123456, 'process_birth': 'verified birth'},
            'current_checkpoint': 'recorded.pt', 'best_checkpoint': 'recorded.pt', 'last_checkpoint': 'recorded.pt'}
        path = output / 'journal.json'; path.write_text(json.dumps(state))
        return output, path

    def test_status_is_read_only_and_needs_no_backend_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output, path = self.journal(Path(directory)); before = path.read_bytes()
            stream = io.StringIO()
            with patch.object(sys, 'argv', ['controller', '--action', 'status', '--output', str(output)]), \
                 patch.object(cli, 'LongRunController') as constructor, patch('sys.stdout', stream):
                cli.main()
            constructor.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(stream.getvalue())['allocation']['used_seconds'], 600.)

    def test_pause_signals_only_matching_controller_birth_and_stale_pid_is_untouched(self):
        for birth, expected in [('verified birth', True), ('another birth', False), (None, False)]:
            with self.subTest(birth=birth), tempfile.TemporaryDirectory() as directory:
                output, path = self.journal(Path(directory)); before = path.read_bytes()
                with patch.object(sys, 'argv', ['controller', '--action', 'pause', '--output', str(output)]), \
                     patch.object(control, 'process_birth', return_value=birth), \
                     patch('os.kill') as signal, patch('sys.stdout', io.StringIO()), \
                     patch.object(cli, 'LongRunController') as constructor:
                    cli.main()
                self.assertEqual(signal.called, expected)
                if expected: signal.assert_called_once_with(123456, cli.signal.SIGTERM)
                constructor.assert_not_called(); self.assertEqual(path.read_bytes(), before)

    def test_fake_run_pause_closes_accounting_and_resume_command_cannot_repeat_extension(self):
        for extension in (['--extend-hours', '2'], ['--extend-hours=2']):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); output, journal = self.journal(root)
                state = json.loads(journal.read_text())
                state['identity'] = {'config': {'duration_hours': 10.}}
                journal.write_text(json.dumps(state))
                protocol = root / 'protocol.json'
                protocol.write_text(json.dumps({'checkpoint': {'path': str(root/'source.pt')}}))
                runner = MagicMock(); runner.state = state
                runner.execute.side_effect = KeyboardInterrupt('operator pause')
                argv = ['controller', '--action', 'run', '--execute', '--output', str(output),
                        '--protocol', str(protocol), '--backend-build', str(root/'build'),
                        '--pool-bank', str(root/'bank'), '--pool-panels', str(root/'panels'),
                        '--device', 'cpu', '--session-seconds', '900', '--session-kind', 'pilot', *extension]
                with patch.object(sys, 'argv', argv), patch.object(cli, 'LongRunController', return_value=runner), \
                     patch.object(cli.signal, 'signal'), patch('sys.stdout', io.StringIO()):
                    cli.main()
                runner.execute.assert_called_once_with(session_seconds=900., session_kind='pilot')
                runner.extend_hours.assert_called_once_with(2.)
                runner.set_accounting.assert_called_with(False)
                self.assertEqual(runner.state['status'], 'operator_paused')
                runner.close.assert_called_once()
                saved = json.loads((output/'main_command.json').read_text())['argv']
                self.assertFalse(any(token.startswith('--extend-hours') for token in saved))
                self.assertNotIn('--session-seconds', saved)
                self.assertEqual(saved[-2:], ['--session-kind', 'main'])

    def test_plan_defaults_are_ten_active_hours_dedicated_gpu_and_eight_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol = root / 'protocol.json'
            protocol.write_text(json.dumps({'checkpoint': {'path': str(root/'source.pt')}}))
            argv = ['controller', '--output', str(root/'run'), '--protocol', str(protocol),
                    '--backend-build', str(root/'build'), '--pool-bank', str(root/'bank'),
                    '--pool-panels', str(root/'panels')]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), patch('sys.stdout', stream), \
                 patch.object(cli, 'LongRunController') as constructor:
                cli.main()
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['mode'], 'plan_only')
            self.assertEqual(plan['config']['duration_hours'], 10.)
            self.assertEqual(plan['config']['workers'], 8)
            self.assertEqual(plan['config']['duty_cycle'], 1.)
            self.assertEqual(plan['config']['progress_floor_weight'], .05)
            self.assertEqual(plan['config']['terminal_hp_weight'], 0.)
            constructor.assert_not_called(); self.assertFalse((root/'run').exists())
