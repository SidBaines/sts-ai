from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class QueueRecordingTests(unittest.TestCase):
    def check_completion(self, *, child_code, fresh_done):
        spec=importlib.util.spec_from_file_location('rl_queue_test',Path(__file__).parents[2]/'scripts/rl/queue_runner.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';out.mkdir()
            old='{"event": "done"}\n'
            (out/'log.jsonl').write_text(old+(' {"event": "done"}\n' if fresh_done else ''))
            run={'name':'run','cmd':['python','--out',str(out)],'status':'running','pid':123,
                 'log_start_bytes':len(old)}
            child=Mock();child.poll.return_value=child_code
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(module,'ROOT',root))
                stack.enter_context(patch.object(module,'STATUS',root/'status.json'))
                stack.enter_context(patch.object(module,'Locked',return_value=contextlib.nullcontext()))
                stack.enter_context(patch.object(module,'read_queue',return_value={'runs':[run]}))
                stack.enter_context(patch.object(module,'write_queue'))
                stack.enter_context(patch.object(module,'on_battery',return_value=False))
                stack.enter_context(patch.object(module,'alive',return_value=False))
                stack.enter_context(patch.object(module.os,'waitpid',side_effect=ChildProcessError))
                stack.enter_context(patch.object(module.time,'sleep',side_effect=StopIteration))
                if child_code is not None:
                    module.LAUNCHED[123]=child
                with self.assertRaises(StopIteration): module.runner()
            return run

    def test_known_nonzero_exit_cannot_be_overruled_by_done_marker(self):
        run=self.check_completion(child_code=2,fresh_done=True)
        self.assertEqual(run['returncode'],2)
        self.assertEqual(run['status'],'failed')

    def test_lost_child_exit_does_not_reuse_done_from_prior_launch(self):
        run=self.check_completion(child_code=None,fresh_done=False)
        self.assertIsNone(run['returncode'])
        self.assertEqual(run['status'],'failed')

    def test_boolean_bank_flag_is_not_a_filename(self):
        spec = importlib.util.spec_from_file_location('rl_queue_inputs', Path(__file__).parents[2] / 'scripts/rl/queue_runner.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(module.launch_inputs(['python', '--bank', '--no-teacher-labels', '--n', '256']), [])
        self.assertEqual(module.launch_inputs(['python', '--bank']), [])
        self.assertEqual(module.launch_inputs(['python', '--bank', '/bank/a', '/bank/b', '--out', '/run']), ['/bank/a', '/bank/b'])
        self.assertEqual(module.launch_inputs(['python', '--records', '/rows.json', '--baseline', '/baseline.json']), ['/rows.json', '/baseline.json'])

    def test_last_activity_counts_collection_progress(self):
        spec = importlib.util.spec_from_file_location('rl_queue_activity', Path(__file__).parents[2] / 'scripts/rl/queue_runner.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs = root / 'data/rl_runs'
            with patch.object(module, 'ROOT', root), patch.object(module, 'RUNS', runs):
                for output in (None, 'relative-labels', str(root / 'absolute-labels')):
                    with self.subTest(output=output):
                        run = {'name': 'queued', 'cmd': ['python'], 'started': 100, 'resumed_at': 150}
                        if output is not None:
                            run['cmd'] += ['--out', output]
                        out = root / output if output is not None else runs / run['name']
                        out.mkdir(parents=True, exist_ok=True)
                        stdout = runs / run['name'] / 'stdout.log'
                        stdout.parent.mkdir(parents=True, exist_ok=True)
                        stdout.write_text('working\n')
                        os.utime(stdout, (200, 200))
                        self.assertEqual(module.last_activity(run), (200, False))
                        log = out / 'log.jsonl'
                        log.write_text('{}\n')
                        os.utime(log, (250, 250))
                        self.assertEqual(module.last_activity(run), (250, True))
                        progress = out / 'progress.jsonl'
                        progress.write_text('{}\n')
                        os.utime(progress, (300, 300))
                        self.assertEqual(module.last_activity(run), (300, True))
                        log.unlink()
                        self.assertEqual(module.last_activity(run), (300, True))

    def test_stalled_job_fails_without_rewriting_command_or_retry(self):
        spec = importlib.util.spec_from_file_location('rl_queue_stall', Path(__file__).parents[2] / 'scripts/rl/queue_runner.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        run = {'name': 'owned', 'cmd': ['python', 'fit.py'], 'status': 'running', 'pid': 123, 'started': 0}
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(module, 'ROOT', Path(td)))
            stack.enter_context(patch.object(module, 'STATUS', Path(td) / 'status.json'))
            stack.enter_context(patch.object(module, 'Locked', return_value=contextlib.nullcontext()))
            stack.enter_context(patch.object(module, 'read_queue', return_value={'runs': [run]}))
            stack.enter_context(patch.object(module, 'write_queue'))
            stack.enter_context(patch.object(module, 'on_battery', return_value=False))
            child = Mock(); child.poll.return_value = None; module.LAUNCHED[123] = child
            stack.enter_context(patch.object(module, 'last_activity', return_value=(0, True)))
            stack.enter_context(patch.object(module.time, 'time', return_value=999999))
            kill = stack.enter_context(patch.object(module.os, 'killpg'))
            stack.enter_context(patch.object(module.time, 'sleep', side_effect=StopIteration))
            with self.assertRaises(StopIteration): module.runner()
            kill.assert_called_once_with(123, module.signal.SIGKILL)
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(run['cmd'], ['python', 'fit.py'])
        self.assertNotIn('retried', run)


class RunnerLockTests(unittest.TestCase):
    def test_second_runner_cannot_take_the_lock_until_the_first_releases_it(self):
        spec = importlib.util.spec_from_file_location('rl_queue_lock', Path(__file__).parents[2] / 'scripts/rl/queue_runner.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runner.lock'
            first = module.hold_runner_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(module.hold_runner_lock(path))
            first.close()
            again = module.hold_runner_lock(path)
            self.assertIsNotNone(again)
            again.close()
