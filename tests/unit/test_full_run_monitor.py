from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import monitor_full_run_eval as monitor


class MonitorTests(unittest.TestCase):
    def test_snapshot_immutable_sha_and_reuses_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.pt'; source.write_bytes(b'checkpoint')
            sha = monitor.sha_file(source)
            target = monitor.snapshot_checkpoint(source, root / 'snapshots', sha)
            source.unlink()
            self.assertEqual(monitor.snapshot_checkpoint(source, root / 'snapshots', sha), target)
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'modified'):
                monitor.snapshot_checkpoint(source, root / 'snapshots', sha)

    def test_snapshot_rejects_wrong_completed_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.pt'; source.write_bytes(b'incomplete')
            with self.assertRaisesRegex(RuntimeError, 'completed artifact SHA'):
                monitor.snapshot_checkpoint(source, root / 'snapshots', '0' * 64)
            self.assertEqual(list((root / 'snapshots').iterdir()), [])

    def test_completed_roles_ignore_partial_latest(self):
        state = {'best_checkpoint': 'best', 'initial_checkpoint': 'initial', 'global_round': 8,
                 'identity': {'initial_checkpoint_sha256': 'init'}, 'jobs': [
                     {'kind': 'train', 'status': 'complete', 'finished': 1, 'checkpoint': 'best', 'artifacts': {'best': 'A'}},
                     {'kind': 'train', 'status': 'complete', 'finished': 2, 'checkpoint': 'latest', 'artifacts': {'latest': 'B'}},
                     {'kind': 'train', 'status': 'running', 'checkpoint': 'partial'}]}
        rows = monitor.completed_candidates(state)
        self.assertEqual([(r['role'], r['checkpoint_sha256']) for r in rows], [('selected_best', 'A'), ('latest_candidate', 'B')])
        state['best_checkpoint'] = 'partial'
        with self.assertRaises(RuntimeError): monitor.completed_candidates(state)

    def test_pauses_allocation_liveness_and_explicit_hard_stop(self):
        state = {'status': 'running', 'allocation': {'used_seconds': 9, 'authorized_seconds': 10},
                 'identity': {'config': {'hard_wall_deadline': None}}, 'controller': {'pid': 12, 'process_birth': 'birth'}}
        with patch.object(monitor, 'process_birth', return_value='birth'):
            self.assertIsNone(monitor.learner_permission(state))
            for status in ('operator_paused', 'paused_resource', 'attention_required', 'pilot_complete', 'complete'):
                state['status'] = status
                self.assertEqual(monitor.learner_permission(state), 'learner_' + status)
            state['status'] = 'running'; state['allocation']['used_seconds'] = 10
            self.assertEqual(monitor.learner_permission(state), 'allocation_exhausted')
            state['allocation']['used_seconds'] = 9; state['identity']['config']['hard_wall_deadline'] = 100
            self.assertEqual(monitor.learner_permission(state, now=100), 'hard_deadline')
        state['identity']['config']['hard_wall_deadline'] = None
        with patch.object(monitor, 'process_birth', return_value=None):
            self.assertEqual(monitor.learner_permission(state), 'controller_not_live')

    def test_command_pins_cpu_fixed_dev3_and_constant_reward(self):
        protocol = {'panels': {'dev_act3': {'path': '/dev/tasks', 'task_ids': list(range(48))}}}
        command = monitor.build_command(Path('/snapshot'), Path('/metrics'), protocol)
        options = {command[i]: command[i + 1] for i in range(len(command) - 1) if command[i].startswith('--')}
        for key, value in {'--device': 'cpu', '--inference-device': 'cpu', '--workers': '4', '--threads': '1',
                           '--max-act': '3', '--horizon': '4096', '--episodes': '48', '--seed': '777',
                           '--simulations': '200', '--progress-floor-weight': '.05', '--terminal-hp-weight': '0'}.items():
            self.assertEqual(options[key], value)
        self.assertEqual(command[:3], ['/usr/bin/nice', '-n', '15'])
        self.assertNotIn('--split final', ' '.join(command)); self.assertIn('--preserve-task-prefixes', command)

    def test_exact_coverage_faults_not_defeats_or_truncations(self):
        metrics = {'rounds': [{'task_ids': ['a', 'b', 'c'], 'cohort_completed': True, 'unstarted_tasks': 0,
                             'episodes': [{'task_id': 'a', 'outcome': 'act_clear', 'floor': 51},
                                          {'task_id': 'b', 'outcome': 'defeat', 'floor': 10}],
                             'faults': [{'task_id': 'c', 'error': 'native fault'}]}]}
        result = monitor.outcome_record(metrics, ['a', 'b', 'c'])
        self.assertTrue(result['complete']); self.assertEqual((result['clears'], result['defeats'], result['faults'], result['truncated']), (1, 1, 1, 0))
        with self.assertRaisesRegex(RuntimeError, 'different DEV cohort'):
            monitor.outcome_record(metrics, ['a', 'b', 'd'])
        metrics['rounds'][0]['unstarted_tasks'] = 1
        self.assertFalse(monitor.outcome_record(metrics, ['a', 'b', 'c'])['complete'])

    def test_identical_checkpoint_condition_reuses_measured_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'snapshot.pt'; checkpoint.write_bytes(b'weights')
            metrics_path = root / 'done.json'; metrics_path.write_text('{}')
            observer = monitor.Observer.__new__(monitor.Observer)
            observer.output = root; observer.seconds = 300; observer.pins = {'backend': 'bound'}
            observer.protocol = {'panels': {'dev_act3': {'path': str(root / 'tasks'), 'task_ids': ['a']}}}
            (root / 'tasks').write_text('tasks')
            observer.validate_inputs = lambda: None
            capture = {'snapshot': str(checkpoint), 'checkpoint_sha256': monitor.sha_file(checkpoint)}
            signature = monitor.condition_signature(monitor.build_command(checkpoint, root / 'jobs/placeholder.metrics.json', observer.protocol, seconds=observer.seconds))
            result = {'condition_key': monitor.observer_condition_key(signature, observer.pins),
                      'metrics_path': str(metrics_path), 'metrics_sha256': monitor.sha_file(metrics_path)}
            observer.state = {'histories': [result]}
            with patch.object(monitor.subprocess, 'Popen', side_effect=AssertionError('No repeated evaluation')):
                got, reused = observer.evaluate(capture)
            self.assertIs(got, result); self.assertTrue(reused)
            metrics_path.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'metrics changed'): observer.evaluate(capture)

    def test_once_limit_keeps_next_due_zero_for_unmeasured_second_role(self):
        observer = monitor.Observer.__new__(monitor.Observer)
        observer.state = {'captures': [], 'latest': {}, 'next_due': 0}; observer.output = Path('/output')
        observer.allowed = lambda: None; observer.save = lambda: None
        observer.learner = {'best_checkpoint': 'best', 'initial_checkpoint': 'initial', 'global_round': 0,
                            'identity': {'initial_checkpoint_sha256': 'I'}, 'jobs': [
                                {'kind': 'train', 'status': 'complete', 'finished': 1, 'checkpoint': 'best', 'artifacts': {'best': 'B'}},
                                {'kind': 'train', 'status': 'complete', 'finished': 2, 'checkpoint': 'last', 'artifacts': {'last': 'L'}}]}
        observer.evaluate = lambda capture: ({'roles': [], 'checkpoint_sha256': capture['checkpoint_sha256']}, False)
        with patch.object(monitor, 'snapshot_checkpoint', return_value=Path('/snapshot')):
            self.assertEqual(observer.cycle(max_panels=1), 1)
        self.assertEqual(observer.state['next_due'], 0)
        self.assertEqual(set(observer.state['latest']), {'selected_best'})

    def test_observer_only_repair_preserves_measurement_key_but_science_change_rejects(self):
        old = {'observer_source_sha256': 'old', 'backend': 'bound'}
        new = {'observer_source_sha256': 'new', 'backend': 'bound'}
        signature = {'digest': 'same-scientific-command'}
        previous = {'pins': old, 'histories': [{'condition_signature': signature, 'evaluated_at': 12}]}
        result = monitor.restore_state(previous, new)
        self.assertEqual(result['histories'][0]['condition_key'], monitor.observer_condition_key(signature, old))
        self.assertEqual(result['histories'][0]['evaluated_at'], 12)
        self.assertEqual(len(result['observer_source_amendments']), 1)
        with self.assertRaisesRegex(RuntimeError, 'scientific inputs changed'):
            monitor.restore_state(previous, dict(new, backend='different'))

    def test_once_publishes_no_active_evaluation_after_return(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = monitor.Observer.__new__(monitor.Observer)
            observer.output = Path(directory); observer.journal_path = observer.output / 'journal.json'
            observer.state = {'status': 'evaluating'}; observer.stop = False
            observer.recover = lambda: None; observer.save = lambda: None
            observer.cycle = lambda **kwargs: 1
            observer.run(once=True, max_panels=1)
            self.assertEqual(observer.state['status'], 'one_shot_complete')

    def test_retention_race_retries_capture_in_30_seconds_not_30_minutes(self):
        observer = monitor.Observer.__new__(monitor.Observer)
        observer.state = {'captures': [], 'latest': {}, 'next_due': 0}; observer.output = Path('/output')
        observer.interval = 1800; observer.allowed = lambda: None; observer.save = lambda: None
        observer.learner = {'best_checkpoint': 'initial', 'initial_checkpoint': 'initial', 'global_round': 0,
                            'identity': {'initial_checkpoint_sha256': 'I'}, 'jobs': []}
        with patch.object(monitor, 'snapshot_checkpoint', side_effect=FileNotFoundError), patch.object(monitor.time, 'time', return_value=100):
            self.assertEqual(observer.cycle(), 0)
        self.assertEqual(observer.state['next_due'], 130)
        self.assertEqual(observer.state['status'], 'capture_retry')

    def test_failure_publishes_attention_under_observer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = monitor.Observer.__new__(monitor.Observer)
            observer.output = Path(directory); observer.journal_path = observer.output / 'journal.json'
            observer.state = {'status': 'starting'}; observer.stop = False
            saved = []
            observer.recover = lambda: None
            observer.save = lambda: saved.append(dict(observer.state))
            def fail(**kwargs): raise RuntimeError('bad panel input')
            observer.cycle = fail
            with self.assertRaisesRegex(RuntimeError, 'bad panel input'): observer.run(once=True)
            self.assertEqual(saved[-1]['status'], 'attention_required')
            self.assertEqual(saved[-1]['error'], 'bad panel input')


if __name__ == '__main__': unittest.main()
