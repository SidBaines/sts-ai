"""Native-free joint-loop regressions; all outputs and queue records are temporary."""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / 'experiments/2026-10-22-joint-run-2/joint_loop.py'
spec = importlib.util.spec_from_file_location('joint_run_2', SCRIPT)
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def summary(start=100, n=4, wins=1, floor=20):
    return {'completed': n, 'cohort_complete': True, 'confidence_claims_allowed': True,
            'records': [{'world_seed': start + i, 'status': 'completed', 'win': i < wins,
                         'floor': floor} for i in range(n)]}


class DecisionTests(unittest.TestCase):
    def test_fit_requires_both_completed_status_and_checkpoint(self):
        for status in ({}, {'status': 'running'}, {'status': 'preencoding'}):
            self.assertEqual(joint.classify_fit_outcome(status, True)[0], 'pending')
        self.assertEqual(joint.classify_fit_outcome({'status': 'completed'}, True), ('success', 'completed'))
        self.assertEqual(joint.classify_fit_outcome({'status': 'completed'}, False)[0], 'failed')

    def test_terminal_fit_statuses_fail_even_with_best_checkpoint(self):
        for status in ('interrupted', 'failed', 'error', 'cancelled', 'incomplete'):
            with self.subTest(status=status):
                self.assertEqual(joint.classify_fit_outcome({'status': status}, True)[0], 'failed')
        self.assertEqual(joint.classify_fit_outcome({'status': 'interrupted', 'reason': 'disk_below_6_GiB'}, False),
                         ('failed', 'disk_below_6_GiB'))
        self.assertIn('bad weights', joint.classify_fit_outcome(
            {'status': 'failed', 'reason': 'fit_error', 'error': 'ValueError: bad weights'}, False)[1])

    def test_completed_fit_summary_overrides_stale_status_but_requires_best(self):
        for state in ('running', 'preencoding', 'interrupted'):
            self.assertEqual(joint.classify_fit_outcome({'status': state}, True,
                             summary={'status': 'completed'}), ('success', 'completed'))
        self.assertEqual(joint.classify_fit_outcome({}, False, summary={'status': 'completed'})[0], 'failed')

    def test_fit_failure_retains_errors_even_when_heartbeat_left_running_status(self):
        status = {'status': 'running', 'error': 'ValueError: bad weights'}
        for kwargs in ({'queue_status': 'done'}, {'timed_out': True}):
            with self.subTest(kwargs=kwargs):
                outcome, reason = joint.classify_fit_outcome(status, False, **kwargs)
                self.assertEqual(outcome, 'failed')
                self.assertIn('ValueError: bad weights', reason)

    def test_every_terminal_queue_state_without_fit_artifacts_fails(self):
        for state in ('done', 'failed', 'exited', 'cancelled', 'error', 'missing', 'unknown_terminal'):
            with self.subTest(state=state):
                self.assertEqual(joint.classify_fit_outcome({}, False, state)[0], 'failed')
        for state in ('pending', 'running', 'paused'):
            self.assertEqual(joint.classify_fit_outcome({}, False, state)[0], 'pending')
        self.assertEqual(joint.classify_fit_outcome({}, False, 'running', True), ('failed', 'timeout'))

    def test_label_threshold_uses_completed_worlds_only_and_accepts_equality(self):
        self.assertTrue(joint.label_accepted([{'completed': 3}, {'faults': 1}], 4))
        self.assertFalse(joint.label_accepted([{'completed': 2, 'faults': 2}], 4))
        self.assertTrue(joint.label_accepted([{'completed': 2}], 4, .5))
        self.assertFalse(joint.label_accepted([], 0))

    def test_gate_rule_and_selection_are_unchanged(self):
        def result(wins, floor):
            return {'wins': wins, 'floor': {'difference': floor}}
        for wins, floor, accepted in [([4, 4], 0, True), ([4, 5], 0, True), ([4, 3], 5, False),
                                      ([4, 5], -.01, False), ([4, 4], -.01, False)]:
            self.assertEqual(joint.gate_decision({7: result(wins, floor)}), (7, accepted))
        results = {1: result([4, 5], 1), 2: result([4, 5], 2), 3: result([4, 4], 3)}
        self.assertEqual(joint.gate_decision(results), (2, True))
        # A lower-ranked passing student must not replace the selected best one.
        self.assertEqual(joint.gate_decision({1: result([4, 5], -1), 2: result([4, 4], 1)}), (1, False))
        self.assertEqual(joint.gate_decision({}), (None, False))

    def test_cycle_start_boundaries_and_stop_conditions(self):
        self.assertTrue(joint.should_start_cycle(0, 2, 24, 21))
        self.assertFalse(joint.should_start_cycle(0, 2, 24, 22))
        self.assertFalse(joint.should_start_cycle(2, 2, 24, 0))
        self.assertFalse(joint.should_start_cycle(0, 2, 24, 0, stopped=True))
        self.assertFalse(joint.should_start_cycle(0, 2, 24, 0, trainer_done=True))
        self.assertFalse(joint.should_start_cycle(0, 2, None, 0))
        self.assertTrue(joint.should_start_cycle(0, 2, 24, 22, min_iterations_left=2))

    def test_paired_arithmetic_and_exact_mcnemar(self):
        base, arm = summary(wins=1, floor=10), summary(wins=3, floor=12)
        arm['records'].reverse()
        result = joint.compare_summaries(base, arm, range(100, 104))
        self.assertEqual(result['wins'], [1, 3])
        self.assertEqual(result['floor'], {'baseline': 10, 'arm': 12, 'difference': 2})
        self.assertEqual(result['only_baseline'], 0)
        self.assertEqual(result['only_arm'], 2)
        self.assertAlmostEqual(result['mcnemar_p'], .5)
        self.assertEqual(joint.compare_summaries(base, base)['mcnemar_p'], 1.)

    def test_invalid_and_excessively_incomplete_gate_cohorts_are_rejected(self):
        base = summary()
        variants = []
        for key, value in [('status', 'fault'), ('win', 1), ('floor', float('nan')), ('cut', True),
                           ('floor', float('inf')), ('ub', True)]:
            arm = deepcopy(base)
            arm['records'][0][key] = value
            variants.append(arm)
        variants += [{'records': []}, {'error': 'failed', **base}, {**base, 'programming_errors': 1},
                     {**base, 'unstarted': 1}]
        for status in ('programming_error', 'unstarted'):
            arm = deepcopy(base)
            arm['records'][0]['status'] = status
            variants.append(arm)
        arm = deepcopy(base)
        arm['records'].append(arm['records'][0])
        variants.append(arm)
        arm = deepcopy(base)
        arm['records'].pop()
        variants.append(arm)
        arm = deepcopy(base)
        arm['records'][0]['world_seed'] = 999
        variants.append(arm)
        for arm in variants:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                joint.compare_summaries(base, arm, range(100, 104))
        with self.assertRaisesRegex(ValueError, 'intersection missing'):
            joint.compare_summaries(base, summary(start=200))

    def test_gate_uses_completed_intersection_at_one_percent_missing(self):
        for status in ('fault', 'censored'):
            with self.subTest(status=status):
                base, arm = summary(n=100, wins=2, floor=10), summary(n=100, wins=3, floor=12)
                arm.update(completed=99, faults=1, cuts=1, censored=1,
                           confidence_claims_allowed=False, cohort_complete=False)
                arm['records'][0].update(status=status, win=False, floor=0, cut=True)
                arm['records'].reverse()
                result = joint.compare_summaries(base, arm, range(100, 200))
                self.assertEqual((result['worlds'], result['dropped_worlds']), (99, 1))
                self.assertEqual(result['wins'], [1, 2])
                self.assertEqual(result['floor'], {'baseline': 10, 'arm': 12, 'difference': 2})
                self.assertEqual((result['only_baseline'], result['only_arm']), (0, 1))

    def test_missing_limit_counts_intersection_not_each_arm_separately(self):
        base, arm = summary(n=100), summary(n=100)
        base['records'][0]['status'] = 'fault'
        arm['records'][1]['status'] = 'censored'
        with self.assertRaisesRegex(ValueError, 'missing 2/100'):
            joint.compare_summaries(base, arm, range(100, 200))

    def test_duplicate_or_outside_seed_is_invalid_even_in_dropped_records(self):
        for seed in (100, 999):
            arm = summary(n=100)
            arm['records'].append({'world_seed': seed, 'status': 'censored'})
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                joint.compare_summaries(summary(n=100), arm, range(100, 200))

    def test_missing_ranges_include_holes_and_do_not_repeat_completed_worlds(self):
        self.assertEqual(joint.missing_ranges(100, 8, {100, 102, 106}), [(101, 1), (103, 3), (107, 1)])


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(out=str(self.root / 'out'), code=str(SCRIPT.parents[2]), python='/fake/python',
                        build='/fake/build', queue_root=str(self.root / 'queue'), name='joint-test',
                        lane_prefix='test-', trainer_out=str(self.root / 'trainer'),
                        start=str(self.root / 'start.pt'), follow=str(self.root / 'follow.pt'),
                        max_cycles=1, worlds_per_cycle=32, label_world_start=1000, label_world_stride=100,
                        teacher_probabilities={'BOSS2': 1}, seeds=[1], base_corpus_shards=['base-shard'],
                        probe='probe', gate_world_start=100, gate_worlds=4)
        Path(self.cfg['start']).write_bytes(b'player')
        Path(self.cfg['follow']).write_bytes(b'incumbent-combat')
        write_json(Path(self.cfg['trainer_out']) / 'config.json', {'iterations': 24})
        (Path(self.cfg['trainer_out']) / 'log.jsonl').touch()
        self.loop = joint.Loop(self.cfg)
        self.queue_path = Path(self.cfg['queue_root']) / 'data/rl_runs/queue.json'
        self.runs, self.calls = [], []
        self.on_add = self.default_artifacts
        self.now = 0.
        self.addCleanup(patch.stopall)
        patch.object(joint.subprocess, 'run', side_effect=self.fake_run).start()
        patch.object(joint.time, 'monotonic', side_effect=lambda: self.now).start()
        patch.object(joint.time, 'sleep', side_effect=self.advance).start()
        patch.object(self.loop, 'free_gib', return_value=100).start()
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def advance(self, seconds):
        self.now += seconds

    def events(self):
        path = self.loop.out / 'decisions.jsonl'
        return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

    def save_queue(self):
        write_json(self.queue_path, {'runs': self.runs})

    def fake_run(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[1] == self.loop.runner:
            action = cmd[2]
            if action == 'add':
                name = cmd[3]
                if any(r['name'] == name for r in self.runs):
                    raise AssertionError(f'colliding name submitted: {name}')
                child = cmd[cmd.index('--') + 1:]
                out = Path(child[child.index('--out') + 1])
                out.mkdir(parents=True, exist_ok=True)
                run = {'name': name, 'cmd': child, 'status': 'done'}
                self.runs.append(run)
                self.on_add(child, out, run)
                self.save_queue()
            elif action == 'cancel':
                next(r for r in self.runs if r['name'] == cmd[3])['status'] = 'cancelled'
                self.save_queue()
            elif action != 'start':
                raise AssertionError(cmd)
        elif cmd[1].endswith('combine_heads.py'):
            Path(cmd[cmd.index('--output') + 1]).write_bytes(b'combined')
        else:
            raise AssertionError(f'Unexpected subprocess: {cmd}')
        return subprocess.CompletedProcess(cmd, 0)

    def default_artifacts(self, cmd, out, run):
        if any(s.endswith('fresh_collect.py') for s in cmd):
            n, start = int(cmd[cmd.index('--n') + 1]), int(cmd[cmd.index('--world-start') + 1])
            write_json(out / 'completion.json', {'completed': n, 'status': 'completed'})
            write_json(out / 'summary.json', summary(start, n))
        elif any(s.endswith('fresh_train.py') for s in cmd):
            write_json(out / 'status.json', {'status': 'completed'})
            (out / 'best.pt').write_bytes(b'student')
        else:
            write_json(out / 'summary.json', summary(wins=1 if 'incumbent' in out.name else 2,
                                                     floor=20 if 'incumbent' in out.name else 21))

    def corpus(self):
        path = self.loop.out / 'corpus_00'
        write_json(path / 'corpus.json', {'accepted': True, 'shards': ['new-shard']})
        return path

    def adds(self, script=None):
        return [c for c in self.calls if len(c) > 2 and c[2] == 'add'
                and (script is None or any(s.endswith(script) for s in c))]

    def test_zero_exit_interrupted_fit_retries_into_fresh_directory(self):
        def artifact(cmd, out, run):
            if '_retry1' not in out.name:
                write_json(out / 'status.json', {'status': 'interrupted', 'reason': 'disk_below_6_GiB'})
            else:
                self.default_artifacts(cmd, out, run)
        self.on_add = artifact
        best = self.loop.fit(0, [self.corpus()], 7)
        self.assertEqual(best.parent.name, 'fit_c00_seed7_retry1')
        self.assertEqual([e['event'] for e in self.events()], ['fit_failed', 'fit_retry', 'fit_ok'])
        self.assertEqual(len(self.adds()), 2)
        self.assertTrue(self.adds()[1][3].endswith('-retry1'))

    def test_two_failed_fits_drop_seed_and_never_wait_for_best(self):
        self.on_add = lambda cmd, out, run: write_json(out / 'status.json', {'status': 'error'})
        self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertEqual(len(self.adds()), 2)
        self.assertEqual(self.now, 0)

    def test_done_queue_without_success_artifacts_fails_instead_of_hanging(self):
        self.on_add = lambda *args: None
        self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertEqual(len(self.adds()), 2)
        self.assertEqual(self.now, 0)

    def test_completed_status_without_best_fails(self):
        self.on_add = lambda cmd, out, run: write_json(out / 'status.json', {'status': 'completed'})
        self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertEqual(self.now, 0)

    def test_fit_accepts_completed_summary_and_reports_status_error_text(self):
        def artifact(cmd, out, run):
            write_json(out / 'status.json', {'status': 'running'})
            write_json(out / 'summary.json', {'status': 'completed'})
            (out / 'best.pt').write_bytes(b'student')
        self.on_add = artifact
        self.assertIsNotNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertEqual(len(self.adds()), 1)
        self.on_add = lambda cmd, out, run: write_json(
            out / 'status.json', {'status': 'failed', 'error': 'ValueError: broken fit'})
        self.assertIsNone(self.loop.fit(1, [self.corpus()], 1))
        failures = [e for e in self.events() if e['event'] == 'fit_failed']
        self.assertEqual(len(failures), 2)
        self.assertTrue(all('ValueError: broken fit' in e['reason'] for e in failures))

    def test_fit_timeout_cancels_each_attempt_before_retry(self):
        self.cfg['fit_timeout_seconds'] = 20
        self.on_add = lambda cmd, out, run: run.update(status='running')
        self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        actions = [c[2] for c in self.calls]
        self.assertEqual(actions, ['add', 'start', 'cancel', 'add', 'start', 'cancel'])
        self.assertEqual(self.now, 40)

    def test_unique_names_and_existing_output_preservation(self):
        name = 'joint-test-c00-fit1'
        self.runs.append({'name': name, 'status': 'running'})
        self.save_queue()
        old = self.loop.out / 'fit_c00_seed1'
        old.mkdir()
        (old / 'keep').write_text('preserved')
        best = self.loop.fit(0, [self.corpus()], 1)
        self.assertEqual(best.parent.name, 'fit_c00_seed1_rerun1')
        self.assertEqual(self.adds()[0][3], name + '-rerun1')
        self.assertEqual((old / 'keep').read_text(), 'preserved')
        self.assertEqual(self.runs[0]['status'], 'running')

    def test_failed_start_cancels_only_new_unique_job(self):
        name = 'joint-test-c00-fit1'
        self.runs.append({'name': name, 'status': 'running'})
        self.save_queue()
        original = self.fake_run
        def fail_start(cmd, **kwargs):
            if cmd[2] == 'start':
                self.calls.append(cmd)
                raise subprocess.CalledProcessError(1, cmd)
            return original(cmd, **kwargs)
        with patch.object(joint.subprocess, 'run', side_effect=fail_start):
            self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        cancelled = [c[3] for c in self.calls if c[2] == 'cancel']
        self.assertEqual(cancelled, [name + '-rerun1', name + '-retry1'])
        self.assertEqual(self.runs[0]['status'], 'running')

    def test_name_collision_during_add_is_rechecked_without_cancelling_old_job(self):
        original = self.fake_run
        raced = False
        def race(cmd, **kwargs):
            nonlocal raced
            if cmd[2] == 'add' and not raced:
                raced = True
                self.runs.append({'name': cmd[3], 'status': 'running'})
                self.save_queue()
                raise subprocess.CalledProcessError(1, cmd, stderr=b'already queued')
            return original(cmd, **kwargs)
        with patch.object(joint.subprocess, 'run', side_effect=race):
            self.assertIsNotNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertTrue(self.adds()[0][3].endswith('-rerun1'))
        self.assertEqual(self.runs[0]['status'], 'running')
        self.assertFalse(any(c[2] == 'cancel' for c in self.calls))

    def test_queue_errors_fail_each_attempt_without_crashing(self):
        with patch.object(self.loop, 'queue', side_effect=RuntimeError('queue unavailable')):
            self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        failures = [e for e in self.events() if e['event'] == 'queue_failed']
        self.assertEqual(len(failures), 2)
        self.assertTrue(all('queue unavailable' in e['reason'] for e in failures))

    def test_poll_cancel_errors_are_logged_and_fit_still_retries(self):
        self.cfg['fit_timeout_seconds'] = 20
        self.on_add = lambda cmd, out, run: run.update(status='running')
        with patch.object(self.loop, 'cancel', side_effect=OSError('cancel unavailable')):
            self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertEqual(len(self.adds()), 2)
        self.assertEqual(len([e for e in self.events() if e['event'] == 'cancel_failed']), 2)
        self.assertTrue(all('cancel unavailable' in e['reason'] for e in self.events()
                            if e['event'] == 'fit_failed'))
        self.assertEqual(set(self.loop.active_tasks), {c[3] for c in self.adds()})

    def test_start_and_cancel_errors_do_not_crash_launch_or_lose_owned_jobs(self):
        original = self.fake_run
        def fail_start(cmd, **kwargs):
            if cmd[2] == 'start':
                raise RuntimeError('runner start failed')
            return original(cmd, **kwargs)
        self.on_add = lambda cmd, out, run: run.update(status='running')
        with patch.object(joint.subprocess, 'run', side_effect=fail_start), \
                patch.object(self.loop, 'cancel', side_effect=OSError('cancel unavailable')):
            self.assertIsNone(self.loop.fit(0, [self.corpus()], 1))
        self.assertEqual(len(self.adds()), 2)
        self.assertEqual(self.loop.owned_jobs, {c[3] for c in self.adds()})
        self.assertEqual(len([e for e in self.events() if e['event'] == 'cancel_failed']), 2)

    def test_unavailable_queue_snapshot_preserves_previous_states(self):
        self.on_add = lambda cmd, out, run: run.update(status='running')
        out = self.loop.out / 'fit'
        task = self.loop.launch(0, 'fit', 'waiting', 'fit', out, self.loop.fit_command([], out, 1))
        self.loop.poll_tasks()
        self.assertEqual(self.loop.queue_states['waiting'], 'running')
        for broken in ('missing', 'invalid_json', 'permission_denied'):
            with self.subTest(broken=broken):
                self.save_queue()
                if broken == 'missing':
                    self.queue_path.unlink()
                elif broken == 'invalid_json':
                    self.queue_path.write_text('{')
                original = Path.read_text
                def read_text(path, *args, **kwargs):
                    if path == self.queue_path and broken == 'permission_denied':
                        raise PermissionError('unreadable queue')
                    return original(path, *args, **kwargs)
                with patch.object(Path, 'read_text', read_text):
                    self.loop.poll_tasks()
                self.assertNotIn('result', task)
                self.assertEqual(self.loop.queue_states['waiting'], 'running')
        self.assertFalse(any(c[2] == 'cancel' for c in self.calls))
        self.default_artifacts(self.runs[0]['cmd'], out, self.runs[0])
        self.runs[0]['status'] = 'done'
        self.save_queue()
        self.loop.poll_tasks()
        self.assertEqual(task['result'][0], 'success')

    def test_unreadable_queue_does_not_prevent_bounded_timeout_cancellation(self):
        self.cfg['fit_timeout_seconds'] = 20
        self.on_add = lambda cmd, out, run: run.update(status='running')
        out = self.loop.out / 'fit'
        task = self.loop.launch(0, 'fit', 'waiting', 'fit', out, self.loop.fit_command([], out, 1))
        with patch.object(self.loop, 'queue_runs', return_value=None):
            result = self.loop.wait([task], 0, 'fit')
        self.assertEqual(result['waiting'], ('failed', 'timeout'))
        self.assertEqual(self.runs[0]['status'], 'cancelled')

    def test_failures_with_unknown_queue_end_cancel_before_retry(self):
        for state in ('missing', 'unknown_terminal'):
            with self.subTest(state=state):
                cycle = len(self.runs)
                def artifact(cmd, out, run):
                    run['status'] = state
                    write_json(out / 'status.json', {'status': 'interrupted', 'reason': 'fit stopped'})
                self.on_add = artifact
                self.assertIsNone(self.loop.fit(cycle, [self.corpus()], 1))
                actions = [c[2] for c in self.calls[-6:]]
                self.assertEqual(actions, ['add', 'start', 'cancel', 'add', 'start', 'cancel'])

    def test_disk_guard_uses_hysteresis_and_logs_once_per_pause(self):
        self.loop.free_gib.side_effect = [14, 16, 16.9, 17]
        self.loop.before_step(2, 'fit')
        self.assertEqual(self.now, 45)
        self.assertEqual([e['event'] for e in self.events()], ['waiting_for_disk', 'resumed'])
        heartbeat = joint.read_json(self.loop.out / 'heartbeat.json')
        self.assertEqual((heartbeat['cycle'], heartbeat['step']), (2, 'fit'))
        self.assertEqual(heartbeat['free_disk_gib'], 14)

    def test_no_job_launch_until_disk_recovers(self):
        self.loop.free_gib.side_effect = [14, 16, 17, 100, 100]
        self.loop.fit(0, [self.corpus()], 1)
        self.assertEqual(self.now, 30)
        self.assertEqual([e['event'] for e in self.events()][:2], ['waiting_for_disk', 'resumed'])

    def test_wait_heartbeats_and_times_out_even_during_disk_pause(self):
        self.cfg['fit_timeout_seconds'] = 125
        self.on_add = lambda cmd, out, run: run.update(status='paused')
        out = self.loop.out / 'fit'
        task = self.loop.launch(0, 'fit', 'timed', 'test-fit', out, self.loop.fit_command(['data'], out, 1))
        self.loop.free_gib.return_value = 14
        result = self.loop.wait([task], 0, 'fit')
        self.assertEqual(result['timed'], ('failed', 'timeout'))
        heartbeat = joint.read_json(self.loop.out / 'heartbeat.json')
        self.assertEqual(heartbeat['seconds_in_step'], 120)
        self.assertEqual(heartbeat['jobs'], ['timed'])
        self.assertEqual(heartbeat['free_disk_gib'], 14)
        self.assertEqual(len([c for c in self.calls if c[2] == 'cancel']), 1)
        self.assertEqual(len([e for e in self.events() if e['event'] == 'waiting_for_disk']), 1)

    def test_disk_pause_before_next_submission_still_cancels_active_timeout(self):
        self.cfg['label_timeout_seconds'] = 20
        self.on_add = lambda cmd, out, run: run.update(status='running')
        out = self.loop.out / 'label'
        task = self.loop.launch(0, 'label', 'first-shard', 'label0', out,
                                self.loop.label_command(Path(self.cfg['start']), out, 1000, 4))
        self.loop.free_gib.side_effect = [14, 14, 14, 17]
        self.loop.before_step(0, 'label')
        self.assertEqual(task['result'], ('failed', 'timeout'))
        self.assertEqual(self.runs[0]['status'], 'cancelled')

    def test_label_accepts_aggregate_threshold_without_retry(self):
        def artifact(cmd, out, run):
            if int(out.name[-2:]) < 6:
                self.default_artifacts(cmd, out, run)
            else:
                write_json(out / 'completion.json', {'completed': 0, 'faults': 4})
        self.on_add = artifact
        corpus = self.loop.label(0, Path(self.cfg['start']))
        self.assertEqual(len(self.adds()), 8)
        self.assertEqual(len(joint.read_json(corpus / 'corpus.json')['shards']), 6)
        self.assertEqual(self.events()[-1]['completed'], 24)

    def partial_labels(self, holes=False, summaries=True):
        def artifact(cmd, out, run):
            self.default_artifacts(cmd, out, run)
            if any(s.endswith('fresh_collect.py') for s in cmd) and 'retry1' not in out.name:
                data = joint.read_json(out / 'summary.json')
                data['records'] = data['records'][::2] if holes else data['records'][:2]
                write_json(out / 'completion.json', {'completed': 2, 'status': 'interrupted'})
                if summaries:
                    write_json(out / 'summary.json', data)
                else:
                    (out / 'summary.json').unlink()
        self.on_add = artifact

    def test_label_retries_only_uncompleted_worlds_and_fits_all_sources(self):
        self.partial_labels()
        corpus = self.loop.label(0, Path(self.cfg['start']))
        retried = self.adds()[8:]
        self.assertEqual(len(retried), 8)
        for i, cmd in enumerate(retried):
            self.assertEqual(cmd[cmd.index('--world-start') + 1], str(1000 + 4 * i + 2))
            self.assertEqual(cmd[cmd.index('--n') + 1], '2')
        self.assertEqual(self.events()[-1]['completed'], 32)
        self.assertEqual(len(joint.read_json(corpus / 'corpus.json')['shards']), 16)
        self.loop.fit(0, [corpus], 1)
        fit = self.adds('fresh_train.py')[0]
        self.assertEqual(len(fit[fit.index('--data') + 1:fit.index('--probe')]), 17)

    def test_label_retry_handles_noncontiguous_completed_worlds(self):
        self.partial_labels(holes=True)
        corpus = self.loop.label(0, Path(self.cfg['start']))
        retries = self.adds()[8:]
        self.assertEqual(len(retries), 16)
        worlds = [int(c[c.index('--world-start') + 1]) for c in retries]
        self.assertEqual(set(worlds), set(range(1001, 1032, 2)))
        self.assertTrue(joint.read_json(corpus / 'corpus.json')['accepted'])

    def test_whole_shard_fallback_excludes_original_sources(self):
        self.partial_labels(summaries=False)
        corpus = self.loop.label(0, Path(self.cfg['start']))
        sources = joint.read_json(corpus / 'corpus.json')['shards']
        self.assertEqual(len(sources), 8)
        self.assertTrue(all('retry1' in p for p in sources))
        self.assertTrue(all(c[c.index('--n') + 1] == '4' for c in self.adds()[8:]))

    def test_label_timeouts_cancel_before_full_shard_retry(self):
        self.cfg['label_timeout_seconds'] = 20
        def artifact(cmd, out, run):
            if 'retry1' in out.name:
                self.default_artifacts(cmd, out, run)
            else:
                run['status'] = 'running'
        self.on_add = artifact
        corpus = self.loop.label(0, Path(self.cfg['start']))
        self.assertTrue(joint.read_json(corpus / 'corpus.json')['accepted'])
        self.assertEqual(len([c for c in self.calls if c[2] == 'cancel']), 8)
        self.assertTrue(all('retry1' in s for s in joint.read_json(corpus / 'corpus.json')['shards']))

    def test_timeout_preserves_atomic_world_files_and_retries_only_missing_worlds(self):
        self.cfg['label_timeout_seconds'] = 20
        def artifact(cmd, out, run):
            if 'retry1' in out.name:
                self.default_artifacts(cmd, out, run)
            else:
                run['status'] = 'running'
                start = int(cmd[cmd.index('--world-start') + 1])
                for seed in (start, start + 2):
                    path = out / 'worlds' / f'{seed}.json.gz'
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b'published completed world')
        self.on_add = artifact
        corpus = self.loop.label(0, Path(self.cfg['start']))
        sources = joint.read_json(corpus / 'corpus.json')['shards']
        self.assertEqual(sum(s.endswith('.json.gz') for s in sources), 16)
        retries = self.adds()[8:]
        self.assertEqual(len(retries), 16)
        self.assertEqual({int(c[c.index('--world-start') + 1]) for c in retries}, set(range(1001, 1032, 2)))
        self.assertEqual(self.events()[-1]['completed'], 32)
        # The training input is a snapshot: a late file must not enter it implicitly.
        late = self.loop.out / 'corpus_00/shard_00/worlds/1001.json.gz'
        late.write_bytes(b'late publication')
        self.assertNotIn(str(late), sources)

    def test_gate_artifact_failures_without_summary_retry_once(self):
        self.on_add = lambda *args: None  # exit zero, no output
        self.assertFalse(self.loop.gate(0, self.students()))
        self.assertEqual(len(self.adds()), 4)
        self.assertEqual(self.now, 0)
        self.assertEqual(Path(self.cfg['follow']).read_bytes(), b'incumbent-combat')

    def test_label_second_shortfall_skips_cycle(self):
        self.on_add = lambda cmd, out, run: write_json(out / 'completion.json', {'completed': 0})
        self.assertIsNone(self.loop.label(0, Path(self.cfg['start'])))
        self.assertEqual(len(self.adds()), 16)
        self.assertEqual(self.events()[-1]['event'], 'cycle_skipped')
        self.assertFalse(joint.read_json(self.loop.out / 'corpus_00/corpus.json')['accepted'])

    def test_all_requested_worlds_are_labelled_when_not_divisible_by_eight(self):
        self.cfg['worlds_per_cycle'] = 10
        corpus = self.loop.label(0, Path(self.cfg['start']))
        worlds = []
        for command in self.adds():
            start, n = int(command[command.index('--world-start') + 1]), int(command[command.index('--n') + 1])
            worlds += list(range(start, start + n))
        self.assertEqual(worlds, list(range(1000, 1010)))
        self.assertEqual(self.events()[-1]['completed'], 10)
        self.assertIsNotNone(corpus)

    def students(self):
        student = self.root / 'student.pt'
        student.write_bytes(b'new-combat')
        return {1: student}

    def test_gate_retries_failed_evaluation_once_and_adopts_atomically(self):
        def artifact(cmd, out, run):
            if out.name == 'incumbent':
                write_json(out / 'summary.json', {'error': 'interrupted', 'records': []})
            else:
                self.default_artifacts(cmd, out, run)
        self.on_add = artifact
        self.assertTrue(self.loop.gate(0, self.students()))
        self.assertEqual(Path(self.cfg['follow']).read_bytes(), b'new-combat')
        self.assertEqual(len(self.adds()), 3)
        self.assertTrue(self.adds()[-1][3].endswith('-retry1'))
        failure = next(e for e in self.events() if e['event'] == 'gate_eval_failed')
        self.assertEqual(failure['reason'], 'interrupted')
        self.assertTrue(failure['retry'])
        self.assertEqual(self.events()[-1]['event'], 'adopted')
        self.assertFalse(Path(self.cfg['follow']).with_suffix('.tmp').exists())

    def test_gate_timeouts_retry_then_prevent_adoption(self):
        self.cfg['gate_timeout_seconds'] = 20
        self.on_add = lambda cmd, out, run: run.update(status='running')
        self.assertFalse(self.loop.gate(0, self.students()))
        self.assertEqual(len(self.adds()), 4)
        self.assertEqual(len([c for c in self.calls if c[2] == 'cancel']), 4)
        self.assertEqual(Path(self.cfg['follow']).read_bytes(), b'incumbent-combat')
        self.assertEqual(self.events()[-1]['reason'], 'evaluation_failed_twice')

    def test_more_than_one_percent_missing_is_validation_failure_without_retry(self):
        self.on_add = lambda cmd, out, run: write_json(out / 'summary.json', summary(n=3))
        self.assertFalse(self.loop.gate(0, self.students()))
        self.assertEqual(len(self.adds()), 2)
        self.assertEqual(self.events()[-1]['event'], 'cycle_skipped')
        failure = next(e for e in self.events() if e['event'] == 'gate_eval_failed')
        self.assertIn('missing 1/4', failure['reason'])
        self.assertFalse(failure['retry'])
        self.assertFalse(self.loop.gate_completed)

    def test_gate_compares_one_cut_world_on_intersection_and_logs_dropped_worlds(self):
        self.cfg['gate_worlds'] = 100
        def artifact(cmd, out, run):
            data = summary(n=100, wins=1 if out.name == 'incumbent' else 2,
                           floor=20 if out.name == 'incumbent' else 21)
            if out.name != 'incumbent':
                data.update(completed=99, cuts=1, censored=1,
                            cohort_complete=False, confidence_claims_allowed=False)
                data['records'][0] = {'world_seed': 100, 'status': 'censored', 'reason': 'cut:horizon'}
            write_json(out / 'summary.json', data)
        self.on_add = artifact
        self.assertTrue(self.loop.gate(0, self.students()))
        gate = next(e for e in self.events() if e['event'] == 'gate')
        self.assertEqual(gate['results']['1']['worlds'], 99)
        self.assertEqual(gate['results']['1']['dropped_worlds'], 1)
        self.assertEqual(gate['results']['1']['wins'], [0, 1])
        self.assertEqual(len(self.adds()), 2)

    def test_programming_error_is_not_retried(self):
        def artifact(cmd, out, run):
            data = summary()
            data['records'][0]['status'] = 'programming_error'
            write_json(out / 'summary.json', data)
        self.on_add = artifact
        self.assertFalse(self.loop.gate(0, self.students()))
        self.assertEqual(len(self.adds()), 2)
        failures = [e for e in self.events() if e['event'] == 'gate_eval_failed']
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(not e['retry'] for e in failures))

    def test_combine_failure_drops_only_that_seed_with_ten_minute_timeout(self):
        students = self.students()
        students[2] = self.root / 'other-student.pt'
        students[2].write_bytes(b'other-combat')
        original = self.fake_run
        def combine(cmd, **kwargs):
            if cmd[1].endswith('combine_heads.py'):
                self.assertEqual(kwargs['timeout'], 600)
                if cmd[cmd.index('--combat') + 1] == str(students[1]):
                    raise subprocess.TimeoutExpired(cmd, 600)
            return original(cmd, **kwargs)
        with patch.object(joint.subprocess, 'run', side_effect=combine):
            self.assertTrue(self.loop.gate(0, students))
        self.assertEqual(len(self.adds()), 2)
        gate = next(e for e in self.events() if e['event'] == 'gate')
        self.assertEqual(set(gate['results']), {'2'})
        self.assertEqual(Path(self.cfg['follow']).read_bytes(), b'other-combat')
        failure = next(e for e in self.events() if e['event'] == 'combine_failed')
        self.assertEqual(failure['seed'], 1)

    def test_all_combine_failures_skip_gate_without_evaluating(self):
        with patch.object(joint.subprocess, 'run', side_effect=OSError('combine unavailable')):
            self.assertFalse(self.loop.gate(0, self.students()))
        self.assertFalse(self.adds())
        self.assertEqual(self.events()[-1]['reason'], 'no_combined_candidates')

    def test_losing_gate_is_logged_as_rejected(self):
        self.on_add = lambda cmd, out, run: write_json(out / 'summary.json',
                                  summary(wins=2 if out.name == 'incumbent' else 1))
        self.assertFalse(self.loop.gate(0, self.students()))
        self.assertEqual(self.events()[-1]['event'], 'rejected')
        self.assertEqual(Path(self.cfg['follow']).read_bytes(), b'incumbent-combat')

    def test_comparison_reads_fake_files_without_any_subprocess(self):
        base, arm = self.root / 'base.json', self.root / 'arm.json'
        write_json(base, summary())
        write_json(arm, summary(wins=2, floor=22))
        self.assertEqual(self.loop.compare(base, arm)['wins'], [1, 2])
        self.assertFalse(self.calls)

    def test_stop_file_and_trainer_done_stop_before_launch(self):
        for mode in ('STOP', 'trainer_done'):
            with self.subTest(mode=mode):
                stop = self.loop.out / 'STOP'
                log = Path(self.cfg['trainer_out']) / 'log.jsonl'
                if mode == 'STOP':
                    stop.touch()
                else:
                    stop.unlink()
                    log.write_text('{"event":"done"}\n')
                self.loop.run()
                self.assertEqual(self.events()[-1]['reason'], mode)
                self.assertFalse(self.calls)

    def test_startup_waits_for_config_and_log_then_runs(self):
        trainer = Path(self.cfg['trainer_out'])
        (trainer / 'config.json').unlink()
        (trainer / 'log.jsonl').unlink()
        def appear(seconds):
            self.advance(seconds)
            if self.now == 15:
                write_json(trainer / 'config.json', {'iterations': 24})
            elif self.now == 75:
                (trainer / 'log.jsonl').touch()
            self.assertFalse(self.calls)
        with patch.object(joint.time, 'sleep', side_effect=appear):
            self.loop.run()
        self.assertEqual(self.now, 75)
        self.assertTrue(self.adds())
        self.assertEqual(self.events()[-1]['reason'], 'max_cycles')
        heartbeat = joint.read_json(self.loop.out / 'heartbeat.json')
        self.assertEqual(heartbeat['step'], 'trainer_startup')
        self.assertEqual(heartbeat['seconds_in_step'], 60)

    def test_startup_checks_iteration_limit_only_after_files_appear(self):
        trainer = Path(self.cfg['trainer_out'])
        (trainer / 'config.json').unlink()
        def appear(seconds):
            self.advance(seconds)
            write_json(trainer / 'config.json', {'iterations': 24})
            (trainer / 'log.jsonl').write_text('{"event":"train","iteration":22}\n')
        with patch.object(joint.time, 'sleep', side_effect=appear):
            self.loop.run()
        self.assertEqual(self.now, 15)
        self.assertFalse(self.calls)
        self.assertEqual(self.events()[-1]['reason'], 'too_few_iterations_left')

    def test_startup_timeout_is_thirty_minutes_and_heartbeats(self):
        (Path(self.cfg['trainer_out']) / 'config.json').unlink()
        self.loop.run()
        self.assertEqual(self.now, 1800)
        self.assertFalse(self.calls)
        self.assertEqual(self.events()[-1]['reason'], 'trainer_startup_timeout')
        self.assertEqual(joint.read_json(self.loop.out / 'heartbeat.json')['seconds_in_step'], 1800)

    def test_startup_wait_honors_stop(self):
        (Path(self.cfg['trainer_out']) / 'config.json').unlink()
        def stop(seconds):
            self.advance(seconds)
            (self.loop.out / 'STOP').touch()
        with patch.object(joint.time, 'sleep', side_effect=stop):
            self.loop.run()
        self.assertEqual(self.now, 15)
        self.assertEqual(self.events()[-1]['reason'], 'STOP')
        self.assertFalse(self.calls)

    def test_stop_during_wait_cancels_all_active_jobs_and_logs_names(self):
        self.on_add = lambda cmd, out, run: run.update(status='running')
        def stop(seconds):
            self.advance(seconds)
            (self.loop.out / 'STOP').touch()
        with patch.object(joint.time, 'sleep', side_effect=stop):
            self.loop.run()
        names = {c[3] for c in self.adds()}
        self.assertEqual(len(names), 8)
        self.assertEqual(self.now, 15)
        self.assertEqual(set(self.events()[-1]['cancelled_jobs']), names)
        self.assertEqual(self.events()[-1]['reason'], 'STOP')
        self.assertFalse(self.loop.active_tasks)
        self.assertTrue(all(r['status'] == 'cancelled' for r in self.runs))

    def test_trainer_done_during_wait_cancels_jobs_without_waiting_for_timeout(self):
        self.on_add = lambda cmd, out, run: run.update(status='running')
        def done(seconds):
            self.advance(seconds)
            (Path(self.cfg['trainer_out']) / 'log.jsonl').write_text('{"event":"done"}\n')
        with patch.object(joint.time, 'sleep', side_effect=done):
            self.loop.run()
        self.assertEqual(self.now, 15)
        self.assertEqual(self.events()[-1]['reason'], 'trainer_done')
        self.assertEqual(set(self.events()[-1]['cancelled_jobs']), {c[3] for c in self.adds()})
        self.assertTrue(all(r['status'] == 'cancelled' for r in self.runs))

    def test_exception_exit_cancels_active_jobs_and_reraises(self):
        self.on_add = lambda cmd, out, run: run.update(status='running')
        def broken_label(*args):
            out = self.loop.out / 'shard'
            self.loop.launch(0, 'label', 'unfinished-label', 'label', out,
                             self.loop.label_command(Path(self.cfg['start']), out, 1000, 4))
            raise ValueError('broken label')
        with patch.object(self.loop, 'label', side_effect=broken_label), \
                self.assertRaisesRegex(ValueError, 'broken label'):
            self.loop.run()
        self.assertEqual(self.events()[-1]['cancelled_jobs'], ['unfinished-label'])
        self.assertEqual(self.runs[0]['status'], 'cancelled')

    def test_normal_exit_cancels_jobs_still_running_after_publishing_artifacts(self):
        def artifact(cmd, out, run):
            self.default_artifacts(cmd, out, run)
            run['status'] = 'running'
        self.on_add = artifact
        self.loop.run()
        self.assertEqual(self.events()[-1]['reason'], 'max_cycles')
        self.assertEqual(set(self.events()[-1]['cancelled_jobs']), {c[3] for c in self.adds()})
        self.assertFalse(self.loop.active_tasks)
        self.assertFalse(self.loop.owned_jobs)

    def test_final_cancel_failure_does_not_prevent_cancelling_other_jobs(self):
        self.on_add = lambda cmd, out, run: run.update(status='running')
        original = self.loop.cancel
        def cancel(name):
            if name.endswith('label0'):
                raise OSError('cancel unavailable')
            return original(name)
        def stop(seconds):
            self.advance(seconds)
            (self.loop.out / 'STOP').touch()
        with patch.object(joint.time, 'sleep', side_effect=stop), \
                patch.object(self.loop, 'cancel', side_effect=cancel):
            self.loop.run()
        final = self.events()[-1]
        self.assertEqual(final['cancel_failed_jobs'], ['joint-test-c00-label0'])
        self.assertEqual(len(final['cancelled_jobs']), 7)
        self.assertEqual(final['reason'], 'STOP')

    def test_default_trainer_stall_limit_is_three_hours(self):
        log = Path(self.cfg['trainer_out']) / 'log.jsonl'
        old = joint.time.time() - 10801
        os.utime(log, (old, old))
        self.loop.run()
        self.assertEqual([e['event'] for e in self.events()], ['trainer_stalled', 'stopped'])
        self.assertEqual(self.events()[-1]['reason'], 'trainer_stalled')
        self.assertFalse(self.calls)

    def test_trainer_stall_during_wait_cancels_active_jobs(self):
        self.cfg['trainer_stall_seconds'] = 20
        epoch = joint.time.time()
        log = Path(self.cfg['trainer_out']) / 'log.jsonl'
        os.utime(log, (epoch, epoch))
        self.on_add = lambda cmd, out, run: run.update(status='running')
        with patch.object(joint.time, 'time', side_effect=lambda: epoch + self.now):
            self.loop.run()
        self.assertEqual(self.now, 30)
        self.assertEqual(self.events()[-1]['reason'], 'trainer_stalled')
        self.assertEqual(self.events()[-2]['event'], 'trainer_stalled')
        self.assertEqual(set(self.events()[-1]['cancelled_jobs']), {c[3] for c in self.adds()})

    def test_min_iterations_left_reads_latest_train_event_and_ignores_partial_line(self):
        (Path(self.cfg['trainer_out']) / 'log.jsonl').write_text(
            '{"event":"train","iteration":21}\n{"event":"eval","iteration":24}\n'
            '{"event":"train","iteration":22}\n{"event":')
        self.loop.run()
        self.assertFalse(self.calls)
        self.assertEqual(self.events()[-1]['reason'], 'too_few_iterations_left')

    def test_stop_file_ends_disk_pause_without_launching_a_job(self):
        self.loop.free_gib.return_value = 14
        def stop(seconds):
            self.advance(seconds)
            (self.loop.out / 'STOP').touch()
        with patch.object(joint.time, 'sleep', side_effect=stop):
            self.loop.run()
        self.assertFalse(self.calls)
        self.assertEqual(self.events()[-1]['reason'], 'STOP')

    def test_stop_file_between_label_and_fit_prevents_fit(self):
        original = self.loop.label
        def label(*args):
            corpus = original(*args)
            (self.loop.out / 'STOP').touch()
            return corpus
        with patch.object(self.loop, 'label', side_effect=label):
            self.loop.run()
        self.assertEqual(len(self.adds()), 8)
        self.assertEqual(self.events()[-1]['reason'], 'STOP')

    def test_trainer_done_between_fit_and_gate_prevents_gate(self):
        original = self.loop.fit
        def fit(*args):
            best = original(*args)
            (Path(self.cfg['trainer_out']) / 'log.jsonl').write_text('{"event":"done"}\n')
            return best
        with patch.object(self.loop, 'fit', side_effect=fit):
            self.loop.run()
        self.assertEqual(len(self.adds()), 9)
        self.assertEqual(self.events()[-1]['reason'], 'trainer_done')

    def test_disk_wait_can_stop_and_rechecks_iteration_limit_before_cycle(self):
        def advance(seconds):
            self.now += seconds
            (Path(self.cfg['trainer_out']) / 'log.jsonl').write_text('{"event":"train","iteration":22}\n')
        self.loop.free_gib.side_effect = [14, 17]
        with patch.object(joint.time, 'sleep', side_effect=advance):
            self.loop.run()
        self.assertFalse(self.calls)
        self.assertEqual(self.events()[-1]['reason'], 'too_few_iterations_left')

    def test_no_successful_seed_skips_gate_and_max_cycles_stops(self):
        default = self.default_artifacts
        def artifact(cmd, out, run):
            if any(s.endswith('fresh_train.py') for s in cmd):
                write_json(out / 'status.json', {'status': 'failed'})
            else:
                default(cmd, out, run)
        self.on_add = artifact
        self.loop.run()
        self.assertEqual(len(self.adds()), 10)
        self.assertEqual(self.events()[-2]['reason'], 'no_successful_fits')
        self.assertEqual(self.events()[-1]['reason'], 'max_cycles')

    def test_three_consecutive_skipped_cycles_stop_the_loop(self):
        self.cfg['max_cycles'] = 10
        self.on_add = lambda cmd, out, run: write_json(out / 'completion.json', {'completed': 0})
        self.loop.run()
        self.assertEqual(self.events()[-1]['reason'], 'too_many_skipped_cycles')
        self.assertEqual(self.events()[-1]['cycles'], 3)
        self.assertEqual(self.events()[-2]['event'], 'too_many_skipped_cycles')
        self.assertEqual(self.events()[-2]['consecutive'], 3)
        self.assertEqual(len(self.adds()), 48)

    def test_skipped_cycle_cap_is_configurable(self):
        self.cfg.update(max_cycles=10, max_consecutive_skipped_cycles=1)
        self.on_add = lambda cmd, out, run: write_json(out / 'completion.json', {'completed': 0})
        self.loop.run()
        self.assertEqual(self.events()[-1]['reason'], 'too_many_skipped_cycles')
        self.assertEqual(self.events()[-1]['cycles'], 1)

    def test_valid_rejected_gate_resets_consecutive_skipped_cycles(self):
        self.cfg.update(max_cycles=4, max_consecutive_skipped_cycles=2)
        default = self.default_artifacts
        def artifact(cmd, out, run):
            if any(s.endswith('fresh_collect.py') for s in cmd) and 'corpus_01' not in str(out):
                write_json(out / 'completion.json', {'completed': 0})
            elif any(s.endswith('fast_eval.py') for s in cmd):
                write_json(out / 'summary.json', summary(wins=2 if out.name == 'incumbent' else 1))
            else:
                default(cmd, out, run)
        self.on_add = artifact
        self.loop.run()
        self.assertEqual(self.events()[-1]['reason'], 'too_many_skipped_cycles')
        self.assertEqual(self.events()[-1]['cycles'], 4)
        self.assertEqual(len([e for e in self.events() if e['event'] == 'rejected']), 1)

    def test_failed_seed_is_dropped_while_successful_seed_reaches_gate(self):
        self.cfg['seeds'] = [1, 2]
        default = self.default_artifacts
        def artifact(cmd, out, run):
            if any(s.endswith('fresh_train.py') for s in cmd) and 'seed1' in out.name:
                write_json(out / 'status.json', {'status': 'failed'})
            else:
                default(cmd, out, run)
        self.on_add = artifact
        self.loop.run()
        gate = next(e for e in self.events() if e['event'] == 'gate' and 'results' in e)
        self.assertEqual(set(gate['results']), {'2'})
        self.assertEqual(self.events()[-1]['reason'], 'max_cycles')

    def test_successful_cycle_and_restart_reuse_only_accepted_corpora(self):
        write_json(self.loop.out / 'corpus_00/corpus.json', {'accepted': False, 'shards': ['bad']})
        write_json(self.loop.out / 'corpus_01/corpus.json', {'accepted': True, 'shards': ['preserved']})
        self.cfg['max_cycles'] = 3
        self.loop.run()
        fit = self.adds('fresh_train.py')[0]
        data = fit[fit.index('--data') + 1:fit.index('--probe')]
        self.assertIn('preserved', data)
        self.assertNotIn('bad', data)
        self.assertTrue(any('corpus_02' in s for s in data))
        self.assertEqual(self.events()[-1]['cycles'], 3)

    def test_dry_run_prints_one_cycle_without_creating_output_or_subprocesses(self):
        cfg = {**self.cfg, 'out': str(self.root / 'nonexistent' / 'dry-run'), 'seeds': [1, 2]}
        path = self.root / 'config.json'
        write_json(path, cfg)
        joint.main([str(path), '--dry-run'])
        output = self.output.getvalue()
        self.assertEqual(output.count('fresh_collect.py'), 8)
        self.assertEqual(output.count('fresh_train.py'), 2)
        self.assertEqual(output.count('fast_eval.py'), 3)
        self.assertEqual(output.count('combine_heads.py'), 2)
        self.assertFalse(Path(cfg['out']).exists())
        self.assertFalse(self.calls)
        self.assertFalse(self.queue_path.exists())


if __name__ == '__main__':
    unittest.main()
