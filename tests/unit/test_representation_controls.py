"""Synthetic filesystem/queue tests. Never submit work or read study outcomes."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sts_rl import representation_controls as rc


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return rc.file_identity(path)


def synthetic_cache(paths, manifest, data_identity):
    # No pickle or encoding: the controller only verifies file identities.
    dataset = write(paths.cache_receipt.parent / 'dataset.pkl', {'synthetic': 'encoded FIT/VAL'})
    receipt = write(paths.cache_receipt, {'synthetic': 'cache receipt'})
    return dict(schema='controlled_preencoded_cache_v1', receipt=receipt, dataset=dataset,
                manifest=manifest, data_identity=data_identity)


class FakeValidator:
    def __init__(self, paths, gate, *, trigger=False):
        self.paths, self.gate, self.trigger = paths, gate, trigger
        self.calls = []
        self.fail = None

    def grid(self):
        return {'targets': rc.TARGETS}

    def fit(self, spec):
        self.calls.append(('fit', spec['name']))
        if self.fail == spec['name']:
            raise rc.Blocked('synthetic invalid/incomplete checkpoint')
        path = self.paths.base / 'synthetic_checkpoints' / (spec['name'] + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not a real checkpoint: ' + spec['name'].encode())
        lr = spec['condition']['lr']
        loss = (0.4 if lr == .0003 else 0.6) if self.trigger else (0.6 if lr == .0003 else 0.4)
        if spec['secondary']:
            loss = 0.01  # must NEVER enter primary LR/full-game choice
        cp = rc.file_identity(path)
        return dict(checkpoint=cp, architecture=spec['architecture'], condition=spec['condition'],
                    best_step=100, best_validation_loss=loss, updates=2000, examples=250000,
                    data_identity='synthetic', bindings=[cp])

    def score(self, spec, frozen, selection):
        self.calls.append(('score', spec['name']))
        if self.fail == ('score', spec['name']):
            raise rc.Blocked('synthetic incomplete score')
        return {'bindings': [frozen['checkpoint'], selection]}

    def panel(self, spec, frozen, grid):
        self.calls.append(('panel', spec['name']))
        if self.fail == ('panel', spec['name']):
            raise rc.Blocked('synthetic invalid panel: faults/missing results')
        return {'bindings': [frozen['checkpoint']]}

    def cohort(self, arm, checkpoint, shard=None):
        self.calls.append(('cohort', arm, shard))
        if self.fail == ('cohort', arm, shard):
            raise rc.Blocked('synthetic incomplete cohort')
        return {'bindings': [checkpoint] if checkpoint else [], 'common': {'native': 'synthetic'}}


class Harness:
    def __init__(self, root, *, phase='fits', trigger=False):
        self.paths = rc.Paths(root)
        self.paths.base.mkdir(parents=True)
        self.queue_path = root / 'data/rl_runs/queue.json'
        write(self.queue_path, {'runs': []})
        self.control(phase)
        manifest = write(self.paths.base / 'synthetic_manifest.json', {'synthetic': True})
        champion = write(self.paths.base / 'synthetic_champion.pt', {'synthetic': True})
        self.gate = dict(manifest=manifest, champion=champion, data_identity='synthetic',
                         source_files=[], label_source_files=[], probe_source_files=[], native_files=[])
        self.gate['preencoded_cache'] = synthetic_cache(self.paths, manifest, self.gate['data_identity'])
        self.validator = FakeValidator(self.paths, self.gate, trigger=trigger)
        self.submissions = []
        self.controller = rc.Controller(root, submit=self.submit, now=lambda: rc.utc('2026-10-16T12:00:00Z'))
        self.controller.gates = self.gates

    def gates(self):
        self.controller.state.setdefault('pins', {'synthetic': True})
        return self.gate, self.validator, self.validator.grid()

    def control(self, phase='fits', *, stop=False, deadline=rc.DEADLINE):
        write(self.paths.base / 'control.json', dict(allowed_phase=phase, stop=stop, safety_deadline_utc=deadline))

    def submit(self, argv):
        self.submissions.append(argv)
        queue = rc.read_json(self.queue_path)
        queue['runs'].append(dict(name=argv[3], lane=argv[5], cmd=argv[7:], status='pending'))
        write(self.queue_path, queue)

    def finish(self, status='done', code=0):
        queue = rc.read_json(self.queue_path)
        for run in queue['runs']:
            if run['status'] in rc.ACTIVE:
                run.update(status=status, returncode=code)
        write(self.queue_path, queue)

    def tick(self):
        state = self.controller.tick()
        return deepcopy(state)

    def fitted(self):
        count = 26 if self.validator.trigger else 18
        for _ in range(count):
            state = self.tick()
            if state['status'] == 'blocked':
                raise AssertionError(state['reason'])
            self.finish()
        state = self.tick()
        if state['status'] == 'blocked':
            raise AssertionError(state['reason'])
        return state


class PlanTests(unittest.TestCase):
    def test_all_controlled_fits_share_one_cache_old_fits_have_no_flag(self):
        p = rc.Paths('/synthetic/root')
        cache = str(p.base / 'controlled_cache_v1/receipt.json')
        with patch.object(Path, 'open', side_effect=AssertionError('no file reads')), \
             patch.object(Path, 'read_text', side_effect=AssertionError('no file reads')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('no file reads')), \
             patch.object(Path, 'exists', side_effect=AssertionError('no cache checks')):
            plan = rc.build_plan(p.root)
        self.assertEqual(str(p.cache_receipt), cache)
        self.assertEqual(plan['preencoded_cache'], cache)
        fits = [j for j in plan['jobs'] if j['stage'] == 'fits']
        controlled = [j for j in fits if '--preencoded-cache' in j['command']]
        self.assertEqual(len(controlled), 24)
        self.assertEqual(sum(not j['secondary'] for j in controlled), 16)
        self.assertEqual(sum(j['secondary'] for j in controlled), 8)
        specs = rc.main_fits() + rc.extension_fits() + rc.references()
        for spec in specs:
            cmd = rc.fit_command(p, spec, rc.DEADLINE)
            self.assertFalse(any('cache' in k for k in spec['condition']))
            self.assertEqual(cmd[cmd.index('--manifest') + 1], str(p.manifest))
            self.assertEqual(cmd[cmd.index('--probe') + 1], str(rc.OLD / 'probe_native_schema_v2'))
            self.assertEqual(cmd[cmd.index('--data') + 1:cmd.index('--manifest')],
                             [str(rc.OLD / f'teacher_corpus_2000_v1/shard_{i:02d}') for i in range(8)])
            if spec['architecture'] == 'controlled':
                self.assertEqual(cmd.count('--preencoded-cache'), 1)
                self.assertEqual(cmd[cmd.index('--preencoded-cache') + 1], cache)
            else:
                self.assertNotIn('--preencoded-cache', cmd)
                self.assertNotIn(cache, cmd)
        self.assertEqual(plan['immutable_optimizer'], dict(name='AdamW', weight_decay=.01, gradient_clip=1,
                         dropout=0, critic='untrained/frozen', bucketing='inherited_two_axis'))

    def test_exact_unique_main_factorial_and_settings(self):
        specs = rc.main_fits()
        self.assertEqual(len(specs), 18)
        self.assertEqual(len({s['name'] for s in specs}), 18)
        controlled = [s for s in specs if s['architecture'] == 'controlled']
        self.assertEqual({(s['condition']['legacy_action_features'], s['condition']['width'],
                           s['condition']['lr'], s['seed']) for s in controlled},
                         {(f, w, lr, seed) for f in (False, True) for w in (128, 216)
                          for lr in (.0003, .001) for seed in (1919, 2020)})
        self.assertEqual(specs[0], rc.fit_spec('controlled', 1919, .0003))
        self.assertEqual({(s['seed'], s['condition']['lr']) for s in specs if s['architecture'] == 'old'},
                         {(2020, .0003), (2020, .001)})
        for spec in specs + rc.extension_fits() + rc.references():
            c = spec['condition']
            self.assertEqual({k: c[k] for k in ('phase', 'steps', 'min_steps', 'eval_every', 'patience', 'batch', 'device', 'fit_size', 'defer_test')},
                             dict(phase='combat', steps=12000, min_steps=2000, eval_every=100, patience=6,
                                  batch=128, device='mps', fit_size='full', defer_test=True))

    def test_complete_commands_explicit_native_and_original_paths(self):
        p = rc.Paths()
        plan = rc.build_plan()
        self.assertEqual(str(p.base), str(rc.DEFAULT_ROOT / 'data/rl_runs/representation_controls_20261016'))
        self.assertEqual(rc.NATIVE, rc.OLD.parent.parent.parent / 'external/sts_lightspeed_rl/build015')
        self.assertEqual(len(plan['jobs']), 26 + 28 + 28)
        for job in plan['jobs']:
            cmd = job['command']
            self.assertEqual(cmd[:7], ['/usr/bin/env', 'STS_LIGHTSPEED_BUILD=' + str(rc.NATIVE),
                             'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
                             'VECLIB_MAXIMUM_THREADS=1', str(p.python), str(p.script(
                                 {'fits': 'fresh_train.py', 'scores': 'fresh_score.py', 'panels': 'fresh_panel.py'}[job['stage']]))])
            self.assertEqual(job['enqueue'], p.add(job['name'], job['lane'], cmd))
            self.assertNotIn('--resume', cmd)
            self.assertIn('--deadline-utc', cmd)
            if job['stage'] == 'fits':
                self.assertEqual(job['lane'], 'fit')
                self.assertEqual(cmd[cmd.index('--data') + 1:cmd.index('--manifest')],
                                 [str(rc.OLD / f'teacher_corpus_2000_v1/shard_{i:02d}') for i in range(8)])
                self.assertIn('--defer-test', cmd)
            if job['stage'] == 'panels':
                self.assertEqual(json.loads(cmd[cmd.index('--starts-per-group') + 1]), rc.TARGETS)
                for flag, value in [('--bank-world-start', rc.BANK_START), ('--bank-n', 1024),
                                    ('--workers', 6), ('--resamples', 8), ('--timeout', 30)]:
                    self.assertEqual(cmd[cmd.index(flag) + 1], str(value))
        for spec in rc.references():
            self.assertTrue(str(rc.fit_directory(p, spec)).startswith(str(rc.OLD / 'fit_old_combat_lr')))

    def test_exact_eight_game_arms_ranges_lanes_and_no_secondary_choices(self):
        games = rc.build_plan()['games']
        self.assertEqual([g['arm'] for g in games], ['champion', 'old1919', 'off128_s1919', 'on128_s1919',
                        'off216_s1919', 'on216_s1919', 'old2020', 'off128_s2020'])
        for game in games:
            self.assertEqual(len(game['variants']), 1 if game['arm'] == 'champion' else 2)
            for variant in game['variants']:
                self.assertNotIn('lr1e4', variant['condition'] or '')
                for i, shard in enumerate(variant['shards']):
                    cmd = shard['command']
                    self.assertEqual(shard['lane'], f'game{i}')
                    for flag, value in [('--world-start', rc.GAME_START + 512 * i), ('--n', 512),
                                        ('--workers', 1), ('--timeout', 30), ('--self-consistency-every', 0)]:
                        self.assertEqual(cmd[cmd.index(flag) + 1], str(value))
                    self.assertIn('--no-teacher-labels', cmd)
                    self.assertEqual('--combat' in cmd, game['arm'] != 'champion')
            self.assertIn('fresh_merge.py', game['merge']['command'][6])
            self.assertEqual(game['merge']['command'][-3:], ['4096', '--out', str(rc.Paths().base / f'games/{game["arm"]}/merged')])

    def test_plan_cli_is_pure_exclusive_and_requires_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'plan.json'
            with patch.object(Path, 'read_text', side_effect=AssertionError('no input reads')), \
                 patch.object(rc.subprocess, 'run', side_effect=AssertionError('no process')), \
                 patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(rc.main(['--plan', '--out', str(out)]), 0)
            self.assertEqual(list(Path(tmp).iterdir()), [out])
            self.assertEqual(out.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(FileExistsError):
                rc.main(['--plan', '--out', str(out)])
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                rc.main(['--plan'])

    def test_extension_all_four_strict_and_tie_break(self):
        records = {s['name']: {'best_validation_loss': .4 if s['condition']['lr'] == .0003 else .5}
                   for s in rc.main_fits()}
        self.assertTrue(rc.extension_decision(records)['triggered'])
        name = rc.fit_spec('controlled', 2020, .0003, width=216, features=True)['name']
        records[name]['best_validation_loss'] = .5
        self.assertFalse(rc.extension_decision(records)['triggered'])
        self.assertEqual(len(rc.extension_fits()), 8)
        self.assertTrue(all(s['secondary'] and s['condition']['lr'] == .0001 for s in rc.extension_fits()))
        all_specs = rc.main_fits() + rc.references() + rc.extension_fits()
        records = {s['name']: {'best_validation_loss': 0 if s['secondary'] else 1} for s in all_specs}
        choices = rc.primary_choices(all_specs, records)
        self.assertTrue(all('lr3e4' in name for name in choices.values()))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.h = Harness(Path(self.tmp.name).resolve())

    def test_preparation_stop_deadline_and_missing_gates_never_launch(self):
        h = self.h
        h.control('preparation_only')
        self.assertEqual(h.tick()['status'], 'waiting')
        h.control(stop=True)
        self.assertEqual(h.tick()['status'], 'blocked')
        h.control(deadline='2026-10-16T11:00:00Z')
        self.assertEqual(h.tick()['status'], 'blocked')
        h.control()
        (h.paths.base / 'STOP').touch()
        self.assertEqual(h.tick()['status'], 'blocked')
        (h.paths.base / 'STOP').unlink()
        real = rc.Controller(h.paths.root, submit=h.submit, now=h.controller.now)
        self.assertEqual(real.tick()['status'], 'waiting')
        self.assertFalse(h.submissions)

    def test_one_pending_fit_idempotence_and_resume(self):
        h = self.h
        self.assertEqual(h.tick()['status'], 'waiting')
        self.assertEqual(len(h.submissions), 1)
        for status in ('pending', 'running', 'paused'):
            queue = rc.read_json(h.queue_path)
            queue['runs'][0]['status'] = status
            write(h.queue_path, queue)
            h.tick()
        self.assertEqual(len(h.submissions), 1)
        self.assertEqual(h.validator.calls, [])
        # Recreate the controller against the same durable ledger and pending queue.
        h.controller = rc.Controller(h.paths.root, submit=h.submit, now=h.controller.now)
        h.controller.gates = h.gates
        h.tick()
        self.assertEqual(len(h.submissions), 1)
        h.finish()
        h.tick()
        self.assertEqual(len(h.submissions), 2)

    def test_exit_zero_does_not_establish_completion_and_no_auto_retry(self):
        h = self.h
        h.tick()
        h.validator.fail = rc.main_fits()[0]['name']
        h.finish()
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertEqual(len(h.submissions), 1)
        self.assertNotIn('--resume', h.submissions[0])
        h.tick()
        self.assertEqual(len(h.submissions), 1)

    def test_failed_queue_and_mutated_command_block_even_with_artifacts(self):
        h = self.h
        h.tick()
        h.finish('failed', 1)
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertEqual(h.validator.calls, [])
        queue = rc.read_json(h.queue_path)
        queue['runs'][0].update(status='done', returncode=0, retried=True)
        write(h.queue_path, queue)
        self.assertEqual(h.tick()['status'], 'blocked')
        queue['runs'][0].pop('retried')
        queue['runs'][0]['cmd'].append('--resume')
        write(h.queue_path, queue)
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertEqual(len(h.submissions), 1)

    def test_submission_crash_is_not_repeated_when_queue_missing(self):
        h = self.h
        h.controller.submit = Mock(side_effect=OSError('uncertain submit'))
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertEqual(h.controller.submit.call_count, 1)

    def test_control_deadline_is_frozen_per_existing_command(self):
        h = self.h
        h.tick()
        first = deepcopy(h.submissions[0])
        h.control(deadline='2026-10-16T20:00:00Z')
        h.tick()
        self.assertEqual(h.submissions, [first])
        h.finish()
        h.tick()
        self.assertEqual(h.submissions[0], first)
        self.assertIn('2026-10-16T20:00:00Z', h.submissions[1])

    def test_phase_change_after_intent_prevents_submission(self):
        h = self.h
        save = h.controller.save

        def change_phase_after_intent():
            save()
            if h.controller.state['jobs']:
                h.control('preparation_only')

        h.controller.save = change_phase_after_intent
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertFalse(h.submissions)
        self.assertEqual(h.controller.state['jobs'][rc.main_fits()[0]['name']]['status'], 'submitting')

    def test_fits_gate_freezes_all_twenty_before_explicit_evaluations(self):
        h = self.h
        state = h.fitted()
        self.assertEqual(len(h.submissions), 18)
        self.assertEqual(len(state['selections']), 20)
        self.assertEqual(len(rc.read_json(h.paths.base / 'global_freeze_v1.json')['conditions']), 20)
        self.assertEqual(rc.read_json(h.paths.base / 'global_freeze_v1.json')['preencoded_cache'],
                         h.gate['preencoded_cache'])
        self.assertEqual(rc.read_json(h.paths.base / 'cohort.json')['worlds'], list(range(rc.GAME_START, rc.GAME_START + 4096)))
        identity = state['global_freeze']
        with patch.object(rc, 'primary_choices', side_effect=AssertionError('never choose again')):
            h.tick()
        self.assertEqual(h.controller.state['global_freeze'], identity)
        h.control('evaluations')
        state = h.tick()
        self.assertEqual(len(h.submissions), 19)
        self.assertIn('fresh_score.py', h.submissions[-1][13])
        self.assertEqual(state['jobs'][h.submissions[-1][3]]['global_freeze'], identity)

    def test_extension_all_eight_before_freeze_and_never_changes_game_lr(self):
        h = self.h
        h.validator.trigger = True
        h.control('complete_pipeline')
        state = h.fitted()
        self.assertEqual(len(h.submissions), 27)  # 26 fits + first score
        self.assertTrue(rc.read_json(h.paths.base / 'lower_lr_extension_v1.json')['triggered'])
        frozen = rc.read_json(h.paths.base / 'global_freeze_v1.json')
        self.assertEqual(len(frozen['conditions']), 28)
        self.assertTrue(all('lr1e4' not in (v or '') for v in frozen['game_arms'].values()))
        self.assertEqual(len(state['selections']), 28)
        stages = [Path(c[13]).name for c in h.submissions]
        self.assertEqual(stages, ['fresh_train.py'] * 26 + ['fresh_score.py'])

    def test_checkpoint_mutation_fails(self):
        h = self.h
        state = h.fitted()
        first = state['jobs'][rc.main_fits()[0]['name']]['validated']['checkpoint']
        Path(first['path']).write_bytes(b'changed')
        self.assertEqual(h.tick()['status'], 'blocked')

    def test_selection_mutation_or_missing_recorded_freezes_never_republish(self):
        for kind in ('selection_changed', 'selection_removed', 'global_removed', 'cohort_removed'):
            with self.subTest(kind=kind):
                h = Harness(Path(self.tmp.name).resolve() / kind)
                state = h.fitted()
                if kind.startswith('selection'):
                    target = Path(next(iter(state['selections'].values()))['path'])
                else:
                    target = h.paths.base / ('global_freeze_v1.json' if kind == 'global_removed' else 'cohort.json')
                if kind == 'selection_changed':
                    target.chmod(0o644)
                    write(target, {'changed': True})
                else:
                    target.unlink()
                with patch.object(rc, 'primary_choices', side_effect=AssertionError('no new LR choice')):
                    self.assertEqual(h.tick()['status'], 'blocked')
                self.assertEqual(len(h.submissions), 18)

    def test_crash_after_global_publication_recovers_without_choosing_again(self):
        h = self.h
        state = h.fitted()
        identity = state.pop('global_freeze')
        state.pop('selections')
        write(h.paths.base / 'controller_state_v1.json', state)
        with patch.object(rc, 'primary_choices', side_effect=AssertionError('must reuse published choice')):
            state = h.tick()
        self.assertEqual(state['status'], 'waiting')
        self.assertEqual(state['global_freeze'], identity)
        self.assertEqual(len(state['selections']), 20)

    def test_global_freeze_mutation_never_reselects(self):
        h = self.h
        h.fitted()
        target = h.paths.base / 'global_freeze_v1.json'
        target.chmod(0o644)
        value = rc.read_json(target)
        value['primary_choices']['off128_s1919'] = 'wrong'
        write(target, value)
        with patch.object(rc, 'primary_choices', side_effect=AssertionError('no reselect')):
            self.assertEqual(h.tick()['status'], 'blocked')

    def test_freeze_recovery_rejects_changed_cache_provenance(self):
        h = self.h
        state = h.fitted()
        state.pop('global_freeze')  # publication crash recovery must check the cache too
        state.pop('selections')
        write(h.paths.base / 'controller_state_v1.json', state)
        h.gate['preencoded_cache']['dataset']['sha256'] = 'e' * 64
        with patch.object(rc, 'primary_choices', side_effect=AssertionError('no reselect')):
            self.assertEqual(h.tick()['status'], 'blocked')
        self.assertEqual(len(h.submissions), 18)

    def test_scores_panels_then_one_game_arm_eight_lanes_and_strict_merge(self):
        h = self.h
        h.control('complete_pipeline')
        h.fitted()  # first score pending
        for _ in range(40):  # 20 scores, 20 panels; then first game arm
            h.finish()
            state = h.tick()
            self.assertNotEqual(state['status'], 'blocked', state['reason'])
        queue = rc.read_json(h.queue_path)['runs']
        active = [r for r in queue if r['status'] in rc.ACTIVE]
        self.assertEqual(len(active), 8)
        self.assertEqual({r['lane'] for r in active}, {f'game{i}' for i in range(8)})
        self.assertTrue(all(r['name'].startswith('rc_game_champion_') for r in active))
        self.assertTrue(all('--combat' not in r['cmd'] for r in active))
        h.tick()
        self.assertEqual(len(h.submissions), 18 + 20 + 20 + 8)
        h.finish()
        h.tick()
        self.assertEqual(h.submissions[-1][3], 'rc_merge_champion')
        h.validator.fail = ('cohort', 'champion', None)
        h.finish()
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertFalse(any('rc_game_old1919_' in c[3] for c in h.submissions))

    def test_panel_invalid_stops_games(self):
        h = self.h
        h.control('evaluations')
        h.fitted()
        for _ in range(20):
            h.finish()
            h.tick()
        h.validator.fail = ('panel', rc.main_fits()[0]['name'])
        h.finish()
        self.assertEqual(h.tick()['status'], 'blocked')
        self.assertFalse(any(c[3].startswith('rc_game_') for c in h.submissions))

    def test_full_pipeline_completes_only_after_eight_merges_and_audit(self):
        h = self.h
        h.control('complete_pipeline')
        for _ in range(90):
            state = h.tick()
            self.assertNotEqual(state['status'], 'blocked', state['reason'])
            if state['status'] == 'complete':
                break
            h.finish('exited', None)  # unknown exit status is allowed ONLY with valid artifacts
        else:
            self.fail('did not complete bounded synthetic pipeline')
        self.assertTrue(state['final_audit'])
        self.assertEqual(len([c for c in h.submissions if c[3].startswith('rc_merge_')]), 8)
        self.assertEqual(len([c for c in h.submissions if c[3].startswith('rc_game_')]), 64)
        submitted = len(h.submissions)
        self.assertEqual(h.tick()['status'], 'complete')
        self.assertEqual(len(h.submissions), submitted)


class SourceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve() / 'root'
        self.p = rc.Paths(root)
        self.old = Path(self.tmp.name).resolve() / 'old'
        self.native = Path(self.tmp.name).resolve() / 'native'
        for symbol, value in [('OLD', self.old), ('NATIVE', self.native)]:
            manager = patch.object(rc, symbol, value)
            manager.start()
            self.addCleanup(manager.stop)
        for path in [self.p.script('queue_runner.py'), root / 'src/example.py',
                     root / 'src/sts_policy/entity_vocabulary.json',
                     root / 'docs/representation_controls_plan_2026-10-16.md', self.p.python,
                     root / 'pyproject.toml', self.p.manifest, self.p.champion, self.native / 'fake.so']:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic only\n')
        label = write(self.old / 'teacher_corpus_2000_v1/shard_00/labels.json', {})
        probe = write(self.old / 'probe_native_schema_v2/probes.json', {})
        write(self.p.grid, {})
        self.plan = rc.build_plan(root)
        write(self.p.base / 'controller_plan_v1.json', self.plan)
        native_id = rc.file_identity(self.native / 'fake.so')
        self.gate = dict(schema='representation_controls_preflight_v1', status='PASS', root=str(root),
                         checks={c: 'PASS' for c in rc.CHECKS}, source_files=[rc.file_identity(x) for x in rc.source_paths(root)],
                         manifest=rc.file_identity(self.p.manifest), champion=rc.file_identity(self.p.champion),
                         python=rc.file_identity(self.p.python), label_source_files=[label], probe_source_files=[probe],
                         native_files=[native_id], native={'module_sha256': native_id['sha256']})
        self.gate['data_identity'] = 'd' * 64
        self.gate['preencoded_cache'] = synthetic_cache(self.p, self.gate['manifest'], self.gate['data_identity'])
        self.path = self.p.base / 'preflight_pass_v1.json'
        write(self.path, self.gate)
        self.controller = rc.Controller(root, validator_factory=FakeValidator)
        self.controller.state = {'schema': 'representation_controls_state_v1', 'jobs': {}}

    def test_missing_checks_and_gate_source_mutation_fail(self):
        self.controller.gates()
        self.p.script('queue_runner.py').write_text('# changed\n')
        with self.assertRaisesRegex(rc.Blocked, 'identity changed'):
            self.controller.gates()

    def test_even_pass_gate_cannot_override_inherited_queue_auto_retry(self):
        self.p.script('queue_runner.py').write_text("cmd = cmd + ['--resume']\n")
        self.gate['source_files'] = [rc.file_identity(x) for x in rc.source_paths(self.p.root)]
        write(self.path, self.gate)
        with self.assertRaisesRegex(rc.Blocked, 'automatic retries'):
            self.controller.gates()

    def test_source_inventory_and_pinned_preflight_cannot_change(self):
        self.controller.gates()
        self.gate['checks']['source_review'] = 'FAIL'
        write(self.path, self.gate)
        with self.assertRaises(rc.Blocked):
            self.controller.gates()
        self.gate['checks']['source_review'] = 'PASS'
        self.gate['new_unreviewed_field'] = True
        write(self.path, self.gate)
        with self.assertRaisesRegex(rc.Blocked, 'Pinned'):
            self.controller.gates()

    def test_missing_malformed_or_wrong_cache_bindings_block_before_pinning(self):
        other = write(self.p.base / 'other.json', {})
        mutations = [lambda g: g.pop('preencoded_cache'),
                     lambda g: g.update(preencoded_cache=None),
                     lambda g: g['preencoded_cache'].update(schema='wrong'),
                     lambda g: g['preencoded_cache'].pop('dataset'),
                     lambda g: g['preencoded_cache'].update(extra=True),
                     lambda g: g['preencoded_cache']['receipt'].update(extra=True),
                     lambda g: g['preencoded_cache'].update(receipt=other),
                     lambda g: g['preencoded_cache'].update(manifest=other),
                     lambda g: g['preencoded_cache'].update(data_identity='e' * 64),
                     lambda g: (g.update(data_identity='bad'), g['preencoded_cache'].update(data_identity='bad')),
                     lambda g: g['preencoded_cache']['dataset'].update(sha256='e' * 64)]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                gate = deepcopy(self.gate)
                mutate(gate)
                write(self.path, gate)
                with self.assertRaises(rc.Blocked):
                    self.controller.gates()
                self.assertNotIn('pins', self.controller.state)

    def test_cache_receipt_dataset_manifest_tampering_at_start_and_on_recheck(self):
        for key in ('receipt', 'dataset', 'manifest'):
            path = Path(self.gate['preencoded_cache'][key]['path'])
            original = path.read_bytes()
            for recheck in (False, True):
                with self.subTest(key=key, recheck=recheck):
                    self.controller.state = {'schema': 'representation_controls_state_v1', 'jobs': {}}
                    if recheck:
                        self.controller.gates()
                    path.write_bytes(b'tampered')
                    with self.assertRaisesRegex(rc.Blocked, 'identity changed'):
                        self.controller.gates()
                    path.write_bytes(original)

    def test_missing_cache_files_block_without_pinning(self):
        for key in ('receipt', 'dataset'):
            with self.subTest(key=key):
                path = Path(self.gate['preencoded_cache'][key]['path'])
                original = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    self.controller.gates()
                self.assertNotIn('pins', self.controller.state)
                path.write_bytes(original)

    def test_cache_is_rechecked_immediately_before_submission(self):
        write(self.p.base / 'control.json', dict(allowed_phase='fits', stop=False, safety_deadline_utc=rc.DEADLINE))
        self.controller.now = lambda: rc.utc('2026-10-16T12:00:00Z')
        self.controller.submit = Mock()
        save = self.controller.save

        def tamper_after_intent():
            save()
            if self.controller.state['jobs']:
                Path(self.gate['preencoded_cache']['dataset']['path']).write_bytes(b'changed')

        self.controller.save = tamper_after_intent
        state = self.controller.tick()
        self.assertEqual(state['status'], 'blocked')
        self.assertIn('identity changed', state['reason'])
        self.controller.submit.assert_not_called()


class ValidationTests(unittest.TestCase):
    def synthetic_fit(self, root, *, architecture='old', seed=2020):
        import torch
        from sts_policy.training import CHECKPOINT_VERSION
        p = rc.Paths(root)
        spec = rc.fit_spec(architecture, seed, .0003)
        out = rc.fit_directory(p, spec)
        out.mkdir(parents=True)
        manifest = write(p.manifest, {})
        label = write(root / 'synthetic_labels.json', {})
        probe = write(root / 'synthetic_probes.json', {})
        archive = write(out / 'source.zip', {})
        (out / 'fake.so').write_bytes(b'synthetic only; never loaded')
        native = {'module_file': 'fake.so', 'module_sha256': rc.file_identity(out / 'fake.so')['sha256']}
        write(out / 'build_manifest.json', native)
        config = dict(spec['condition'], architecture=architecture, manifest=str(p.manifest),
                      data=[str(rc.OLD / f'teacher_corpus_2000_v1/shard_{i:02d}') for i in range(8)],
                      probe=[str(rc.OLD / 'probe_native_schema_v2')])
        data_identity = 'd' * 64 if architecture == 'controlled' else 'synthetic'
        inputs = [manifest, label, probe]
        if architecture == 'controlled':
            cache = synthetic_cache(p, manifest, data_identity)
            config.update(preencoded_cache=str(p.cache_receipt), preencoded_cache_provenance=cache)
            inputs += [cache['receipt'], cache['dataset']]
        launch_id = write(out / 'launch.json', dict(effective_config=config, parent_checkpoint=None,
                              native=native, source_archive=archive, data_inputs=inputs, source_files=[],
                              runtime=dict(python='synthetic', platform='synthetic', packages=dict(torch='synthetic', numpy='synthetic'))))
        source = dict(data_identity=data_identity, weight_parent=None, defer_test=True,
                      label_source_files=[label], probe_source_files=[probe], launch=launch_id)
        if architecture == 'controlled':
            source['preencoded_cache'] = cache
        write(out / 'receipt.json', source)
        payload = dict(format_version=CHECKPOINT_VERSION, model_config=dict(size='large', context='none'),
                       model_state={'synthetic': torch.tensor([1.0])}, step=100,
                       optimizer_state={'param_groups': [{'lr': .0003, 'weight_decay': .01}]},
                       metadata=dict(config=config, data_identity=data_identity, initialization='from_scratch',
                           parent_checkpoint=None, filters=['synthetic'], vocabulary_sha256='synthetic',
                           critic_status='head_frozen_shared_actor_features_trained',
                           training_state={'best_step': 100, 'best_loss': .5}, source_receipt=source))
        if architecture == 'controlled':
            from sts_policy.controlled_entity import _source_identity, _config_identity
            model = dict(phase='combat', width=128, layers=2, heads=4, dropout=0, legacy_action_features=False)
            payload['model_config'] = {'model_config': model}
            payload['metadata'].update(_source_identity(), kind='entity_controlled_v1', model_config=model,
                                       model_config_sha256=_config_identity(model), critic_status='frozen_untrained')
        torch.save(payload, out / 'best.pt')
        summary = dict(status='completed', test_deferred=True, test_evaluated=False, actual_fit_rows=113042,
                       validation_rows=17989, selected_checkpoint=rc.file_identity(out / 'best.pt'),
                       best_validation_loss=.5, best_step=100, updates=2000, examples=250000)
        write(out / 'summary.json', summary)
        gate = dict(data_identity=data_identity, filters=['synthetic'], vocabulary_sha256='synthetic',
                    label_source_files=[label], probe_source_files=[probe], native=native, manifest=manifest, source_files=[])
        if architecture == 'controlled':
            gate['preencoded_cache'] = deepcopy(cache)
            gate['vocabulary_sha256'] = payload['metadata']['vocabulary_sha256']
        return p, spec, out, payload, summary, gate

    def test_controlled_checkpoint_requires_exact_cache_path_and_provenance(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(rc, 'OLD', root / 'old'):
                p, spec, out, payload, summary, gate = self.synthetic_fit(root, architecture='controlled')
                validator = rc.Validator(p, gate)
                record = validator.fit(spec)
                self.assertEqual(record['condition'], spec['condition'])
                self.assertFalse(any('cache' in k for k in record['condition']))
                config = lambda x: x['metadata']['config']
                source = lambda x: x['metadata']['source_receipt']
                mutations = [lambda x: config(x).pop('preencoded_cache'),
                             lambda x: config(x).update(preencoded_cache=''),
                             lambda x: config(x).update(preencoded_cache=str(root / 'other/receipt.json')),
                             lambda x: config(x).pop('preencoded_cache_provenance'),
                             lambda x: source(x).pop('preencoded_cache'),
                             lambda x: source(x).update(preencoded_cache=dict(gate['preencoded_cache'], data_identity='e' * 64))]
                for key in ('schema', 'receipt', 'dataset', 'manifest', 'data_identity'):
                    bad = deepcopy(gate['preencoded_cache'])
                    if key in ('receipt', 'dataset', 'manifest'):
                        bad[key]['sha256'] = 'e' * 64
                    else:
                        bad[key] = 'wrong'
                    mutations.append(lambda x, bad=bad: config(x).update(preencoded_cache_provenance=bad))
                for i, mutate in enumerate(mutations):
                    with self.subTest(case=i):
                        changed = deepcopy(payload)
                        mutate(changed)
                        write(out / 'receipt.json', changed['metadata']['source_receipt'])
                        torch.save(changed, out / 'best.pt')
                        summary['selected_checkpoint'] = rc.file_identity(out / 'best.pt')
                        write(out / 'summary.json', summary)
                        with self.assertRaisesRegex(rc.Blocked, 'Checkpoint preencoded cache mismatch'):
                            validator.fit(spec)

    def test_controlled_launch_requires_cache_provenance_and_both_cache_inputs(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(rc, 'OLD', root / 'old'):
                p, spec, out, payload, summary, gate = self.synthetic_fit(root, architecture='controlled')
                validator = rc.Validator(p, gate)
                source = payload['metadata']['source_receipt']
                launch_path = Path(source['launch']['path'])
                original = rc.read_json(launch_path)
                for key in ('receipt', 'dataset', 'wrong_identity', 'effective_config'):
                    with self.subTest(key=key):
                        launch = deepcopy(original)
                        if key in ('receipt', 'dataset'):
                            launch['data_inputs'].remove(gate['preencoded_cache'][key])
                        elif key == 'wrong_identity':
                            launch['data_inputs'][-1]['sha256'] = 'e' * 64
                        else:
                            launch['effective_config'].pop('preencoded_cache_provenance')
                        source['launch'] = write(launch_path, launch)
                        write(out / 'receipt.json', source)
                        torch.save(payload, out / 'best.pt')
                        summary['selected_checkpoint'] = rc.file_identity(out / 'best.pt')
                        write(out / 'summary.json', summary)
                        message = 'launch/config/native mismatch' if key == 'effective_config' else 'missing preencoded cache'
                        with self.assertRaisesRegex(rc.Blocked, message):
                            validator.fit(spec)

    def test_controlled_training_inputs_recheck_actual_cache_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(rc, 'OLD', root / 'old'):
                p, spec, _, _, _, gate = self.synthetic_fit(root, architecture='controlled')
                validator = rc.Validator(p, gate)
                validator.fit(spec)
                for key in ('receipt', 'dataset'):
                    with self.subTest(key=key):
                        path = Path(gate['preencoded_cache'][key]['path'])
                        original = path.read_bytes()
                        path.write_bytes(b'changed')
                        with self.assertRaisesRegex(rc.Blocked, 'identity changed'):
                            validator.fit(spec)
                        path.write_bytes(original)

    def test_old_1919_references_and_2020_fits_require_cache_fields_absent(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (1919, 2020):
                root = Path(tmp).resolve() / str(seed)
                with patch.object(rc, 'OLD', root / 'old'):
                    p, spec, out, payload, summary, gate = self.synthetic_fit(root, seed=seed)
                    validator = rc.Validator(p, gate)
                    self.assertEqual(validator.fit(spec)['condition'], spec['condition'])
                    for field in ('preencoded_cache', 'preencoded_cache_provenance', 'source_receipt'):
                        with self.subTest(seed=seed, field=field):
                            changed = deepcopy(payload)
                            if field == 'source_receipt':
                                changed['metadata'][field]['preencoded_cache'] = {}
                            else:
                                changed['metadata']['config'][field] = None
                            torch.save(changed, out / 'best.pt')
                            summary['selected_checkpoint'] = rc.file_identity(out / 'best.pt')
                            write(out / 'summary.json', summary)
                            with self.assertRaisesRegex(rc.Blocked, 'Old checkpoint must not use a preencoded cache'):
                                validator.fit(spec)

    def test_actual_synthetic_checkpoint_condition_best_step_and_data_binding(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(rc, 'OLD', root / 'old'):
                p, spec, out, payload, summary, gate = self.synthetic_fit(root)
                validator = rc.Validator(p, gate)
                record = validator.fit(spec)
                self.assertEqual(record['best_step'], 100)
                self.assertEqual(record['examples'], 250000)
                original = deepcopy(payload)
                for mutate in [lambda x: x['metadata']['config'].update(seed=1919),
                               lambda x: x.update(step=200),
                               lambda x: x['metadata'].update(data_identity='wrong'),
                               lambda x: x['metadata'].update(parent_checkpoint={'parent': True})]:
                    changed = deepcopy(original)
                    mutate(changed)
                    torch.save(changed, out / 'best.pt')
                    summary['selected_checkpoint'] = rc.file_identity(out / 'best.pt')
                    write(out / 'summary.json', summary)
                    with self.assertRaises((rc.Blocked, ValueError)):
                        validator.fit(spec)
                torch.save(original, out / 'best.pt')
                summary['selected_checkpoint'] = rc.file_identity(out / 'best.pt')
                for field, bad in [('status', 'interrupted'), ('test_deferred', False),
                                   ('test_evaluated', True), ('actual_fit_rows', 113041)]:
                    changed = dict(summary, **{field: bad})
                    write(out / 'summary.json', changed)
                    with self.assertRaises(rc.Blocked):
                        validator.fit(spec)

    def test_strict_score_rows_are_full_frozen_cohort_without_computing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(rc, 'OLD', root / 'old'):
                p, spec, out, payload, _, gate = self.synthetic_fit(root)
                original = dict(phase='combat', split='TEST', has_target=True, source_row_id='synthetic_row')
                gate['manifest'] = write(p.manifest, {'rows': [original]})
                validator = rc.Validator(p, gate)
                cp = rc.file_identity(out / 'best.pt')
                frozen = dict(checkpoint=cp, condition=spec['condition'], data_identity='synthetic')
                sid = write(rc.selection_path(p, spec), dict(selected_checkpoint=cp, manifest=gate['manifest']))
                score = p.base / 'scores' / spec['name']
                score.mkdir(parents=True)
                row = dict(original, selected_checkpoint=cp['sha256'], architecture='old',
                           training_condition=spec['condition'], data_identity='synthetic', inference=rc.INFERENCE,
                           correct=True, ce=.5, target=[.25, .75], target_sha256=rc.digest([.25, .75]))
                rows = score / 'test_rows.jsonl'
                rows.write_text(json.dumps(row) + '\n')
                snapshot = score / 'records/synthetic'
                snapshot.mkdir(parents=True)
                binary = snapshot / 'fake.so'
                binary.write_bytes(b'synthetic; never loaded')
                native = dict(module_file='fake.so', module_sha256=rc.file_identity(binary)['sha256'])
                gate['native'] = native
                write(snapshot / 'build_manifest.json', native)
                archive_id = write(snapshot / 'source.zip', {})
                effective = dict(stage='deferred_test_scoring', training=False, weight_updates=0,
                                 architecture='old', phase='combat', device='cpu', selection=sid,
                                 selected_checkpoint=cp, manifest=gate['manifest'], selection_criterion=rc.CRITERION,
                                 data_identity='synthetic')
                launch_id = write(snapshot / 'launch.json', dict(schema='sts_training_launch_v1',
                    effective_config=effective, parent_checkpoint=cp, source_archive=archive_id, native=native,
                    source_files=[],
                    runtime=dict(python='synthetic', platform='synthetic', packages=dict(torch='synthetic', numpy='synthetic')),
                    data_inputs=[sid, gate['manifest'], *gate['label_source_files']]))
                write(score / 'receipt.json', dict(effective, launch=launch_id, filters=gate['filters'],
                      vocabulary_sha256=gate['vocabulary_sha256'], source_receipt=payload['metadata']['source_receipt'],
                      label_source_files=gate['label_source_files']))
                summary = dict(status='completed', training=False, weight_updates=0, selected_checkpoint=cp,
                               selection=sid, data_identity='synthetic', test_evaluated=True,
                               test_protocol=rc.INFERENCE, test_device='cpu', test_rows=1,
                               test={'groups': {'all_rows': {'rows': 1}}})
                write(score / 'summary.json', summary)
                self.assertIn('bindings', validator.score(spec, frozen, sid))
                for text in ['', json.dumps(row) + '\n' + json.dumps(row) + '\n',
                             json.dumps(dict(row, training_condition={})) + '\n']:
                    rows.write_text(text)
                    with self.assertRaises(rc.Blocked):
                        validator.score(spec, frozen, sid)
                rows.write_text(json.dumps(row) + '\n')
                write(score / 'summary.json', dict(summary, test_protocol='batched'))
                with self.assertRaises(rc.Blocked):
                    validator.score(spec, frozen, sid)

    def test_grid_exact_coverage_salts_and_bank_range(self):
        from sts_rl.fresh_panel_grid import freeze_grid, canonical_grid_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            p = rc.Paths(root)
            provenance = dict(environment_version='0.1.5', observation_version='sts_env_public_v3',
                              native_provenance={'module_sha256': 'synthetic'})
            identity = write(root / 'synthetic_bank.json', {})
            bank = dict(world_start=rc.BANK_START, n=1024, science_identity_hash='0' * 64,
                        expected_provenance=provenance, inputs=[identity], shards=[identity])
            rows = [dict(world_seed=rc.BANK_START + i, start_id=j, group=group, public_hash='0' * 64,
                         first_combat_hash='1' * 64, prefix=[], task={}, entering_action=0,
                         entering_description='synthetic', expected_provenance=provenance)
                    for j, (group, n) in enumerate(rc.TARGETS.items()) for i in range(n)]
            _, grid = freeze_grid(rows, rc.TARGETS, 8, bank_identity=bank)
            write(p.grid, grid)
            validator = rc.Validator(p, {'native': provenance['native_provenance']})
            self.assertEqual(len(validator.grid()['grid']), 600 * 9)
            for mutate in [lambda x: x['coverage'].update(BOSS2=399), lambda x: x.update(resamples=7),
                           lambda x: x['bank_identity'].update(world_start=rc.BANK_START - 1),
                           lambda x: x['grid'].pop()]:
                broken = deepcopy(grid)
                mutate(broken)
                broken['grid_hash'] = canonical_grid_hash(broken)
                write(p.grid, broken)
                with self.assertRaises((rc.Blocked, ValueError)):
                    validator.grid()

    def test_manifest_rows_streaming_handles_boundaries_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            expected = [{'id': i, 'padding': 'x' * 40000} for i in range(4)]
            write(path, {'before': 'header', 'rows': expected, 'after': [1, 2]})
            self.assertEqual(list(rc.manifest_rows(path)), expected)
            write(path, {'rows': []})
            self.assertEqual(list(rc.manifest_rows(path)), [])
            path.write_text('{"rows":[{"id":1}')
            with self.assertRaises(rc.Blocked):
                list(rc.manifest_rows(path))

    def test_daemon_bounded_sleep_and_cli_does_not_start_queue(self):
        fake = Mock()
        fake.tick.side_effect = [dict(status='waiting', reason='pending', last_tick_utc='synthetic'),
                                 dict(status='blocked', reason='STOP', last_tick_utc='synthetic')]
        with patch.object(rc, 'Controller', return_value=fake), patch.object(rc.time, 'sleep') as sleep, \
             patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(rc.main(['--daemon', '--interval', '30']), 2)
            sleep.assert_called_once_with(30)
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            rc.main(['--daemon', '--interval', '31'])


if __name__ == '__main__':
    unittest.main()
