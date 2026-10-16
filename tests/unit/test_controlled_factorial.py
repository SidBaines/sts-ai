"""Synthetic factorial arithmetic and strict complete-panel filesystem contracts.

Tiny model loader and cohort counts stand in for production weights/131k labels;
the inherited checkpoint, selection, grid and completed-run validators stay real.
No fitting, inference, native execution, or study artifacts are used.
"""
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from sts_rl import controlled_factorial as factorial
from sts_rl.fresh_training import FILTERS, checkpoint_training_condition, digest
from sts_rl.fresh_scoring import SELECTION_CRITERION
from sts_rl.recording import file_identity
from tests.unit.controlled_fixture import builder_stub, checkpoint
from tests.unit.test_fresh_panel_comparison import PanelFixture, WORLD, read, write


def condition(seed, feature, width, lr):
    return dict(factorial.FIXED, seed=seed, legacy_action_features=feature, width=width, lr=lr)


def synthetic(outcome):
    conditions = {factorial.condition_key(c): c for c in
                  (condition(*cell) for cell in product(factorial.SEEDS, *factorial.LEVELS))}
    starts = [dict(world_seed=WORLD+w, start_id=s, group=g, first_combat_hash='a'*64)
              for w, s, g in ((0, 10, 'BOSS2'), (1, 11, 'BOSS2'), (2, 12, 'BOSS2'), (0, 20, 'BOSS1'))]
    frozen = dict(starts=starts, grid_hash='b'*64, grid=[
        {k: start[k] for k in ('world_seed', 'start_id', 'group')} | {'salt': salt}
        for start in starts for salt in (-1, *range(8))])
    rows = {}
    for key, c in conditions.items():
        rows[key] = []
        for cell in frozen['grid']:
            lost = outcome(c, cell)
            rows[key].append(dict(cell, policy='replacement', win=True, entry_hp=1000.,
                source_entry_hp=1000., hp_after=1000-lost, hp_lost=float(lost), decisions=7,
                cut=False, ub=False, first_combat_hash='a'*64, grid_hash=frozen['grid_hash']))
    return rows, conditions, frozen


def analyze(fixture, resamples=32):
    rows, conditions, frozen = fixture
    return factorial.factorial_metrics(rows, conditions, frozen, resamples=resamples)


class ArithmeticTests(unittest.TestCase):
    def test_known_main_effects_fixed_levels_and_standard_interaction_scales(self):
        def outcome(c, row):
            f, w, lr = int(c['legacy_action_features']), int(c['width'] == 216), int(c['lr'] == .001)
            return (50 + 2*f + 4*w + 8*lr + 10*f*w + 14*f*lr + 18*w*lr + 22*f*w*lr +
                    5*(c['seed'] == 2020) + row['salt'] + row['world_seed']-WORLD)
        report = analyze(synthetic(outcome))
        metric = report['groups']['BOSS2:resampled']['hp_lost']
        self.assertEqual(metric['main_effect_role'], 'primary')
        for f, expected in (('feature', 19.5), ('capacity', 23.5), ('lr', 29.5)):
            effect = metric['main_effects'][f]
            self.assertEqual(effect['mean_of_seeds']['point'], expected)
            self.assertEqual(effect['direction_status'], 'replicated')
            self.assertEqual(effect['direction'], 'positive')
        interactions = metric['secondary']['interactions']
        for name, expected in (('feature:capacity', 21), ('feature:lr', 25),
                               ('capacity:lr', 29), ('feature:capacity:lr', 22)):
            self.assertEqual(interactions[name]['mean_of_seeds']['point'], expected)
        fixed = metric['secondary']['fixed_level_effects']
        self.assertEqual(fixed['feature|capacity=128,lr=0.0003']['mean_of_seeds']['point'], 2)
        self.assertEqual(fixed['feature|capacity=216,lr=0.001']['mean_of_seeds']['point'], 48)
        self.assertEqual(len(metric['secondary']['seed_2020_minus_1919']), 8)
        self.assertTrue(all(v['point'] == 5 for v in metric['secondary']['seed_2020_minus_1919'].values()))
        self.assertEqual(report['groups']['BOSS2:resampled']['win']['main_effect_role'], 'descriptive')
        self.assertEqual(report['groups']['BOSS2:unsalted']['hp_lost']['main_effect_role'], 'descriptive')
        self.assertFalse(report['filesystem_validated'])
        self.assertFalse(report['confidence_claims_allowed'])

    def test_salt_means_equal_world_weighting_and_separate_unsalted(self):
        def outcome(c, row):
            delta = 2*(row['world_seed']-WORLD+1) + 16*(row['group'] == 'BOSS1') + max(0, row['salt'])
            return 30 + int(c['legacy_action_features'])*delta
        result = analyze(synthetic(outcome))['groups']
        def mean(tag):
            return result[tag]['hp_lost']['main_effects']['feature']['mean_of_seeds']
        self.assertAlmostEqual(mean('BOSS2:resampled')['point'], 7.5)
        self.assertAlmostEqual(mean('all:resampled')['point'], (13.5+7.5+9.5)/3)
        self.assertNotAlmostEqual(mean('all:resampled')['point'], (5.5+21.5+7.5+9.5)/4)
        self.assertAlmostEqual(mean('BOSS2:unsalted')['point'], 4)
        self.assertEqual(mean('all:resampled')['worlds'], 3)

    def test_identical_seed_world_deltas_have_identical_intervals_not_twice_n(self):
        def outcome(c, row):
            return 30 + 5*(c['seed'] == 2020) + int(c['legacy_action_features'])*(row['world_seed']-WORLD+1)
        effect = analyze(synthetic(outcome), resamples=128)['groups']['BOSS2:resampled']['hp_lost']['main_effects']['feature']
        for seed in ('1919', '2020'):
            self.assertEqual(effect['seeds'][seed], effect['mean_of_seeds'])
        self.assertEqual(effect['mean_of_seeds']['worlds'], 3)
        self.assertEqual(effect['mean_of_seeds']['rows'], 3)
        self.assertGreater(effect['mean_of_seeds']['se'], 0)

    def test_opposite_seed_effects_are_never_replicated(self):
        for second, status, point in ((-2, 'suggestive', 2), (-6, 'inconclusive', 0), (2, 'replicated', 4)):
            with self.subTest(second=second):
                data = synthetic(lambda c, r: 30 + int(c['legacy_action_features'])*(6 if c['seed'] == 1919 else second))
                effect = analyze(data)['groups']['BOSS2:resampled']['hp_lost']['main_effects']['feature']
                self.assertEqual(effect['direction_status'], status)
                self.assertEqual(effect['mean_of_seeds']['point'], point)
        data = synthetic(lambda c, r: 30 + int(c['legacy_action_features'])*(r['world_seed']-WORLD))
        effect = analyze(data, resamples=128)['groups']['BOSS2:resampled']['hp_lost']['main_effects']['feature']
        self.assertEqual(effect['direction_status'], 'inconclusive')  # interval touches zero

    def test_win_effects_are_probabilities(self):
        rows, conditions, grid = synthetic(lambda c, r: 30)
        for key, values in rows.items():
            for row in values:
                row['win'] = conditions[key]['legacy_action_features']
                if not row['win']:
                    row.update(hp_after=0, hp_lost=row['entry_hp'])
        metric = analyze((rows, conditions, grid))['groups']['BOSS2:resampled']['win']
        self.assertEqual(metric['main_effects']['feature']['mean_of_seeds']['point'], 1)

    def test_inventory_missing_extra_duplicate_and_wrong_actual_conditions(self):
        _, original, _ = synthetic(lambda c, r: 30)
        first, second = list(original)[:2]
        variants = []
        missing = deepcopy(original)
        missing.pop(first)
        variants.append(missing)
        variants.append(dict(original, extra=original[first]))
        duplicate = deepcopy(original)
        duplicate[second] = deepcopy(duplicate[first])
        variants.append(duplicate)
        wrong = deepcopy(original)
        wrong[first]['lr'] = .0001
        variants.append(wrong)
        variants.append({**{k: v for k, v in original.items() if k != first}, 'caller_label': original[first]})
        for field, value in (('fit_size', '20000'), ('phase', 'ooc'), ('batch', True), ('seed', True),
                              ('device', 'cpu'), ('min_steps', 1999), ('layers', 1), ('defer_test', False)):
            bad = deepcopy(original)
            bad[first][field] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad.get(first)):
                with self.assertRaises(ValueError):
                    factorial._inventory(bad)

    def test_exact_cell_pairing_faults_hashes_and_finite_values(self):
        data = synthetic(lambda c, r: 30)
        first = next(iter(data[0]))
        for mutation in ('missing', 'duplicate', 'extra', 'fault', 'nan', 'hash', 'hp', 'salt'):
            rows, conditions, grid = deepcopy(data)
            values = rows[first]
            if mutation == 'missing':
                values.pop()
            elif mutation in ('duplicate', 'extra'):
                row = deepcopy(values[0])
                if mutation == 'extra':
                    row['start_id'] = 999
                values.append(row)
            elif mutation == 'fault':
                values[1]['fault'] = True
            elif mutation == 'nan':
                values[1]['hp_lost'] = float('nan')
            elif mutation == 'hash':
                values[1]['first_combat_hash'] = 'f'*64
            elif mutation == 'salt':
                values[1]['salt'] = 99
            else:
                values[1].update(entry_hp=1001., hp_lost=31.)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze((rows, conditions, grid))


class FactorialFixture(PanelFixture):
    def __init__(self, root):
        super().__init__(root)
        manifest = dict(schema='fresh_data_manifest_v2', filters=FILTERS,
            source_files=[file_identity(self.root / 'bank-shard.json')],
            public_surface={'provenance': {'native_provenance': self.native}},
            rows=[dict(source_row_id=str(i), phase='combat', split=split, has_target=True)
                  for i, split in enumerate(['FIT']*3 + ['VALIDATION']*2)],
            subsets={'combat': {'full': ['0', '1', '2']}})
        manifest['data_identity'] = digest(manifest)
        self.manifest = self.root / 'manifest.json'
        write(self.manifest, manifest)
        self.arms = {}
        for cell in product(factorial.SEEDS, *factorial.LEVELS):
            c = condition(*cell)
            key = factorial.condition_key(c)
            path = self.root / (key + '.pt')
            checkpoint(path, feature=c['legacy_action_features'])
            payload = torch.load(path, weights_only=False)
            meta = payload['metadata']
            meta['config'] = dict(c, architecture='controlled')
            meta['model_config'].update(width=c['width'], layers=2, heads=4)
            meta.update(data_identity=manifest['data_identity'], critic_status='frozen_untrained',
                        loss_normalization='actual_choice_batch_mean', optimization_rows={'manifest_fit_rows': 3})
            payload['step'] = meta['training_state']['best_step'] = 2000
            snapshot = self.root / ('train-' + key) / 'records' / 'launch-fixture'
            snapshot.mkdir(parents=True)
            (snapshot / 'source.zip').write_bytes(b'synthetic frozen trainer source')
            (snapshot / self.native['module_file']).write_bytes((self.root / 'native-fixture').read_bytes())
            write(snapshot / 'build_manifest.json', self.native)
            write(snapshot / 'launch.json', dict(schema='sts_training_launch_v1',
                effective_config=meta['config'], parent_checkpoint=None, native=self.native,
                runtime=self.runtime, environment={}, source_archive=file_identity(snapshot / 'source.zip')))
            meta['source_receipt'] = dict(launch=file_identity(snapshot / 'launch.json'),
                data_identity=manifest['data_identity'], weight_parent=None, label_source_files=manifest['source_files'],
                filters=meta['filters'], vocabulary_sha256=meta['vocabulary_sha256'], architecture='controlled', defer_test=True)
            torch.save(payload, path)
            spec = dict(checkpoint=file_identity(path), architecture='controlled',
                        condition=checkpoint_training_condition(payload))
            selection = self.root / (key + '-selection.json')
            write(selection, dict(selected_checkpoint=spec['checkpoint'], manifest=file_identity(self.manifest),
                selection_criterion=SELECTION_CRITERION, architecture='controlled', training_condition=spec['condition']))
            policy = dict(self.policies['new'], combat=spec['checkpoint'])
            self.policies[key] = policy
            directory = self.root / key
            setattr(self, key, directory)
            self.make_run(key, self.root / 'native-fixture')
            self.arms[key] = dict(spec, run_directory=str(directory), policy=policy, selection=file_identity(selection))
        self.freeze = self.root / 'global-freeze.json'
        write(self.freeze, {'checkpoints': {k: a['checkpoint'] for k, a in self.arms.items()},
                            'old_reference_checkpoints': []})
        self.expected = dict(schema=factorial.EXPECTATIONS_SCHEMA, version=1, mode='panels',
            grid=file_identity(self.grid), manifest=file_identity(self.manifest),
            global_freeze=file_identity(self.freeze), arms=self.arms)
        write(self.expectations, self.expected)

    def compare(self, resamples=8):
        return factorial.compare_panel_runs(self.expectations, resamples=resamples)

    def change_checkpoint(self, key, change):
        spec = self.arms[key]
        path = Path(spec['checkpoint']['path'])
        payload = torch.load(path, weights_only=False)
        change(payload)
        torch.save(payload, path)
        spec['checkpoint'] = spec['policy']['combat'] = file_identity(path)
        selection = read(spec['selection']['path'])
        selection['selected_checkpoint'] = spec['checkpoint']
        write(spec['selection']['path'], selection)
        spec['selection'] = file_identity(spec['selection']['path'])
        freeze = read(self.freeze)
        freeze['checkpoints'][key] = spec['checkpoint']
        write(self.freeze, freeze)
        self.expected['global_freeze'] = file_identity(self.freeze)
        write(self.expectations, self.expected)


class FilesystemTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(builder_stub())
        self.enterContext(patch.multiple(factorial, FIT_ROWS=3, VALIDATION_ROWS=2,
            BANK_START=WORLD, BANK_WORLDS=3, PANEL_TARGETS={'BOSS2': 3, 'EVENT2': 1}))
        self.f = FactorialFixture(root)
        self.first, self.last = list(self.f.arms)[0], list(self.f.arms)[-1]

    def test_complete_sixteen_panels_with_real_inherited_validators(self):
        report = self.f.compare()
        self.assertTrue(report['filesystem_validated'])
        self.assertTrue(report['confidence_claims_allowed'])
        self.assertTrue(report['control_exact_match'])
        self.assertEqual(len(report['sources']), 16)
        self.assertEqual(report['expectations_file'], file_identity(self.f.expectations))
        self.assertEqual(report['expected_bindings']['global_freeze'], file_identity(self.f.freeze))
        self.assertIn(file_identity(self.f.manifest), report['verified_files'])
        effect = report['groups']['BOSS2:resampled']['hp_lost']['main_effects']['feature']
        self.assertEqual(effect['mean_of_seeds']['worlds'], 3)
        self.assertEqual(report['bootstrap_resamples'], 8)

    def test_actual_checkpoint_condition_cannot_be_relabelled(self):
        self.f.change_checkpoint(self.first, lambda p: p['metadata']['config'].update(lr=.0001))
        with self.assertRaisesRegex(ValueError, 'actual checkpoint metadata'):
            self.f.compare()

    def test_deferred_validation_best_selection_is_required(self):
        self.f.change_checkpoint(self.first, lambda p: p.update(step=2100))
        with self.assertRaisesRegex(ValueError, 'validation-selected best step'):
            self.f.compare()

    def test_expectation_schema_inventory_freeze_policy_and_manifest_bindings(self):
        original = self.f.expectations.read_bytes()
        mutations = [lambda e: e.update(version=True), lambda e: e.update(mode='labels'),
            lambda e: e['arms'].pop(self.first),
            lambda e: e['arms'].update(extra=deepcopy(e['arms'][self.first])),
            lambda e: e['arms'][self.last].update(run_directory=e['arms'][self.first]['run_directory']),
            lambda e: e['arms'][self.first].update(checkpoint=e['arms'][self.last]['checkpoint']),
            lambda e: e['arms'][self.first]['policy'].update(ooc=e['arms'][self.first]['checkpoint']),
            lambda e: e['arms'][self.first]['condition'].update(width=216),
            lambda e: e['manifest'].update(sha256='f'*64)]
        for change in mutations:
            expected = json.loads(original)
            change(expected)
            write(self.f.expectations, expected)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.f.compare()
        self.f.expectations.write_bytes(original)
        freeze = read(self.f.freeze)
        freeze['checkpoints'][self.first] = freeze['checkpoints'][self.last]
        write(self.f.freeze, freeze)
        self.f.expected['global_freeze'] = file_identity(self.f.freeze)
        write(self.f.expectations, self.f.expected)
        with self.assertRaisesRegex(ValueError, 'root global freeze'):
            self.f.compare()

    def test_duplicate_json_keys_and_nonfinite_expectations_fail_closed(self):
        original = self.f.expectations.read_text()
        for text, error in ((original.replace('"version": 1', '"version": 1, "version": 1'), 'Duplicate JSON key'),
                            (original.replace('"version": 1', '"version": NaN'), 'Nonfinite JSON')):
            self.f.expectations.write_text(text)
            with self.assertRaisesRegex(ValueError, error):
                self.f.compare()
        self.f.expectations.write_text(original)

    def test_complete_reader_rejects_fault_cut_wrong_grid_and_missing_cells(self):
        directory = getattr(self.f, self.first)
        original = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        for change in (lambda r: r.update(fault=True), lambda r: r.update(cut=True),
                       lambda r: r.update(grid_hash='f'*64), lambda r: r.update(hp_lost=float('inf'))):
            self.f.change_row(self.first, change)
            with self.assertRaises(ValueError):
                self.f.compare()
            for p, content in original.items():
                p.write_bytes(content)
        for variant in ('missing', 'duplicate', 'extra'):
            summary = read(directory / 'summary.json')
            if variant == 'missing':
                summary['records'].pop()
            else:
                row = deepcopy(summary['records'][0])
                if variant == 'extra':
                    row['start_id'] = 999
                summary['records'].append(row)
            write(directory / 'summary.json', summary)
            self.f.reseal(self.first)
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.f.compare()
            for p in (directory / 'results').iterdir():
                if p not in original:
                    p.unlink()
            for p, content in original.items():
                p.write_bytes(content)

    def test_wrong_predeclared_grid_is_rejected(self):
        with patch.object(factorial, 'PANEL_TARGETS', {'BOSS2': 400, 'BOSS1': 100, 'MONSTER2': 100}):
            with self.assertRaisesRegex(ValueError, 'predeclared factorial panel grid'):
                self.f.compare()

    def test_controls_and_initial_states_match_across_all_arms(self):
        directory = getattr(self.f, self.last)
        original = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        self.f.change_row(self.last, lambda r: r.update(decisions=99), policy='champion')
        with self.assertRaisesRegex(ValueError, 'Champion control exact-match'):
            self.f.compare()
        for p, content in original.items():
            p.write_bytes(content)
        self.f.change_row(self.last, lambda r: r.update(entry_hp=51., hp_lost=51-r['hp_after']))
        with self.assertRaisesRegex(ValueError, 'Paired start/HP'):
            self.f.compare()

    def test_all_runtime_and_environment_identities_match(self):
        directory = getattr(self.f, self.last)
        original = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        for field, value in (('runtime', {'python': 'other', 'platform': 'other',
                                        'packages': {'torch': 'other', 'numpy': 'other'}}),
                             ('environment', {'OMP_NUM_THREADS': '99'})):
            receipt = read(directory / 'receipt.json')
            launch_path = Path(receipt['launch']['path'])
            launch = read(launch_path)
            launch[field] = value
            write(launch_path, launch)
            self.f.change_receipt(self.last, lambda r: r.update(launch=file_identity(launch_path)))
            with self.assertRaisesRegex(ValueError, 'Panel native/runtime/environment'):
                self.f.compare()
            for p, content in original.items():
                p.write_bytes(content)

    def test_pinned_inputs_and_result_directories_rechecked_after_statistics(self):
        real = factorial.clustered_mean
        for target in (self.f.freeze, getattr(self.f, self.last) / 'summary.json', None):
            changed = False
            def mutate(*args, **kwargs):
                nonlocal changed
                if not changed:
                    if target is None:
                        write(getattr(self.f, self.last) / 'results' / 'late.json', {'late': True})
                    else:
                        target.write_bytes(target.read_bytes() + b' ')
                    changed = True
                return real(*args, **kwargs)
            original = target.read_bytes() if target else None
            with patch.object(factorial, 'clustered_mean', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed during validation'):
                    self.f.compare()
            if target is None:
                (getattr(self.f, self.last) / 'results' / 'late.json').unlink()
            else:
                target.write_bytes(original)

    def test_cli_publishes_exclusively_and_failure_writes_no_report(self):
        path = Path(__file__).resolve().parents[2] / 'scripts' / 'rl' / 'controlled_factorial.py'
        spec = importlib.util.spec_from_file_location('factorial_cli_fixture', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        out = self.f.root / 'factorial-report.json'
        argv = ['panels', '--expectations', str(self.f.expectations), '--out', str(out)]
        real = factorial.compare_panel_runs
        with patch.object(cli, 'compare_panel_runs', side_effect=lambda path: real(path, resamples=8)) as compare:
            with redirect_stdout(io.StringIO()) as stream:
                cli.main(argv)
            self.assertEqual(read(out)['schema'], 'controlled_factorial_v1')
            self.assertIn('primary', json.loads(stream.getvalue()))
            original = out.read_bytes()
            with self.assertRaises(FileExistsError):
                cli.main(argv)
            self.assertEqual(compare.call_count, 1)
            self.assertEqual(out.read_bytes(), original)
            out.unlink()
            self.f.expectations.write_text('{}')
            with self.assertRaises(ValueError):
                cli.main(argv)
            self.assertFalse(out.exists())
        self.assertFalse(list(out.parent.glob('.contrast-*')))


if __name__ == '__main__':
    unittest.main()
