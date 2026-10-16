"""Bounded synthetic controlled inference/scoring/paired-comparison contracts."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from sts_rl import controlled_policy, controlled_scoring, controlled_comparison, fresh_panel, fresh_training, fresh_scoring
from sts_rl.fresh_study import StudyPolicy, atomic_json, policy_observation, RunControl
from sts_rl.recording import file_identity
from tests.unit.controlled_fixture import (builder_stub, champion, checkpoint, ScoreFixture,
                                           write_rows, complete_game_fixture)
from tests.integration.test_fresh_policy import observation
from tests.unit.test_fresh_panel_comparison import PanelFixture, read, write


class ControlledTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        torch.set_num_threads(1)
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.stub = self.enterContext(builder_stub())
        self.champion = champion(self.root / 'champion.pt')

    def replace(self, architecture='controlled', phase='combat', feature=False):
        path = self.root / f'{architecture}-{phase}-{feature}.pt'
        checkpoint(path, architecture, phase, feature)
        return path

    def mutate(self, path, change):
        payload = torch.load(path, weights_only=False)
        change(payload)
        torch.save(payload, path)

    def test_controlled_exact_singleton_first_tie_original_indices_and_inherited_methods(self):
        path = self.replace()
        policy = controlled_policy.ControlledStudyPolicy(self.champion, combat=path)
        self.assertIs(type(policy).choose_action, StudyPolicy.choose_action)
        self.assertIs(type(policy).distribution, StudyPolicy.distribution)
        raw = observation('combat')
        saved = deepcopy(raw)
        self.assertEqual(policy.choose_action(raw), 3)
        indices, probabilities = policy.distribution(raw)
        self.assertEqual(indices, [3, 17, 64])
        for probability in probabilities:
            self.assertAlmostEqual(probability, 1/3)
        self.assertEqual(raw, saved)
        self.assertEqual(policy.identity['combat'], file_identity(path))
        self.assertEqual(policy.identity['champion'], file_identity(self.champion))
        self.assertEqual(set(self.stub.observed[-1]), set(raw))

    def test_ooc_inventory_derived_once_no_refilter_and_champion_unchanged(self):
        policy = controlled_policy.ControlledStudyPolicy(self.champion, ooc=self.replace(phase='ooc', feature=True))
        raw = observation('ooc')
        raw['text'] = 'Potions: EMPTY_POTION_SLOT'
        raw['legal_actions'] += [dict(index=99, description='discard potion Fire Potion (slot 1)')]
        with patch('sts_rl.fresh_study.policy_observation', wraps=policy_observation) as filtered:
            self.assertEqual(policy.choose_action(raw), 3)
        self.assertEqual(filtered.call_count, 1)
        seen = self.stub.observed[-1]
        self.assertEqual(seen['adapter_public_ooc_potions'], [dict(slot=1, name='Fire Potion')])
        self.assertNotIn(99, [a['index'] for a in seen['legal_actions']])
        reference = StudyPolicy(self.champion)
        self.assertEqual(policy.choose_action(observation('combat')), reference.choose_action(observation('combat')))

    def test_original_old_new_mixed_routing_and_return_to_champion(self):
        old, new = self.replace('old'), self.replace('new', phase='ooc')
        policy = controlled_policy.ControlledStudyPolicy(self.champion, combat=old, ooc=new)
        reference = StudyPolicy(self.champion, combat=old, ooc=new)
        for phase in ('combat', 'ooc'):
            self.assertEqual(policy.distribution(observation(phase)), reference.distribution(observation(phase)))
        routed = controlled_policy.policy_factory(self.champion, combat=old)
        self.assertIs(type(routed), StudyPolicy)
        self.assertIs(type(controlled_policy.policy_factory(self.champion)), StudyPolicy)
        self.assertIsInstance(controlled_policy.policy_factory(self.champion, combat=self.replace()),
                              controlled_policy.ControlledStudyPolicy)

    def test_phase_hash_finite_metadata_and_feature_fail_closed(self):
        path = self.replace()
        original = path.read_bytes()
        changes = [lambda p: p['metadata']['model_config'].update(phase='ooc'),
                   lambda p: p['metadata'].update(encoder_sha256='f' * 64),
                   lambda p: p['metadata'].update(vocabulary_sha256='f' * 64),
                   lambda p: p['metadata']['model_config'].update(legacy_action_features=1),
                   lambda p: p['model_state']['weight'].fill_(float('nan')),
                   lambda p: p['metadata'].update(kind='unknown')]
        for change in changes:
            with self.subTest(change=change):
                path.write_bytes(original)
                self.mutate(path, change)
                with self.assertRaises(ValueError):
                    controlled_policy.ControlledStudyPolicy(self.champion, combat=path)

    def test_indices_feature_shape_logits_shape_and_nonfinite_rejected(self):
        policy = controlled_policy.ControlledStudyPolicy(self.champion, combat=self.replace())
        encode, collate = self.stub.encoder.encode_observation, self.stub.encoder.collate_observations
        def wrong_indices(obs):
            x = encode(obs)
            x.action_indices = x.action_indices[::-1].copy()
            return x
        with patch.object(self.stub.encoder, 'encode_observation', side_effect=wrong_indices):
            with self.assertRaisesRegex(ValueError, 'index mismatch'):
                policy.choose_action(observation('combat'))
        def wrong_shape(rows, device):
            batch = collate(rows, device)
            batch['legacy_action_features'] = torch.zeros(1, 3, 127)
            return batch
        with patch.object(self.stub.encoder, 'collate_observations', side_effect=wrong_shape):
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                policy.choose_action(observation('combat'))
        for logits in (torch.zeros(3), torch.full((1, 3), float('nan'))):
            with patch.object(policy.replacements['combat'], 'actor_logits', return_value=logits):
                with self.assertRaisesRegex(ValueError, 'logits/menu'):
                    policy.choose_action(observation('combat'))

    def test_changed_checkpoint_during_dispatch_and_load_is_rejected(self):
        path = self.replace()
        real = controlled_policy.replacement_from_payload
        def changing(payload, phase):
            result = real(payload, phase)
            path.write_bytes(path.read_bytes() + b'changed')
            return result
        with patch.object(controlled_policy, 'replacement_from_payload', side_effect=changing):
            with self.assertRaisesRegex(RuntimeError, 'changed during load'):
                controlled_policy.ControlledStudyPolicy(self.champion, combat=path)

    def test_parent_and_fake_native_child_call_actual_factory(self):
        from tests.unit.test_fresh_panel_grid import rows
        path = self.replace()
        calls = []
        class Worker:
            def __init__(self, **kwargs):
                self.episode = object.__new__(fresh_panel.FreshPanelEpisode)
                self.episode._preinitialization = lambda bank: (object(), object())
                def scratch(raw, base, salt, bank, policy=None):
                    if policy is None:
                        return {}  # unsalted replay verification, without native execution
                    combat = policy.identity['combat']
                    calls.append((combat, type(policy), policy.choose_action(observation('combat'))))
                    return dict(win=True, hp_lost=3, first_combat_hash=str(salt), decisions=1)
                self.episode._scratch = scratch
            def fresh_fight(worker, bank, salt, champ, combat, expected):
                return worker.episode.fresh_fight(bank, salt, champ, combat, expected)
            def close(self):
                pass
        def prepare(out, config, policy, **kwargs):
            receipt = dict(protocol='fresh_start_v1', policy=policy.identity, config=config)
            atomic_json(Path(out) / 'receipt.json', receipt)
            return receipt
        with patch.object(fresh_panel, '_POLICIES', {}), \
             patch.object(fresh_panel, '_panel_bank', return_value=({'inputs': [], 'shards': []}, rows(1)[:1], {})), \
             patch.object(fresh_panel, 'prepare_run', side_effect=prepare), \
             patch.object(fresh_panel, 'RunControl', side_effect=lambda *a, **k: RunControl(*a, **k, ac=lambda: True)), \
             patch.object(fresh_panel, 'policy_factory', wraps=controlled_policy.policy_factory) as factory, redirect_stdout(io.StringIO()):
            result = fresh_panel.run_panel(dict(checkpoint=str(self.champion), combat=str(path), workers=1,
                timeout=1., resamples=1, starts_per_group={'BOSS2': 1}, out=str(self.root / 'panel')), worker_factory=Worker)
        self.assertTrue(result['results_complete'])
        self.assertEqual(factory.call_count, 3)  # parent, cached-child champion, cached-child controlled
        self.assertEqual([t for p, t, _ in calls if p], [controlled_policy.ControlledStudyPolicy] * 2)


class ControlledScoringTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        torch.set_num_threads(1)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.stub = self.enterContext(builder_stub())
        self.f = ScoreFixture(self.root, self.stub)

    def test_scorer_rejects_unfrozen_or_incompatible_selection_before_encoding(self):
        path = self.f.selections['left']
        saved = path.read_bytes()
        cases = [lambda s: s.update(selection_criterion=dict(metric='test_soft_ce', direction='min', tie_break='lower_lr')),
                 lambda s: s.update(training_condition={}), lambda s: s.update(architecture='old'),
                 lambda s: s['selected_checkpoint'].update(sha256='0' * 64),
                 lambda s: s['manifest'].update(sha256='0' * 64)]
        with patch.object(fresh_scoring, 'preencode', side_effect=AssertionError('raw TEST encoding')) as encode:
            for change in cases:
                path.write_bytes(saved)
                selected = json.loads(saved)
                change(selected)
                atomic_json(path, selected)
                with self.assertRaises(ValueError):
                    fresh_scoring.score_deferred(path, self.root / 'out')
            self.assertFalse(encode.called)
            self.assertFalse((self.root / 'out').exists())

    def test_scorer_rejects_nonbest_nondeferred_and_source_metadata_mismatch(self):
        path = Path(self.f.checkpoints['left']['path'])
        saved = path.read_bytes()
        cases = [lambda p: p.update(step=11),
                 lambda p: p['metadata']['config'].update(defer_test=False),
                 lambda p: p['metadata']['source_receipt'].update(data_identity='f' * 64),
                 lambda p: p['metadata']['config'].update(legacy_action_features=True)]
        selection_path = self.f.selections['left']
        selected = read(selection_path)
        with patch.object(fresh_scoring, 'preencode', side_effect=AssertionError('raw TEST encoding')):
            for change in cases:
                path.write_bytes(saved)
                payload = torch.load(path, weights_only=False)
                change(payload)
                torch.save(payload, path)
                atomic_json(selection_path, {**selected, 'selected_checkpoint': file_identity(path)})
                with self.assertRaises(ValueError):
                    fresh_scoring.score_deferred(selection_path, self.root / 'out')

    def test_controlled_score_calls_only_builder_test_helpers_and_receipts(self):
        class Control:
            def __init__(self, *args):
                self.status = {}
            def start(self): pass
            def check(self): pass
            def close(self): pass
            def emit(self): pass
            def stats(self): return dict(active_ac_seconds=0)
        marker = object()
        class CheckedManifest(dict):
            pass
        manifests = []
        frozen_selection = controlled_scoring.frozen_selection
        def checked_selection(*args):
            selected, identity, manifest, meta, condition = frozen_selection(*args)
            checked = CheckedManifest(manifest)
            manifests.append(weakref.ref(checked))
            return selected, identity, checked, meta, condition
        def freeze(*args, **kwargs):
            self.assertTrue(manifests)
            self.assertTrue(all(ref() is None for ref in manifests), 'Duplicate parsed manifest retained')
            return self.f.manifest
        def begin(out, effective, checkpoint, inputs):
            return self.f.launch(Path(out), effective, file_identity(checkpoint), [file_identity(p) for p in inputs])
        def score(checkpoint_path, examples, architecture, out, **kwargs):
            self.assertIs(examples, marker)
            self.assertEqual(architecture, 'controlled')
            write_rows(Path(out) / 'test_rows.jsonl', self.f.rows['left'])
            return dict(test_evaluated=True, test_protocol='single_observation_exact_argmax_actor_only_v1',
                        test_device='cpu', test_rows=3)
        with ExitStack() as stack:
            stack.enter_context(patch.object(controlled_scoring, 'frozen_selection', side_effect=checked_selection))
            stack.enter_context(patch.object(fresh_scoring, 'TrainingControl', Control))
            stack.enter_context(patch.object(fresh_scoring, 'freeze_manifest', side_effect=freeze))
            encode = stack.enter_context(patch.object(fresh_scoring, 'preencode', return_value={'TEST': marker}))
            stack.enter_context(patch.object(fresh_scoring, 'score_test_checkpoint', side_effect=score))
            stack.enter_context(patch.object(fresh_scoring, 'begin_run', side_effect=begin))
            stack.enter_context(patch.object(fresh_training, 'fit', side_effect=AssertionError('fit')))
            result = fresh_scoring.score_deferred(self.f.selections['left'], self.root / 'score-out')
        self.assertEqual(result['weight_updates'], 0)
        self.assertEqual(encode.call_args.kwargs['splits'], ('TEST',))
        self.assertEqual(result['test_rows_file'], file_identity(self.root / 'score-out' / 'test_rows.jsonl'))

    def test_old_score_retains_original_path(self):
        with patch.object(controlled_scoring, 'frozen_selection', side_effect=AssertionError('controlled validation')), \
             patch.object(fresh_scoring, 'TrainingControl', side_effect=RuntimeError('original control boundary')):
            with self.assertRaisesRegex(RuntimeError, 'original control boundary'):
                fresh_scoring.score_deferred(self.f.selections['right'], self.root / 'out')

    def test_strict_labels_known_agreement_and_ce_both_world_bootstrapped(self):
        result = controlled_comparison.compare_labels_runs(self.root / 'left', self.root / 'right',
                                                           self.f.cohort, self.f.expectations)
        group = result['groups']['high_confidence_choice']
        self.assertAlmostEqual(group['agreement']['point'], 1/3)
        self.assertAlmostEqual(group['soft_ce']['point'], -.5)
        self.assertEqual(group['agreement']['bootstrap_resamples'], 10_000)
        self.assertEqual(group['agreement']['bootstrap_seed'], 1919)
        self.assertIn('act:2', result['groups'])
        self.assertIn('room:BOSS', result['groups'])
        self.assertTrue(result['filesystem_validated'])
        self.assertNotIn('gate', json.dumps(result))

    def test_labels_compare_manifest_bindings_without_retaining_or_serializing_full_manifests(self):
        class CheckedManifest(dict):
            pass
        manifests = []
        frozen_selection = controlled_comparison.frozen_selection
        same = controlled_comparison._same
        def checked_selection(*args):
            self.assertTrue(all(ref() is None for ref in manifests), 'Previous arm manifest retained')
            selected, identity, manifest, meta, condition = frozen_selection(*args)
            checked = CheckedManifest(manifest)
            manifests.append(weakref.ref(checked))
            return selected, identity, checked, meta, condition
        def compact_same(a, b):
            for value in (a, b):
                self.assertFalse(isinstance(value, dict) and value.get('schema') == 'fresh_data_manifest_v2',
                                 'Full manifest serialized for equality')
            return same(a, b)
        with patch.object(controlled_comparison, 'frozen_selection', side_effect=checked_selection), \
             patch.object(controlled_comparison, '_same', side_effect=compact_same):
            result = controlled_comparison.compare_labels_runs(self.root / 'left', self.root / 'right',
                                                               self.f.cohort, self.f.expectations)
        self.assertTrue(result['filesystem_validated'])
        self.assertEqual(len(manifests), 2)
        self.assertTrue(all(ref() is None for ref in manifests))

    def test_label_comparison_swapped_missing_misaligned_and_tampered_arms_reject(self):
        compare = lambda: controlled_comparison.compare_labels_runs(self.root / 'left', self.root / 'right',
                                                                    self.f.cohort, self.f.expectations)
        with self.assertRaises(ValueError):
            controlled_comparison.compare_labels_runs(self.root / 'right', self.root / 'left', self.f.cohort, self.f.expectations)
        original = deepcopy(self.f.rows['left'])
        for change in (lambda r: r.pop(), lambda r: r[0].update(target=[.2, .8]),
                       lambda r: r[0].update(architecture='new'),
                       lambda r: r[0].update(source_identity='f' * 64),
                       lambda r: r[0].update(target_sha256='f' * 64),
                       lambda r: r[0].update(world_seed=91_200_000_000)):
            self.f.rows['left'] = deepcopy(original)
            change(self.f.rows['left'])
            with self.assertRaises((ValueError, AssertionError)):
                self.f.reseal_score('left')
                compare()
        self.f.rows['left'] = original
        self.f.reseal_score('left')
        summary_path = self.root / 'left' / 'summary.json'
        value = read(summary_path)
        nonfinite_rows = deepcopy(original)
        nonfinite_rows[0]['ce'] = float('nan')
        rows_path = self.root / 'left' / 'test_rows.jsonl'
        # Write malformed input directly: the valid-fixture writer rejects NaN.
        rows_path.write_text(''.join(json.dumps(row) + '\n' for row in nonfinite_rows))
        atomic_json(summary_path, {**value, 'test_rows_file': file_identity(rows_path)})
        with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
            compare()
        self.f.reseal_score('left')
        missing_binding = deepcopy(value)
        missing_binding.pop('test_rows_file')
        atomic_json(summary_path, missing_binding)
        with self.assertRaisesRegex(ValueError, 'row file binding'):
            compare()
        value['test']['groups']['choice']['soft_ce'] = 999
        atomic_json(summary_path, value)
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            compare()


class GenericPanelTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        torch.set_num_threads(1)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.stub = self.enterContext(builder_stub())
        self.f = PanelFixture(self.root)
        self.generic = self.root / 'generic.json'
        self.expected = dict(schema='controlled_comparison_expectations_v1', version=1, mode='panel', grid=file_identity(self.f.grid),
            **{arm: dict(policy=self.f.policies[role], replacements=dict(combat=dict(
                checkpoint=self.f.policies[role]['combat'], architecture=role, condition=self.f.conditions[role])))
               for arm, role in (('left', 'new'), ('right', 'old'))})
        atomic_json(self.generic, self.expected)

    def compare(self):
        return controlled_comparison.compare_panel_runs(self.f.new, self.f.old, self.f.grid, self.generic)

    def test_old_new_generic_direct_panel_preserves_primary_and_unsalted(self):
        result = self.compare()
        self.assertAlmostEqual(result['groups']['BOSS2:resampled']['win']['point'], 1/3)
        self.assertEqual(result['groups']['BOSS2:unsalted']['win']['point'], -1)
        self.assertEqual(result['groups']['BOSS2:resampled']['role'], 'primary')
        self.assertEqual(result['groups']['EVENT2:resampled']['role'], 'descriptive')
        self.assertTrue(result['control_exact_match'])
        self.assertIn('left_mean', result['groups']['BOSS2:resampled']['win'])
        self.assertNotIn('new_mean', result['groups']['BOSS2:resampled']['win'])

    def test_null_grid_identity_rejected_before_read(self):
        self.expected['grid'] = None
        atomic_json(self.generic, self.expected)
        read_input = controlled_comparison._Inputs.read
        def no_grid_read(inputs, path, expected=None):
            self.assertNotEqual(Path(path).resolve(), self.f.grid.resolve(), 'Unbound grid read')
            return read_input(inputs, path, expected)
        with patch.object(controlled_comparison._Inputs, 'read', new=no_grid_read):
            with self.assertRaisesRegex(ValueError, 'file identity'):
                self.compare()

    def test_controlled_panel_uses_real_generic_architecture_and_conditions(self):
        path = self.root / 'controlled.pt'
        identity = checkpoint(path)
        self.f.policies['new']['combat'] = identity
        spec = self.expected['left']
        spec['policy'] = self.f.policies['new']
        spec['replacements']['combat'] = dict(checkpoint=identity, architecture='controlled',
            condition=self.stub.condition(torch.load(path, weights_only=False)))
        atomic_json(self.generic, self.expected)
        # Refresh only the run's actual policy/config/launch bindings, retaining
        # the fixture outcomes and original independent champion control rows.
        def change(receipt):
            receipt['policy'] = spec['policy']
            receipt['config']['combat'] = str(path)
        self.f.change_receipt('new', change)
        launch_path = Path(read(self.f.new / 'receipt.json')['launch']['path'])
        launch = read(launch_path)
        launch['data_inputs'][-1] = identity
        write(launch_path, launch)
        self.f.change_receipt('new', lambda r: None)
        self.assertEqual(self.compare()['expected_bindings']['left']['replacements']['combat']['architecture'], 'controlled')

    def test_swapped_missing_grid_control_fault_and_wrong_condition_rejected(self):
        with self.assertRaises(ValueError):
            controlled_comparison.compare_panel_runs(self.f.old, self.f.new, self.f.grid, self.generic)
        wrong_grid = self.root / 'wrong-grid.json'
        frozen = deepcopy(self.f.frozen)
        frozen['grid'][0]['salt'] = 100
        atomic_json(wrong_grid, frozen)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            controlled_comparison.compare_panel_runs(self.f.new, self.f.old, wrong_grid, self.generic)
        self.f.change_row('new', lambda r: r.update(decisions=r['decisions'] + 1), policy='champion')
        with self.assertRaisesRegex(ValueError, 'control exact-match'):
            self.compare()
        self.f.change_row('new', lambda r: r.update(decisions=r['decisions'] - 1), policy='champion')
        self.expected['left']['replacements']['combat']['condition']['seed'] += 1
        atomic_json(self.generic, self.expected)
        with self.assertRaisesRegex(ValueError, 'Frozen condition'):
            self.compare()
        self.expected['left']['replacements']['combat']['condition']['seed'] -= 1
        atomic_json(self.generic, self.expected)
        self.f.change_row('new', lambda r: r.update(cut=True))
        with self.assertRaises(ValueError):
            self.compare()
        self.f.change_row('new', lambda r: r.update(cut=False))
        next((self.f.new / 'results').iterdir()).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.compare()


class GenericPolicyBindingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        torch.set_num_threads(1)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.stub = self.enterContext(builder_stub())
        base = StudyPolicy(champion(self.root / 'champion.pt')).identity
        self.specs = {}
        for arm, feature in (('left', False), ('right', True)):
            identity = checkpoint(self.root / (arm + '.pt'), feature=feature)
            self.specs[arm] = dict(policy=dict(base, combat=identity), replacements=dict(combat=dict(
                checkpoint=identity, architecture='controlled',
                condition=self.stub.condition(torch.load(identity['path'], weights_only=False)))))
        self.expected_path = self.root / 'expected.json'

    def compare(self, mode, specs):
        atomic_json(self.expected_path, dict(schema='controlled_comparison_expectations_v1', version=1,
                                            mode=mode, **specs))
        compare = controlled_comparison.compare_panel_runs if mode == 'panel' else controlled_comparison.compare_games_runs
        # Invalid expectations must fail before any run/grid/cohort access.
        return compare(self.root / 'left', self.root / 'right', self.root / 'unopened.json', self.expected_path)

    def test_panel_and_games_null_architecture_rejected_before_checkpoint_details(self):
        checkpoint_details = controlled_comparison.checkpoint_details
        def declared_architecture(payload, *, architecture, **kwargs):
            self.assertIn(architecture, ('old', 'new', 'controlled'), 'Undeclared checkpoint architecture')
            return checkpoint_details(payload, architecture=architecture, **kwargs)
        with patch.object(controlled_comparison, 'checkpoint_details', side_effect=declared_architecture):
            for mode in ('panel', 'games'):
                for arm in ('left', 'right'):
                    with self.subTest(mode=mode, arm=arm):
                        specs = deepcopy(self.specs)
                        specs[arm]['replacements']['combat']['architecture'] = None
                        with self.assertRaisesRegex(ValueError, 'frozen checkpoint architecture'):
                            self.compare(mode, specs)

    def test_panel_and_games_known_checkpoint_copies_rejected_despite_different_paths(self):
        for mode in ('panel', 'games'):
            with self.subTest(mode=mode):
                specs = deepcopy(self.specs)
                for phase in (('combat',) if mode == 'panel' else ('combat', 'ooc')):
                    if phase == 'ooc':
                        identity = checkpoint(self.root / 'left-ooc.pt', phase='ooc')
                        specs['left']['policy'][phase] = identity
                        specs['left']['replacements'][phase] = dict(checkpoint=identity, architecture='controlled',
                            condition=self.stub.condition(torch.load(identity['path'], weights_only=False)))
                    original = specs['left']['policy'][phase]
                    copy_path = self.root / (mode + '-' + phase + '-copy.pt')
                    copy_path.write_bytes(Path(original['path']).read_bytes())
                    copied = file_identity(copy_path)
                    self.assertNotEqual(original['path'], copied['path'])
                    self.assertEqual(original['sha256'], copied['sha256'])
                    specs['right']['policy'][phase] = copied
                    specs['right']['replacements'][phase] = dict(specs['left']['replacements'][phase], checkpoint=copied)
                with self.assertRaisesRegex(ValueError, 'different LEFT/RIGHT policy checkpoint SHAs'):
                    self.compare(mode, specs)

    def test_games_champion_versus_replacement_remains_allowed(self):
        specs = deepcopy(self.specs)
        specs['left']['policy']['combat'] = None
        specs['left']['replacements'] = {}
        atomic_json(self.expected_path, dict(schema='controlled_comparison_expectations_v1', version=1,
                                            mode='games', **specs))
        controlled_comparison._expectations(self.expected_path, 'games', controlled_comparison._Inputs())


class GenericGameTests(unittest.TestCase):
    def game_pairs(self, outcomes):
        seeds = list(range(91_001_000_000, 91_001_000_000 + len(outcomes)))
        def row(seed, win):
            return dict(world_seed=seed, status='completed', win=win, floor=10,
                        bosses={}, expected_provenance={'fixture': 'synthetic'})
        return ([row(seed, a) for seed, (a, b) in zip(seeds, outcomes)],
                [row(seed, b) for seed, (a, b) in zip(seeds, outcomes)], seeds)

    def test_game_exact_mcnemar_known_discordant_and_concordant_counts(self):
        left, right, seeds = self.game_pairs([(True, False)] * 7 + [(False, True)] +
                                            [(True, True)] * 2 + [(False, False)] * 3)
        # Scramble RIGHT to establish pairing by world rather than row position.
        result = controlled_comparison.game_metrics(left, right[::-1], seeds, resamples=32)
        win = result['win']
        for field, expected in dict(left_only_wins=7, right_only_wins=1, left_wins=9,
                                    right_wins=3, both_wins=2, neither_wins=3).items():
            self.assertEqual(win[field], expected, field)
        self.assertEqual(result['paired_worlds'], 13)
        self.assertAlmostEqual(win['point'], 6 / 13)
        # Two-sided Binomial(8, .5): 2 * (C(8, 0) + C(8, 1)) / 2**8.
        self.assertAlmostEqual(win['exact_mcnemar_p'], 18 / 256)

    def test_game_exact_mcnemar_is_symmetric_under_arm_swap(self):
        left, right, seeds = self.game_pairs([(True, False)] * 7 + [(False, True)] +
                                            [(True, True)] * 2 + [(False, False)] * 3)
        forward = controlled_comparison.game_metrics(left, right, seeds, resamples=32)['win']
        reverse = controlled_comparison.game_metrics(right, left, seeds, resamples=32)['win']
        for a, b in (('left_only_wins', 'right_only_wins'), ('left_wins', 'right_wins')):
            self.assertEqual(forward[a], reverse[b])
            self.assertEqual(forward[b], reverse[a])
        for field in ('both_wins', 'neither_wins', 'exact_mcnemar_p'):
            self.assertEqual(forward[field], reverse[field])
        self.assertEqual(forward['point'], -reverse['point'])
        for a, b in zip(forward['ci95'], reversed(reverse['ci95'])):
            self.assertAlmostEqual(a, -b)

    def test_game_all_concordant_has_unit_p_and_zero_paired_interval(self):
        for outcomes in ([False] * 4, [True] * 4, [True, False, True, False]):
            with self.subTest(outcomes=outcomes):
                left, right, seeds = self.game_pairs([(win, win) for win in outcomes])
                win = controlled_comparison.game_metrics(left, right, seeds, resamples=32)['win']
                self.assertEqual(win['left_only_wins'], 0)
                self.assertEqual(win['right_only_wins'], 0)
                self.assertEqual(win['left_wins'], sum(outcomes))
                self.assertEqual(win['right_wins'], sum(outcomes))
                self.assertEqual(win['both_wins'], sum(outcomes))
                self.assertEqual(win['neither_wins'], len(outcomes) - sum(outcomes))
                self.assertEqual(win['exact_mcnemar_p'], 1.0)
                self.assertEqual(win['point'], 0)
                self.assertEqual(win['ci95'], [0, 0])

    def test_generic_game_metrics_have_direct_pairing_and_no_gate(self):
        from tests.unit.test_fresh_cohort import world
        worlds = [world(91_000_000_000 + i) for i in range(3)]
        other = deepcopy(worlds)
        for row in other:
            row['win'], row['floor'] = False, row['floor'] - 1
        result = controlled_comparison.game_metrics(worlds, other, [r['world_seed'] for r in worlds], resamples=32)
        self.assertEqual(result['win']['point'], 1)
        self.assertEqual(result['floor']['point'], 1)
        self.assertNotIn('gate', json.dumps(result))
        with self.assertRaises(ValueError):
            controlled_comparison.game_metrics(worlds, other[:-1], [r['world_seed'] for r in worlds], resamples=32)

    def test_games_validate_complete_cohorts_actual_policies_and_namespace(self):
        import tempfile
        from tests.unit.test_fresh_cohort import world, rewrite_world
        with tempfile.TemporaryDirectory() as directory, builder_stub() as stub:
            root = Path(directory)
            original = champion(root / 'champion.pt')
            champ_id = file_identity(original)
            policies, specs = {}, {}
            base = dict(protocol='fresh_start_v1', filter_ids=['ooc_potion_drink_discard_hidden_v1', 'legacy_ooc_dead_key_recall_v1'],
                public_context_id='ooc_potion_public_menu_inventory_v1', inference='single_observation_exact_argmax_actor_only_v1',
                champion=champ_id, ooc=None)
            for arm, feature in (('left', False), ('right', True)):
                identity = checkpoint(root / (arm + '.pt'), feature=feature)
                policies[arm] = dict(base, combat=identity)
                specs[arm] = dict(policy=policies[arm], replacements=dict(combat=dict(checkpoint=identity,
                    architecture='controlled', condition=stub.condition(torch.load(identity['path'], weights_only=False)))))
                complete_game_fixture(root / arm, [world(91_000_000_000 + i) for i in range(2)],
                                      policies[arm], b'never loaded native fixture')
            cohort, expectations = root / 'cohort.json', root / 'expected.json'
            atomic_json(cohort, [91_000_000_000, 91_000_000_001])
            expected = dict(schema='controlled_comparison_expectations_v1', version=1, mode='games',
                            cohort=file_identity(cohort), **specs)
            atomic_json(expectations, expected)
            compare = lambda: controlled_comparison.compare_games_runs(root / 'left', root / 'right', cohort, expectations)
            self.assertEqual(compare()['win']['point'], 0)
            with self.assertRaises(ValueError):
                controlled_comparison.compare_games_runs(root / 'right', root / 'left', cohort, expectations)
            atomic_json(cohort, [91_200_000_000])
            expected['cohort'] = file_identity(cohort)
            atomic_json(expectations, expected)
            with patch.object(controlled_comparison, '_games', side_effect=AssertionError('embargoed world read')):
                with self.assertRaisesRegex(ValueError, 'development evaluation namespace'):
                    compare()
            atomic_json(cohort, [91_000_000_000, 91_000_000_001])
            expected['cohort'] = file_identity(cohort)
            atomic_json(expectations, expected)
            rewrite_world(root / 'left', lambda r: r.update(cut=True))
            with self.assertRaises(ValueError):
                compare()


class ControlledCLITests(unittest.TestCase):
    def script(self, name):
        directory = Path(__file__).resolve().parents[2] / 'scripts' / 'rl'
        spec = importlib.util.spec_from_file_location(name, directory / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(directory), *sys.path]):
            spec.loader.exec_module(module)
        return module

    def test_eval_uses_protected_runner_and_validates_completed_cohort(self):
        module = self.script('controlled_eval')
        summary = dict(event='done', status='completed', cohort_complete=True, completed=1,
                       faults=0, cuts=0, censored=0, unstarted=0, wins=1)
        args = ['--mode', 'eval', '--checkpoint', 'champion', '--combat', 'head', '--out', 'out',
                '--world-start', '91000000000', '--n', '1']
        policy = object()
        with patch.object(module, 'policy_factory', return_value=policy), \
             patch.object(module, 'run_collection', return_value=summary) as run, \
             patch.object(module, 'validate_cohorts') as validate, redirect_stdout(io.StringIO()):
            module.main(args)
        self.assertIs(run.call_args.args[1], policy)
        validate.assert_called_once_with(['out'], world_start=91_000_000_000, n=1)
        with patch.object(module, 'policy_factory', side_effect=AssertionError('embargoed launch')):
            with self.assertRaises(ValueError):
                module.main([v if v != '91000000000' else '91200000000' for v in args])

    def test_new_dev_full_range_accepted_without_execution_and_confirmation_rejected(self):
        module = self.script('controlled_eval')
        args = ['--mode', 'eval', '--checkpoint', 'champion', '--combat', 'head', '--out', 'out',
                '--world-start', str(91_001_000_000), '--n', '4096']
        summary = dict(event='done', status='completed', cohort_complete=True, completed=4096,
                       faults=0, cuts=0, censored=0, unstarted=0, wins=0)
        with patch.object(module, 'policy_factory', return_value=object()) as policy, \
             patch.object(module, 'run_collection', return_value=summary) as run, \
             patch.object(module, 'validate_cohorts') as validate, redirect_stdout(io.StringIO()):
            module.main(args)
            self.assertEqual(run.call_args.args[0]['world_start'], 91_001_000_000)
            self.assertEqual(run.call_args.args[0]['n'], 4096)
            validate.assert_called_once_with(['out'], world_start=91_001_000_000, n=4096)
            policy.reset_mock()
            run.reset_mock()
            validate.reset_mock()
            with self.assertRaisesRegex(ValueError, 'embargoed'):
                module.main([str(91_200_000_000) if v == str(91_001_000_000) else v for v in args])
            policy.assert_not_called()
            run.assert_not_called()
            validate.assert_not_called()

    def test_new_bank_full_range_accepted_by_validation_without_execution(self):
        from sts_rl.fresh_study import validate_world_range
        module = self.script('fresh_collect')
        config = vars(module.parser().parse_args([
            '--mode', 'collect', '--checkpoint', 'champion', '--out', 'out', '--bank',
            '--world-start', str(92_100_000_000), '--n', '1024']))
        with patch.object(module, 'StudyPolicy', side_effect=AssertionError('policy load')), \
             patch.object(module, 'run_collection', side_effect=AssertionError('native execution')):
            module.validate_config(config)
            validate_world_range(config['world_start'], config['n'])
            self.assertTrue(config['bank'])
            self.assertEqual((config['world_start'], config['n']), (92_100_000_000, 1024))
            with self.assertRaisesRegex(ValueError, 'embargoed'):
                module.validate_config(dict(config, world_start=91_200_000_000))

    def test_scorer_dispatch_and_comparison_missing_input_fail_closed(self):
        self.assertIs(self.script('fresh_score').score_deferred, fresh_scoring.score_deferred)
        module = self.script('controlled_compare')
        with redirect_stdout(io.StringIO()), patch.object(sys, 'stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                module.main(['labels', '--left', 'left', '--right', 'right', '--expectations', 'expected', '--out', 'out'])
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
