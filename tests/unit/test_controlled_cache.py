"""Small synthetic serialization/contracts only; no native execution or fitting."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields, is_dataclass
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from sts_policy import controlled_encoding as ce
from sts_rl import controlled_cache as cc, fresh_training as ft
from sts_rl.recording import file_identity
from tests.unit.test_fresh_training import fixture


def script(name):
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'rl' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name + '_cache_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_json(path, value):
    path.chmod(0o644)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')
    path.chmod(0o444)


class ControlledCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.fixture_tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture_tmp.name)
        cls.data, cls.source = fixture(cls.root)
        cls.manifest = ft.freeze_manifest([cls.data])
        cls.manifest_path = cls.root / 'manifest.json'
        ft.atomic_json(cls.manifest_path, cls.manifest)
        cls.manifest_id = file_identity(cls.manifest_path)
        cls.dataset = ft.preencode([cls.data], cls.manifest, 'combat', 'controlled', 'full',
                                  splits=('FIT', 'VALIDATION'))
        cls.counts = {s: len(v) for s, v in cls.dataset.items()}
        # Launch fixture contains inert native bytes; no native import or recording call.
        snapshot = cls.root / 'cache-launch'
        snapshot.mkdir()
        native = cls.manifest['public_surface']['provenance']['native_provenance']
        (snapshot / native['module_file']).write_bytes((cls.root / native['module_file']).read_bytes())
        with zipfile.ZipFile(snapshot / 'source.zip', 'x') as archive:
            for name, item in cc.logic_sources().items():
                archive.write(item['path'], name)
        ft.atomic_json(snapshot / 'launch.json', dict(parent_checkpoint=None, native=native,
            effective_config=dict(stage='controlled_cache_creation', manifest=cls.manifest_id,
                                  encoding=cc.ENCODING, data_identity=cls.manifest['data_identity'],
                                  logic_sources=cc.logic_sources()),
            data_inputs=[cls.manifest_id, *cls.manifest['source_files']],
            source_archive=file_identity(snapshot / 'source.zip')))
        cls.launch = file_identity(snapshot / 'launch.json')

    @classmethod
    def tearDownClass(cls):
        cls.fixture_tmp.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def write(self, name='cache', dataset=None):
        out = self.out / name
        out.mkdir()
        cc._write_cache(out, self.dataset if dataset is None else dataset, self.manifest,
                        self.manifest_id, self.launch, counts=self.counts)
        return out / 'receipt.json'

    def prepare(self, path):
        return cc._prepare_cache(path, self.manifest, self.manifest_path, [self.data], counts=self.counts)

    def load(self, path):
        return cc._load_cache(self.prepare(path), self.manifest, counts=self.counts)[0]

    def mutate_dataset(self, path, change):
        receipt = json.loads(path.read_text())
        data_path = Path(receipt['dataset']['path'])
        data = deepcopy(self.dataset)
        change(data)
        data_path.chmod(0o644)
        with data_path.open('wb') as stream:
            pickle.dump(data, stream, protocol=5)
        data_path.chmod(0o444)
        receipt['dataset'] = file_identity(data_path)
        replace_json(path, receipt)

    def exact(self, left, right):
        self.assertIs(type(left), type(right))
        if isinstance(left, np.ndarray):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.strides, right.strides)
            self.assertEqual(left.tobytes(), right.tobytes())
        elif is_dataclass(left):
            for field in fields(left):
                self.exact(getattr(left, field.name), getattr(right, field.name))
        elif isinstance(left, dict):
            self.assertEqual(list(left), list(right))
            for key in left:
                self.exact(left[key], right[key])
        elif isinstance(left, list):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.exact(a, b)
        else:
            self.assertEqual(left, right)

    def test_byte_exact_valid_load_no_raw_reader_and_rng_unchanged(self):
        path = self.write()
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
        with patch.object(ft, 'iter_rows', side_effect=AssertionError('raw reader on load')), \
             patch.object(ce, 'encode_observation', side_effect=AssertionError('re-encoding on load')):
            loaded = self.load(path)
        self.exact(self.dataset, loaded)
        self.assertEqual(random.getstate(), python_rng)
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
        self.assertEqual(actual_numpy[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        if mps_rng is not None:
            self.assertTrue(torch.equal(mps_rng, torch.mps.get_rng_state()))
        self.assertEqual(list(loaded), ['FIT', 'VALIDATION'])
        self.assertTrue(all(x.meta['split'] != 'TEST' for rows in loaded.values() for x in rows))
        receipt = json.loads(path.read_text())
        for item in [path, Path(receipt['dataset']['path'])]:
            self.assertEqual(item.stat().st_mode & 0o222, 0)
        self.assertEqual(receipt['row_contract'], 'synthetic_fixture_only')

    def test_pooled_diagnostics_survive_serialization_and_validate_menu_alignment(self):
        data,_=fixture(self.out/'values',record_search=True)
        manifest=ft.freeze_manifest([data],target_mode='pooled_values',value_scale=32.)
        dataset=ft.preencode([data],manifest,'combat','controlled',splits=('FIT','VALIDATION'))
        loaded=pickle.loads(pickle.dumps(dataset,protocol=5))
        self.exact(dataset,loaded)
        cc.validate_dataset(loaded,manifest,counts=self.counts)
        for rows in loaded.values():
            for example in rows:
                self.assertEqual(example.pooled_position,7)
                np.testing.assert_array_equal(example.value_regrets,[1.]*7+[0.])
        example=loaded['FIT'][0]
        for regrets,position in ((example.value_regrets.astype(np.float32),7),
                                 (example.value_regrets,0),(None,7)):
            invalid=deepcopy(example)
            invalid.value_regrets,invalid.pooled_position=regrets,position
            with self.assertRaisesRegex(ValueError,'value_regrets|pooled'):
                cc._example(invalid)

    def test_identical_collation_minibatches_and_initial_weights(self):
        path = self.write()
        # Recompute only the small synthetic fixture, never a study corpus.
        torch.manual_seed(1919)
        uncached = ft.preencode([self.data], self.manifest, 'combat', 'controlled', 'full', splits=('FIT', 'VALIDATION'))
        original = ft.make_model('controlled', 'combat', width=16, layers=1)
        torch.manual_seed(1919)
        cached = self.load(path)
        after_load = ft.make_model('controlled', 'combat', width=16, layers=1)
        self.assertTrue(all(torch.equal(v, after_load.state_dict()[k]) for k, v in original.state_dict().items()))
        for flag in (False, True):
            for seed in (1919, 2020):
                order1 = ft.minibatch_order(uncached['FIT'], 'controlled', 2, random.Random(seed))
                order2 = ft.minibatch_order(cached['FIT'], 'controlled', 2, random.Random(seed))
                self.assertEqual(order1, order2)
                for indices in order1:
                    a = ft.collate([uncached['FIT'][i] for i in indices], 'controlled', 'cpu')
                    b = ft.collate([cached['FIT'][i] for i in indices], 'controlled', 'cpu')
                    self.assertEqual(set(a[0]), set(b[0]))
                    self.assertTrue(all(torch.equal(a[0][k], b[0][k]) for k in a[0]))
                    self.assertTrue(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
            # ON and OFF consume the same serialized legacy arrays.
            model = ft.make_model('controlled', 'combat', width=16, layers=1, legacy_action_features=flag)
            self.assertIs(model.config.legacy_action_features, flag)

    def test_fixed_full_counts_and_fixture_cache_never_production_loadable(self):
        self.assertEqual(cc.FULL_COUNTS, {'FIT': 113042, 'VALIDATION': 17989})
        with self.assertRaisesRegex(ValueError, 'row counts'):
            cc.manifest_rows(self.manifest)
        path = self.write()
        with self.assertRaises(ValueError):
            cc.prepare_cache(path, self.manifest, self.manifest_path, [self.data])
        with self.assertRaisesRegex(ValueError, 'fixture'):
            cc.load_cache(self.prepare(path), self.manifest)

    def test_invalid_arrays_targets_classes_indices_and_order_rejected(self):
        def changed_meta(data):
            data['FIT'][0].meta['world_seed'] += 1
        def bad_source(data):
            data['FIT'][0].encoded.entity.action_source[0] = 1000000
        def nan(data):
            data['FIT'][0].encoded.legacy_action_features[0, 0] = np.nan
        def wrong_dtype(data):
            object.__setattr__(data['FIT'][0].encoded.entity, 'entity_ids', np.zeros(1, dtype=np.int32))
        def wrong_shape(data):
            object.__setattr__(data['FIT'][0].encoded, 'legacy_action_features', np.zeros((1, 1), np.float32))
        def index_permutation(data):
            indices = data['FIT'][0].encoded.action_indices
            indices[:] = indices[::-1].copy()
        def duplicate_indices(data):
            data['FIT'][0].encoded.action_indices[:] = 1
        def target(data):
            data['FIT'][0].target[:] = 1 / len(data['FIT'][0].target)
        def wrong_class(data):
            data['FIT'][0] = dict(vars(data['FIT'][0]))
        def test_row(data):
            data['FIT'][0].meta['split'] = 'TEST'
        def wrong_phase(data):
            data['FIT'][0].meta['phase'] = 'ooc'
        def test_split(data):
            data['TEST'] = data['FIT'][:1]
        def source_mask(data):
            data['FIT'][0].encoded.entity.action_source_mask[0, 0] ^= True
        changes = [changed_meta, bad_source, nan, wrong_dtype, wrong_shape, index_permutation,
            duplicate_indices, target, wrong_class, test_row, wrong_phase, test_split, source_mask,
            lambda d: d['FIT'].reverse(), lambda d: d['FIT'].pop(),
            lambda d: d['FIT'].append(d['FIT'][0]), lambda d: d['FIT'].__setitem__(1, d['FIT'][0]),
            lambda d: d['FIT'].clear()]
        for i, change in enumerate(changes):
            with self.subTest(change=change.__name__, i=i):
                path = self.write(str(i))
                self.mutate_dataset(path, change)
                with self.assertRaises(ValueError):
                    self.load(path)

    def test_wrong_receipt_split_counts_rows_hashes_and_logic_fail_before_unpickle(self):
        changes = [
            lambda r: r['encoding'].__setitem__('splits', ['FIT', 'VALIDATION', 'TEST']),
            lambda r: r['encoding'].__setitem__('architecture', 'old'),
            lambda r: r['row_counts'].__setitem__('FIT', 4),
            lambda r: r['row_ids']['FIT'].__setitem__(0, 'missing'),
            lambda r: r['row_ids']['FIT'].__setitem__(0, r['row_ids']['FIT'][1]),
            lambda r: r['row_ids'].__setitem__('TEST', []),
            lambda r: r['source_files'][0].__setitem__('sha256', '1' * 64),
            lambda r: r['manifest'].__setitem__('sha256', '2' * 64),
            lambda r: r['dataset'].__setitem__('sha256', '3' * 64),
            lambda r: r.__setitem__('data_identity', '4' * 64),
            lambda r: r['runtime'].__setitem__('numpy', 'stale')]
        for name in cc.logic_sources():
            changes.append(lambda r, name=name: r['logic_sources'][name].__setitem__('sha256', '0' * 64))
        for i, change in enumerate(changes):
            with self.subTest(i=i):
                path = self.write(str(i))
                receipt = json.loads(path.read_text())
                change(receipt)
                replace_json(path, receipt)
                with patch.object(pickle, 'load', side_effect=AssertionError('unpickle before hashes')):
                    with self.assertRaises(ValueError):
                        self.load(path)

    def test_missing_truncated_and_tampered_file_fails_before_deserialization(self):
        for kind in ('missing', 'tampered', 'truncated'):
            path = self.write(kind)
            receipt = json.loads(path.read_text())
            target = Path(receipt['dataset']['path'])
            target.chmod(0o644)
            if kind == 'missing':
                target.unlink()
            elif kind == 'tampered':
                with target.open('ab') as stream:
                    stream.write(b'bad')
            else:
                target.write_bytes(b'\x80\x05')
            with patch.object(pickle, 'load', side_effect=AssertionError('unpickle before hashes')):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load(path)

    def test_invalid_receipt_json_fails_before_deserialization(self):
        for i, text in enumerate(('{"schema":"one","schema":"two"}', '{"timings":NaN}', '{broken')):
            path = self.write(str(i))
            path.chmod(0o644)
            path.write_text(text)
            path.chmod(0o444)
            with patch.object(pickle, 'load', side_effect=AssertionError('invalid receipt deserialized')):
                with self.assertRaises(ValueError):
                    self.load(path)

    def test_after_load_and_before_train_rechecks_detect_changes(self):
        path = self.write()
        cache = self.prepare(path)
        real_load = pickle.load
        def changed_during_load(*args, **kwargs):
            value = real_load(*args, **kwargs)
            changed = dict(cache.receipt, schema='tampered')
            replace_json(path, changed)
            return value
        with patch.object(pickle, 'load', side_effect=changed_during_load):
            with self.assertRaisesRegex(ValueError, 'file identity'):
                cc._load_cache(cache, self.manifest, counts=self.counts)
        path = self.write('before-train')
        cache = self.prepare(path)
        cc._load_cache(cache, self.manifest, counts=self.counts)
        replace_json(path, dict(cache.receipt, schema='tampered'))
        with self.assertRaisesRegex(ValueError, 'file identity'):
            cache.recheck()

    def test_source_and_original_manifest_changes_rejected(self):
        path = self.write()
        # Change only synthetic copies of source bytes, restoring after each check.
        for target in (Path(self.manifest['source_files'][0]['path']), self.manifest_path):
            original = target.read_bytes()
            try:
                target.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    self.prepare(path)
            finally:
                target.write_bytes(original)
        cache = self.prepare(path)
        target = Path(self.manifest['source_files'][0]['path'])
        original = target.read_bytes()
        try:
            target.write_bytes(original + b' ')
            with self.assertRaises(ValueError):
                cache.recheck()
        finally:
            target.write_bytes(original)

    def test_exclusive_and_interrupted_writes_have_no_success_receipt(self):
        path = self.write()
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            cc._write_cache(path.parent, self.dataset, self.manifest, self.manifest_id, self.launch, counts=self.counts)
        self.assertEqual(before, path.read_bytes())
        out = self.out / 'interrupted'
        out.mkdir()
        def interrupted(*args, **kwargs):
            args[1].write(b'partial')
            raise ft.Interrupted('STOP fixture')
        with patch.object(pickle, 'dump', side_effect=interrupted):
            with self.assertRaises(ft.Interrupted):
                cc._write_cache(out, self.dataset, self.manifest, self.manifest_id, self.launch, counts=self.counts)
        self.assertFalse((out / 'receipt.json').exists())
        self.assertFalse((out / 'dataset.pkl').exists())
        with self.assertRaises(FileExistsError):
            cc._write_cache(out, self.dataset, self.manifest, self.manifest_id, self.launch, counts=self.counts)
        # A stop after dataset publication still cannot publish a success receipt.
        out = self.out / 'interrupted-final'
        out.mkdir()
        publish = cc._exclusive_publish
        def stop_before_receipt(temporary, final):
            if final.name == 'receipt.json':
                raise ft.Interrupted('deadline fixture')
            publish(temporary, final)
        with patch.object(cc, '_exclusive_publish', side_effect=stop_before_receipt):
            with self.assertRaises(ft.Interrupted):
                cc._write_cache(out, self.dataset, self.manifest, self.manifest_id, self.launch, counts=self.counts)
        self.assertTrue((out / 'dataset.pkl').exists())
        self.assertFalse((out / 'receipt.json').exists())
        # Atomic linking never overwrites a preexisting historical destination.
        first, second = self.out / 'first', self.out / 'second'
        first.write_bytes(b'new')
        second.write_bytes(b'historical')
        with self.assertRaises(FileExistsError):
            publish(first, second)
        self.assertEqual(second.read_bytes(), b'historical')

    def test_stream_controls_checked_and_errors_propagate(self):
        class Stopped:
            def check(self):
                raise ft.Interrupted('disk_below_6_GiB fixture')
        with self.assertRaises(ft.Interrupted):
            cc.identity(self.manifest_path, Stopped())
        # Streaming callbacks propagate the existing TrainingControl exceptions.
        from io import BytesIO
        with self.assertRaises(ft.Interrupted):
            cc._ControlledStream(BytesIO(), Stopped()).write(b'x' * (8 << 20))

    def test_creator_calls_only_existing_validation_and_exact_fit_validation_preencode(self):
        cfg = dict(data=[str(self.data)], manifest=str(self.manifest_path), out=str(self.out / 'new'),
                   deadline_utc='2099-01-01T00:00:00Z')
        original_control = ft.TrainingControl
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, STS_LIGHTSPEED_BUILD='/synthetic-unused'))
            stack.enter_context(patch.object(ft, 'TrainingControl', side_effect=lambda out, deadline:
                original_control(out, deadline, ac=lambda: True, caffeinate=False)))
            cohort = stack.enter_context(patch.object(cc, 'manifest_rows'))
            freeze = stack.enter_context(patch.object(ft, 'freeze_manifest', return_value=self.manifest))
            begin = stack.enter_context(patch.object(ft, 'begin_run', return_value=self.launch))
            # Full source/native evidence has separate synthetic tests; never execute native recording here.
            stack.enter_context(patch.object(cc, '_launch_evidence', return_value=[]))
            encode = stack.enter_context(patch.object(ft, 'preencode', return_value=self.dataset))
            write = stack.enter_context(patch.object(cc, '_write_cache', side_effect=ft.Interrupted('STOP fixture')))
            stack.enter_context(patch.object(ft, 'make_model', side_effect=AssertionError('cache created model')))
            stack.enter_context(patch.object(ft, 'fit', side_effect=AssertionError('cache fitted model')))
            with self.assertRaisesRegex(ft.Interrupted, 'STOP'):
                cc.build_cache(cfg)
        cohort.assert_called_once()  # Production calls the default fixed row contract, without an override.
        freeze.assert_called_once()
        self.assertEqual(encode.call_args.args, (cfg['data'], self.manifest, 'combat', 'controlled', 'full'))
        self.assertEqual(encode.call_args.kwargs['splits'], ('FIT', 'VALIDATION'))
        self.assertIn('control', encode.call_args.kwargs)
        self.assertIsNone(begin.call_args.kwargs['checkpoint'])
        self.assertIn(str(self.manifest_path), begin.call_args.kwargs['inputs'])
        self.assertEqual(write.call_args.kwargs['logic'], cc.logic_sources())
        self.assertFalse((Path(cfg['out']) / 'receipt.json').exists())
        with self.assertRaises(FileExistsError):
            with patch.dict(os.environ, STS_LIGHTSPEED_BUILD='/synthetic-unused'):
                cc.build_cache(cfg)

    def test_creator_passes_mixed_sampling_and_target_options_from_manifest(self):
        first,_=fixture(self.out/'first',record_search=True,config={'teacher_probabilities':{'BOSS2':1.0,'MONSTER1':.04}})
        second,_=fixture(self.out/'second',record_search=True,config={'world_start':92_000_001_000,
            'teacher_probabilities':{'BOSS2':1.0,'MONSTER1':0.0}})
        sources=[str(first),str(second)]
        original_control=ft.TrainingControl
        for options in ({'relabel_vote_rule':'pooled_mean'}, {'target_mode':'pooled_values','value_scale':32.},
                        {'target_mode':'pooled_values_or_votes','value_scale':32.}):
            with self.subTest(options=options),ExitStack() as stack:
                manifest=ft.freeze_manifest(sources,mixed_sources=True,mixed_sampling=True,**options)
                path=self.out/'mixed_manifest.json'
                ft.atomic_json(path,manifest)
                cfg=dict(data=sources,manifest=str(path),out=str(self.out/('mixed_cache_'+manifest.get('target_mode','votes'))),
                         deadline_utc=None)
                stack.enter_context(patch.dict(os.environ,STS_LIGHTSPEED_BUILD='/synthetic-unused'))
                stack.enter_context(patch.object(ft,'TrainingControl',side_effect=lambda out,deadline:
                    original_control(out,deadline,ac=lambda:True,caffeinate=False)))
                stack.enter_context(patch.object(cc,'manifest_rows'))
                stack.enter_context(patch.object(ft,'begin_run',side_effect=ft.Interrupted('STOP after manifest validation')))
                freeze=stack.enter_context(patch.object(ft,'freeze_manifest',wraps=ft.freeze_manifest))
                with self.assertRaisesRegex(ft.Interrupted,'STOP after manifest validation'):
                    cc.build_cache(cfg)
                self.assertIs(freeze.call_args.kwargs['mixed_sources'],True)
                self.assertIs(freeze.call_args.kwargs['mixed_sampling'],True)
                self.assertEqual(freeze.call_args.kwargs['relabel_vote_rule'],options.get('relabel_vote_rule'))
                self.assertEqual(freeze.call_args.kwargs['target_mode'],options.get('target_mode','votes'))
                self.assertEqual(freeze.call_args.kwargs['value_scale'],options.get('value_scale',100.))

    def test_original_cli_defaults_and_cache_restrictions(self):
        train, creator = script('fresh_train'), script('fresh_cache')
        args = ['--data', 'fixture', '--out', 'out', '--architecture', 'old', '--phase', 'combat',
                '--deadline-utc', '2099-01-01T00:00:00Z']
        config = vars(train.parser().parse_args(args))
        self.assertNotIn('preencoded_cache', config)
        self.assertNotIn('legacy_action_features', config)
        self.assertEqual(config, dict(data=['fixture'], manifest=None, probe=None, out='out', architecture='old',
            phase='combat', fit_size='full', lr=.0003, old_size='large', old_context='none', width=128, layers=2,
            heads=4, steps=12000, min_steps=2000, batch=None, defer_test=False, eval_every=100, patience=6,
            seed=1919, device='cpu', deadline_utc='2099-01-01T00:00:00Z'))
        new_args = [x if x != 'old' else 'new' for x in args]
        self.assertEqual(vars(train.parser().parse_args(new_args)), dict(config, architecture='new'))
        good = dict(config, architecture='controlled', defer_test=True, manifest='manifest.json',
                    preencoded_cache='receipt.json', probe=['fixture'])
        ft.validate_training_config(good)
        for update in ({'architecture': 'old'}, {'architecture': 'new'}, {'phase': 'ooc'},
                       {'fit_size': '5000'}, {'fit_size': '20000'}, {'defer_test': False},
                       {'manifest': None}, {'preencoded_cache': None}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'preencoded-cache'):
                ft.validate_training_config(dict(good, **update))
        self.assertEqual(set(vars(creator.parser().parse_args(['--data', 'fixture', '--manifest', 'manifest.json',
            '--out', 'out', '--deadline-utc', '2099-01-01T00:00:00Z']))),
            {'data', 'manifest', 'out', 'deadline_utc'})

    def test_training_hook_preserves_raw_checks_provenance_and_rechecks_before_fit(self):
        path = self.write()
        cache = self.prepare(path)
        cfg = dict(data=[str(self.data)], manifest=str(self.manifest_path), out=str(self.out / 'run'),
            architecture='controlled', phase='combat', fit_size='full', width=16, layers=1, heads=4, steps=2000,
            min_steps=2000, batch=2, eval_every=100, patience=6, lr=.001, seed=1919, device='cpu',
            deadline_utc=None, defer_test=True, probe=['not-read-fixture'], preencoded_cache=str(path))
        events = []
        original_control = ft.TrainingControl
        # All heavy operational boundaries are mocks. No fit, probes or native code executes.
        def begin(out, effective, **kwargs):
            events.append('launch')
            self.assertIn(path.resolve(), [Path(p).resolve() for p in kwargs['inputs']])
            self.assertIn(Path(cache.receipt['dataset']['path']), list(map(Path, kwargs['inputs'])))
            launch_path = self.out / 'training-launch.json'
            ft.atomic_json(launch_path, dict(native=self.manifest['public_surface']['provenance']['native_provenance']))
            return file_identity(launch_path)
        def fake_load(handle, manifest, control):
            events.append('load')
            return cc._load_cache(handle, manifest, control, counts=self.counts)
        def fake_fit(model, dataset, effective, out, manifest, receipt, control, **kwargs):
            events.append('fit')
            self.exact(dataset, self.dataset)
            self.assertEqual(receipt['label_source_files'], self.manifest['source_files'])
            self.assertEqual(effective['data_identity'], self.manifest['data_identity'])
            self.assertEqual(receipt['preencoded_cache'], effective['preencoded_cache_provenance'])
            return dict(status='completed', test_evaluated=False)
        real_recheck = cache.recheck
        def recheck(control=None):
            events.append('recheck')
            real_recheck(control)
        with ExitStack() as stack:
            freeze = stack.enter_context(patch.object(ft, 'freeze_manifest', return_value=self.manifest))
            stack.enter_context(patch.object(ft, 'TrainingControl', side_effect=lambda out, deadline:
                original_control(out, deadline, ac=lambda: True, caffeinate=False)))
            stack.enter_context(patch('sts_rl.fresh_probes.select_probes', return_value=(
                {'selection_identity': 'synthetic', 'source_files': []}, [])))
            stack.enter_context(patch('sts_rl.fresh_probes.encode_probes', return_value=[]))
            stack.enter_context(patch.object(ft, 'shard_files', side_effect=lambda sources:
                [] if sources == ['not-read-fixture'] else [Path(f['path']) for f in self.manifest['source_files']] ))
            stack.enter_context(patch.object(ft, 'begin_run', side_effect=begin))
            stack.enter_context(patch.object(cc, 'prepare_cache', return_value=cache))
            stack.enter_context(patch.object(cc, 'load_cache', side_effect=fake_load))
            stack.enter_context(patch.object(cache, 'recheck', side_effect=recheck))
            stack.enter_context(patch.object(ft, 'preencode', side_effect=AssertionError('cached run encoded')))
            stack.enter_context(patch.object(ft, 'fit', side_effect=fake_fit))
            result = ft.run_training(cfg)
        freeze.assert_called_once()
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(events[-2:], ['recheck', 'fit'])
        usage = json.loads((Path(cfg['out']) / 'cache_usage.json').read_text())
        self.assertEqual(usage['receipt'], file_identity(path))


if __name__ == '__main__':
    unittest.main()
