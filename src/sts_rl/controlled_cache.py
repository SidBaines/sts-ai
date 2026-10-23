"""Optional, trusted local FIT/VALIDATION preencoding; never a label authority.

Only the creator calls the existing preencode. Readers hash before unpickling,
check the canonical manifest and array ABI, and do not read raw observations.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import zipfile

import numpy as np
import torch

from sts_policy import controlled_encoding as ce, entity_encoding as ee
from sts_rl import fresh_training as ft
from sts_rl.fresh_study import read_json

SCHEMA = 'controlled_preencoded_cache_v2'
ENCODING = dict(architecture='controlled', phase='combat', fit_size='full', splits=['FIT', 'VALIDATION'])
FULL_COUNTS = {'FIT': 113_042, 'VALIDATION': 17_989}
_ROOT = Path(__file__).resolve().parents[2]
_LOGIC = (
    'src/sts_rl/controlled_cache.py', 'src/sts_rl/fresh_training.py',
    'src/sts_rl/fresh_study.py', 'src/sts_rl/recording.py', 'src/sts_rl/teacher_votes.py',
    'src/sts_policy/controlled_encoding.py', 'src/sts_policy/entity_encoding.py',
    'src/sts_policy/encoding.py', 'src/sts_policy/curriculum.py',
    'src/sts_policy/entity_vocabulary.json', 'src/sts_env/_backend/glossary.py',
    'src/sts_env/_backend/glossary_v4.py', 'src/sts_env/_backend/glossary_v5.py',
    'scripts/rl/fresh_cache.py', 'scripts/rl/fresh_train.py',
)


def _require(condition, message):
    if not condition:
        raise ValueError('Controlled cache: ' + message)


def validate_optin(config):
    if 'preencoded_cache' not in config:
        return
    _require(config.get('architecture') == 'controlled' and config.get('phase') == 'combat'
             and config.get('fit_size', 'full') == 'full' and config.get('defer_test') is True,
             '--preencoded-cache requires controlled/combat/full/--defer-test')
    _require(isinstance(config['preencoded_cache'], str) and bool(config['preencoded_cache'])
             and bool(config.get('manifest')), '--preencoded-cache requires a receipt and --manifest')


def _check(control):
    if control is not None:
        control.check()


def identity(path, control=None):
    """Same identity as recording.file_identity, with bounded control boundaries."""
    path = Path(path)
    sha, size = hashlib.sha256(), 0
    _check(control)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            sha.update(chunk)
            size += len(chunk)
            _check(control)
    _require(size == path.stat().st_size, 'file changed while hashing')
    return dict(path=str(path.resolve()), bytes=size, sha256=sha.hexdigest())


def _verify(item, control=None):
    _require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'invalid file identity')
    _require(type(item['path']) is str and type(item['bytes']) is int and item['bytes'] >= 0
             and type(item['sha256']) is str and len(item['sha256']) == 64
             and all(c in '0123456789abcdef' for c in item['sha256']), 'invalid file identity fields')
    _require(identity(item['path'], control) == item, 'file identity mismatch: ' + str(item['path']))


def logic_sources(control=None):
    return {name: identity(_ROOT / name, control) for name in _LOGIC}


def runtime():
    return dict(python=sys.version, platform=platform.platform(), byteorder=sys.byteorder,
                numpy=np.__version__, torch=torch.__version__)


def _row_contract(counts):
    return 'approved_full_combat_v1' if counts == FULL_COUNTS else 'synthetic_fixture_only'


def manifest_rows(manifest, counts=FULL_COUNTS):
    """Pure cohort check. Explicit small counts are for synthetic unit fixtures only."""
    _require(manifest.get('schema') == 'fresh_data_manifest_v2' and manifest.get('filters') == ft.FILTERS,
             'manifest contract mismatch')
    wanted = {s: {} for s in ENCODING['splits']}
    seen = set()
    for meta in manifest['rows']:
        rid = meta['source_row_id']
        _require(rid not in seen, 'duplicate manifest row ID')
        seen.add(rid)
        if meta['phase'] == 'combat' and meta['has_target'] and meta['split'] in wanted:
            _require(meta['abstention'] is False and meta['target_sha256'] is not None,
                     'invalid target metadata')
            wanted[meta['split']][rid] = meta
    _require({s: len(v) for s, v in wanted.items()} == counts, 'wrong full FIT/VALIDATION row counts')
    fit = manifest['subsets']['combat']['full']
    _require(len(fit) == counts['FIT'] and set(fit) == set(wanted['FIT']), 'wrong full FIT subset IDs')
    return wanted


def _array(value, dtype, shape, name):
    _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype) and value.shape == shape,
             'array dtype/shape mismatch: ' + name)
    if value.dtype.kind == 'f':
        _require(bool(np.isfinite(value).all()), 'nonfinite array: ' + name)
    return value.nbytes


def _range(array, low, high, name):
    _require(not array.size or (int(array.min()) >= low and int(array.max()) < high),
             'index outside encoder contract: ' + name)


def _example(example):
    """Validate the existing unpadded encoder ABI, without encoding observations."""
    _require(type(example) is ft.Example and set(vars(example)) == {f.name for f in fields(ft.Example)},
             'Example class/schema mismatch')
    x = example.encoded
    _require(type(x) is ce.EncodedControlledObservation and
             set(vars(x)) == {'entity', 'legacy_action_features'}, 'controlled class/schema mismatch')
    e = x.entity
    _require(type(e) is ee.EncodedEntityObservation and
             set(vars(e)) == {f.name for f in fields(ee.EncodedEntityObservation)}, 'entity class/schema mismatch')
    _require(all(type(getattr(e, n)) is np.ndarray and getattr(e, n).ndim == 1 for n in
                 ('entity_ids', 'action_indices', 'power_ids', 'recent_ids')), 'invalid array axes')
    n, a, p, r = map(len, (e.entity_ids, e.action_indices, e.power_ids, e.recent_ids))
    _require(n > 0 and a > 0, 'empty entities/menu')
    shapes = dict(entity_ids=('int64', (n,)), entity_types=('int64', (n,)),
        entity_aux=('int64', (n, 3)), entity_numeric=('float32', (n, ee.NUMERIC_DIM)), entity_mask=('bool', (n,)),
        power_owner=('int64', (p,)), power_ids=('int64', (p,)), power_amounts=('float32', (p,)),
        power_amount_available=('bool', (p,)), power_mask=('bool', (p,)),
        recent_ids=('int64', (r,)), recent_types=('int64', (r,)),
        recent_numeric=('float32', (r, ee.NUMERIC_DIM)), recent_mask=('bool', (r,)),
        action_ids=('int64', (a,)), action_types=('int64', (a,)), action_aux=('int64', (a, 3)),
        action_numeric=('float32', (a, ee.NUMERIC_DIM)), action_text=('float32', (a, ee.TEXT_DIM)),
        action_source=('int64', (a,)), action_target=('int64', (a,)), action_mask=('bool', (a,)),
        action_indices=('int64', (a,)))
    size = sum(_array(getattr(e, k), dtype, shape, k) for k, (dtype, shape) in shapes.items())
    _require(type(e.action_source_members) is np.ndarray and e.action_source_members.ndim == 2
             and e.action_source_members.shape[0] == a and e.action_source_members.shape[1] >= 1,
             'source membership axes')
    size += _array(e.action_source_members, 'int64', e.action_source_members.shape, 'action_source_members')
    size += _array(e.action_source_mask, 'bool', e.action_source_members.shape, 'action_source_mask')
    _require(np.array_equal(e.action_source_mask, e.action_source_members >= 0), 'source mask alignment')
    for prefix in ('entity', 'power', 'recent', 'action'):
        _require(bool(getattr(e, prefix + '_mask').all()), 'unexpected padding in cached example')
    for name in ('entity_ids', 'entity_aux', 'power_ids', 'recent_ids', 'action_ids', 'action_aux'):
        _range(getattr(e, name), 0, ee.VOCAB_SIZE, name)
    for name in ('entity_types', 'action_types', 'recent_types'):
        _range(getattr(e, name), 1, 1 + max((ee.ENTITY_TYPES if name == 'entity_types' else ee.ACTION_TYPES).values()), name)
    _range(e.power_owner, 0, n, 'power_owner')
    for name in ('action_source', 'action_target', 'action_source_members'):
        _range(getattr(e, name), -1, n, name)
    _require(bool((e.action_indices >= 0).all()) and len(set(e.action_indices.tolist())) == a,
             'duplicate/negative action indices')
    _require(type(e.coverage) is dict, 'coverage class mismatch')
    # JSON finiteness also checks all nested coverage counters; nothing is rewritten.
    ft.digest(e.coverage)
    _require(e.coverage.get('entity_count') == n and e.coverage.get('action_count') == a,
             'coverage/menu alignment')
    size += _array(x.legacy_action_features, 'float32', (a, ce.LEGACY_ACTION_DIM), 'legacy_action_features')
    size += _array(example.target, 'float32', (a,), 'target')
    _require(bool((example.target >= 0).all()) and abs(float(example.target.sum()) - 1.) <= 1e-6,
             'invalid target mass')
    _require(type(example.meta) is dict and example.meta.get('choice') == (a > 1)
             and example.meta.get('confident') == (float(example.target.max()) >= .75), 'target/menu metadata alignment')
    _require(ft.digest(example.target.tolist()) == example.meta.get('target_sha256'), 'canonical target mismatch')
    if example.value_regrets is None:
        _require(example.pooled_position is None, 'missing pooled value regrets')
    else:
        size += _array(example.value_regrets, 'float64', (a,), 'value_regrets')
        _require(type(example.pooled_position) is int and 0 <= example.pooled_position < a
                 and example.value_regrets[example.pooled_position] == 0., 'pooled winner/menu alignment')
    return size


def validate_dataset(dataset, manifest, *, counts=FULL_COUNTS, row_ids=None, alignment=None, control=None):
    """Pure validation; counts overrides are only used by explicit synthetic fixtures."""
    wanted = manifest_rows(manifest, counts)
    _require(type(dataset) is dict and list(dataset) == ENCODING['splits'], 'FIT/VALIDATION only, in original split order')
    orders, alignments, size = {}, {}, 0
    for split in ENCODING['splits']:
        rows = dataset[split]
        _require(type(rows) is list and len(rows) == counts[split], 'wrong dataset row counts')
        order, sha = [], hashlib.sha256()
        for i, x in enumerate(rows):
            if i % 128 == 0:
                _check(control)
            size += _example(x)
            rid = x.meta['source_row_id']
            _require(rid in wanted[split] and ft.digest(x.meta) == ft.digest(wanted[split].pop(rid)),
                     'extra/duplicate/TEST/wrong phase/changed canonical metadata')
            order.append(rid)
            sha.update(rid.encode('ascii'))
            sha.update(x.encoded.action_indices.tobytes())
            sha.update(x.target.tobytes())
        _require(not wanted[split], 'missing canonical rows')
        orders[split], alignments[split] = order, sha.hexdigest()
    if row_ids is not None:
        _require(orders == row_ids, 'original row ordering mismatch')
    if alignment is not None:
        _require(alignments == alignment, 'original action/target index alignment mismatch')
    return dict(row_ids=orders, alignment_sha256=alignments, array_bytes=size)


@dataclass
class CacheInput:
    receipt: dict
    receipt_id: dict
    identities: list

    @property
    def provenance(self):
        return dict(schema=SCHEMA, receipt=self.receipt_id, dataset=self.receipt['dataset'],
                    manifest=self.receipt['manifest'], data_identity=self.receipt['data_identity'])

    @property
    def input_paths(self):
        return [self.receipt_id['path'], self.receipt['dataset']['path'], self.receipt['manifest']['path']]

    def recheck(self, control=None):
        for item in self.identities:
            _verify(item, control)


def _launch_evidence(launch_id, manifest, manifest_id, logic, control=None):
    _verify(launch_id, control)
    with Path(launch_id['path']).open() as stream:
        launch = json.load(stream)
    effective = launch['effective_config']
    _require(launch.get('parent_checkpoint') is None and effective.get('stage') == 'controlled_cache_creation'
             and effective.get('manifest') == manifest_id and effective.get('encoding') == ENCODING
             and effective.get('logic_sources') == logic
             and effective.get('data_identity') == manifest['data_identity'], 'creation launch binding mismatch')
    _require(all(item in launch['data_inputs'] for item in [manifest_id, *manifest['source_files']]),
             'creation launch missing original manifest/raw inputs')
    native = launch['native']
    _require(not native.get('preparation_only') and bool(native.get('module_sha256')) and
             native == manifest['public_surface']['provenance']['native_provenance'], 'creation native mismatch')
    snapshot = identity(Path(launch_id['path']).parent / native['module_file'], control)
    _require(snapshot['sha256'] == native['module_sha256'], 'creation native snapshot mismatch')
    _verify(launch['source_archive'], control)
    with zipfile.ZipFile(launch['source_archive']['path']) as archive:
        for name, item in logic.items():
            _check(control)
            _require(hashlib.sha256(archive.read(name)).hexdigest() == item['sha256'],
                     'creation source snapshot/encoding logic mismatch: ' + name)
    return [launch_id, launch['source_archive'], snapshot]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate receipt JSON key: ' + key)
        result[key] = value
    return result


def _nonfinite_json(value):
    raise ValueError('Controlled cache: nonfinite receipt JSON value: ' + value)


def _prepare_cache(path, manifest, manifest_path, sources, control=None, *, counts=FULL_COUNTS):
    receipt_id = identity(path, control)
    with Path(path).open() as stream:
        receipt = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_nonfinite_json)
    _verify(receipt_id, control)
    _require(type(receipt) is dict and set(receipt) == {'schema', 'row_contract', 'encoding', 'dataset', 'manifest', 'data_identity',
             'source_files', 'logic_sources', 'runtime', 'row_counts', 'row_ids', 'alignment_sha256',
             'array_bytes', 'launch', 'timings'}, 'receipt schema mismatch')
    _require(receipt['schema'] == SCHEMA and receipt['encoding'] == ENCODING
             and receipt['row_contract'] == _row_contract(counts) and receipt['row_counts'] == counts,
             'encoding/splits/full cohort contract mismatch')
    wanted = manifest_rows(manifest, counts)
    _require(receipt['data_identity'] == manifest['data_identity'] and
             receipt['manifest'] == identity(manifest_path, control), 'original manifest identity mismatch')
    _require(receipt['source_files'] == manifest['source_files'] ==
             [identity(p, control) for p in ft.shard_files(sources)], 'raw source identities mismatch')
    _require(receipt['logic_sources'] == logic_sources(control) and receipt['runtime'] == runtime(),
             'stale encoder/trainer/cache/schema/vocabulary/glossary/runtime identity')
    _require(set(receipt['row_ids']) == set(wanted) and set(receipt['alignment_sha256']) == set(wanted),
             'invalid row inventory splits')
    for split in wanted:
        ids = receipt['row_ids'][split]
        _require(type(ids) is list and len(ids) == counts[split] and set(ids) == set(wanted[split]),
                 'receipt row IDs missing/extra/duplicate/TEST')
    evidence = _launch_evidence(receipt['launch'], manifest, receipt['manifest'], receipt['logic_sources'], control)
    result = CacheInput(receipt, receipt_id, [receipt_id, receipt['dataset'], receipt['manifest'],
        *receipt['source_files'], *receipt['logic_sources'].values(), *evidence])
    result.recheck(control)  # Includes the cache hash BEFORE deserialization.
    return result


def prepare_cache(path, manifest, manifest_path, sources, control=None):
    """Production entry point: no row-count override and no TEST data reader."""
    return _prepare_cache(path, manifest, manifest_path, sources, control)


class _ControlledStream:
    """Pickle streams directly to disk; check controls at most every 8 MiB."""
    def __init__(self, stream, control):
        self.stream, self.control, self.pending = stream, control, 0

    def _tick(self, size):
        self.pending += size
        if self.pending >= 8 << 20:
            _check(self.control)
            self.pending = 0

    def write(self, data):
        # Large individual buffers are also bounded; no corpus-sized BytesIO.
        for offset in range(0, len(data), 8 << 20):
            chunk = memoryview(data)[offset:offset + (8 << 20)]
            self.stream.write(chunk)
            self._tick(len(chunk))

    def read(self, size=-1):
        data = self.stream.read(size)
        self._tick(len(data))
        return data

    def readline(self):
        data = self.stream.readline()
        self._tick(len(data))
        return data


def _load_cache(cache, manifest, control=None, *, counts=FULL_COUNTS):
    start = time.monotonic()
    cache.recheck(control)
    verified = time.monotonic()
    with Path(cache.receipt['dataset']['path']).open('rb') as stream:
        dataset = pickle.load(_ControlledStream(stream, control))  # Trusted self-created data only.
        _require(stream.read(1) == b'', 'trailing cache data')
    loaded = time.monotonic()
    info = validate_dataset(dataset, manifest, counts=counts, row_ids=cache.receipt['row_ids'],
                            alignment=cache.receipt['alignment_sha256'], control=control)
    _require(info['array_bytes'] == cache.receipt['array_bytes'], 'array byte inventory mismatch')
    cache.recheck(control)
    return dataset, dict(verify_before_seconds=verified-start, deserialize_seconds=loaded-verified,
                         validate_recheck_seconds=time.monotonic()-loaded,
                         cache_bytes=cache.receipt['dataset']['bytes'], array_bytes=info['array_bytes'])


def load_cache(cache, manifest, control=None):
    _require(cache.receipt['row_contract'] == _row_contract(FULL_COUNTS), 'fixture cache forbidden in production')
    return _load_cache(cache, manifest, control)


def _exclusive_publish(temporary, final):
    os.chmod(temporary, 0o444)
    os.link(temporary, final)  # Atomic and exclusive: never replace a historical file.
    temporary.unlink()


def _write_cache(out, dataset, manifest, manifest_id, launch, control=None, *, counts=FULL_COUNTS,
                 timings=None, logic=None):
    """Internal streaming writer; explicit small counts support unit fixtures only."""
    out = Path(out)
    _require(not (out/'receipt.json').exists() and not (out/'dataset.pkl').exists(), 'cache already exists')
    logic = logic if logic is not None else logic_sources(control)
    evidence = _launch_evidence(launch, manifest, manifest_id, logic, control)
    if control is not None:
        control.status.update(status='validating_encoded_cache')
    info = validate_dataset(dataset, manifest, counts=counts, control=control)
    if control is not None:
        control.status.update(status='writing_encoded_cache')
    start = time.monotonic()
    temporary = out/'.dataset.pending'
    with temporary.open('xb') as stream:
        pickle.dump(dataset, _ControlledStream(stream, control), protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    write_seconds = time.monotonic()-start
    _check(control)
    start = time.monotonic()
    dataset_id = identity(temporary, control)
    hash_seconds = time.monotonic()-start
    # Recheck inputs before publishing. Receipt is the final success marker.
    for item in [manifest_id, *evidence, *manifest['source_files'], *logic.values()]:
        _verify(item, control)
    _exclusive_publish(temporary, out/'dataset.pkl')
    dataset_id['path'] = str((out/'dataset.pkl').resolve())
    receipt = dict(schema=SCHEMA, row_contract=_row_contract(counts), encoding=ENCODING,
        dataset=dataset_id, manifest=manifest_id, data_identity=manifest['data_identity'],
        source_files=manifest['source_files'], logic_sources=logic, runtime=runtime(), row_counts=counts,
        row_ids=info['row_ids'], alignment_sha256=info['alignment_sha256'], array_bytes=info['array_bytes'], launch=launch,
        timings=dict(timings or {}, cache_write_seconds=write_seconds, cache_hash_read_seconds=hash_seconds,
                     estimated_sequential_read_seconds=hash_seconds,
                     estimated_load_array_bytes=info['array_bytes'], cache_bytes=dataset_id['bytes']))
    _check(control)
    pending_receipt = out/'.receipt.pending'
    with pending_receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    _check(control)
    _exclusive_publish(pending_receipt, out/'receipt.json')
    return receipt


def build_cache(config):
    """Heavy creator, for root's queue only. No models, policies or native calls."""
    _require(Path(sys.prefix).name == '.venv' and sys.dont_write_bytecode and
             bool(os.environ.get('STS_LIGHTSPEED_BUILD')), 'use frozen .venv, PYTHONDONTWRITEBYTECODE=1 and STS_LIGHTSPEED_BUILD')
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)
    control = ft.TrainingControl(out, config['deadline_utc'])
    control.start()
    try:
        _check(control)
        start = time.monotonic()
        logic = logic_sources(control)
        manifest_id = identity(config['manifest'], control)
        original = read_json(config['manifest'])
        original_identity = original['data_identity']
        mixed_sources = original.get('mixed_sources',False)
        mixed_sampling = original.get('mixed_sampling',False)
        relabel_vote_rule = original.get('relabel_vote_rule')
        target_mode, value_scale = original.get('target_mode','votes'), original.get('value_scale',100.)
        _require(original_identity == ft.digest({k: v for k, v in original.items() if k != 'data_identity'}),
                 'original manifest self identity mismatch')
        manifest_rows(original)
        del original  # Never keep two parsed copies of the large original manifest.
        manifest = ft.freeze_manifest(config['data'], control=control, mixed_sources=mixed_sources,
                                      mixed_sampling=mixed_sampling, relabel_vote_rule=relabel_vote_rule,
                                      target_mode=target_mode, value_scale=value_scale)
        _require(manifest['data_identity'] == original_identity, 'Frozen manifest/source identities differ')
        _verify(manifest_id, control)
        freeze_seconds = time.monotonic()-start
        teachers = sorted({s['policy']['champion']['path'] for s in manifest['collection_sources'].values()})
        effective = dict(config, stage='controlled_cache_creation', manifest=manifest_id,
                         encoding=ENCODING, data_identity=manifest['data_identity'], logic_sources=logic)
        launch = ft.begin_run(out, effective, checkpoint=None,
            inputs=[config['manifest'], _ROOT/'src/sts_policy/entity_vocabulary.json',
                    *teachers, *ft.shard_files(config['data'])])
        _launch_evidence(launch, manifest, manifest_id, logic, control)
        control.status.update(status='preencoding', data_collected=manifest['collected_total'])
        start = time.monotonic()
        dataset = ft.preencode(config['data'], manifest, 'combat', 'controlled', 'full',
                               splits=('FIT', 'VALIDATION'), control=control)
        encode_seconds = time.monotonic()-start
        receipt = _write_cache(out, dataset, manifest, manifest_id, launch, control,
                              timings=dict(manifest_validation_seconds=freeze_seconds, raw_encode_seconds=encode_seconds),
                              logic=logic)
        summary = dict(status='completed', cache_receipt=identity(out/'receipt.json', control),
                       dataset=receipt['dataset'], row_counts=receipt['row_counts'], timings=receipt['timings'],
                       test_encoded=False, test_serialized=False, **control.stats())
        ft.atomic_json(out/'summary.json', summary)
        control.status.update(summary)
        control.emit()
        return summary
    except Exception as exc:
        control.status.update(status='interrupted' if isinstance(exc, ft.Interrupted) else 'failed',
                              error=f'{type(exc).__name__}: {exc}')
        control.emit()
        raise
    finally:
        control.close()
