"""Conservative orchestration for the approved 2026-10-16 representation study.

Importing this module, and building a plan, use only the standard library. No
training, inference, native policy, or outcome-dependent arm selection lives here.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


DEFAULT_ROOT = Path('/Users/sidbaines/.codex/worktrees/representation-controls/SlayTheSpireAI')
OLD = Path('/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/nonllm-fresh-start/data/rl_runs/fresh_start_20261015')
NATIVE = OLD.parent.parent.parent / 'external/sts_lightspeed_rl/build015'
DEADLINE = '2026-10-17T08:00:00Z'
TARGETS = {'BOSS2': 400, 'BOSS1': 100, 'MONSTER2': 100}
BANK_START, GAME_START = 92_100_000_000, 91_001_000_000
CRITERION = {'metric': 'validation_soft_ce', 'direction': 'min', 'tie_break': 'lower_lr'}
INFERENCE = 'single_observation_exact_argmax_actor_only_v1'
CHECKS = ('source_review', 'same_head_native', 'controlled_child_pilot', 'cpu_mps_parity',
          'capacity_width216', 'fresh_ranges_disjoint', 'no_queue_auto_retry')
ACTIVE = {'pending', 'running', 'paused'}


class Blocked(ValueError):
    """Needs root diagnosis; never a reason to restart or change the science."""


class Waiting(Blocked):
    """An external dependency has not finished; leave the daemon available."""


def require(ok, message):
    if not ok:
        raise Blocked(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}


_VERIFIED = {}


def verify(identity):
    # Rehash after replacement, write, permission change or restart; avoid rereading
    # the full frozen corpus/checkpoints on every idle tick. ctime is included.
    path = Path(identity['path'])
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _VERIFIED.get(str(path))
    if cached != (signature, identity):
        require(file_identity(path) == identity, 'File identity changed: ' + identity['path'])
        _VERIFIED[str(path)] = (signature, dict(identity))
    return identity


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise Blocked('Nonfinite JSON: ' + value)


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_object, parse_constant=_nonfinite)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def exclusive_json(path, value):
    """Atomic, exclusive publication: a crash cannot publish a truncated freeze."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)  # fails if already published, never replaces
    finally:
        os.unlink(temporary)
    return file_identity(path)


def utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, 'Require UTC deadline')
    return stamp.timestamp()


@dataclass(frozen=True)
class Paths:
    root: Path = DEFAULT_ROOT

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).absolute())

    @property
    def base(self):
        return self.root / 'data/rl_runs/representation_controls_20261016'

    @property
    def python(self):
        return self.root / '.venv/bin/python'

    @property
    def champion(self):
        return OLD / 'inputs/champion_3d135d9bd53b.pt'

    @property
    def manifest(self):
        return OLD / 'common_manifest_v1.json'

    @property
    def cache_receipt(self):
        return self.base / 'controlled_cache_v1/receipt.json'

    @property
    def grid(self):
        return self.base / 'frozen_panel_grid_v1.json'

    def script(self, name):
        return self.root / 'scripts/rl' / name

    def workload(self, script, args):
        return ['/usr/bin/env', 'STS_LIGHTSPEED_BUILD=' + str(NATIVE),
                'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
                'VECLIB_MAXIMUM_THREADS=1', str(self.python), str(self.script(script)), *map(str, args)]

    def add(self, name, lane, command):
        return [str(self.python), str(self.script('queue_runner.py')), 'add', name,
                '--lane', lane, '--', *command]


def fit_spec(architecture, seed, lr, *, width=128, features=False, secondary=False):
    rate = {0.0003: '3e4', 0.001: '1e3', 0.0001: '1e4'}[lr]
    profile = ('on' if features else 'off') + str(width) if architecture == 'controlled' else 'old'
    name = f'rc_{profile}_lr{rate}_s{seed}'
    condition = dict(phase='combat', lr=lr, seed=seed, steps=12000, min_steps=2000,
                     eval_every=100, patience=6, device='mps', fit_size='full', batch=128, defer_test=True)
    if architecture == 'controlled':
        condition.update(width=width, layers=2, heads=4, legacy_action_features=features)
    else:
        condition.update(old_size='large', old_context='none')
    return dict(name=name, architecture=architecture, profile=profile, seed=seed,
                secondary=secondary, condition=condition)


def main_fits():
    # Alternating feature, width, seed and LR blocks; the prescribed plumbing replicate leads.
    result = []
    cells = [(False, 128), (True, 216), (True, 128), (False, 216)]
    for repeat in range(2):
        for cell, (features, width) in enumerate(cells):
            for offset in range(2):
                seed = (1919, 2020)[(cell + offset) % 2]
                lr = (0.0003, 0.001)[(repeat + cell + offset) % 2]
                result.append(fit_spec('controlled', seed, lr, width=width, features=features))
        result.append(fit_spec('old', 2020, (0.0003, 0.001)[repeat]))
    return result


def extension_fits():
    return [fit_spec('controlled', seed, 0.0001, width=width, features=features, secondary=True)
            for seed in (1919, 2020) for width in (128, 216) for features in (False, True)]


def references():
    return [fit_spec('old', 1919, lr) for lr in (0.0003, 0.001)]


def fit_directory(paths, spec):
    if spec['architecture'] == 'old' and spec['seed'] == 1919:
        rate = {0.0003: '3e4', 0.001: '1e3'}[spec['condition']['lr']]
        return OLD / f'fit_old_combat_lr{rate}_v1'
    return paths.base / 'fits' / spec['name']


def fit_command(paths, spec, deadline):
    c = spec['condition']
    args = ['--data', *[OLD / f'teacher_corpus_2000_v1/shard_{i:02d}' for i in range(8)],
            '--manifest', paths.manifest, '--probe', OLD / 'probe_native_schema_v2',
            '--out', fit_directory(paths, spec), '--architecture', spec['architecture']]
    for key in ('phase', 'fit_size', 'lr', 'seed', 'steps', 'min_steps', 'eval_every', 'patience', 'batch', 'device'):
        args += ['--' + key.replace('_', '-'), c[key]]
    args += ['--old-size', 'large', '--old-context', 'none', '--width', c.get('width', 128),
             '--layers', 2, '--heads', 4, '--defer-test', '--deadline-utc', deadline]
    if spec['architecture'] == 'controlled':
        args += ['--preencoded-cache', paths.cache_receipt]
        args += ['--legacy-action-features' if c['legacy_action_features'] else '--no-legacy-action-features']
    return paths.workload('fresh_train.py', args)


def selection_path(paths, spec):
    return paths.base / 'selections' / (spec['name'] + '.json')


def evaluation_command(paths, stage, spec, deadline):
    out = paths.base / stage / spec['name']
    if stage == 'scores':
        return paths.workload('fresh_score.py', ['--selection', selection_path(paths, spec),
                                               '--out', out, '--deadline-utc', deadline])
    require(stage == 'panels', 'Unknown evaluation stage')
    return paths.workload('fresh_panel.py', [
        '--bank', *[paths.base / f'fresh_bank_v1/shard_{i:02d}' for i in range(4)],
        '--bank-world-start', BANK_START, '--bank-n', 1024, '--grid', paths.grid,
        '--starts-per-group', canonical(TARGETS), '--resamples', 8, '--workers', 6, '--timeout', 30,
        '--checkpoint', paths.champion, '--combat', fit_directory(paths, spec) / 'best.pt',
        '--out', out, '--deadline-utc', deadline])


def game_arms():
    return [('champion', None, None), ('old1919', 'old', 1919),
            ('off128_s1919', 'off128', 1919), ('on128_s1919', 'on128', 1919),
            ('off216_s1919', 'off216', 1919), ('on216_s1919', 'on216', 1919),
            ('old2020', 'old', 2020), ('off128_s2020', 'off128', 2020)]


def game_command(paths, arm, shard, checkpoint, deadline):
    args = ['--mode', 'eval', '--checkpoint', paths.champion, '--no-teacher-labels', '--no-bank',
            '--self-consistency-every', 0, '--world-start', GAME_START + 512 * shard, '--n', 512,
            '--workers', 1, '--timeout', 30, '--out', paths.base / f'games/{arm}/shard_{shard:02d}',
            '--deadline-utc', deadline]
    if checkpoint is not None:
        args += ['--combat', checkpoint]
    return paths.workload('controlled_eval.py', args)


def merge_command(paths, arm):
    return paths.workload('fresh_merge.py', ['--inputs',
        *[paths.base / f'games/{arm}/shard_{i:02d}' for i in range(8)],
        '--world-start', GAME_START, '--n', 4096, '--out', paths.base / f'games/{arm}/merged'])


def build_plan(root=DEFAULT_ROOT, deadline=DEADLINE):
    """Pure command/spec generation. No source gates, queues or outcomes are read."""
    require(utc(deadline) <= utc(DEADLINE), 'Plan deadline exceeds approved safety boundary')
    p = Paths(root)
    fits, extension = main_fits(), extension_fits()
    conditions = fits + references() + extension
    jobs = []
    for spec in fits + extension:
        name, lane, command = spec['name'], 'fit', fit_command(p, spec, deadline)
        jobs.append(dict(stage='fits', condition=spec['name'], secondary=spec['secondary'],
                         name=name, lane=lane, command=command, enqueue=p.add(name, lane, command)))
    for stage, lane in [('scores', 'score'), ('panels', 'panel')]:
        for spec in conditions:
            name, command = f'rc_{stage}_{spec["name"]}', evaluation_command(p, stage, spec, deadline)
            jobs.append(dict(stage=stage, condition=spec['name'], secondary=spec['secondary'],
                             name=name, lane=lane, command=command, enqueue=p.add(name, lane, command)))
    games = []
    for arm, profile, seed in game_arms():
        candidates = [s for s in fits + references() if s['profile'] == profile and s['seed'] == seed] if profile else [None]
        variants = []
        for spec in candidates:
            checkpoint = fit_directory(p, spec) / 'best.pt' if spec else None
            shards = []
            for i in range(8):
                name, lane = f'rc_game_{arm}_{i:02d}', f'game{i}'
                cmd = game_command(p, arm, i, checkpoint, deadline)
                shards.append(dict(name=name, lane=lane, command=cmd, enqueue=p.add(name, lane, cmd)))
            variants.append(dict(condition=spec['name'] if spec else None, shards=shards))
        cmd, name = merge_command(p, arm), f'rc_merge_{arm}'
        games.append(dict(arm=arm, profile=profile, seed=seed, variants=variants,
                          merge=dict(name=name, lane='game0', command=cmd, enqueue=p.add(name, 'game0', cmd))))
    return dict(schema='representation_controls_plan_v1', root=str(p.root), base=str(p.base),
                old=str(OLD), native=str(NATIVE), python=str(p.python), deadline_utc=deadline,
                preencoded_cache=str(p.cache_receipt),
                approved_plan=str(p.root / 'docs/representation_controls_plan_2026-10-16.md'),
                main_fits=fits, extension_fits=extension, references=references(), jobs=jobs, games=games,
                extension_rule='ALL four width216 feature x seed validation CE(0.0003) < CE(0.001)',
                selection_criterion=CRITERION, targets=TARGETS, resamples=8,
                bank_world_start=BANK_START, bank_n=1024, game_world_start=GAME_START, game_n=4096,
                immutable_optimizer=dict(name='AdamW', weight_decay=0.01, gradient_clip=1,
                                         dropout=0, critic='untrained/frozen', bucketing='inherited_two_axis'),
                gates=list(CHECKS), confirmation='closed',
                ordering='main fits; extension decision/fits; global freeze; all labels; all panels; eight game arms')


def extension_decision(records):
    cells = []
    for features in (False, True):
        for seed in (1919, 2020):
            names = [fit_spec('controlled', seed, lr, width=216, features=features)['name']
                     for lr in (0.0003, 0.001)]
            losses = [records[n]['best_validation_loss'] for n in names]
            cells.append(dict(names=names, losses=losses, lower_rate_better=losses[0] < losses[1]))
    return dict(schema='representation_controls_extension_v1', validation_only=True,
                triggered=all(c['lower_rate_better'] for c in cells), cells=cells,
                secondary_only=True, changes_primary_lr_choices=False)


def primary_choices(specs, records):
    choices = {}
    for spec in specs:
        if spec['secondary']:
            continue
        key = f'{spec["profile"]}_s{spec["seed"]}'
        previous = choices.get(key)
        rank = lambda s: (records[s['name']]['best_validation_loss'], s['condition']['lr'])
        if previous is None or rank(spec) < rank(previous):
            choices[key] = spec
    return {key: spec['name'] for key, spec in choices.items()}


def manifest_rows(path):
    """Stream the large manifest's rows; never retain the 393MB JSON document."""
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_nonfinite)
    with Path(path).open() as stream:
        buffer, ended = '', False

        def refill():
            nonlocal buffer, ended
            chunk = stream.read(65536)
            ended = not chunk
            buffer += chunk

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not ended:
                refill()
                buffer = buffer.lstrip()

        def token(expected):
            nonlocal buffer
            whitespace()
            require(buffer.startswith(expected), 'Invalid manifest JSON token: ' + expected)
            buffer = buffer[len(expected):]

        def value():
            nonlocal buffer
            whitespace()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    return result
                except json.JSONDecodeError:
                    require(not ended, 'Truncated manifest JSON')
                    refill()

        token('{')
        seen, found = set(), False
        while True:
            key = value()
            require(isinstance(key, str) and key not in seen, 'Invalid/duplicate manifest key')
            seen.add(key)
            token(':')
            if key == 'rows':
                found = True
                token('[')
                whitespace()
                if not buffer.startswith(']'):
                    while True:
                        yield value()
                        whitespace()
                        if buffer.startswith(']'):
                            break
                        token(',')
                token(']')
            else:
                value()
            whitespace()
            if buffer.startswith('}'):
                token('}')
                break
            token(',')
        whitespace()
        require(found and not buffer, 'Missing rows or trailing manifest JSON')


def _bindings(paths):
    return [file_identity(p) for p in paths]


class Validator:
    """Filesystem/checkpoint validation only. Never calls a model or native API."""

    def __init__(self, paths, gate):
        self.paths, self.gate = paths, gate

    def root_launch(self, launch):
        design = str((self.paths.root / 'docs/representation_controls_plan_2026-10-16.md').resolve())
        expected = sorted((i for i in self.gate['source_files'] if i['path'] != design), key=lambda i: i['path'])
        require(sorted(launch['source_files'], key=lambda i: i['path']) == expected,
                'Launch source files differ from pinned ROOT source gate')

    def grid(self):
        from sts_rl.fresh_panel_comparison import _Inputs, _grid
        frozen = read_json(self.paths.grid)
        require(frozen['targets'] == TARGETS and frozen['coverage'] == TARGETS, 'Require 400/100/100 grid')
        bank = frozen['bank_identity']
        require(bank['world_start'] == BANK_START and bank['n'] == 1024, 'Wrong fresh bank range')
        require(bank['expected_provenance']['native_provenance'] == self.gate['native'], 'Bank/native mismatch')
        inputs = _Inputs()
        _grid(frozen, inputs)
        inputs.recheck()
        return frozen

    def fit(self, spec):
        # Lazy CPU inspection; no actor is constructed or initialized, no inference.
        import torch
        from sts_rl.fresh_training import checkpoint_training_condition
        from sts_policy.training import load_checkpoint
        directory = fit_directory(self.paths, spec)
        summary = read_json(directory / 'summary.json')
        require(summary.get('status') == 'completed' and summary.get('test_deferred') is True
                and summary.get('test_evaluated') is False, 'Fit incomplete/interrupted or TEST already evaluated')
        require(summary['actual_fit_rows'] == 113042 and summary['validation_rows'] == 17989,
                'Fit/validation rows differ from approved full corpus')
        cp = file_identity(directory / 'best.pt')
        require(cp == summary['selected_checkpoint'], 'Fit summary/checkpoint mismatch')
        payload = load_checkpoint(cp['path'], expected_data_identity=self.gate['data_identity'])
        meta, c = payload['metadata'], spec['condition']
        require(checkpoint_training_condition(payload) == c and meta['config']['architecture'] == spec['architecture'],
                'Actual checkpoint training condition mismatch')
        require(meta.get('initialization') == 'from_scratch' and 'parent_checkpoint' in meta
                and meta['parent_checkpoint'] is None, 'Checkpoint inherited weights')
        require(meta['filters'] == self.gate['filters'] and meta['vocabulary_sha256'] == self.gate['vocabulary_sha256'],
                'Checkpoint filtering/vocabulary mismatch')
        expected_critic = 'head_frozen_shared_actor_features_trained' if spec['architecture'] == 'old' else 'frozen_untrained'
        groups = payload['optimizer_state']['param_groups']
        require(meta['critic_status'] == expected_critic and groups and
                all(g['lr'] == c['lr'] and g['weight_decay'] == .01 for g in groups),
                'Actual optimizer/critic condition mismatch')
        loss, step = summary['best_validation_loss'], summary['best_step']
        require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0, 'Invalid validation loss')
        require(type(step) is int and step > 0 and step % 100 == 0 and payload['step'] == step
                and meta['training_state']['best_step'] == step and meta['training_state']['best_loss'] == loss,
                'Not the actual validation-selected best checkpoint')
        require(type(summary['updates']) is int and 2000 <= summary['updates'] <= 12000
                and step <= summary['updates'] and summary['updates'] % 100 == 0 and summary['examples'] > 0,
                'Invalid completed optimizer budget/example presentations')
        if spec['architecture'] == 'controlled':
            from sts_policy.controlled_entity import _source_identity, _config_identity
            require(all(meta.get(k) == v for k, v in _source_identity().items()), 'Controlled source identity mismatch')
            require(meta['model_config']['dropout'] == 0 and
                    meta['model_config_sha256'] == _config_identity(meta['model_config']) and
                    payload['model_config']['model_config'] == meta['model_config'], 'Model config identity mismatch')
        require(payload['model_state'] and all(bool(torch.isfinite(t).all()) for t in payload['model_state'].values()),
                'Empty/nonfinite checkpoint')
        source = meta['source_receipt']
        cache = self.gate['preencoded_cache'] if spec['architecture'] == 'controlled' else None
        if spec['architecture'] == 'controlled':
            require(isinstance(cache, dict) and meta['config'].get('preencoded_cache') == str(self.paths.cache_receipt)
                    and meta['config'].get('preencoded_cache_provenance') == cache
                    and source.get('preencoded_cache') == cache, 'Checkpoint preencoded cache mismatch')
        else:
            require('preencoded_cache' not in meta['config'] and 'preencoded_cache_provenance' not in meta['config']
                    and 'preencoded_cache' not in source, 'Old checkpoint must not use a preencoded cache')
        require(read_json(directory / 'receipt.json') == source and source['data_identity'] == self.gate['data_identity']
                and source.get('weight_parent') is None and source['defer_test'] is True
                and source['label_source_files'] == self.gate['label_source_files'], 'Checkpoint source/data receipt mismatch')
        expected_data = [str(OLD / f'teacher_corpus_2000_v1/shard_{i:02d}') for i in range(8)]
        require(meta['config']['data'] == expected_data and meta['config']['manifest'] == str(self.paths.manifest)
                and meta['config']['probe'] == [str(OLD / 'probe_native_schema_v2')]
                and source['probe_source_files'] == self.gate['probe_source_files'], 'Checkpoint corpus/probe arguments mismatch')
        verify(source['launch'])
        launch = read_json(source['launch']['path'])
        require(launch['effective_config'] == meta['config'] and launch['parent_checkpoint'] is None
                and launch['native'] == self.gate['native'], 'Checkpoint launch/config/native mismatch')
        if spec['seed'] != 1919 or spec['architecture'] != 'old':
            self.root_launch(launch)
        from sts_rl.controlled_comparison import _native_snapshot
        from sts_rl.fresh_panel_comparison import _Inputs
        native_inputs = _Inputs()
        _native_snapshot(source['launch'], launch, native_inputs, self.gate['native'])
        native_inputs.recheck()
        verify(launch['source_archive'])
        require(launch['data_inputs'] and source['probe_source_files'], 'Missing training/probe data identities')
        if cache is not None:
            require(all(cache[k] in launch['data_inputs'] for k in ('receipt', 'dataset')),
                    'Training data inputs missing preencoded cache receipt/dataset identities')
        for identity in [*launch['data_inputs'], *source['probe_source_files']]:
            verify(identity)
        verify(cp)
        return dict(checkpoint=cp, architecture=spec['architecture'], condition=c, best_step=step,
                    best_validation_loss=loss, data_identity=meta['data_identity'],
                    updates=summary['updates'], examples=summary['examples'],
                    bindings=_bindings([directory / 'summary.json', directory / 'receipt.json']) +
                             [cp, source['launch'], launch['source_archive']])

    def score(self, spec, frozen, selection_id):
        out = self.paths.base / 'scores' / spec['name']
        summary, receipt = read_json(out / 'summary.json'), read_json(out / 'receipt.json')
        cp = frozen['checkpoint']
        for value in (summary, receipt):
            require(value.get('training') is False and type(value.get('weight_updates')) is int
                    and value['weight_updates'] == 0 and value['selected_checkpoint'] == cp
                    and value['selection'] == selection_id and value['data_identity'] == frozen['data_identity'],
                    'Score does not bind inference-only frozen selection')
        require(summary.get('status') == 'completed' and summary.get('test_evaluated') is True
                and summary.get('test_protocol') == INFERENCE and summary.get('test_device') == 'cpu',
                'Score incomplete or wrong singleton CPU protocol')
        require(receipt['manifest'] == self.gate['manifest'] and receipt['selection_criterion'] == CRITERION
                and receipt['architecture'] == spec['architecture'] and receipt['phase'] == 'combat'
                and receipt['stage'] == 'deferred_test_scoring' and receipt['device'] == 'cpu'
                and receipt['label_source_files'] == self.gate['label_source_files'], 'Score receipt mismatch')
        require(receipt['filters'] == self.gate['filters'] and receipt['vocabulary_sha256'] == self.gate['vocabulary_sha256']
                and receipt['source_receipt'] == read_json(fit_directory(self.paths, spec) / 'receipt.json'),
                'Score training/source metadata mismatch')
        # Strict stdlib parsing avoids the scanner's repeated partial decoding.
        wanted = {r['source_row_id']: r for r in read_json(self.paths.manifest)['rows']
                  if r['phase'] == 'combat' and r['split'] == 'TEST' and r['has_target']}
        expected_count = len(wanted)
        require(expected_count > 0 and summary['test_rows'] == expected_count, 'Score TEST cohort count mismatch')
        require(summary['test']['groups']['all_rows']['rows'] == expected_count, 'Score summary row count mismatch')
        rows_path = out / 'test_rows.jsonl'
        rows_id = file_identity(rows_path)
        if spec['architecture'] == 'controlled' or 'test_rows_file' in summary:
            require(summary.get('test_rows_file') == rows_id, 'Score row file binding mismatch')
        with rows_path.open() as stream:
            for line in stream:
                row = json.loads(line, object_pairs_hook=_object, parse_constant=_nonfinite)
                original = wanted.pop(row['source_row_id'], None)
                require(original is not None and all(row.get(k) == v for k, v in original.items()),
                        'Duplicate/unexpected/mismatched TEST row')
                require(row['selected_checkpoint'] == cp['sha256'] and row['architecture'] == spec['architecture']
                        and row['training_condition'] == frozen['condition'] and row['data_identity'] == frozen['data_identity']
                        and row['inference'] == INFERENCE and type(row['correct']) is bool
                        and type(row['ce']) in (int, float) and math.isfinite(row['ce']) and row['ce'] >= -1e-6
                        and digest(row['target']) == row['target_sha256'], 'Invalid score row protocol/condition/target')
        require(not wanted, 'Incomplete TEST rows')
        verify(rows_id)
        from sts_rl.controlled_comparison import _launch
        from sts_rl.fresh_panel_comparison import _Inputs
        inputs = _Inputs()
        launch = _launch(receipt, out.resolve(), inputs, dict(source_files=self.gate['label_source_files'],
                        public_surface={'provenance': {'native_provenance': self.gate['native']}}))
        inputs.recheck()
        launch_id = receipt['launch']
        self.root_launch(launch)
        require(not any((out / 'quarantine').rglob('*')), 'Score contains quarantined attempts')
        return dict(bindings=_bindings([out / 'summary.json', out / 'receipt.json']) +
                    [rows_id, launch_id, launch['source_archive']])

    def panel(self, spec, frozen, grid):
        from sts_rl.fresh_panel_comparison import _Inputs, _grid, _run
        from sts_rl.fresh_study import PROTOCOL, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID
        inputs = _Inputs()
        starts = _grid(grid, inputs)
        policy = dict(protocol=PROTOCOL, filter_ids=[FILTER_ID, DEAD_FILTER_ID],
                      public_context_id=PUBLIC_CONTEXT_ID, inference=INFERENCE,
                      champion=self.gate['champion'], combat=frozen['checkpoint'], ooc=None)
        result = _run(self.paths.base / 'panels' / spec['name'], policy, grid,
                      file_identity(self.paths.grid), starts, inputs)
        self.root_launch(read_json(result['identities']['launch']['path']))
        inputs.recheck()
        return dict(bindings=list(result['identities'].values()))

    def cohort(self, arm, checkpoint, shard=None):
        from sts_rl.fresh_cohort import validate_cohorts
        from sts_rl.controlled_comparison import _native_snapshot
        from sts_rl.fresh_panel_comparison import _Inputs
        out = self.paths.base / 'games' / arm / ('merged' if shard is None else f'shard_{shard:02d}')
        start, n = (GAME_START, 4096) if shard is None else (GAME_START + 512 * shard, 512)
        cohort = validate_cohorts([out], world_start=start, n=n)
        science, c = cohort['science_identity'], cohort['config']
        policy = science['policy']
        require(policy['champion'] == self.gate['champion'] and policy['combat'] == checkpoint and policy['ooc'] is None
                and science['native'] == self.gate['native'], 'Game frozen policy/native mismatch')
        require(c['mode'] == 'eval' and c['teacher_labels'] is False and c['bank'] is False
                and c['self_consistency_every'] == 0 and c['workers'] == 1 and c['timeout'] == 30,
                'Game evaluation settings mismatch')
        inputs, environment = _Inputs(), None
        for source in cohort['sources']:
            launch_id = inputs.verify(source['launch'])
            launch, _ = inputs.read(launch_id['path'], launch_id)
            self.root_launch(launch)
            _native_snapshot(launch_id, launch, inputs, self.gate['native'])
            require(launch['data_inputs'] == ([checkpoint] if checkpoint else []), 'Game replacement inputs mismatch')
            require(environment is None or environment == launch['environment'], 'Game launch environment differs')
            environment = launch['environment']
            for key, identity in [('checkpoint', self.gate['champion']), ('combat', checkpoint), ('ooc', None)]:
                actual = source['config'].get(key)
                require((actual is None and identity is None) or
                        (actual is not None and identity is not None and Path(actual).resolve() == Path(identity['path'])),
                        'Game config/checkpoint path mismatch')
        inputs.recheck()
        common = {**science, 'config': {k: v for k, v in science['config'].items() if k != 'combat'},
                  'policy': {k: v for k, v in policy.items() if k != 'combat'}, 'environment': environment}
        return dict(bindings=_bindings(cohort['inputs']), common=common)


def source_paths(root):
    # Match recording.begin_run's archived inventory, plus the approved design.
    paths = {p for section in ('src', 'scripts', 'tests', 'patches') for p in (root / section).rglob('*')
             if p.is_file() and p.suffix in {'.py', '.json', '.patch', '.md', '.sh'} and '__pycache__' not in p.parts}
    paths |= {root / 'pyproject.toml', root / 'docs/representation_controls_plan_2026-10-16.md'}
    return sorted(paths)


class Controller:
    def __init__(self, root=DEFAULT_ROOT, *, submit=None, validator_factory=Validator, now=time.time):
        self.paths = Paths(root)
        self.submit = submit or self._submit
        self.validator_factory, self.now = validator_factory, now
        self.state = None
        self._grid_cache = None
        self._audited = False

    def _submit(self, argv):
        # The only subprocess the controller may create: queue_runner add, never start.
        subprocess.run(argv, cwd=self.paths.root, check=True, timeout=30, shell=False,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    def save(self):
        atomic_json(self.paths.base / 'controller_state_v1.json', self.state)

    @contextmanager
    def locked(self):
        self.paths.base.mkdir(parents=True, exist_ok=True)
        with (self.paths.base / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def queue(self):
        path = self.paths.root / 'data/rl_runs/queue.json'
        return read_json(path)['runs'] if path.exists() else []

    def gates(self):
        p = self.paths
        gate_path, plan_path = p.base / 'preflight_pass_v1.json', p.base / 'controller_plan_v1.json'
        if not (gate_path.exists() and p.grid.exists() and plan_path.exists()):
            raise Waiting('Waiting for root preflight, grid and plan')
        gate, plan = read_json(gate_path), read_json(plan_path)
        require(plan == build_plan(p.root, plan['deadline_utc']), 'Plan differs from exact approved command/spec matrix')
        require(gate.get('status') == 'PASS' and gate.get('schema') == 'representation_controls_preflight_v1'
                and gate.get('root') == str(p.root), 'Root preflight is not PASS for this ROOT')
        require(all(gate.get('checks', {}).get(k) == 'PASS' for k in CHECKS), 'Root preflight checks missing')
        expected = {str(path.resolve()) for path in source_paths(p.root)}
        actual = [i['path'] for i in gate['source_files']]
        require(len(actual) == len(set(actual)) and set(actual) == expected, 'Source gate must cover exact ROOT source inventory')
        for identity in gate['source_files']:
            verify(identity)
        # Keep rejecting the known auto-retry path even after root repairs the runner.
        queue_ast = ast.parse(p.script('queue_runner.py').read_text())
        require(not any(isinstance(n, ast.Constant) and n.value == '--resume' for n in ast.walk(queue_ast)),
                'Queue source still permits --resume/automatic retries; root must repair and repin')
        require(gate['manifest']['path'] == str(p.manifest) and gate['champion']['path'] == str(p.champion),
                'Wrong manifest/champion paths')
        cache = gate.get('preencoded_cache')
        require(isinstance(cache, dict) and set(cache) == {'schema', 'receipt', 'dataset', 'manifest', 'data_identity'}
                and cache['schema'] == 'controlled_preencoded_cache_v1', 'Missing/invalid preencoded cache provenance')
        for key in ('receipt', 'dataset', 'manifest'):
            require(isinstance(cache[key], dict) and set(cache[key]) == {'path', 'bytes', 'sha256'},
                    'Invalid preencoded cache file identity: ' + key)
        require(cache['receipt']['path'] == str(p.cache_receipt.resolve())
                and cache['manifest'] == gate['manifest'] and cache['data_identity'] == gate['data_identity']
                and isinstance(cache['data_identity'], str) and len(cache['data_identity']) == 64
                and all(c in '0123456789abcdef' for c in cache['data_identity']),
                'Preencoded cache path/manifest/data identity mismatch')
        for key in ('receipt', 'dataset', 'manifest'):
            verify(cache[key])
        for identity in [gate['manifest'], gate['champion'], gate['python'], *gate['label_source_files'],
                         *gate['probe_source_files'], *gate['native_files']]:
            verify(identity)
        require(gate['python'] == file_identity(p.python) and gate['native_files'] and gate['probe_source_files']
                and gate['label_source_files'], 'Missing frozen runtime/native/data identity')
        require(all(Path(i['path']).is_relative_to(NATIVE) for i in gate['native_files'])
                and gate['native']['module_sha256'] in {i['sha256'] for i in gate['native_files']}
                and not gate['native'].get('preparation_only'), 'Wrong native build identity')
        pins = dict(preflight=file_identity(gate_path), grid=file_identity(p.grid), plan=file_identity(plan_path))
        if 'pins' in self.state:
            require(self.state['pins'] == pins, 'Pinned preflight/grid/plan changed')
        else:
            self.state['pins'] = pins
            self.save()
        validator = self.validator_factory(p, gate)
        if self._grid_cache is None:
            self._grid_cache = validator.grid()
        # The grid validator checked all bank bindings on first use. Recheck their
        # file signatures while idle, and fully hash again after process restart.
        for identity in self._grid_cache.get('bank_identity', {}).get('inputs', []) + self._grid_cache.get('bank_identity', {}).get('shards', []):
            verify(identity)
        return gate, validator, self._grid_cache

    def job(self, name, lane, command, validate):
        """Return true only for a completed, scientifically valid artifact set."""
        runs = self.queue()
        matches = [r for r in runs if r['name'] == name]
        require(len(matches) <= 1, 'Duplicate queue job: ' + name)
        record = self.state['jobs'].get(name)
        if record is None:
            self.authorize_submission(command)
            require(not matches, 'Unowned pre-existing queue job: ' + name)
            if '--out' in command:
                out = Path(command[command.index('--out') + 1])
                require(not out.exists() or not any(out.iterdir()), 'Unowned/nonempty output: ' + str(out))
            if not getattr(self, '_game_submission', False) and any(r['status'] in ACTIVE for r in runs):
                raise Waiting('Wait for existing workloads before submitting another phase')
            require(shutil.disk_usage(self.paths.base).free >= 6 * 1024**3, 'Less than 6GiB free')
            # Persist intent before add. Ambiguous submissions never auto-repeat.
            record = dict(name=name, lane=lane, command=command, status='submitting',
                          deadline_utc=self.control['safety_deadline_utc'],
                          global_freeze=self.state.get('global_freeze'))
            self.state['jobs'][name] = record
            self.save()
            self.authorize_submission(command)
            self.submit(self.paths.add(name, lane, command))
            matches = [r for r in self.queue() if r['name'] == name]
            require(len(matches) == 1, 'Ambiguous queue submission; root must reconcile: ' + name)
        require(matches, 'Submission missing from queue; no automatic retry: ' + name)
        run = matches[0]
        require(run['cmd'] == record['command'] and run.get('lane', 'main') == record['lane']
                and '--resume' not in run['cmd'] and not run.get('retried'), 'Queue command/lane changed or retried: ' + name)
        require(run['status'] not in {'failed', 'cancelled', 'interrupted', 'incomplete'}
                and not run.get('cancel'), 'Job failed/cancelled; root repair required: ' + name)
        if run['status'] in ACTIVE:
            record['status'] = run['status']
            self.state['waiting_for'] = name
            return False
        require(run['status'] in {'done', 'exited'} and run.get('returncode') in (None, 0), 'Unknown/failed terminal queue status')
        if 'validated' not in record:
            record['validated'] = validate()
        for identity in record['validated']['bindings']:
            verify(identity)
        record['status'] = 'validated'
        return True

    def authorize_submission(self, command):
        # Validation of a prior artifact may take time. Recheck live source and
        # permission immediately before add, including phase changes during a tick.
        self.gates()
        self.check_control()
        is_fit = str(self.paths.script('fresh_train.py')) in command
        require(is_fit or self.control['allowed_phase'] in {'evaluations', 'complete_pipeline'},
                'Evaluations explicitly disabled')
        require(is_fit or self.state.get('global_freeze'), 'Cannot evaluate before the global freeze')
        require(self.control['allowed_phase'] != 'preparation_only', 'Fits explicitly disabled')
        if not is_fit:
            verify(self.state['global_freeze'])
            verify(self.state['cohort'])
            for identity in self.state['selections'].values():
                verify(identity)
            for record in read_json(self.state['global_freeze']['path'])['conditions'].values():
                verify(record['checkpoint'])
        if '--deadline-utc' in command:
            require(command[command.index('--deadline-utc') + 1] == self.control['safety_deadline_utc'],
                    'Control changed before submission; root must reconcile any recorded intent')

    def check_control(self):
        p = self.paths
        require(not (p.base / 'STOP').exists(), 'STOP present; no new submissions')
        c = read_json(p.base / 'control.json')
        require(c.get('stop') is False, 'Control stop/missing explicit stop=false')
        require(c.get('allowed_phase') in {'preparation_only', 'fits', 'evaluations', 'complete_pipeline'}, 'Invalid allowed_phase')
        deadline = utc(c['safety_deadline_utc'])
        require(self.now() < deadline <= utc(DEADLINE), 'Safety deadline reached or exceeds approved boundary')
        self.control = c
        return c

    def frozen_command(self, name, make):
        # Changes to control affect new commands only, never already submitted ones.
        existing = self.state['jobs'].get(name)
        if existing:
            return existing['command']
        self.check_control()
        return make(self.control['safety_deadline_utc'])

    def freeze(self, specs, records, gate, decision_id):
        p = self.paths
        global_path = p.base / 'global_freeze_v1.json'
        if global_path.exists():
            frozen = read_json(global_path)
            identity = file_identity(global_path)
            if self.state.get('global_freeze'):
                require(identity == self.state['global_freeze'], 'Global freeze changed')
            # Recover publication crashes without choosing LRs again.
            require(frozen['pins'] == self.state['pins'] and frozen['extension_decision'] == decision_id
                    and frozen['conditions'] == records and frozen['manifest'] == gate['manifest']
                    and frozen.get('preencoded_cache') == gate['preencoded_cache'],
                    'Existing freeze no longer matches immutable fit/source bindings')
        else:
            require(not self.state.get('global_freeze'), 'Global freeze was removed; never reselect')
            choices = primary_choices(specs, records)
            arms = {arm: (choices[f'{profile}_s{seed}'] if profile else None) for arm, profile, seed in game_arms()}
            frozen = dict(schema='representation_controls_global_freeze_v1', pins=self.state['pins'],
                          conditions=records, specs=specs, primary_choices=choices, game_arms=arms,
                          manifest=gate['manifest'], champion=gate['champion'], data_identity=gate['data_identity'],
                          preencoded_cache=gate['preencoded_cache'],
                          source_files=gate['source_files'], label_source_files=gate['label_source_files'],
                          probe_source_files=gate['probe_source_files'], native_files=gate['native_files'],
                          selection_criterion=CRITERION, extension_decision=decision_id,
                          worlds=list(range(GAME_START, GAME_START + 4096)))
            identity = exclusive_json(global_path, frozen)
        self.state['global_freeze'] = identity
        self.save()  # bind before publishing selections, and before any evaluation submission
        for spec in specs:
            r = records[spec['name']]
            selection = dict(selected_checkpoint=r['checkpoint'], manifest=gate['manifest'],
                             selection_criterion=CRITERION, architecture=spec['architecture'],
                             training_condition=r['condition'], global_freeze=identity, secondary=spec['secondary'])
            target = selection_path(p, spec)
            if target.exists():
                require(read_json(target) == selection, 'Frozen selection changed: ' + str(target))
            else:
                require(spec['name'] not in self.state.get('selections', {}), 'Frozen selection removed')
                exclusive_json(target, selection)
            sid = file_identity(target)
            require(self.state.get('selections', {}).get(spec['name'], sid) == sid, 'Selection file identity changed')
            self.state.setdefault('selections', {})[spec['name']] = sid
        cohort = dict(schema='representation_controls_cohort_v1', global_freeze=identity,
                      worlds=frozen['worlds'], world_start=GAME_START, n=4096, arms=frozen['game_arms'])
        target = p.base / 'cohort.json'
        if target.exists():
            require(read_json(target) == cohort, 'Cohort freeze changed')
        else:
            require('cohort' not in self.state, 'Cohort freeze removed')
            exclusive_json(target, cohort)
        cid = file_identity(target)
        require(self.state.get('cohort', cid) == cid, 'Cohort file identity changed')
        self.state['cohort'] = cid
        self.save()
        return frozen

    def advance(self):
        p = self.paths
        c = self.check_control()
        if c['allowed_phase'] == 'preparation_only':
            self.state.update(status='waiting', reason='preparation_only: fits explicitly disabled')
            return
        gate, validator, grid = self.gates()
        require(not any(r['status'] in ACTIVE and any(str(p.script('representation_controls.py')) == a for a in r['cmd'])
                        for r in self.queue()), 'Controller cannot run inside queue (nested deadlock)')
        specs, records = main_fits(), {}
        for spec in specs:
            name = spec['name']
            command = self.frozen_command(name, lambda deadline: fit_command(p, spec, deadline))
            if not self.job(name, 'fit', command, lambda s=spec: validator.fit(s)):
                self.state.update(status='waiting', reason='fit queued/running; no learning-health claim')
                return
            records[name] = self.state['jobs'][name]['validated']
        decision = extension_decision(records)
        decision_path = p.base / 'lower_lr_extension_v1.json'
        if decision_path.exists():
            require(read_json(decision_path) == decision, 'Validation-only extension decision changed')
        else:
            require('extension_decision' not in self.state, 'Extension decision removed')
            exclusive_json(decision_path, decision)
        decision_id = file_identity(decision_path)
        require(self.state.get('extension_decision', decision_id) == decision_id, 'Extension decision identity changed')
        self.state['extension_decision'] = decision_id
        self.save()
        if decision['triggered']:
            for spec in extension_fits():
                specs.append(spec)
                name = spec['name']
                command = self.frozen_command(name, lambda deadline: fit_command(p, spec, deadline))
                if not self.job(name, 'fit', command, lambda s=spec: validator.fit(s)):
                    self.state.update(status='waiting', reason='secondary fit queued/running')
                    return
                records[name] = self.state['jobs'][name]['validated']
        for spec in references():
            specs.append(spec)
            cached = self.state.setdefault('references', {}).get(spec['name'])
            if cached is None:
                cached = validator.fit(spec)  # validation summary only; never original TEST files
                self.state['references'][spec['name']] = cached
            for identity in cached['bindings']:
                verify(identity)
            records[spec['name']] = cached
        frozen = self.freeze(specs, records, gate, decision_id)
        if self.check_control()['allowed_phase'] == 'fits':
            self.state.update(status='waiting', reason='All fits/freeze complete; evaluations explicitly disabled')
            return
        # Default deliberately serial: ALL labels, then ALL panels, then full games.
        for stage, lane in [('scores', 'score'), ('panels', 'panel')]:
            for spec in specs:
                name = f'rc_{stage}_{spec["name"]}'
                command = self.frozen_command(name, lambda deadline: evaluation_command(p, stage, spec, deadline))
                def validate(s=spec):
                    r = records[s['name']]
                    return (validator.score(s, r, self.state['selections'][s['name']]) if stage == 'scores'
                            else validator.panel(s, r, grid))
                if not self.job(name, lane, command, validate):
                    self.state.update(status='waiting', reason=stage + ' queued/running')
                    return
        for arm, _, _ in game_arms():
            selected = frozen['game_arms'][arm]
            cp = records[selected]['checkpoint'] if selected else None
            # Eight independent lanes; no later arm until this arm strictly merges.
            complete = True
            for i in range(8):
                name = f'rc_game_{arm}_{i:02d}'
                command = self.frozen_command(name, lambda deadline: game_command(p, arm, i, cp['path'] if cp else None, deadline))
                # Permit only this arm's already submitted shards to coexist.
                if name not in self.state['jobs']:
                    busy = [r for r in self.queue() if r['status'] in ACTIVE]
                    require(all(r['name'].startswith(f'rc_game_{arm}_') for r in busy), 'Other workload active before game arm')
                    if busy:
                        self._game_submission = True
                try:
                    done = self.job(name, f'game{i}', command, lambda index=i: validator.cohort(arm, cp, index))
                finally:
                    self._game_submission = False
                complete = complete and done
                if done:
                    self.bind_game_common(self.state['jobs'][name]['validated']['common'])
            if not complete:
                self.state.update(status='waiting', reason='Eight shards of one frozen game arm queued/running')
                return
            name = f'rc_merge_{arm}'
            if not self.job(name, 'game0', merge_command(p, arm), lambda: validator.cohort(arm, cp)):
                self.state.update(status='waiting', reason='Strict game merge queued/running')
                return
            self.bind_game_common(self.state['jobs'][name]['validated']['common'])
        # Revalidate all completed evaluation files once before declaring completion.
        if not self.state.get('final_audit') or not self._audited:
            for spec in specs:
                validator.score(spec, records[spec['name']], self.state['selections'][spec['name']])
                validator.panel(spec, records[spec['name']], grid)
            for arm, _, _ in game_arms():
                selected = frozen['game_arms'][arm]
                cp = records[selected]['checkpoint'] if selected else None
                self.bind_game_common(validator.cohort(arm, cp)['common'])
            self.state['final_audit'] = True
            self._audited = True
        self.state.update(status='complete', reason='All required artifacts strictly validated; root analysis remains separate')

    def bind_game_common(self, common):
        require(self.state.get('game_common', common) == common, 'Game source/runtime/native/filter identities differ across arms')
        self.state['game_common'] = common

    def tick(self):
        with self.locked():
            path = self.paths.base / 'controller_state_v1.json'
            self.state = read_json(path) if path.exists() else dict(schema='representation_controls_state_v1', jobs={})
            require(self.state.get('schema') == 'representation_controls_state_v1', 'Unknown state schema')
            self.state.pop('waiting_for', None)
            try:
                self.advance()
            except Waiting as exc:
                self.state.update(status='waiting', reason=str(exc))
            except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                self.state.update(status='blocked', reason=f'{type(exc).__name__}: {exc}')
            self.state['last_tick_utc'] = datetime.fromtimestamp(self.now(), timezone.utc).isoformat()
            self.save()
            return self.state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--tick', action='store_true')
    mode.add_argument('--daemon', action='store_true')
    parser.add_argument('--out', type=Path, help='Required exclusive plan output, valid only with --plan')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=float, default=15, help='Daemon sleep, 0 < seconds <= 30')
    args = parser.parse_args(argv)
    if args.plan:
        if args.out is None:
            parser.error('--plan requires --out')
        identity = exclusive_json(args.out, build_plan(args.root))
        print(canonical(dict(status='planned', plan=identity, launched=False)))
        return 0
    if args.out is not None:
        parser.error('--out is only valid with --plan')
    if not 0 < args.interval <= 30:
        parser.error('--interval must be >0 and <=30 seconds')
    controller = Controller(args.root)
    while True:
        state = controller.tick()
        print(canonical({k: state[k] for k in ('status', 'reason', 'last_tick_utc')}), flush=True)
        if not args.daemon or state['status'] in ('complete', 'blocked'):
            return 0 if state['status'] == 'complete' else (2 if state['status'] == 'blocked' else 0)
        time.sleep(args.interval)
