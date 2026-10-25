"""Fresh supervised actor warm-up only. Raw worlds are immutable inputs.

TEST is evaluated with the validation-selected checkpoint on singleton CPU
inference, or deferred until an external selection receipt is frozen. Deferred
runs never encode or evaluate TEST. The old control matches the champion's
large/none architecture by default and always initializes from scratch.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
import torch

from sts_policy.curriculum import public_hash
from sts_policy.entity_encoding import VOCABULARY_SHA256
from sts_policy.training import save_checkpoint, load_checkpoint, seed_training
from sts_rl.fresh_collection import plurality
from sts_rl.fresh_study import (PROTOCOL, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID, INFERENCE, MIN_DISK,
                                Interrupted, atomic_json, fit_manifest_path, read_json,
                                on_ac, policy_observation, validate_world_range)
from sts_rl.recording import begin_run, capture_rng, file_identity
from sts_rl.teacher_votes import RULES, pooled_edges, pooled_value_target, votes_from_details

SPLIT_VERSION = 'fresh-start-split-v1'
TEACHER_PROTOCOL = 'canonical_rng_resampled_native_v2'
FILTERS = [FILTER_ID, DEAD_FILTER_ID]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def world_split(world):
    bucket = int(hashlib.sha256(f'{SPLIT_VERSION}:{world}'.encode()).hexdigest(), 16) % 10
    return 'TEST' if bucket < 2 else 'VALIDATION' if bucket == 2 else 'FIT'


def training_world(world):
    # Label/rollout namespaces and TRAIN-derived panel worlds; DEV, confirmation
    # and the separate unlabelled 93e9 probe namespace are excluded.
    return 90_000_000_000 <= world < 91_000_000_000 or 92_000_000_000 <= world < 93_000_000_000


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    return int(value)


def _target_options(target_mode='votes', value_scale=100.):
    if target_mode not in ('votes', 'pooled_values'):
        raise ValueError('target_mode must be one of votes, pooled_values')
    if (isinstance(value_scale, bool) or not isinstance(value_scale, (int, float))
            or not math.isfinite(value_scale) or value_scale <= 0):
        raise ValueError('value_scale must be finite and positive')
    if target_mode == 'votes':
        if value_scale != 100.:
            raise ValueError('value_scale requires target_mode=pooled_values')
        return {}
    return {'target_mode': target_mode, 'value_scale': float(value_scale)}


def _target_config(config):
    options = _target_options(config.get('target_mode', 'votes'), config.get('value_scale', 100.))
    return {**{k:v for k,v in config.items() if k not in ('target_mode', 'value_scale')}, **options}


def _teacher_index(key, indices, mapped):
    try:
        index = int(key)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid teacher action index') from exc
    if isinstance(key, bool) or str(index) != str(key) or index not in indices or index in mapped:
        raise ValueError('Illegal/duplicate teacher index')
    return index


def target_distribution(row, *, check_label=True, target_mode='votes', value_scale=100.):
    """Map original possibly gapped display indices into filtered menu positions."""
    _target_options(target_mode, value_scale)
    observation = policy_observation(row['observation'])
    indices = [_integer(a['index'], 'action index') for a in observation['legal_actions']]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Empty or duplicate legal action indices')
    if 'source_action' in row:
        _integer(row['source_action'], 'source action')
    if observation['phase'] == 'combat':
        if row.get('label_kind') != TEACHER_PROTOCOL:
            raise ValueError('Combat requires original native vote targets')
        if target_mode == 'pooled_values':
            if 'search_details' not in row:
                raise ValueError('pooled_values requires search_details on every combat row; collect with --record-search')
            if not isinstance(row['search_details'], list):
                raise ValueError('pooled_values requires search_details to be a list')
        votes = row.get('votes')
        # A deliberately recorded teacher abstention is a raw record, not a target.
        if row.get('label') is None and votes == {}:
            return None, indices
        if not isinstance(votes, dict) or not votes:
            raise ValueError('Empty votes')
        if check_label:
            _integer(row.get('label'), 'teacher label')
        mapped = {}
        for key, value in votes.items():
            index = _teacher_index(key, indices, mapped)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value)!=value:
                raise ValueError('Negative/nonfinite vote count')
            mapped[index] = float(value)
        target = np.array([mapped.get(index, 0.) for index in indices], dtype=np.float64)
        total = float(target.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError('Empty vote mass')
        target /= total
        if check_label and row.get('label') != indices[int(target.argmax())]:
            raise ValueError('Teacher plurality/index mapping mismatch')
        search = row.get('search', {})
        samples = search.get('samples', [])
        if not samples or len(set(samples))!=len(samples) or total != len(samples):
            raise ValueError('Incomplete teacher histogram')
        if target_mode == 'pooled_values':
            mapped = {}
            for key, value in pooled_value_target(row['search_details'], value_scale).items():
                index = _teacher_index(key, indices, mapped)
                mapped[index] = value
            if not mapped:
                raise ValueError('Labelled combat row requires visited search_details for pooled_values')
            target = np.array([mapped.get(index, 0.) for index in indices], dtype=np.float64)
            if not np.isfinite(target).all():
                raise ValueError('Nonfinite pooled value target')
    else:
        _integer(row.get('label'), 'teacher label')
        if row.get('label_kind') != 'champion_hard' or row.get('label') not in indices:
            raise ValueError('OOC requires legal champion hard action')
        if row['label'] != row.get('source_action'):
            raise ValueError('Hard teacher/source action mismatch')
        target = np.zeros(len(indices), dtype=np.float64)
        target[indices.index(row['label'])] = 1.
    return target.astype(np.float32), indices


def _verify_file(identity):
    if not isinstance(identity, dict) or file_identity(identity['path']) != identity:
        raise ValueError('Source file identity mismatch')


def _archived_policy_contract(archive):
    """Check the archived public adapter without executing archived Python."""
    from sts_rl import fresh_study
    old = ast.parse(archive.read('src/sts_rl/fresh_study.py'))
    current = ast.parse(Path(fresh_study.__file__).read_text())
    def contract(tree):
        names = {'PROTOCOL', 'FILTER_ID', 'DEAD_FILTER_ID', 'PUBLIC_CONTEXT_ID', 'INFERENCE', '_OOC_POTION_LABEL'}
        result = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in names):
                result[node.targets[0].id] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.FunctionDef) and node.name in {'policy_observation', 'filter_observation'}:
                result[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.ClassDef) and node.name == 'StudyPolicy':
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name in {'choose_action', 'distribution', '_logits'}:
                        result['StudyPolicy.'+method.name] = ast.dump(method, include_attributes=False)
        return result
    if contract(old) != contract(current):
        raise ValueError('Archived public input-context/filter/inference contract mismatch')


def validate_source(source):
    if source.get('config', {}).get('teacher_plays') is True:
        raise ValueError('Benchmark corpus with teacher_plays true cannot be used for training')
    return _validate_collection_source(source, labelled=True)


def _scientific_collection_config(config):
    return {k:v for k,v in config.items() if k not in {
        'out', 'n', 'world_start', 'workers', 'deadline_utc', 'max_labels'}}


def _validate_collection_source(source, *, labelled):
    if (source.get('protocol') != PROTOCOL or source['policy'].get('protocol') != PROTOCOL or
            source['policy'].get('filter_ids') != FILTERS or
            source['policy'].get('public_context_id') != PUBLIC_CONTEXT_ID):
        raise ValueError('Source/filter protocol mismatch')
    policy = source['policy']
    if source['config'].get('mode') != 'collect' or source['config'].get('teacher_labels') is not labelled:
        raise ValueError('Source must be '+('labelled' if labelled else 'unlabelled probe')+' collection')
    if policy.get('inference') != INFERENCE or policy.get('combat') or policy.get('ooc'):
        raise ValueError('Source must be frozen champion trajectories')
    _verify_file(policy['champion'])
    _verify_file(source['launch'])
    launch = json.loads(Path(source['launch']['path']).read_text())
    _verify_file(launch['source_archive'])
    # Reconstruct the environment interface hash from the immutable source
    # archive, rather than trusting a self-reported row provenance string.
    interface = hashlib.sha256()
    with zipfile.ZipFile(launch['source_archive']['path']) as archive:
        _archived_policy_contract(archive)
        versions = {node.targets[0].id: ast.literal_eval(node.value)
                    for node in ast.parse(archive.read('src/sts_env/env.py')).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {'ENVIRONMENT_VERSION', 'OBSERVATION_VERSION'}}
        names = sorted(n for n in archive.namelist() if n.startswith('src/sts_env/') and n.endswith('.py'))
        if not names or len(set(names)) != len(names):
            raise ValueError('Missing/duplicate archived public interface sources')
        for name in names:
            interface.update(name.removeprefix('src/sts_env/').encode())
            interface.update(b'\0')
            interface.update(archive.read(name))
            interface.update(b'\0')
    launch = {**launch, 'verified_interface_sha256':interface.hexdigest(),
              'verified_environment_version':versions['ENVIRONMENT_VERSION'],
              'verified_observation_version':versions['OBSERVATION_VERSION']}
    if launch.get('parent_checkpoint') != policy['champion']:
        raise ValueError('Teacher checkpoint provenance mismatch')
    if launch.get('effective_config') != source['config']:
        raise ValueError('Collection source config mismatch')
    if ('checkpoint' in source['config'] and
            Path(source['config']['checkpoint']).resolve() != Path(policy['champion']['path']).resolve()):
        raise ValueError('Collection config checkpoint does not match the recorded champion')
    native = launch['native']
    if not native.get('module_sha256') or native.get('preparation_only'):
        raise ValueError('Missing actual collection native identity')
    binary = Path(source['launch']['path']).parent / native['module_file']
    if file_identity(binary)['sha256'] != native['module_sha256']:
        raise ValueError('Native snapshot checksum mismatch')
    # Operational shard bounds may differ; scientific settings must be identical.
    config = _scientific_collection_config(source['config'])
    scientific = {'policy': policy, 'native': native, 'config': config,
                  'source_archive_sha256': launch['source_archive']['sha256']}
    return digest(scientific), launch


def validate_provenance(prov, launch):
    required = ('python_interface_sha256', 'runtime_python_version', 'environment_version',
                'observation_version', 'native_provenance')
    if (any(not prov.get(k) for k in required) or prov['native_provenance'] != launch['native']
            or prov['python_interface_sha256'] != launch['verified_interface_sha256']
            or prov['environment_version'] != launch['verified_environment_version']
            or prov['observation_version'] != launch['verified_observation_version']
            or prov['runtime_python_version'] != launch['runtime']['python'].split()[0]):
        raise ValueError('Missing/mismatched native/public provenance')


def public_surface(source, prov):
    """The compatibility boundary deliberately excludes encoder/archive bytes."""
    return {'provenance': prov, 'policy': {k: source['policy'][k] for k in
            ('protocol', 'filter_ids', 'public_context_id', 'inference')}}


def _mixed_source_contract(source, prov, launch, *, mixed_sampling=False):
    # The champion and archived Python bytes may differ, with each source still
    # validated separately. Search recording adds diagnostics, not different votes.
    config = {k:v for k,v in _scientific_collection_config(source['config']).items()
              if k not in {'checkpoint', 'record_search'}}
    # Match older collectors that omitted these options to explicit no-effect
    # defaults. Normalize only this comparison, preserving source/manifest identities.
    for key, defaults in {'vote_rule': ('best_line',), 'teacher_plays': (False,),
                          'search_options': (None, {}), 'exclude_action_prefixes': (None, []),
                          'teacher_plays_prefixes': (None, [])}.items():
        if key in config and any(type(config[key]) is type(default) and config[key] == default
                                 for default in defaults):
            config.pop(key)
    if mixed_sampling:
        # Sampling selects states to label; it does not change their teacher targets.
        config.pop('teacher_probabilities', None)
    return {'native': launch['native'], 'config': config,
            'provenance': {k:v for k,v in prov.items() if k!='python_interface_sha256'},
            'policy': {k:v for k,v in source['policy'].items() if k!='champion'}}


def _require_mixed_match(actual, expected, field=''):
    if digest(actual) == digest(expected):
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(actual.keys() | expected.keys()):
            name = field+'.'+key if field else key
            if key not in actual or key not in expected:
                raise ValueError('Incompatible mixed source field: '+name)
            _require_mixed_match(actual[key], expected[key], name)
    raise ValueError('Incompatible mixed source field: '+field)


def shard_files(sources):
    paths = []
    for source in sources:
        path = Path(source)
        paths.extend(sorted((path/'worlds').glob('*.json.gz')) if path.is_dir() else [path])
    if not paths or len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError('Missing/duplicate source shards')
    return paths


def _relabel_combat_row(row, rule):
    if row['observation']['phase'] != 'combat':
        return row
    if 'search_details' not in row:
        raise ValueError('Combat relabelling requires search_details on every combat row; collect with --record-search')
    details = row['search_details']
    if not isinstance(details, list):
        raise ValueError('Combat relabelling requires search_details to be a list')
    if not details:
        if row.get('label') is not None or row.get('votes') != {}:
            raise ValueError('Labelled combat row requires non-empty search_details for relabelling')
        return row  # Preserve recorded abstentions and their exclusion from targets.
    votes = votes_from_details(details, rule)
    return {**row, 'votes': votes, 'label': plurality(votes, row['observation']), 'vote_rule': rule}


def iter_rows(sources, *, control=None, mixed_sources=False, mixed_sampling=False, relabel_vote_rule=None,
              target_mode='votes', value_scale=100.):
    """Accept C1 world JSON objects and JSONL exports; hold at most one world."""
    if mixed_sampling and not mixed_sources:
        raise ValueError('mixed_sampling requires mixed_sources')
    if relabel_vote_rule is not None and relabel_vote_rule not in RULES:
        raise ValueError('relabel_vote_rule must be one of '+', '.join(RULES))
    _target_options(target_mode, value_scale)
    if relabel_vote_rule is not None and target_mode == 'pooled_values':
        raise ValueError('relabel_vote_rule cannot be combined with target_mode=pooled_values')
    verified, worlds = {}, set()
    scientific, expected = None, None
    for path in shard_files(sources):
        if control:
            control.check()
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            for line in stream:
                if not line.strip():
                    continue
                shard = json.loads(line)
                nested = 'labels' in shard
                if nested:
                    if shard.get('source', {}).get('config', {}).get('teacher_plays') is True:
                        raise ValueError('Benchmark corpus with teacher_plays true cannot be used for training')
                    if shard.get('protocol') != PROTOCOL or shard.get('status') != 'completed':
                        raise ValueError('Nonterminal/faulted source world')
                    world = _integer(shard['world_seed'], 'world')
                    validate_world_range(world, 1)
                    if not training_world(world):
                        raise ValueError('Training data requires a fresh TRAIN namespace')
                    if world in worlds:
                        raise ValueError('Duplicate source world')
                    worlds.add(world)
                    rows = shard['labels']
                    if 'sampling' in shard:
                        # Collector counters distinguish informative labels from saved raw rows.
                        for suffix, predicate in (
                                (':labelled', lambda r: r.get('label_kind')==TEACHER_PROTOCOL and r.get('label') is not None),
                                (':abstain', lambda r: r.get('label_kind')==TEACHER_PROTOCOL and r.get('label') is None)):
                            recorded = sum(v for k,v in shard['sampling'].items() if k.endswith(suffix))
                            if recorded != sum(predicate(r) for r in rows):
                                raise ValueError('Recorded informative/abstention counts mismatch')
                else:
                    rows = [shard]
                for position, row in enumerate(rows):
                    if row.get('use_for_training') is False:
                        raise ValueError('Sample with use_for_training false cannot be used for training')
                    if control and position % 128 == 0:
                        control.check()
                    world = _integer(row['world_seed'], 'world')
                    validate_world_range(world, 1)
                    if not training_world(world):
                        raise ValueError('Training data requires a fresh TRAIN namespace')
                    source = row['source']
                    key = digest(source)
                    if key not in verified:
                        verified[key] = validate_source(source)
                    identity, launch = verified[key]
                    prov = row['expected_provenance']
                    validate_provenance(prov, launch)
                    if mixed_sources:
                        contract = _mixed_source_contract(source, prov, launch, mixed_sampling=mixed_sampling)
                        if expected is None:
                            expected = contract
                        _require_mixed_match(contract, expected)
                    else:
                        if scientific is None:
                            scientific, expected = identity, prov
                        if identity != scientific or expected != prov:
                            raise ValueError('Incompatible scientific source/provenance identities')
                    if nested:
                        n = len(row['prefix'])
                        if (world != shard['world_seed'] or row['source'] != shard['source'] or
                                prov != shard['expected_provenance'] or row['prefix'] != shard['action_prefix'][:n]
                                or n >= len(shard['action_prefix']) or row['source_action'] != shard['action_prefix'][n]):
                            raise ValueError('Source trajectory/prefix mismatch')
                    obs = row['observation']
                    if public_hash(obs) != row['public_hash']:
                        raise ValueError('Public hash mismatch')
                    if row['source_action'] not in [a['index'] for a in policy_observation(obs)['legal_actions']]:
                        raise ValueError('Source action outside legal menu')
                    if any(isinstance(i,bool) or not isinstance(i,int) for i in row['prefix']):
                        raise ValueError('Invalid source action prefix')
                    state = obs['state']
                    group = f"{str(state.get('room', '')).rsplit('.',1)[-1]}{int(state.get('act',0))}"
                    if row['group'] != group:
                        raise ValueError('Act/room group mismatch')
                    if relabel_vote_rule is not None:
                        row = _relabel_combat_row(row, relabel_vote_rule)
                    target, indices = target_distribution(row, target_mode=target_mode, value_scale=value_scale)
                    row_id = digest([world, len(row['prefix']), row['public_hash']])
                    yield row, {'source_row_id': row_id, 'world_seed': world, 'group': group,
                        'phase': 'combat' if obs['phase'] == 'combat' else 'ooc',
                        'public_hash': row['public_hash'], 'target_sha256': digest(target.tolist()) if target is not None else None,
                        'has_target': target is not None, 'abstention': target is None,
                        'choice': len(indices)>1, 'confident': target is not None and float(target.max())>=.75,
                        'shard': str(path.resolve()), 'source_identity': identity,
                        'split': world_split(world)}


def interleave_fit(rows):
    """Full-mix quota deficits across groups; hashed world round-robin within.

    At prefix k, choose the nonempty group most behind k * full_fraction.
    Integer numerators avoid floating-point ties; group names break exact ties.
    Only the supplied FIT rows determine quotas and the nested prefix order.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['group']][row['world_seed']].append(row)
    queues = {}
    counts = {g: sum(map(len, worlds.values())) for g, worlds in groups.items()}
    total, used = sum(counts.values()), Counter()
    for group, worlds in sorted(groups.items()):
        queues[group] = deque(deque(sorted(v, key=lambda r: digest(['fit-order-v1',r['source_row_id']])))
                              for _,v in sorted(worlds.items(), key=lambda p: digest(['world-order-v1',p[0]])))
    result = []
    while queues:
        k = len(result) + 1
        group = min(queues, key=lambda g: (-(k * counts[g] - used[g] * total), g))
        worlds = queues[group]
        shard = worlds.popleft()
        result.append(shard.popleft()['source_row_id'])
        used[group] += 1
        if shard:
            worlds.append(shard)
        if not worlds:
            del queues[group]
    return result


def freeze_manifest(sources, *, control=None, mixed_sources=False, mixed_sampling=False, relabel_vote_rule=None,
                    target_mode='votes', value_scale=100.):
    files = [file_identity(p) for p in shard_files(sources)]
    rows, seen, collection_sources = [], set(), {}
    interfaces = set()
    for raw, meta in iter_rows(sources, control=control, mixed_sources=mixed_sources, mixed_sampling=mixed_sampling,
                               relabel_vote_rule=relabel_vote_rule, target_mode=target_mode, value_scale=value_scale):
        if meta['source_row_id'] in seen:
            raise ValueError('Duplicate source row identity')
        seen.add(meta['source_row_id'])
        rows.append(meta)
        collection_sources[digest(raw['source'])] = raw['source']
        if mixed_sources:
            interfaces.add(raw['expected_provenance']['python_interface_sha256'])
    rows.sort(key=lambda r: (r['world_seed'], r['source_row_id']))
    splits = {s: {'worlds': sorted({r['world_seed'] for r in rows if r['split']==s}),
                  'row_ids': [r['source_row_id'] for r in rows if r['split']==s and r['has_target']]}
              for s in ('FIT','VALIDATION','TEST')}
    fit = {phase: interleave_fit([r for r in rows if r['split']=='FIT' and r['phase']==phase and r['has_target']])
           for phase in ('combat','ooc')}
    subsets = {p: {'5000': ids[:5000], '20000': ids[:20000], 'full': ids} for p, ids in fit.items()}
    row_groups = {r['source_row_id']: r['group'] for r in rows}
    subset_counts = {p: {size: dict(sorted(Counter(row_groups[i] for i in ids).items()))
                         for size, ids in sizes.items()} for p, sizes in subsets.items()}
    counts = Counter()
    for r in rows:
        counts['raw_rows'] += 1
        counts[r['phase']+'_raw_rows'] += 1
        counts['target_rows'] += int(r['has_target'])
        counts['combat_informative_labels'] += int(r['phase']=='combat' and r['has_target'] and r['choice'])
        counts['combat_abstentions'] += int(r['abstention'])
        counts['ooc_choice_targets'] += int(r['phase']=='ooc' and r['choice'])
        counts['ooc_forced_targets'] += int(r['phase']=='ooc' and not r['choice'])
    result = {'schema': 'fresh_data_manifest_v2', 'split_version': SPLIT_VERSION,
        'filters': FILTERS, 'source_files': files, 'rows': rows, 'splits': splits,
        'collection_sources':collection_sources,
        'public_surface': public_surface(raw['source'], raw['expected_provenance']) if rows else None,
        'counts': dict(counts),
        'fit_order': fit, 'subsets': subsets,
        'fit_order_method': 'full_group_quota_deficit_hashed_world_round_robin_v1',
        'fit_subset_group_counts': subset_counts,
        'collected_total': len(rows), 'collected_worlds': len({r['world_seed'] for r in rows})}
    if mixed_sources:
        result['mixed_sources'] = True
        if rows:
            result['public_surface']['provenance'] = {**raw['expected_provenance'],
                'python_interface_sha256': sorted(interfaces)}
    if mixed_sampling:
        result['mixed_sampling'] = True
        result['teacher_probability_profiles'] = sorted({
            json.dumps(s['config'].get('teacher_probabilities'), sort_keys=True, separators=(',', ':'), allow_nan=False)
            for s in collection_sources.values()})
    if relabel_vote_rule is not None:
        result['relabel_vote_rule'] = relabel_vote_rule
    result.update(_target_options(target_mode, value_scale))
    result['data_identity'] = digest(result)
    if files != [file_identity(p) for p in shard_files(sources)]:
        raise ValueError('Data changed during manifest freeze')
    return result


@dataclass
class Example:
    encoded: object
    target: np.ndarray
    meta: dict
    value_regrets: np.ndarray | None = None
    pooled_position: int | None = None


def value_diagnostics(details, indices):
    means = {}
    for edge in pooled_edges(details):
        index = _teacher_index(edge['index'], indices, means)
        if edge['visits'] > 0:
            means[index] = edge['mean_evaluation']
    if not means:
        raise ValueError('Combat search_details have no visited edges')
    winner = next(iter(votes_from_details(details, 'pooled_mean')))
    best, worst = means[winner], min(means.values())
    regrets = np.array([best - means.get(index, worst) for index in indices], dtype=np.float64)
    if not np.isfinite(regrets).all():
        raise ValueError('Nonfinite pooled value regret')
    return regrets, indices.index(winner)


def observation_encoder(architecture):
    """Explicit dispatch; the archived public adapter remains unchanged."""
    from sts_policy import entity_encoding, encoding
    if architecture == 'controlled':
        from sts_policy import controlled_encoding
        return controlled_encoding
    if architecture == 'new':
        return entity_encoding
    if architecture == 'old':
        return encoding
    raise ValueError('architecture must be new, old or controlled')


def preencode(sources, manifest, phase, architecture, fit_size='full', *, control=None,
              splits=('FIT','VALIDATION','TEST')):
    encoder = observation_encoder(architecture)
    if fit_size not in ('5000','20000','full'):
        raise ValueError('Unknown nested FIT subset')
    if not splits or len(set(splits)) != len(splits) or set(splits)-{'FIT','VALIDATION','TEST'}:
        raise ValueError('Invalid encoding splits')
    fit_ids = set(manifest['subsets'][phase][fit_size])
    wanted = {r['source_row_id']: r for r in manifest['rows'] if r['split'] in splits and
              r['phase']==phase and r['has_target'] and
              (r['split']!='FIT' or r['source_row_id'] in fit_ids)}
    result = {s: [] for s in splits}
    options = _target_options(manifest.get('target_mode', 'votes'), manifest.get('value_scale', 100.))
    for row, meta in iter_rows(sources, control=control, mixed_sources=manifest.get('mixed_sources',False),
                              mixed_sampling=manifest.get('mixed_sampling',False),
                              relabel_vote_rule=manifest.get('relabel_vote_rule'), **options):
        if meta['source_row_id'] not in wanted:
            continue
        if meta != wanted.pop(meta['source_row_id']):
            raise ValueError('Dataset no longer matches frozen manifest')
        target, indices = target_distribution(row, **options)
        # The encoders see ONLY the full filtered public observation.
        encoded = encoder.encode_observation(policy_observation(row['observation']))
        if encoded.action_indices.tolist() != indices:
            raise ValueError('Encoder action/menu alignment failure')
        diagnostics = (value_diagnostics(row['search_details'], indices)
                       if phase == 'combat' and 'search_details' in row else (None, None))
        result[meta['split']].append(Example(encoded, target, meta, *diagnostics))
    if wanted:
        raise ValueError('Frozen rows missing from sources')
    if manifest['source_files'] != [file_identity(p) for p in shard_files(sources)]:
        raise ValueError('Source shards changed since manifest freeze')
    if not all(result.values()):
        raise ValueError('Each requested phase/split needs nonempty target worlds')
    return result


def soft_cross_entropy(logits, targets, mask):
    """Finite masked CE, including zero target mass at padded -inf logits."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1).masked_fill(~mask, 0.)
    return -(targets*logp).sum(-1)


def make_model(architecture, phase, *, width=128, layers=2, heads=4,
               old_size='large', old_context='none', legacy_action_features=False):
    """Initialize independent weights; old defaults match champion architecture.

    This is an architecture-matched old control, not a parameter-count match to
    the entity transformer. No checkpoint or teacher weights are accepted.
    """
    if type(legacy_action_features) is not bool:
        raise ValueError('legacy_action_features must be boolean')
    if architecture != 'controlled' and legacy_action_features:
        raise ValueError('legacy_action_features requires controlled architecture')
    if architecture == 'controlled':
        from sts_policy.controlled_entity import ControlledEntityActorCritic
        model = ControlledEntityActorCritic(width=width, layers=layers, heads=heads, dropout=0., phase=phase,
                                           legacy_action_features=legacy_action_features)
    elif architecture == 'new':
        from sts_policy.entity_model import EntityActorCritic
        model = EntityActorCritic(width=width, layers=layers, heads=heads, dropout=0., phase=phase)
    elif architecture == 'old':
        from sts_policy.model import CandidateActorCritic
        model = CandidateActorCritic(size=old_size, context=old_context)
    else:
        raise ValueError('architecture must be new, old or controlled')
    for p in model.critic.parameters():
        p.requires_grad_(False)
    return model


def collate(examples, architecture, device):
    module = observation_encoder(architecture)
    batch = module.collate_observations([x.encoded for x in examples], device=device)
    mask = batch[2] if architecture == 'old' else batch['action_mask']
    targets = np.zeros(tuple(mask.shape), np.float32)
    for i,x in enumerate(examples):
        targets[i,:len(x.target)] = x.target
    return batch, torch.as_tensor(targets, device=device), mask


def actor_logits(model, batch, architecture, *, singleton=False):
    if architecture in ('new', 'controlled'):
        return model.actor_logits(batch) if singleton else model.training_logits(batch)
    if architecture == 'old':
        return model.actor_logits(*batch)
    raise ValueError('architecture must be new, old or controlled')


def actor_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def rss_tree():
    """Current total RSS, including any descendants (trainer uses no workers)."""
    try:
        result = subprocess.run(['ps','-eo','pid=,ppid=,rss='], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None  # Sandboxed hosts may prohibit process-table reads.
    table = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]
    descendants = {os.getpid()}
    while True:
        expanded = descendants | {pid for pid,parent,_ in table if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    return sum(rss*1024 for pid,_,rss in table if pid in descendants)


def memory_stats():
    current = rss_tree()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'rss_bytes':current, 'rss_scope':'process_tree' if current is not None else 'unavailable',
            'rss_peak_self_bytes':peak if sys.platform=='darwin' else peak*1024,
            'workers':0}


class TrainingControl:
    """Cached power queries; STOP/deadline/disk checked at every boundary."""
    def __init__(self, out, deadline_utc=None, *, ac=on_ac, now=time.time, monotonic=time.monotonic,
                 sleep=time.sleep, disk=shutil.disk_usage, caffeinate=True):
        self.out, self.ac, self.now, self.mono = Path(out), ac, now, monotonic
        self.sleep, self.disk = sleep, disk
        self.deadline = None
        self.deadline_utc = deadline_utc
        if deadline_utc:
            stamp = datetime.fromisoformat(deadline_utc.replace('Z','+00:00'))
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds()!=0:
                raise ValueError('Deadline requires explicit UTC')
            self.deadline = stamp.timestamp()
        self.lock = threading.RLock()
        self.last = monotonic()
        self.power_checked = -math.inf
        self.powered, self.active = False, 0.
        self.inhibitor = None
        self.caffeinate = caffeinate and sys.platform=='darwin'
        self.status = {'phase':'preparing','step':0,'examples':0}
        self.started = self.last
        self.closed = threading.Event()
        self.error = None

    def stats(self):
        with self.lock:
            current = self.mono()
            previous = self.powered
            if current-self.power_checked >= 2:
                self.powered = bool(self.ac())
                self.power_checked = current
            if previous and self.powered:
                self.active += current-self.last
            self.last = current
            if self.caffeinate and self.powered and self.inhibitor is None:
                self.inhibitor = subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
            if not self.powered and self.inhibitor is not None:
                self.inhibitor.terminate()
                self.inhibitor.wait(timeout=5)
                self.inhibitor = None
            return {'active_ac_seconds': self.active, 'on_ac':self.powered,
                    'elapsed_seconds':current-self.started, 'free_disk_bytes':self.disk(self.out).free}

    def check(self):
        while True:
            if self.error:
                raise RuntimeError('Heartbeat failed') from self.error
            if (self.out/'STOP').exists():
                raise Interrupted('STOP')
            if self.deadline is not None and self.now()>=self.deadline:
                raise Interrupted('deadline')
            stats = self.stats()
            if stats['free_disk_bytes'] < MIN_DISK:
                raise Interrupted('disk_below_6_GiB')
            if stats['on_ac']:
                return
            self.status['status'] = 'paused_on_battery'
            self.sleep(1)

    def emit(self):
        atomic_json(self.out/'status.json', {**self.status, **self.stats(),
            'utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': self.deadline_utc,
            **memory_stats(), 'filters':FILTERS})

    def start(self):
        def heartbeat():
            while not self.closed.is_set():
                try:
                    self.emit()
                except Exception as exc:
                    self.error = exc
                    return
                self.closed.wait(10)
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if hasattr(self,'thread'):
            self.thread.join(timeout=6)
        if self.inhibitor is not None:
            self.inhibitor.terminate()
            self.inhibitor.wait(timeout=5)
            self.inhibitor = None


def summarize_predictions(examples, losses, probabilities, positions):
    groups = defaultdict(list)
    for x, loss, p, position in zip(examples, losses, probabilities, positions):
        correct = int(position==np.argmax(x.target))
        values = ((float(x.value_regrets[position]), int(position==x.pooled_position))
                  if x.value_regrets is not None else None)
        entry = (float(loss),correct,float(-(p*np.log(np.maximum(p,1e-30))).sum()),float(max(p)),values)
        groups['all_rows'].append(entry)
        if x.meta['choice']:
            for key in ('choice','group:'+x.meta['group']):
                groups[key].append(entry)
            if x.meta['confident']:
                groups['high_confidence_choice'].append(entry)
    stats = {key: dict(rows=len(v), soft_ce=float(np.mean([r[0] for r in v])),
                      agreement=float(np.mean([r[1] for r in v])),
                      entropy=float(np.mean([r[2] for r in v])), top_probability=float(np.mean([r[3] for r in v])))
             for key,v in groups.items()}
    for key, rows in groups.items():
        values = [r[4] for r in rows if r[4] is not None]
        if values:
            stats[key].update(value_rows=len(values), value_regret=float(np.mean([v[0] for v in values])),
                              pooled_agreement=float(np.mean([v[1] for v in values])))
    return {'loss':stats['all_rows']['soft_ce'], 'groups':stats,
            'primary_agreement':stats.get('choice',{}).get('agreement')}


@torch.no_grad()
def evaluate(model, examples, architecture, *, device='cpu', batch_size=128, singleton=False, control=None,
             predictions_out=None):
    model.eval()
    losses, probabilities, positions = [], [], []
    for offset in range(0,len(examples),1 if singleton else batch_size):
        if control:
            control.check()
        chosen = examples[offset:offset+(1 if singleton else batch_size)]
        batch, targets, mask = collate(chosen,architecture,device)
        logits = actor_logits(model,batch,architecture,singleton=singleton)
        if not torch.isfinite(logits[mask]).all():
            raise ValueError('Nonfinite inference logits')
        losses.extend(soft_cross_entropy(logits,targets,mask).cpu().tolist())
        positions.extend(logits.masked_fill(~mask,-torch.inf).argmax(-1).cpu().tolist())
        prob = logits.masked_fill(~mask,-torch.inf).softmax(-1).cpu().numpy()
        probabilities.extend(p[:len(x.target)].copy() for x,p in zip(chosen,prob))
    if predictions_out is not None:
        predictions_out.extend(positions)
    return summarize_predictions(examples,losses,probabilities,positions), losses, probabilities


def movement(current, previous, start):
    def kl(a,b):
        return float(np.mean([np.sum(p*(np.log(np.maximum(p,1e-30))-np.log(np.maximum(q,1e-30)))) for p,q in zip(a,b)]))
    return {'label':'supervised_update_block_movement', 'kl_previous_to_current':kl(previous,current),
            'kl_start_to_current':kl(start,current),
            'argmax_change_previous':float(np.mean([np.argmax(a)!=np.argmax(b) for a,b in zip(previous,current)])),
            'argmax_change_start':float(np.mean([np.argmax(a)!=np.argmax(b) for a,b in zip(start,current)]))}


def coverage(examples):
    counts = Counter()
    for x in examples:
        for key,value in x.encoded.coverage.items():
            if isinstance(value,dict):
                for k,v in value.items():
                    if isinstance(v,(int,float)):
                        counts[key+':'+k]+=v
            elif isinstance(value,(int,float)):
                counts[key]+=value
    return dict(counts)


def minibatch_order(examples, architecture, batch_size, rng):
    """Bucket both padded axes without dropping or duplicating any example."""
    buckets = defaultdict(list)
    for i, x in enumerate(examples):
        entities = len(x.encoded.entity_ids) if architecture in ('new', 'controlled') else 1
        actions = len(x.encoded.action_indices)
        buckets[((max(1, entities)-1).bit_length(), (max(1, actions)-1).bit_length())].append(i)
    batches = []
    for key in sorted(buckets):
        indices = buckets[key]
        rng.shuffle(indices)
        batches.extend(indices[i:i+batch_size] for i in range(0, len(indices), batch_size))
    rng.shuffle(batches)
    return batches


def fit(model, dataset, config, out, manifest, receipt, control, *, probes=None):
    """Actual minibatch AdamW updates. Fresh-only, no implicit resume."""
    config = {**config, 'old_size':config.get('old_size','large'),
              'old_context':config.get('old_context','none'), 'defer_test':config.get('defer_test',False)}
    config = _target_config(config)
    if config.get('relabel_vote_rule') != manifest.get('relabel_vote_rule'):
        raise ValueError('Fit config/manifest relabel_vote_rule mismatch')
    for key, default in (('target_mode', 'votes'), ('value_scale', 100.)):
        if config.get(key, default) != manifest.get(key, default):
            raise ValueError('Fit config/manifest '+key+' mismatch')
    architecture, device = config['architecture'], config['device']
    if architecture == 'controlled':
        if config['defer_test'] is not True:
            raise ValueError('Controlled fitting requires defer_test=True; freeze validation-only selection before TEST')
        config.setdefault('legacy_action_features', False)
        for key in ('phase', 'width', 'layers', 'heads', 'legacy_action_features'):
            actual = getattr(model.config, key)
            config.setdefault(key, actual)
            if type(config[key]) is not type(actual) or config[key] != actual:
                raise ValueError('Controlled model/config mismatch: ' + key)
    critic_status = ('head_frozen_shared_actor_features_trained' if architecture=='old'
                     else 'frozen_untrained')
    if architecture=='old' and (model.config.size != config['old_size'] or model.config.context != config['old_context']):
        raise ValueError('Old control model/config architecture mismatch')
    manifest_fit, validation = dataset['FIT'], dataset['VALIDATION']
    # Forced targets stay in the manifest and diagnostics/held-out evaluation.
    # They must never consume an AdamW step (including momentum/weight decay).
    fit_rows = [x for x in manifest_fit if len(x.target) > 1]
    row_counts = {'manifest_fit_rows':len(manifest_fit),
                  'optimization_fit_choice_rows':len(fit_rows),
                  'forced_fit_rows_excluded':len(manifest_fit)-len(fit_rows)}
    if not fit_rows:
        raise ValueError('No optimization FIT choice rows: all FIT targets are forced single-action rows')
    control.status.update(row_counts)
    model.to(device)
    parameters = actor_parameters(model)
    optimizer = torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=.01)
    rng = random.Random(config['seed'])
    encoded_coverage = {'FIT':coverage(manifest_fit),'VALIDATION':coverage(validation)}
    diagnostics = sorted(manifest_fit,key=lambda x:x.meta['source_row_id'])[:128]
    _,_,start_probs = evaluate(model,diagnostics,architecture,device=device,control=control)
    previous = start_probs
    probe_monitor = None
    if probes:
        from sts_rl.fresh_probes import ProbeMonitor
        probe_monitor = ProbeMonitor(model, probes, architecture, control)
    initial = [p.detach().cpu().clone() for p in parameters]
    norm_start = math.sqrt(sum(float(p.square().sum()) for p in initial))
    best_loss, best_step, stale = math.inf,0,0
    examples, block_losses, block_grads, history, order = 0,[],[],[],[]
    batch_size = config['batch'] or 128

    def save(name, step):
        metadata = model.metadata()
        if architecture=='old':
            metadata.update(encoder_version='public_candidate_v2',phase=config['phase'],
                            old_size=config['old_size'],old_context=config['old_context'])
        metadata.update(data_identity=manifest['data_identity'], vocabulary_sha256=VOCABULARY_SHA256,
            source_receipt=receipt, config=config, filters=FILTERS, initialization='from_scratch',
            parent_checkpoint=None, probe_selection=receipt.get('probe_selection'),
            optimization_rows=row_counts, loss_normalization='actual_choice_batch_mean',
            critic_status=critic_status, explained_variance='not_applicable',
            training_state={'best_loss':None if not math.isfinite(best_loss) else best_loss,
                'best_step':best_step,'stale':stale,'examples':examples,**row_counts,'rng_state':capture_rng(rng),
                'remaining_minibatch_order':list(order)})
        save_checkpoint(out/name,model,model_config=model.metadata(),optimizer=optimizer,step=step,metadata=metadata)

    completed = 0
    try:
        for step in range(1,config['steps']+1):
            control.check()
            if not order:
                # Every bucket contains actual choice targets only.
                order = minibatch_order(fit_rows, architecture, batch_size, rng)
            chosen = [fit_rows[i] for i in order.pop()]
            batch,targets,mask = collate(chosen,architecture,device)
            model.train()
            logits = actor_logits(model,batch,architecture)
            if not torch.isfinite(logits[mask]).all():
                raise ValueError('Nonfinite training logits')
            loss = soft_cross_entropy(logits,targets,mask).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            completed = step
            examples += len(chosen)
            block_losses.append(float(loss.detach().cpu()))
            block_grads.append(float(grad.detach().cpu()))
            control.status.update(phase=config['phase'],status='supervised_training',step=step,examples=examples)
            if step % config['eval_every'] and step != config['steps']:
                continue
            fit_metrics,_,current = evaluate(model,diagnostics,architecture,device=device,control=control)
            val_metrics,_,_ = evaluate(model,validation,architecture,device=device,batch_size=batch_size,control=control)
            delta = math.sqrt(sum(float((p.detach().cpu()-q).square().sum()) for p,q in zip(parameters,initial)))/max(norm_start,1e-30)
            event = {'step':step,'examples':examples, 'fit_update_block_loss':statistics_mean(block_losses),
                **row_counts, 'loss_normalization':'actual_choice_batch_mean',
                'fit_diagnostic':fit_metrics,'validation':val_metrics,'gradient_norm':statistics_mean(block_grads),
                'parameter_relative_delta':delta,'movement':{**movement(current,previous,start_probs),
                    'label':'fixed FIT diagnostic movement; not 93e9 probes'},
                'encoder_coverage':encoded_coverage,
                'diagnostic_source':'fixed FIT subset; supervised diagnostics only',
                'explained_variance':'not_applicable','critic_status':critic_status,
                'evaluation_mode':'supervised_validation_dropout_disabled',**control.stats(),**memory_stats()}
            if probe_monitor:
                event['fixed93e9_probes'] = {**probe_monitor.measure(model),
                    'selection_identity':receipt['probe_selection']['selection_identity']}
            event['examples_per_active_second'] = examples/max(event['active_ac_seconds'],1e-9)
            previous = current
            block_losses.clear()
            block_grads.clear()
            history.append(event)
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(event,allow_nan=False)+'\n')
            if val_metrics['loss'] < best_loss:
                best_loss,best_step,stale = val_metrics['loss'],step,0
                save('best.pt',step)
            elif step >= config['min_steps']:
                stale += 1
            save('latest.pt',step)
            control.status.update(validation_loss=val_metrics['loss'],best_step=best_step)
            if step>=config['min_steps'] and stale>=config['patience']:
                break
    except Interrupted as exc:
        save('latest.pt',completed)
        return {'status':'interrupted','reason':str(exc),'updates':completed,'examples':examples,
                **row_counts,'test_evaluated':False}
    summary = {'status':'completed','updates':completed,'examples':examples,'best_step':best_step,
               'best_validation_loss':best_loss,'actual_fit_rows':len(manifest_fit),**row_counts,
               'loss_normalization':'actual_choice_batch_mean',
               'validation_rows':len(validation),'selected_checkpoint':file_identity(out/'best.pt'),
               'test_deferred':config['defer_test'],'test_evaluated':False}
    if config['defer_test']:
        return summary
    summary.update(score_test_checkpoint(out/'best.pt',dataset['TEST'],architecture,out,control=control))
    return summary


def checkpoint_training_condition(payload):
    """Scientific settings from the selected checkpoint, never a row-side claim."""
    config = payload['metadata']['config']
    architecture = config['architecture']
    if architecture not in ('new','old','controlled') or config['phase'] not in ('combat','ooc'):
        raise ValueError('Invalid checkpoint architecture/phase')
    condition = {k:config[k] for k in ('phase','lr','seed','steps','min_steps','eval_every','patience','device')}
    condition.update(fit_size=config.get('fit_size','full'), batch=config.get('batch') or 128,
                     defer_test=config.get('defer_test',False))
    if config.get('relabel_vote_rule') is not None:
        condition['relabel_vote_rule'] = config['relabel_vote_rule']
    condition.update(_target_options(config.get('target_mode', 'votes'), config.get('value_scale', 100.)))
    if architecture in ('new','controlled'):
        actual = payload['metadata']['model_config']
        if actual['phase'] != config['phase']:
            raise ValueError('Checkpoint entity phase/config mismatch')
        for k in ('width','layers','heads'):
            if config.get(k,actual[k]) != actual[k]:
                raise ValueError('Checkpoint entity architecture/config mismatch')
            condition[k] = actual[k]
        if architecture == 'controlled':
            flag = actual['legacy_action_features']
            if (payload['metadata'].get('kind') != 'entity_controlled_v1' or type(flag) is not bool
                    or type(config.get('legacy_action_features')) is not bool
                    or config['legacy_action_features'] != flag):
                raise ValueError('Checkpoint controlled legacy_action_features/config mismatch')
            condition['legacy_action_features'] = flag
    else:
        actual = payload['model_config']
        for k, field in (('old_size','size'),('old_context','context')):
            if config.get(k,actual[field]) != actual[field]:
                raise ValueError('Checkpoint old control architecture/config mismatch')
            condition[k] = actual[field]
    return condition


def score_test_checkpoint(checkpoint, examples, architecture, out, *, control=None):
    """Actor-only CPU singleton TEST pass; no optimizer, backward or weight save."""
    if architecture=='controlled':
        from sts_policy.controlled_entity import policy_from_payload
    elif architecture=='new':
        from sts_policy.entity_model import policy_from_payload
    elif architecture=='old':
        from sts_policy.model_agent import policy_from_payload
    else:
        raise ValueError('architecture must be new, old or controlled')
    payload = load_checkpoint(checkpoint)
    if payload['metadata']['config']['architecture'] != architecture:
        raise ValueError('Selected checkpoint architecture mismatch')
    condition = checkpoint_training_condition(payload)
    data_identity = payload['metadata']['data_identity']
    if not data_identity or any(x.meta.get('split') != 'TEST' for x in examples):
        raise ValueError('Final scoring requires manifest identity and TEST rows')
    selected = policy_from_payload(payload)
    selected.to('cpu').eval()
    selected.requires_grad_(False)
    if control:
        control.status.update(status='greedy_selected_TEST')
    predictions = []
    test_metrics,losses,_ = evaluate(selected,examples,architecture,singleton=True,control=control,
                                    predictions_out=predictions)
    selected_identity = file_identity(checkpoint)['sha256']
    # Correctness uses the exact logits from the singleton pass above.
    with (Path(out)/'test_rows.jsonl').open('x') as stream:
        for x,loss,position in zip(examples,losses,predictions):
            if control:
                control.check()
            row = {**x.meta,'seed':x.meta['world_seed'],'correct':position==int(x.target.argmax()),
                   'ce':loss,'prediction_index':int(x.encoded.action_indices[position]),'target':x.target.tolist(),
                   'data_identity':data_identity, 'inference':INFERENCE,'selected_checkpoint':selected_identity,
                   'architecture':architecture,'training_condition':condition}
            if x.value_regrets is not None:
                row.update(value_regret=float(x.value_regrets[position]),
                           pooled_agreement=bool(position==x.pooled_position))
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    return {'test':test_metrics,'test_evaluated':True,'test_protocol':INFERENCE,
            'test_device':'cpu','test_rows':len(examples)}


def statistics_mean(values):
    return sum(values)/len(values) if values else None


def validate_training_config(config, *, synthetic=False):
    if 'preencoded_cache' in config:
        from sts_rl.controlled_cache import validate_optin
        validate_optin(config)
    if type(config.get('legacy_action_features',False)) is not bool:
        raise ValueError('legacy_action_features must be boolean')
    if config.get('architecture') != 'controlled' and config.get('legacy_action_features',False):
        raise ValueError('legacy_action_features requires controlled architecture')
    if config.get('architecture') == 'controlled' and not synthetic and config.get('defer_test') is not True:
        raise ValueError('Production controlled training requires --defer-test; freeze validation-only selection before TEST')
    if config.get('phase')=='ooc' and config.get('fit_size','full')!='full':
        raise ValueError('OOC warm-up is full-only; FIT prefixes apply to informative combat targets')
    if config.get('old_size','large') not in ('small','medium','large') or config.get('old_context','none') not in ('none','mean'):
        raise ValueError('Invalid old control size/context')
    if not isinstance(config.get('defer_test',False),bool):
        raise ValueError('defer_test must be boolean')
    if not isinstance(config.get('mixed_sources',False),bool):
        raise ValueError('mixed_sources must be boolean')
    if not isinstance(config.get('mixed_sampling',False),bool):
        raise ValueError('mixed_sampling must be boolean')
    if config.get('mixed_sampling') and not config.get('mixed_sources'):
        raise ValueError('mixed_sampling requires mixed_sources')
    if config.get('mixed_sources') and config.get('phase') != 'combat':
        # OOC targets are each collecting champion's own choices; mixing would blend policies.
        raise ValueError('mixed_sources is only valid for combat fits')
    if config.get('relabel_vote_rule') is not None:
        if config['relabel_vote_rule'] not in RULES:
            raise ValueError('relabel_vote_rule must be one of '+', '.join(RULES))
        if config.get('phase') != 'combat':
            raise ValueError('relabel_vote_rule is only valid for combat fits')
    _target_options(config.get('target_mode', 'votes'), config.get('value_scale', 100.))
    if config.get('relabel_vote_rule') is not None and config.get('target_mode') == 'pooled_values':
        raise ValueError('relabel_vote_rule cannot be combined with target_mode=pooled_values')
    if ('target_mode' in config or 'value_scale' in config) and config.get('phase') != 'combat':
        raise ValueError('target_mode/value_scale are only valid for combat fits')
    for key in ('steps','min_steps','eval_every','patience','width','layers','heads'):
        if not isinstance(config[key],int) or config[key]<1:
            raise ValueError('Positive integer required: '+key)
    if config['steps']<config['min_steps'] or (not synthetic and config['min_steps']<2000):
        raise ValueError('Production warm-up requires steps >= min-steps >= 2000')
    if not synthetic and not config.get('probe'):
        raise ValueError('Production warm-up requires a dedicated 93e9 probe corpus (--probe)')
    if config.get('batch') is not None and config['batch']<1:
        raise ValueError('Positive batch required')
    if not math.isfinite(config['lr']) or config['lr']<=0 or config['device'] not in ('cpu','mps'):
        raise ValueError('Invalid LR/device')
    if config['device']=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')


def run_training(config):
    config = {**config, 'old_size':config.get('old_size','large'),
              'old_context':config.get('old_context','none'), 'defer_test':config.get('defer_test',False)}
    if config.get('architecture') == 'controlled':
        config.setdefault('legacy_action_features', False)
    validate_training_config(config)
    config = _target_config(config)
    out = Path(config['out'])
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Fresh-only output required; resume is not implemented')
    with (out/'.fresh-training-claim').open('x') as stream:
        stream.write(str(os.getpid()))
    torch.set_num_threads(1)
    seed_training(config['seed'])
    np.random.seed(config['seed'] % 2**32)
    control = TrainingControl(out,config.get('deadline_utc'))
    control.start()
    try:
        control.check()
        manifest = freeze_manifest(config['data'],control=control,mixed_sources=config.get('mixed_sources',False),
                                  mixed_sampling=config.get('mixed_sampling',False),
                                  relabel_vote_rule=config.get('relabel_vote_rule'),
                                  **_target_options(config.get('target_mode', 'votes'), config.get('value_scale', 100.)))
        if config.get('manifest'):
            frozen = read_json(config['manifest'])
            if frozen != manifest:
                raise ValueError('Frozen manifest/source identities differ')
            if 'preencoded_cache' in config:
                del frozen
        atomic_json(out/'manifest.json.gz',manifest,compressed=True)
        probe_selection, probe_rows = None, None
        if config.get('probe'):
            from sts_rl.fresh_probes import select_probes, encode_probes
            probe_selection, raw_probes = select_probes(config['probe'], config['phase'],
                expected_surface=manifest['public_surface'], control=control)
            atomic_json(out/'probe_selection.json',probe_selection)
            probe_rows = encode_probes(raw_probes,probe_selection,config['architecture'])
        probe_input = ({**file_identity(out/'probe_selection.json'),
                        'selection_identity':probe_selection['selection_identity'],
                        'label':'fixed93e9 probes', 'role':'unlabelled movement monitoring'}
                       if probe_selection else None)
        teachers = sorted({s['policy']['champion']['path'] for s in manifest['collection_sources'].values()})
        cache = None
        if 'preencoded_cache' in config:
            from sts_rl.controlled_cache import prepare_cache, load_cache
            cache = prepare_cache(config['preencoded_cache'],manifest,config['manifest'],config['data'],control)
        effective = {**config,'teacher_checkpoints':teachers,'filters':FILTERS,'stage':'supervised_teacher_warmup',
                     'data_identity':manifest['data_identity'],'vocabulary_sha256':VOCABULARY_SHA256,
                     'probe_selection':probe_input, 'initialization':'from_scratch', 'parent_checkpoint':None}
        vocab = Path(__file__).parents[1]/'sts_policy'/'entity_vocabulary.json'
        if cache:
            effective['preencoded_cache_provenance'] = cache.provenance
        launch = begin_run(out,effective,checkpoint=None,
            inputs=[fit_manifest_path(out),vocab,*teachers,*shard_files(config['data']),
                    *([out/'probe_selection.json',*shard_files(config['probe'])] if probe_selection else []),
                    *(cache.input_paths if cache else [])])
        _verify_file(launch)
        recorded = json.loads(Path(launch['path']).read_text())
        if (recorded['native'].get('preparation_only') or not recorded['native'].get('module_sha256') or
                recorded['native'] != manifest['public_surface']['provenance']['native_provenance']):
            raise ValueError('Training requires validated native identity matching the label dataset')
        receipt = {'launch':launch,'filters':FILTERS,'data_identity':manifest['data_identity'],
                   'vocabulary_sha256':VOCABULARY_SHA256,'teacher_checkpoints':teachers,'weight_parent':None,
                   'architecture':config['architecture'],'old_size':config['old_size'],
                   'old_context':config['old_context'],'defer_test':config['defer_test'],
                   'probe_selection':probe_input, 'label_source_files':manifest['source_files'],
                   'probe_source_files':probe_selection['source_files'] if probe_selection else []}
        if config['architecture'] == 'controlled':
            receipt['legacy_action_features'] = config['legacy_action_features']
        if cache:
            receipt['preencoded_cache'] = cache.provenance
        atomic_json(out/'receipt.json',receipt)
        atomic_json(out/'config.json',effective)
        control.status.update(status='preencoding',data_collected=manifest['collected_total'])
        splits = ('FIT','VALIDATION') if config['defer_test'] else ('FIT','VALIDATION','TEST')
        if cache:
            control.status.update(status='loading_preencoded_cache')
            dataset, cache_timings = load_cache(cache,manifest,control)
            atomic_json(out/'cache_usage.json',dict(cache.provenance,**cache_timings))
        else:
            dataset = preencode(config['data'],manifest,config['phase'],config['architecture'],config['fit_size'],
                                control=control,splits=splits)
        atomic_json(out/'coverage.json',{s:{'rows':len(v),'coverage':coverage(v)} for s,v in dataset.items()})
        model = make_model(config['architecture'],config['phase'],width=config['width'],layers=config['layers'],
                           heads=config['heads'],old_size=config['old_size'],old_context=config['old_context'],
                           legacy_action_features=config.get('legacy_action_features',False))
        if cache:
            cache.recheck(control)
        summary = fit(model,dataset,effective,out,manifest,receipt,control,probes=probe_rows)
        summary.update(collected_total=manifest['collected_total'],filters=FILTERS,**control.stats())
        atomic_json(out/'summary.json',summary)
        control.status.update(summary)
        control.emit()
        return summary
    except Interrupted as exc:
        summary = {'status':'interrupted','reason':str(exc),'test_evaluated':False}
        atomic_json(out/'summary.json',summary)
        control.status.update(summary)
        control.emit()
        return summary
    except Exception as exc:
        control.status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        control.emit()
        raise
    finally:
        control.close()
