"""Fixed representation-control factorial analysis. No fitting or native execution.

Only filesystem panel reports certify provenance. Two fixed training seeds share
the same source worlds: average their world deltas BEFORE bootstrapping worlds.
"""
from collections import Counter, defaultdict
from itertools import combinations, product
import json
import math
from pathlib import Path
import statistics

from sts_rl.controlled_comparison import _checkpoint, _policy, _native_snapshot
from sts_rl.controlled_scoring import frozen_selection
from sts_rl.fresh_panel_comparison import (
    _Inputs, _grid, _run, _require, _same, _index, _valid_row,
    CONTROL_FIELDS, write_comparison,
)
from sts_rl.fresh_panel_grid import HP_DEFINITION
from sts_rl.fresh_statistics import BOOTSTRAPS, BOOTSTRAP_SEED, clustered_mean
from sts_rl.fresh_study import PROTOCOL

EXPECTATIONS_SCHEMA = 'controlled_factorial_expectations_v1'
SEEDS = (1919, 2020)
FACTORS = ('feature', 'capacity', 'lr')
LEVELS = ((False, True), (128, 216), (.0003, .001))
PROFILES = tuple(product(*LEVELS))
PANEL_TARGETS = {'BOSS2': 400, 'BOSS1': 100, 'MONSTER2': 100}
BANK_START, BANK_WORLDS = 92_100_000_000, 1024
FIT_ROWS, VALIDATION_ROWS = 113_042, 17_989
FIXED = dict(phase='combat', fit_size='full', steps=12_000, min_steps=2000,
             eval_every=100, patience=6, device='mps', batch=128,
             defer_test=True, layers=2, heads=4)


def _cell(condition):
    _require(isinstance(condition, dict), 'Require actual checkpoint training_condition')
    seed, feature, width, lr = (condition.get(k) for k in
                               ('seed', 'legacy_action_features', 'width', 'lr'))
    _require(type(seed) is int and seed in SEEDS and type(feature) is bool and
             type(width) is int and width in LEVELS[1] and
             type(lr) in (int, float) and lr in LEVELS[2],
             'Condition outside the fixed 16-cell factorial (secondary LRs forbidden)')
    wanted = dict(FIXED, seed=seed, legacy_action_features=feature, width=width, lr=lr)
    _require(_same(condition, wanted), 'Wrong full-FIT combat factorial training condition')
    return seed, feature, width, lr


def condition_key(condition):
    """Canonical arm key, determined by the actual validated checkpoint condition."""
    seed, feature, width, lr = _cell(condition)
    return f's{seed}_f{int(feature)}_w{width}_lr{lr:g}'


def _inventory(conditions):
    _require(isinstance(conditions, dict) and len(conditions) == 16,
             'Require exact 16-arm factorial inventory; no missing/extra cells')
    cells = {key: _cell(value) for key, value in conditions.items()}
    _require(len(set(cells.values())) == 16, 'Duplicate actual factorial condition')
    _require(set(cells.values()) == set(product(SEEDS, *LEVELS)),
             'Missing/extra actual factorial conditions')
    _require(all(key == condition_key(conditions[key]) for key in cells),
             'Condition key differs from actual checkpoint metadata')
    return cells


def _selection(spec, manifest_id, inputs, *, check_cohort):
    """Reuse the full deferred/best/source validator; release its large manifest."""
    inputs.verify(spec['selection'])
    selection, selection_id, manifest, meta, condition = frozen_selection(spec['selection']['path'], inputs)
    _require(_same(selection_id, spec['selection']) and
             _same(selection['selected_checkpoint'], spec['checkpoint']) and
             _same(selection['manifest'], manifest_id) and
             _same(condition, spec['condition']) and spec['architecture'] == 'controlled',
             'Frozen selection/checkpoint/manifest/condition mismatch')
    _require(_same(meta['model_config']['dropout'], 0.0) and
             meta.get('critic_status') == 'frozen_untrained' and
             meta.get('loss_normalization') == 'actual_choice_batch_mean' and
             _same(meta['optimization_rows']['manifest_fit_rows'], FIT_ROWS),
             'Wrong fixed dropout/critic/full-FIT optimization contract')
    if check_cohort:
        rows = [r for r in manifest['rows'] if r['phase'] == 'combat' and r['has_target']]
        counts = Counter(r['split'] for r in rows)
        fit = [r['source_row_id'] for r in rows if r['split'] == 'FIT']
        validation = [r['source_row_id'] for r in rows if r['split'] == 'VALIDATION']
        _require(counts['FIT'] == FIT_ROWS and counts['VALIDATION'] == VALIDATION_ROWS and
                 len(set(fit)) == FIT_ROWS and len(set(validation)) == VALIDATION_ROWS and
                 set(manifest['subsets']['combat']['full']) == set(fit) and
                 len(manifest['subsets']['combat']['full']) == FIT_ROWS,
                 'Frozen manifest differs from full predeclared FIT/VALIDATION cohorts')
    launch_id = meta['source_receipt']['launch']
    launch, _ = inputs.read(launch_id['path'], launch_id)
    _native_snapshot(launch_id, launch, inputs,
                     manifest['public_surface']['provenance']['native_provenance'])
    # Return only compact identities, never a parsed manifest or checkpoint.
    return meta['data_identity'], {k: launch[k] for k in ('native', 'runtime', 'environment')}


def _expectations(path, inputs):
    expected, identity = inputs.read(path)
    _require(set(expected) == {'schema', 'version', 'mode', 'grid', 'manifest', 'global_freeze', 'arms'} and
             expected['schema'] == EXPECTATIONS_SCHEMA and type(expected['version']) is int and
             expected['version'] == 1 and expected['mode'] == 'panels',
             'Require controlled_factorial_expectations_v1 version 1 mode panels')
    arms = expected['arms']
    _require(isinstance(arms, dict) and len(arms) == 16, 'Require exact 16-arm factorial inventory')
    inputs.verify(expected['manifest'])
    freeze, _ = inputs.read(expected['global_freeze']['path'], expected['global_freeze'])
    _require(isinstance(freeze.get('checkpoints'), dict), 'Global freeze requires checkpoints mapping')
    conditions, directories, checkpoints, selections = {}, set(), set(), set()
    training = None
    data_identity = None
    policy_base = None
    for key, spec in arms.items():
        _require(isinstance(spec, dict) and set(spec) == {
            'run_directory', 'checkpoint', 'architecture', 'condition', 'policy', 'selection'},
            'Require run_directory/checkpoint/architecture/condition/policy/selection per arm')
        _require(isinstance(spec['run_directory'], str) and Path(spec['run_directory']).is_absolute(),
                 'Require absolute run_directory')
        directories.add(str(Path(spec['run_directory']).resolve()))
        _require(spec['architecture'] == 'controlled', 'Factorial arms must use controlled architecture')
        _require(_same(freeze['checkpoints'].get(key), spec['checkpoint']),
                 'Checkpoint differs from root global freeze')
        # Generic validators load actual checkpoint metadata and strict model provenance.
        meta, condition = _checkpoint({k: spec[k] for k in ('checkpoint', 'architecture', 'condition')},
                                      inputs, 'combat')
        conditions[key] = condition
        _cell(condition)
        checkpoints.add(spec['checkpoint']['sha256'])
        selections.add(spec['selection']['path'])
        _policy(spec['policy'], inputs)
        policy = spec['policy']
        _require(policy['ooc'] is None and _same(policy['combat'], spec['checkpoint']) and
                 policy['combat']['sha256'] != policy['champion']['sha256'],
                 'Require combat-only policy bound to actual controlled checkpoint')
        common = {k: v for k, v in policy.items() if k != 'combat'}
        _require(policy_base is None or _same(common, policy_base), 'Factorial policy identities differ')
        policy_base = common
        current_data, current_training = _selection(spec, expected['manifest'], inputs,
                                                    check_cohort=data_identity is None)
        _require(current_data == meta['data_identity'] and
                 (data_identity is None or data_identity == current_data), 'Checkpoint data identities differ')
        _require(training is None or _same(training, current_training), 'Training native/runtime/environment differs')
        training, data_identity = current_training, current_data
    cells = _inventory(conditions)
    _require(len(directories) == len(checkpoints) == len(selections) == 16,
             'Duplicate run directory, checkpoint or selection in factorial inventory')
    return expected, identity, cells, data_identity


def _reduce(rows, frozen, baseline=None):
    """One world mean per group/metric; exact cells and initial-state pairing."""
    indexed = _index(rows)
    cells = {(r['world_seed'], r['start_id'], r['salt'], 'replacement'): r for r in frozen['grid']}
    starts = {(r['world_seed'], r['start_id']): r for r in frozen['starts']}
    _require(set(indexed) == set(cells) and bool(cells), 'Missing/unexpected replacement pairs')
    initial = {}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, cell in cells.items():
        row = indexed[key]
        _valid_row(row, cell, starts[key[:2]], frozen['grid_hash'])
        initial[key] = [row[k] for k in ('first_combat_hash', 'entry_hp', 'source_entry_hp')]
        _require(baseline is None or _same(initial[key], baseline[key]), 'Paired start/HP mismatch')
        kind = ':unsalted' if key[2] == -1 else ':resampled'
        for tag in ('all' + kind, row['group'] + kind):
            for metric in ('hp_lost', 'win'):
                value = float(row[metric])
                _require(math.isfinite(value), 'Nonfinite factorial metric')
                groups[tag][metric][key[0]].append(value)
    means = {tag: {m: {w: statistics.mean(v) for w, v in worlds.items()}
                   for m, worlds in metrics.items()} for tag, metrics in groups.items()}
    return means, initial


def _weights(factors, fixed=None):
    fixed = fixed or {}
    divisor = 2 ** (3 - len(factors) - len(fixed))
    return {p: math.prod(1 if p[FACTORS.index(f)] == LEVELS[FACTORS.index(f)][1] else -1
                         for f in factors) / divisor for p in PROFILES
            if all(p[FACTORS.index(f)] == level for f, level in fixed.items())}


def _direction(stats):
    lo, hi = stats['ci95']
    return 1 if lo is not None and lo > 0 else -1 if hi is not None and hi < 0 else 0


def _effect(values, weights, resamples):
    worlds = sorted(next(iter(values.values())))
    deltas = {seed: {w: math.fsum(weight * values[(seed, *p)][w] for p, weight in weights.items())
                     for w in worlds} for seed in SEEDS}
    def interval(v):
        return clustered_mean({w: [x] for w, x in v.items()}, equal_world=True,
                              resamples=resamples, seed=BOOTSTRAP_SEED)
    by_seed = {str(seed): interval(deltas[seed]) for seed in SEEDS}
    mean = interval({w: statistics.mean(deltas[s][w] for s in SEEDS) for w in worlds})
    a, b = (_direction(by_seed[str(s)]) for s in SEEDS)
    replicated = a != 0 and a == b
    sign = a if replicated else _direction(mean)
    return {'seeds': by_seed, 'mean_of_seeds': mean,
            'direction_status': 'replicated' if replicated else 'suggestive' if sign else 'inconclusive',
            'direction': 'positive' if sign > 0 else 'negative' if sign < 0 else None}


def _metrics(reduced, cells, resamples):
    _require(type(resamples) is int and resamples >= 2, 'Require at least two bootstrap resamples')
    reference = next(iter(reduced.values()))
    _require(all(set(v) == set(reference) for v in reduced.values()), 'Factorial group inventory differs')
    groups = {}
    for tag, metrics in sorted(reference.items()):
        groups[tag] = {}
        for metric, worlds in metrics.items():
            values = {cells[k]: reduced[k][tag][metric] for k in cells}
            _require(all(set(v) == set(worlds) and all(math.isfinite(x) for x in v.values())
                         for v in values.values()), 'Incomplete/nonfinite paired world cells')
            primary = tag == 'BOSS2:resampled' and metric == 'hp_lost'
            main = {f: _effect(values, _weights((f,)), resamples) for f in FACTORS}
            fixed = {}
            for f in FACTORS:
                others = [x for x in FACTORS if x != f]
                for levels in product(*(LEVELS[FACTORS.index(x)] for x in others)):
                    context = dict(zip(others, levels))
                    name = f + '|' + ','.join(f'{x}={v}' for x, v in context.items())
                    fixed[name] = _effect(values, _weights((f,), context), resamples)
            interactions = {':'.join(fs): _effect(values, _weights(fs), resamples)
                            for n in (2, 3) for fs in combinations(FACTORS, n)}
            seed_differences = {}
            for p in PROFILES:
                name = f'f{int(p[0])}_w{p[1]}_lr{p[2]:g}'
                seed_differences[name] = clustered_mean(
                    {w: [values[(2020, *p)][w] - values[(1919, *p)][w]] for w in worlds},
                    equal_world=True, resamples=resamples, seed=BOOTSTRAP_SEED)
            groups[tag][metric] = {
                'main_effect_role': 'primary' if primary else 'descriptive', 'main_effects': main,
                'cell_means': {k: statistics.mean(values[c].values()) for k, c in cells.items()},
                'secondary': {'fixed_level_effects': fixed, 'interactions': interactions,
                              'seed_2020_minus_1919': seed_differences},
            }
    return dict(groups=groups, filesystem_validated=False, confidence_claims_allowed=False,
                bootstrap_resamples=resamples, bootstrap_seed=BOOTSTRAP_SEED,
                weighting='equal_world', cluster_unit='source_world',
                factor_directions={'feature': 'ON minus OFF', 'capacity': '216 minus 128',
                                   'lr': '0.001 minus 0.0003'},
                hp_lost_direction='Negative favors the higher factor level; units are HP.',
                win_direction='Positive favors the higher factor level; units are win probability, not percent.',
                interaction_scale='Difference of factor effects; two-way averaged over the third factor. '
                                  'Three-way is the high-minus-low change in a two-way interaction.',
                primary='Three main effects on BOSS2:resampled hp_lost only.',
                interpretation='95% source-world intervals conditional on these two fixed training seeds. '
                               'Average seed deltas within world before resampling; seeds do not double world n. '
                               'Replicated requires both seed intervals strictly excluding zero in the same direction; '
                               'otherwise suggestive requires the mean interval excluding zero, else inconclusive. '
                               'No training-seed population interval, equivalence, gates, or multiplicity adjustment. '
                               'All wins, unsalted, BOSS1/MONSTER2 and pooled endpoints are descriptive; '
                               'fixed-level effects, interactions and seed differences are secondary. '
                               'Constant observed deltas cannot quantify unobserved discordance.')


def factorial_metrics(rows, conditions, frozen, *, resamples=BOOTSTRAPS):
    """Synthetic/in-memory diagnostics only; cannot certify filesystem provenance."""
    cells = _inventory(conditions)
    _require(set(rows) == set(cells), 'Missing/extra factorial row arms')
    reduced, baseline = {}, None
    for key in cells:
        reduced[key], initial = _reduce(rows[key], frozen, baseline)
        if baseline is None:
            baseline = initial
    return _metrics(reduced, cells, resamples)


def compare_panel_runs(expectations, *, resamples=BOOTSTRAPS):
    """Read-only analysis of exactly 16 frozen complete panels, sequentially."""
    inputs = _Inputs()
    try:
        expected, expected_id, cells, data_id = _expectations(expectations, inputs)
        frozen, grid_id = inputs.read(expected['grid']['path'], expected['grid'])
        starts = _grid(frozen, inputs)
        _require(_same(frozen['targets'], PANEL_TARGETS) and
                 frozen['bank_identity']['world_start'] == BANK_START and
                 frozen['bank_identity']['n'] == BANK_WORLDS,
                 'Wrong predeclared factorial panel grid/bank targets')
        reduced, sources = {}, {}
        controls, baseline, provenance = None, None, None
        for key, spec in expected['arms'].items():
            run = _run(spec['run_directory'], spec['policy'], frozen, grid_id, starts, inputs)
            current = {k: run[k] for k in ('native', 'runtime', 'environment')}
            _require(provenance is None or _same(current, provenance), 'Panel native/runtime/environment differs')
            provenance = current
            current_controls = {k[:3]: [r[f] for f in CONTROL_FIELDS]
                                for k, r in run['rows'].items() if k[3] == 'champion'}
            _require(controls is None or _same(list(sorted(current_controls.items())),
                                             list(sorted(controls.items()))),
                     'Champion control exact-match failure across factorial arms')
            controls = current_controls
            for k, row in run['rows'].items():
                champion = run['rows'][(*k[:3], 'champion')]
                _require(all(_same(row[f], champion[f]) for f in
                             ('first_combat_hash', 'entry_hp', 'source_entry_hp')), 'Paired start/HP mismatch')
            reduced[key], initial = _reduce([r for k, r in run['rows'].items() if k[3] == 'replacement'],
                                             frozen, baseline)
            if baseline is None:
                baseline = initial
            sources[key] = run['identities']
            del run
        inputs.recheck()
        result = _metrics(reduced, cells, resamples)
        inputs.recheck()
        return {**result, 'schema': 'controlled_factorial_v1', 'version': 1, 'mode': 'panels',
                'protocol': PROTOCOL, 'filesystem_validated': True, 'confidence_claims_allowed': True,
                'descriptive_dev_only': True, 'confirmation_closed': True,
                'control_exact_match': True, 'grid_complete': True, 'grid_hash': frozen['grid_hash'],
                'targets': frozen['targets'], 'bank_identity': frozen['bank_identity'],
                'hp_definition': HP_DEFINITION, 'data_identity': data_id,
                'expected_bindings': expected, 'expectations_file': expected_id, 'sources': sources,
                'provenance': provenance, 'verified_files': sorted(inputs.files.values(), key=lambda i: i['path'])}
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Missing/invalid factorial input: {exc}') from exc
