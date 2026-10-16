"""Strict generic LEFT-minus-RIGHT development comparisons, without verdicts.

Filesystem entry points validate completed runs, frozen checkpoints/conditions,
and exact cohorts before computing intervals. No fitting, inference or native
execution. Pure metric helpers alone do not certify filesystem provenance.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from sts_rl.controlled_scoring import checkpoint_details, frozen_selection
from sts_rl.fresh_panel_comparison import (_Inputs, _grid, _run, _require, _same, _hash,
    _json_object, _nonfinite, CONTROL_FIELDS, write_comparison)
from sts_rl.fresh_panel_grid import HP_DEFINITION
from sts_rl.fresh_statistics import (BOOTSTRAPS, BOOTSTRAP_SEED, _matched, _valid_game,
    _file_binding, bootstrap_components, clustered_mean, exact_discordance)
from sts_rl.fresh_study import (PROTOCOL, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID,
                                INFERENCE, validate_world_range)
from sts_rl.recording import file_identity

EXPECTATIONS_SCHEMA = 'controlled_comparison_expectations_v1'
ARMS = ('left', 'right')


def _checkpoint(spec, inputs, phase=None):
    from sts_policy.training import load_checkpoint
    _require(isinstance(spec, dict) and set(spec) == {'checkpoint', 'architecture', 'condition'},
             'Require frozen checkpoint, architecture and condition per arm')
    _require(spec['architecture'] in ('old', 'new', 'controlled'), 'Invalid frozen checkpoint architecture')
    identity = inputs.verify(spec['checkpoint'])
    payload = load_checkpoint(identity['path'])
    meta, condition = checkpoint_details(payload, architecture=spec['architecture'], phase=phase)
    _require(_same(condition, spec['condition']), 'Frozen condition differs from actual checkpoint metadata')
    inputs.recheck()
    return meta, condition


def _policy(value, inputs):
    _require(isinstance(value, dict) and set(value) == {
        'protocol', 'filter_ids', 'public_context_id', 'inference', 'champion', 'combat', 'ooc'} and
        value['protocol'] == PROTOCOL and value['filter_ids'] == [FILTER_ID, DEAD_FILTER_ID] and
        value['public_context_id'] == PUBLIC_CONTEXT_ID and value['inference'] == INFERENCE,
        'Require full frozen StudyPolicy identity')
    for phase in ('champion', 'combat', 'ooc'):
        if value[phase] is not None:
            inputs.verify(value[phase])
    _require(value['champion'] is not None, 'Missing champion')


def _expectations(path, mode, inputs):
    value, identity = inputs.read(path)
    _require(value.get('schema') == EXPECTATIONS_SCHEMA and type(value.get('version')) is int and
             value['version'] == 1 and value.get('mode') == mode,
             'Require generic frozen expectations schema/version/mode')
    details = {}
    for arm in ARMS:
        spec = value[arm]
        if mode == 'labels':
            details[arm] = _checkpoint(spec, inputs)
        else:
            _require(set(spec) == {'policy', 'replacements'}, 'Require policy and replacement bindings')
            _policy(spec['policy'], inputs)
            replacements = {p: i for p, i in spec['policy'].items() if p in ('combat', 'ooc') and i is not None}
            _require(set(spec['replacements']) == set(replacements), 'Missing/extra policy checkpoint conditions')
            details[arm] = {}
            for phase, checkpoint in replacements.items():
                _require(_same(spec['replacements'][phase]['checkpoint'], checkpoint),
                         'Policy checkpoint differs from frozen checkpoint condition')
                _require(checkpoint['sha256'] != spec['policy']['champion']['sha256'],
                         'Replacement must differ from champion')
                details[arm][phase] = _checkpoint(spec['replacements'][phase], inputs, phase)
    if mode == 'labels':
        _require(value['left']['checkpoint']['sha256'] != value['right']['checkpoint']['sha256'],
                 'Require different LEFT/RIGHT checkpoints')
        _paired_conditions(details['left'], details['right'])
        inputs.verify(value['manifest'])
    else:
        a, b = (value[arm]['policy'] for arm in ARMS)
        _require(all(_same(a[k], b[k]) for k in ('protocol', 'filter_ids', 'public_context_id', 'inference', 'champion')),
                 'Policies must share champion/filter/context/inference')
        _require(any((a[p]['sha256'] if a[p] is not None else None) !=
                     (b[p]['sha256'] if b[p] is not None else None) for p in ('champion', 'combat', 'ooc')),
                 'Require different LEFT/RIGHT policy checkpoint SHAs')
        all_details = [d for arm in ARMS for d in details[arm].values()]
        _require(bool(all_details), 'Require frozen replacements')
        _require(len({d[0]['data_identity'] for d in all_details}) == 1, 'Checkpoint data identities differ')
        for phase in ('combat', 'ooc'):
            if phase in details['left'] and phase in details['right']:
                _paired_conditions(details['left'][phase], details['right'][phase])
        if mode == 'panel':
            _require(all(set(details[arm]) == {'combat'} for arm in ARMS),
                     'Panel comparison requires combat-only replacements in both arms')
    return value, identity, details


def _paired_conditions(a, b):
    _require(a[0]['data_identity'] == b[0]['data_identity'] and
             all(_same(a[1][k], b[1][k]) for k in ('phase', 'fit_size')),
             'Paired checkpoint data/phase/FIT size mismatch')


def _cohort(path, expected, inputs):
    # _Inputs.read deliberately requires objects; cohorts are exact frozen lists.
    identity = inputs.verify(expected)
    _require(Path(path).resolve() == Path(identity['path']), 'Frozen cohort file mismatch')
    value = json.loads(Path(path).read_text(), object_pairs_hook=_json_object, parse_constant=_nonfinite)
    _require(isinstance(value, list) and bool(value) and len(value) == len(set(value)),
             'Require nonempty unique exact cohort list')
    inputs.recheck()
    return value


def _launch(receipt, directory, inputs, manifest):
    launch, _ = inputs.read(receipt['launch']['path'], receipt['launch'])
    path = Path(receipt['launch']['path'])
    _require(path.name == 'launch.json' and directory / 'records' in path.parents and
             launch['schema'] == 'sts_training_launch_v1', 'Scored launch must belong to this run')
    fields = ('stage', 'training', 'weight_updates', 'architecture', 'phase', 'device', 'selection',
              'selected_checkpoint', 'manifest', 'selection_criterion', 'data_identity')
    _require(_same(launch['effective_config'], {k: receipt[k] for k in fields}) and
             _same(launch['parent_checkpoint'], receipt['selected_checkpoint']), 'Scored launch binding mismatch')
    expected = [receipt['selection'], receipt['manifest'], *manifest['source_files']]
    keys = lambda xs: Counter(json.dumps(i, sort_keys=True) for i in xs)
    _require(keys(launch['data_inputs']) == keys(expected), 'Scored launch source files mismatch')
    for item in launch['data_inputs']:
        inputs.verify(item)
    _native_snapshot(receipt['launch'], launch, inputs,
                     manifest['public_surface']['provenance']['native_provenance'])
    return launch


def _native_snapshot(launch_id, launch, inputs, expected_native):
    path = Path(launch_id['path'])
    archive = inputs.verify(launch['source_archive'])
    _require(Path(archive['path']) == path.parent / 'source.zip', 'Launch source archive binding mismatch')
    native = launch['native']
    _require(_same(native, expected_native) and _hash(native.get('module_sha256')) and
             not native.get('preparation_only'), 'Native provenance mismatch')
    _require(Path(native['module_file']).name == native['module_file'], 'Invalid native module name')
    binary = inputs.verify(file_identity(path.parent / native['module_file']))
    build, _ = inputs.read(path.parent / 'build_manifest.json')
    _require(binary['sha256'] == native['module_sha256'] and _same(build, native),
             'Native snapshot/build mismatch')
    _require(isinstance(launch['runtime'], dict) and all(launch['runtime'].get(k) for k in ('python', 'platform', 'packages')),
             'Runtime identity missing')
    _require(all(launch['runtime']['packages'].get(k) for k in ('torch', 'numpy')), 'Runtime package identity missing')


def _score(directory, spec, manifest_id, inputs):
    directory = Path(directory).resolve()
    _require(directory.is_dir(), 'Require completed scored run directory')
    receipt, receipt_id = inputs.read(directory / 'receipt.json')
    summary, summary_id = inputs.read(directory / 'summary.json')
    selection, selection_id, manifest, meta, condition = frozen_selection(receipt['selection']['path'], inputs)
    _require(_same(receipt['selection'], selection_id) and _same(receipt['manifest'], manifest_id) and
             _same(selection['manifest'], manifest_id) and
             _same(selection['selected_checkpoint'], spec['checkpoint']) and
             _same(condition, spec['condition']) and meta['config']['architecture'] == spec['architecture'],
             'Scored selection/checkpoint/manifest/condition mismatch')
    for value in (receipt, summary):
        _require(value.get('training') is False and type(value.get('weight_updates')) is int and
                 value['weight_updates'] == 0 and _same(value['selected_checkpoint'], spec['checkpoint']) and
                 _same(value['selection'], selection_id) and value['data_identity'] == meta['data_identity'],
                 'Require completed inference-only scoring bound to selected checkpoint')
    _require(summary.get('status') == 'completed' and summary.get('test_evaluated') is True and
             summary.get('test_protocol') == INFERENCE and summary.get('test_device') == 'cpu' and
             receipt.get('stage') == 'deferred_test_scoring' and receipt.get('device') == 'cpu' and
             receipt.get('architecture') == spec['architecture'] and receipt.get('phase') == condition['phase'] and
             _same(receipt['selection_criterion'], selection['selection_criterion']) and
             all(_same(receipt[k], meta[k]) for k in ('filters', 'vocabulary_sha256', 'source_receipt')) and
             _same(receipt['label_source_files'], manifest['source_files']), 'Invalid scored receipt metadata')
    launch = _launch(receipt, directory, inputs, manifest)
    path = directory / 'test_rows.jsonl'
    rows_id = inputs.verify(file_identity(path))
    if spec['architecture'] == 'controlled' or 'test_rows_file' in summary:
        _require(_same(summary.get('test_rows_file'), rows_id), 'Scored row file binding mismatch')
    with path.open() as stream:
        rows = [json.loads(line, object_pairs_hook=_json_object, parse_constant=_nonfinite)
                for line in stream if line.strip()]
    _require(type(summary.get('test_rows')) is int and summary['test_rows'] == len(rows) and bool(rows),
             'Incomplete scored rows')
    wanted = {r['source_row_id']: r for r in manifest['rows'] if
              r['phase'] == condition['phase'] and r['split'] == 'TEST' and r['has_target']}
    _require(len(wanted) == len(rows) and len({r['source_row_id'] for r in rows}) == len(rows) and
             set(wanted) == {r['source_row_id'] for r in rows}, 'Scored rows differ from full frozen TEST cohort')
    for row in rows:
        _require(all(_same(row.get(k), v) for k, v in wanted[row['source_row_id']].items()),
                 'Scored row differs from frozen manifest source/target metadata')
        _require(row.get('selected_checkpoint') == spec['checkpoint']['sha256'] and
                 row.get('architecture') == spec['architecture'] and _same(row.get('training_condition'), condition) and
                 row.get('data_identity') == meta['data_identity'], 'Scored row checkpoint/condition binding mismatch')
    groups = defaultdict(list)
    for row in rows:
        tags = ['all_rows']
        if row['choice']:
            tags += ['choice', 'group:' + row['group']]
            if row['confident']:
                tags += ['high_confidence_choice']
        for tag in tags:
            groups[tag].append(row)
    _require(set(summary['test']['groups']) == set(groups), 'Scored summary group inventory differs')
    for tag, values in groups.items():
        stats = summary['test']['groups'][tag]
        _require(type(stats['rows']) is int and stats['rows'] == len(values) and
                 math.isclose(stats['agreement'], statistics.mean(int(r['correct']) for r in values), abs_tol=1e-7) and
                 math.isclose(stats['soft_ce'], statistics.mean(r['ce'] for r in values), abs_tol=1e-7),
                 'Scored summary differs from actual prediction rows')
    inputs.directory(directory / 'quarantine', set())
    # Both arms pin the same immutable manifest bytes and validate every row
    # against them. Retain only that binding, releasing the parsed manifest.
    manifest_binding = {'file': selection['manifest'], 'data_identity': manifest['data_identity']}
    return rows, manifest_binding, launch, {'receipt': receipt_id, 'summary': summary_id, 'rows': rows_id}


def label_metrics(left, right, expected_ids, *, resamples=BOOTSTRAPS):
    """Paired teacher-plurality agreement AND soft CE, whole-world bootstrap."""
    from sts_rl.fresh_training import digest, world_split
    pairs = _matched(left, right, lambda r: r['source_row_id'], expected_ids)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    identity = None
    for a, b in pairs:
        for row in (a, b):
            _require(_hash(row.get('source_row_id')) and _hash(row.get('public_hash')),
                     'Invalid label row/public identity')
            world = row['world_seed']
            _require(type(world) is int and 90_000_000_000 <= world < 91_000_000_000 and
                     world_split(world) == 'TEST' and row.get('split') == 'TEST' and row.get('inference') == INFERENCE,
                     'Label comparison requires development TEST singleton CPU inference')
            ids = (row.get('data_identity'), row.get('source_identity'))
            _require(all(_hash(i) for i in ids) and (identity is None or ids == identity),
                     'Label data/scientific source identities differ')
            identity = ids
            _require(all(type(row.get(k)) is bool for k in ('correct', 'choice', 'confident')) and
                     row.get('has_target') is True and row.get('abstention') is False, 'Invalid label target/status')
            target = row['target']
            _require(isinstance(target, list) and bool(target) and all(type(x) in (int, float) and
                     math.isfinite(x) and x >= 0 for x in target) and abs(sum(target) - 1) < 1e-6 and
                     digest(target) == row['target_sha256'] and row['choice'] == (len(target) > 1) and
                     row['confident'] == (max(target) >= .75), 'Invalid label target/checksum/confidence')
            _require(type(row.get('ce')) in (int, float) and math.isfinite(row['ce']) and row['ce'] >= -1e-6,
                     'Invalid soft cross entropy')
        for k in ('world_seed', 'group', 'phase', 'target', 'target_sha256', 'public_hash', 'choice', 'confident'):
            _require(_same(a[k], b[k]), 'Paired label/target alignment failure: ' + k)
        group = a['group']
        _require(isinstance(group, str) and len(group) > 1 and group[-1] in '123', 'Missing act/room group')
        tags = ['all_rows']
        if a['choice']:
            tags += ['choice', 'act:' + group[-1], 'room:' + group[:-1], 'group:' + group]
            if a['confident']:
                tags += ['high_confidence_choice']
        for tag in tags:
            for metric, x, y in (('agreement', int(a['correct']), int(b['correct'])), ('soft_ce', a['ce'], b['ce'])):
                for name, value in ((metric, x - y), (metric + '_left', x), (metric + '_right', y)):
                    groups[tag][name][a['world_seed']].append(value)
    result = {}
    for tag, clusters in sorted(groups.items()):
        result[tag] = {}
        for metric in ('agreement', 'soft_ce'):
            result[tag][metric] = {**clustered_mean(clusters[metric], resamples=resamples),
                **{arm + '_mean': statistics.mean(v for vs in clusters[metric + '_' + arm].values() for v in vs)
                   for arm in ARMS}}
    return {'groups': result, 'paired_rows': len(pairs), 'contrast': 'left_minus_right',
            'weighting': 'label_weighted', 'cluster_unit': 'source_world',
            'agreement_direction': 'positive favors LEFT', 'soft_ce_direction': 'negative favors LEFT'}


def panel_metrics(left, right, frozen, *, resamples=BOOTSTRAPS, full_fit=True):
    # Reuse the established paired estimand/CI calculation only; architectural
    # roles are validated generically above and never relabelled old/new.
    from sts_rl.fresh_panel_comparison import contrast_metrics
    result = contrast_metrics(left, right, frozen, resamples=resamples, full_fit=full_fit)
    for group in result['groups'].values():
        for metric in ('win', 'hp_lost'):
            group[metric]['left_mean'] = group[metric].pop('new_mean')
            group[metric]['right_mean'] = group[metric].pop('old_mean')
    result.update(contrast='left_replacement_minus_right_replacement',
                  hp_lost_direction='negative LEFT-minus-RIGHT HP loss favors LEFT')
    return result


def game_metrics(left, right, seeds, *, resamples=BOOTSTRAPS):
    pairs = _matched(left, right, lambda r: r['world_seed'], seeds)
    for a, b in pairs:
        _valid_game(a)
        _valid_game(b)
        _require(_same(a['expected_provenance'], b['expected_provenance']), 'Paired full-game provenance differs')
    difference = lambda field: clustered_mean({a['world_seed']: [float(a[field]) - float(b[field])]
                                              for a, b in pairs}, resamples=resamples)
    win = difference('win')
    left_only = sum(a['win'] and not b['win'] for a, b in pairs)
    right_only = sum(b['win'] and not a['win'] for a, b in pairs)
    both = sum(a['win'] and b['win'] for a, b in pairs)
    neither = sum(not a['win'] and not b['win'] for a, b in pairs)
    win.update(left_only_wins=left_only, right_only_wins=right_only,
               left_wins=left_only + both, right_wins=right_only + both,
               both_wins=both, neither_wins=neither,
               exact_mcnemar_p=exact_discordance(left_only, right_only))
    bosses = {}
    for act in ('1', '2', '3'):
        components = {}
        for a, b in pairs:
            aa, bb = a['bosses'].get(act), b['bosses'].get(act)
            components[a['world_seed']] = [int(aa is not None), int(bb is not None),
                int(bool(aa and aa['pass'])), int(bool(bb and bb['pass'])),
                float(aa['entry_hp']) if aa else 0., float(bb['entry_hp']) if bb else 0., 1]
        estimands = dict(reach=lambda t: (t[0]-t[1])/t[6], pass_marginal=lambda t: (t[2]-t[3])/t[6],
            pass_conditional=lambda t: t[2]/t[0]-t[3]/t[1] if t[0] and t[1] else None,
            entry_hp_conditional=lambda t: t[4]/t[0]-t[5]/t[1] if t[0] and t[1] else None)
        bosses[act] = {k: bootstrap_components(components, f, resamples=resamples) for k, f in estimands.items()}
    return dict(cohort_complete=True, paired_worlds=len(pairs), contrast='left_minus_right',
                win=win, floor=difference('floor'), bosses=bosses,
                conditional_note='Policy-dependent reach cohorts; conditional contrasts are descriptive, not causal.')


def _report(result, expected, expected_id, inputs, sources):
    inputs.recheck()
    return {**result, 'schema': 'controlled_comparison_v1', 'version': 1, 'protocol': PROTOCOL,
            'filesystem_validated': True, 'confidence_claims_allowed': True,
            'descriptive_dev_only': True, 'confirmation_closed': True, 'expected_bindings': expected,
            'expectations_file': expected_id, 'sources': sources,
            'bootstrap_resamples': BOOTSTRAPS, 'bootstrap_seed': BOOTSTRAP_SEED,
            'verified_files': sorted(inputs.files.values(), key=lambda i: i['path'])}


def compare_labels_runs(left, right, cohort, expectations):
    inputs = _Inputs()
    expected, expected_id, _ = _expectations(expectations, 'labels', inputs)
    _require(Path(left).resolve() != Path(right).resolve(), 'Require different scored directories')
    ids = _cohort(cohort, expected['cohort'], inputs)
    runs = {arm: _score(directory, expected[arm], expected['manifest'], inputs)
            for arm, directory in zip(ARMS, (left, right))}
    _require(_same(runs['left'][1], runs['right'][1]), 'Scored manifests differ')
    for field in ('native', 'runtime', 'environment'):
        _require(_same(runs['left'][2][field], runs['right'][2][field]), 'Scored provenance differs: ' + field)
    inputs.recheck()
    result = label_metrics(runs['left'][0], runs['right'][0], ids)
    return _report(result, expected, expected_id, inputs, {arm: runs[arm][3] for arm in ARMS})


def compare_panel_runs(left, right, grid, expectations):
    inputs = _Inputs()
    expected, expected_id, details = _expectations(expectations, 'panel', inputs)
    _require(Path(left).resolve() != Path(right).resolve(), 'Require different panel directories')
    _file_binding(expected['grid'])
    frozen, grid_id = inputs.read(grid, expected['grid'])
    starts = _grid(frozen, inputs)
    runs = {arm: _run(directory, expected[arm]['policy'], frozen, grid_id, starts, inputs)
            for arm, directory in zip(ARMS, (left, right))}
    for field in ('native', 'runtime', 'environment'):
        _require(_same(runs['left'][field], runs['right'][field]), 'Panel provenance differs: ' + field)
    for cell in frozen['grid']:
        key = cell['world_seed'], cell['start_id'], cell['salt']
        controls = [runs[arm]['rows'][(*key, 'champion')] for arm in ARMS]
        _require(all(_same(controls[0][k], controls[1][k]) for k in CONTROL_FIELDS),
                 'Champion control exact-match failure')
        rows = [runs[arm]['rows'][(*key, p)] for arm in ARMS for p in ('champion', 'replacement')]
        _require(all(all(_same(row[k], rows[0][k]) for k in ('first_combat_hash', 'entry_hp', 'source_entry_hp'))
                     for row in rows), 'Paired start/HP mismatch')
    inputs.recheck()
    result = panel_metrics(*[[r for k, r in runs[arm]['rows'].items() if k[3] == 'replacement'] for arm in ARMS],
                           frozen, full_fit=details['left']['combat'][1]['fit_size'] == 'full')
    result.update(grid_complete=True, control_exact_match=True, grid_hash=frozen['grid_hash'],
                  bank_identity=frozen['bank_identity'], hp_definition=HP_DEFINITION)
    return _report(result, expected, expected_id, inputs, {arm: runs[arm]['identities'] for arm in ARMS})


def _games(directory, policy, inputs):
    from sts_rl.fresh_cohort import validate_cohorts
    # Pin every input before validation, then use the original complete-cohort
    # reader. A second read/check closes concurrent rewrite/addition races.
    directory = Path(directory).resolve()
    _require(directory.is_dir(), 'Games require completed cohort directories')
    receipt, _ = inputs.read(directory / 'receipt.json')
    config = receipt['config']
    validate_world_range(config['world_start'], config['n'])
    _require(91_000_000_000 <= config['world_start'] < config['world_start'] + config['n'] <= 91_200_000_000,
             'Games require development evaluation namespace')
    files = set(p for p in directory.rglob('*') if p.is_file())
    for path in files:
        inputs.verify(file_identity(path))
    inputs.directory(directory, set(directory.rglob('*')))
    cohort = validate_cohorts([directory])
    science = cohort['science_identity']
    _require(cohort['config']['mode'] == 'eval' and _same(science['policy'], policy),
             'Full-game cohort policy/mode differs from frozen expectations')
    environment = None
    for source in cohort['sources']:
        for phase, key in (('champion', 'checkpoint'), ('combat', 'combat'), ('ooc', 'ooc')):
            actual, identity = source['config'].get(key), policy[phase]
            _require((actual is None and identity is None) or (actual is not None and identity is not None and
                     Path(actual).resolve() == Path(identity['path'])), 'Game config/checkpoint identity mismatch')
        inputs.verify(source['launch'])
        launch, _ = inputs.read(source['launch']['path'], source['launch'])
        path = Path(source['launch']['path'])
        _require(launch['schema'] == 'sts_training_launch_v1' and path.name == 'launch.json' and
                 Path(source['config']['out']).resolve() / 'records' in path.parents,
                 'Game launch must belong to its actual source run')
        expected_inputs = [policy[p] for p in ('combat', 'ooc') if policy[p] is not None]
        _require(_same(launch['data_inputs'], expected_inputs), 'Game launch replacement inputs mismatch')
        _native_snapshot(source['launch'], launch, inputs, science['expected_provenance']['native_provenance'])
        for item in launch['data_inputs']:
            inputs.verify(item)
        _require(environment is None or _same(environment, launch['environment']), 'Game source environments differ')
        environment = launch['environment']
    inputs.recheck()
    return cohort['records'], {**science, 'environment': environment}


def compare_games_runs(left, right, cohort, expectations):
    inputs = _Inputs()
    expected, expected_id, _ = _expectations(expectations, 'games', inputs)
    seeds = _cohort(cohort, expected['cohort'], inputs)
    for world in seeds:
        _require(type(world) is int and 91_000_000_000 <= world < 91_200_000_000,
                 'Games require development evaluation namespace')
        validate_world_range(world, 1)
    _require(Path(left).resolve() != Path(right).resolve(), 'Require different game cohort directories')
    runs = {arm: _games(directory, expected[arm]['policy'], inputs) for arm, directory in zip(ARMS, (left, right))}
    for field in ('native', 'runtime', 'environment', 'expected_provenance'):
        _require(_same(runs['left'][1][field], runs['right'][1][field]), 'Game provenance differs: ' + field)
    configs = [{k: v for k, v in runs[arm][1]['config'].items() if k not in ('checkpoint', 'combat', 'ooc')}
               for arm in ARMS]
    _require(_same(*configs), 'Game scientific configs differ beyond frozen policy replacements')
    inputs.recheck()
    result = game_metrics(runs['left'][0], runs['right'][0], seeds)
    return _report(result, expected, expected_id, inputs, {arm: runs[arm][1] for arm in ARMS})
