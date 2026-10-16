"""Predeclared Gate C paired statistics. No native code or confirmation access.

Intervals resample source WORLD blocks. MDE is a normal approximation at 80%
power, not an exact power calculation. All differences are student minus teacher.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics

import numpy as np

BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 1919
EXPECTATIONS_SCHEMA = 'fresh_comparison_expectations_v1'


def _file_binding(identity):
    """Validate identity structure here; only the CLI opens the bound files."""
    from pathlib import Path
    if (not isinstance(identity, dict) or set(identity) != {'path','bytes','sha256'} or
            not isinstance(identity['path'], str) or not Path(identity['path']).is_absolute() or
            isinstance(identity['bytes'], bool) or not isinstance(identity['bytes'], int) or identity['bytes'] <= 0 or
            not isinstance(identity['sha256'], str) or len(identity['sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in identity['sha256'])):
        raise ValueError('Invalid frozen checkpoint file identity')


def validate_expectations(expectations, mode):
    """Frozen pre-evaluation inputs are mandatory for final game/label claims."""
    from sts_rl.fresh_study import PROTOCOL, FILTER_ID, DEAD_FILTER_ID, PUBLIC_CONTEXT_ID, INFERENCE
    if (not isinstance(expectations, dict) or expectations.get('schema') != EXPECTATIONS_SCHEMA or
            type(expectations.get('version')) is not int or expectations['version'] != 1 or
            expectations.get('mode') != mode):
        raise ValueError('Require frozen comparison expectations schema/version/mode')
    if mode == 'games':
        policies = []
        for arm in ('student','champion'):
            policy = expectations.get(arm+'_policy')
            if (not isinstance(policy, dict) or set(policy) != {'protocol','filter_ids','public_context_id',
                    'inference','champion','combat','ooc'} or policy['protocol'] != PROTOCOL or
                    policy['filter_ids'] != [FILTER_ID, DEAD_FILTER_ID] or
                    policy['public_context_id'] != PUBLIC_CONTEXT_ID or policy['inference'] != INFERENCE):
                raise ValueError('Invalid frozen StudyPolicy identity')
            _file_binding(policy['champion'])
            for phase in ('combat','ooc'):
                if policy[phase] is not None:
                    _file_binding(policy[phase])
                    if policy[phase]['sha256'] == policy['champion']['sha256']:
                        raise ValueError('Replacement must differ from champion checkpoint')
            policies.append(policy)
        student, champion = policies
        if any(student[k] != champion[k] for k in ('protocol','filter_ids','public_context_id','inference','champion')):
            raise ValueError('Frozen policies must share the same champion/filter/inference/context')
        if champion['combat'] is not None or champion['ooc'] is not None:
            raise ValueError('Champion arm must have no replacements')
        if student['combat'] is None and student['ooc'] is None:
            raise ValueError('Student arm requires at least one frozen replacement')
    elif mode == 'labels':
        for arm, architecture in (('student','new'), ('control','old')):
            _file_binding(expectations.get(arm+'_checkpoint'))
            if expectations.get(arm+'_architecture') != architecture:
                raise ValueError('Label roles require new student / old control architectures')
            condition = expectations.get(arm+'_condition')
            common = {'phase','fit_size','lr','seed','steps','min_steps','eval_every','patience','device','batch','defer_test'}
            fields = common | ({'width','layers','heads'} if architecture=='new' else {'old_size','old_context'})
            if (not isinstance(condition, dict) or set(condition) != fields or
                    condition['phase'] not in ('combat','ooc') or condition['fit_size'] not in ('5000','20000','full')):
                raise ValueError('Missing/invalid frozen training condition')
        if expectations['student_checkpoint']['sha256'] == expectations['control_checkpoint']['sha256']:
            raise ValueError('Label arms require different checkpoints; copies are not a comparison')
        if expectations['student_condition']['phase'] != expectations['control_condition']['phase']:
            raise ValueError('Label arms must compare the same phase')
    else:
        raise ValueError('Expectations only apply to games/labels')


def _bound_label_rows(rows, expectations, arm):
    checkpoint = expectations[arm+'_checkpoint']['sha256']
    for row in rows:
        if (row.get('selected_checkpoint') != checkpoint or
                row.get('architecture') != expectations[arm+'_architecture'] or
                row.get('training_condition') != expectations[arm+'_condition']):
            raise ValueError('Label arm checkpoint/architecture/training condition differs from frozen expectations')


def _bound_game_rows(rows, policy):
    for row in rows:
        source = row.get('source')
        if not isinstance(source, dict) or source.get('policy') != policy:
            raise ValueError('Game arm source.policy differs from frozen expectations')


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite statistic input')
    return value


def bootstrap_components(components, estimate, *, resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """One fixed-size component vector per world; keep every within-world row.

    estimate maps summed components to the estimand. Undefined conditional
    replicates are omitted and counted explicitly, never replaced by zero.
    """
    if resamples < 2:
        raise ValueError('Require at least two bootstrap resamples')
    worlds = sorted(components)
    if not worlds:
        return dict(point=None, se=None, ci95=[None, None], worlds=0,
                    bootstrap_valid=0, bootstrap_resamples=resamples, approx_80pct_power_mde=None)
    values = np.asarray([components[w] for w in worlds], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite cluster components')
    point = estimate(values.sum(0))
    replicates = []
    if len(worlds) >= 2:
        rng = np.random.default_rng(seed)
        # Bound temporary memory even on large cohorts.
        for start in range(0, resamples, 128):
            draws = rng.integers(0, len(worlds), (min(128, resamples-start), len(worlds)))
            for total in values[draws].sum(1):
                result = estimate(total)
                if result is not None:
                    replicates.append(_finite(result))
    se = statistics.stdev(replicates) if len(replicates) > 1 else None
    interval = np.quantile(replicates, [.025, .975]).tolist() if replicates else [None, None]
    return dict(point=None if point is None else _finite(point), se=se, ci95=interval,
                worlds=len(worlds), bootstrap_valid=len(replicates), bootstrap_resamples=resamples,
                bootstrap_degenerate=bool(replicates and min(replicates)==max(replicates)),
                approx_80pct_power_mde=None if se is None else 2.8*se)


def clustered_mean(clusters, *, resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED, equal_world=False):
    clusters = {w: [_finite(v) for v in vs] for w, vs in clusters.items() if len(vs)}
    components = {w: [statistics.mean(vs), 1] if equal_world else [sum(vs), len(vs)]
                  for w, vs in clusters.items()}
    result = bootstrap_components(components, lambda t: t[0]/t[1] if t[1] else None,
                                  resamples=resamples, seed=seed)
    return {**result, 'rows': sum(map(len, clusters.values())),
            'weighting': 'equal_world' if equal_world else 'label_weighted',
            'cluster_unit': 'source_world', 'bootstrap_seed': seed}


def _index(records, key):
    result = {}
    for row in records:
        identity = key(row)
        if identity in result:
            raise ValueError('Duplicate paired identity')
        result[identity] = row
    return result


def _matched(new, old, key, expected=None):
    new, old = _index(new, key), _index(old, key)
    if expected is not None:
        expected = list(expected)
        if len(expected) != len(set(expected)):
            raise ValueError('Duplicate requested cohort identity')
    wanted = set(new) if expected is None else set(expected)
    if not wanted or set(new) != wanted or set(old) != wanted:
        raise ValueError('Missing/unexpected paired cohort identities; intersection pairing forbidden')
    return [(new[k], old[k]) for k in sorted(wanted)]


def compare_labels(new, old, *, expectations, expected_ids=None, resamples=BOOTSTRAPS):
    from sts_rl.fresh_study import INFERENCE
    validate_expectations(expectations, 'labels')
    pairs = _matched(new, old, lambda r: r['source_row_id'], expected_ids)
    _bound_label_rows([a for a,b in pairs], expectations, 'student')
    _bound_label_rows([b for a,b in pairs], expectations, 'control')
    groups = defaultdict(lambda: defaultdict(list))
    identity = None
    for a, b in pairs:
        for row in (a, b):
            if row.get('split') != 'TEST' or row.get('inference') != INFERENCE:
                raise ValueError('Paired labels require TEST singleton CPU inference')
            ids = tuple(row.get(f) for f in ('data_identity', 'source_identity'))
            if any(not isinstance(i, str) or not i for i in ids):
                raise ValueError('Missing manifest/scientific source identity')
            if identity is None:
                identity = ids
            if ids != identity:
                raise ValueError('Paired manifest/scientific source identities differ')
        for field in ('world_seed', 'group', 'target_sha256', 'public_hash', 'choice', 'confident'):
            if field not in a or a[field] != b.get(field):
                raise ValueError('Paired label/target alignment failure: '+field)
        if not a.get('target') or a.get('target') != b.get('target'):
            raise ValueError('Paired targets differ')
        from sts_rl.fresh_training import digest
        if digest(a['target']) != a['target_sha256']:
            raise ValueError('Target checksum mismatch')
        for row in (a, b):
            if row['correct'] not in (True, False, 0, 1):
                raise ValueError('Invalid correctness')
            _finite(row['ce'])
        # Choice-only accuracy is the primary estimand; all rows are also shown.
        tags = ['all_rows']
        if a['choice']:
            tags += ['choice', 'group:'+a['group']]
            if a['confident']:
                tags += ['high_confidence_choice']
        for tag in tags:
            groups[tag][a['world_seed']].append(int(a['correct'])-int(b['correct']))
    return {'metric': 'teacher_plurality_agreement_student_minus_control', 'paired_rows': len(pairs),
            'expected_bindings':expectations,
            'groups': {k: clustered_mean(v, resamples=resamples) for k, v in groups.items()}}


def teacher_consistency(rows, *, resamples=BOOTSTRAPS):
    """Independent repeat histograms only; no within-vote pseudo-ceiling."""
    from sts_rl.fresh_training import target_distribution
    groups, count, abstained = defaultdict(lambda: defaultdict(list)), 0, 0
    for row in rows:
        repeat = row.get('self_consistency')
        if repeat is None:
            continue
        original, other = row['search'], repeat['search']
        if repeat.get('use_for_training') is not False:
            raise ValueError('Repeat query must be excluded from training')
        for field in ('samples', 'draw_order_seeds', 'effective_rng_seeds', 'effective_search_seeds'):
            if not original.get(field) or not other.get(field) or set(original[field]) & set(other[field]):
                raise ValueError('Repeat queries are not independent: '+field)
        if original['search_seed'] == other['search_seed']:
            raise ValueError('Repeat queries share search seed')
        for plan in (original,other):
            samples = plan['samples']
            if (plan['draw_order_seeds'] != samples or
                plan['effective_rng_seeds'] != [plan['search_seed']*1009+s for s in samples] or
                plan['effective_search_seeds'] != [plan['search_seed']+s for s in samples]):
                raise ValueError('Stored repeat query seed plan is inconsistent')
        p, indices = target_distribution(row)
        q, _ = target_distribution({**row, 'search':other, 'votes': repeat['votes'], 'label': None}, check_label=False)
        if p is None or q is None:
            abstained += 1
            continue
        match = int(np.argmax(p) == np.argmax(q))
        for tag in ('all', row['group']):
            groups[tag][row['world_seed']].append(match)
        count += 1
    return {'method': 'empirical independent repeat-query plurality agreement', 'rows': count, 'abstained_pairs':abstained,
            'groups': {k: clustered_mean(v, resamples=resamples) for k, v in groups.items()}}


def exact_discordance(student_only, champion_only):
    """Two-sided exact binomial McNemar p, stable log-sum even at large n."""
    if any(isinstance(x, bool) or int(x) != x or x < 0 for x in (student_only, champion_only)):
        raise ValueError('Discordance counts must be nonnegative integers')
    n, k = student_only+champion_only, min(student_only, champion_only)
    if not n:
        return 1.0
    logs = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
    maximum = max(logs)
    return min(1.0, 2*math.exp(maximum)*sum(math.exp(x-maximum) for x in logs))


def gate_c(interval, margin=.01):
    lo, hi = interval
    if lo is None or hi is None:
        return {'gate': 'INCONCLUSIVE', 'evidence_of_decline': False, 'ni_margin': margin}
    return {'gate': 'PASS' if lo > -margin else 'FAIL' if hi < 0 else 'INCONCLUSIVE',
            'evidence_of_decline': hi < 0, 'ni_margin': margin}


def _valid_game(row):
    if row.get('status') != 'completed' or any(row.get(k) for k in (
            'fault', 'cut', 'ub', 'censored', 'programming_error', 'truncated')):
        raise ValueError('Full-game final claims require completed fault-free exact cohort')
    if not isinstance(row.get('win'), bool):
        raise ValueError('Full-game outcome missing')
    _finite(row['floor'])
    if not isinstance(row.get('expected_provenance'), dict) or not row['expected_provenance']:
        raise ValueError('Missing full-game expected_provenance')
    if not isinstance(row.get('bosses'), dict):
        raise ValueError('Full-game requires an explicit bosses dict (empty means no reach)')
    for act, boss in row['bosses'].items():
        if (act not in ('1', '2', '3') or not isinstance(boss, dict) or
                boss.get('reach') is not True or not isinstance(boss.get('pass'), bool) or
                isinstance(boss.get('entry_hp'), bool) or not isinstance(boss.get('entry_hp'), (int, float))):
            raise ValueError('Malformed boss record')
        if _finite(boss['entry_hp']) < 0:
            raise ValueError('Malformed boss entry HP')


def compare_games(student, champion, requested_seeds, *, expectations, resamples=BOOTSTRAPS):
    validate_expectations(expectations, 'games')
    seeds = list(requested_seeds)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Requested cohort must be nonempty and unique')
    pairs = _matched(student, champion, lambda r: r['world_seed'], seeds)
    _bound_game_rows([a for a,b in pairs], expectations['student_policy'])
    _bound_game_rows([b for a,b in pairs], expectations['champion_policy'])
    for a, b in pairs:
        _valid_game(a)
        _valid_game(b)
        if a['expected_provenance'] != b['expected_provenance']:
            raise ValueError('Paired full-game expected_provenance differs')
    def difference(field):
        return clustered_mean({a['world_seed']: [float(a[field])-float(b[field])] for a,b in pairs},
                              resamples=resamples)
    win = difference('win')
    a_only = sum(a['win'] and not b['win'] for a,b in pairs)
    b_only = sum(b['win'] and not a['win'] for a,b in pairs)
    win.update(gate_c(win['ci95']))
    win.update(student_only=a_only, champion_only=b_only,
               zero_discordance=(a_only+b_only)==0,
               exact_mcnemar_p=exact_discordance(a_only, b_only),
               student_wins=sum(a['win'] for a,b in pairs), champion_wins=sum(b['win'] for a,b in pairs),
               denominator=len(pairs))
    bosses = {}
    for act in ('1', '2', '3'):
        components = {}
        for a,b in pairs:
            aa, bb = a['bosses'].get(act), b['bosses'].get(act)
            components[a['world_seed']] = [int(aa is not None), int(bb is not None),
                int(bool(aa and aa['pass'])), int(bool(bb and bb['pass'])),
                _finite(aa['entry_hp']) if aa else 0., _finite(bb['entry_hp']) if bb else 0., 1]
        total = np.array(list(components.values())).sum(0)
        estimates = {
            'reach': lambda t: (t[0]-t[1])/t[6],
            'pass_marginal': lambda t: (t[2]-t[3])/t[6],
            'pass_conditional': lambda t: t[2]/t[0]-t[3]/t[1] if t[0] and t[1] else None,
            'entry_hp_conditional': lambda t: t[4]/t[0]-t[5]/t[1] if t[0] and t[1] else None}
        bosses[act] = {k: bootstrap_components(components, f, resamples=resamples) for k,f in estimates.items()}
        bosses[act].update(student_reached=int(total[0]), champion_reached=int(total[1]),
            student_passed=int(total[2]), champion_passed=int(total[3]), cohort_denominator=len(pairs),
            student_conditional_pass=total[2]/total[0] if total[0] else None,
            champion_conditional_pass=total[3]/total[1] if total[1] else None,
            student_entry_hp=total[4]/total[0] if total[0] else None,
            champion_entry_hp=total[5]/total[1] if total[1] else None)
    return {'cohort_complete': True, 'win': win, 'floor': difference('floor'), 'bosses': bosses,
            'expected_bindings':expectations,
            'conditional_note': 'Policy-dependent reach cohorts; marginal conditional estimates with same-world resampling, no causal interpretation.',
            'mde_note': '2.8 * paired bootstrap SE: normal approximation, 80% power at two-sided 5%.',
            'bootstrap_note': 'Constant observed differences give degenerate percentile intervals; these cannot quantify unobserved discordance uncertainty. Gate uses the predeclared interval rule.'}


def compare_panel(records, frozen, *, resamples=BOOTSTRAPS):
    grid = _index(frozen['grid'], lambda r: (r['world_seed'], r['start_id'], r['salt']))
    salts = frozen.get('salts', sorted({key[2] for key in grid}))
    if not grid or not salts or len(set(salts))!=len(salts):
        raise ValueError('Empty/duplicate frozen panel grid salts')
    starts = {(w,s) for w,s,_ in grid}
    if set(grid) != {(w,s,t) for w,s in starts for t in salts}:
        raise ValueError('Frozen panel grid lacks declared start/salt cells')
    expected = {(w,s,t,p) for w,s,t in grid for p in ('champion', 'replacement')}
    indexed = _index(records, lambda r: (r['world_seed'], r['start_id'], r['salt'], r['policy']))
    if not set(indexed) <= expected:
        raise ValueError('Unexpected panel grid cells')
    counts = {k: sum(bool(r.get(k)) for r in records) for k in ('fault', 'cut', 'ub', 'censored', 'programming_error')}
    valid = {k: r for k,r in indexed.items() if not any(r.get(f) for f in counts)}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    paired = 0
    for key, item in grid.items():
        a, b = valid.get((*key, 'replacement')), valid.get((*key, 'champion'))
        if a is None or b is None:
            continue
        if not a.get('first_combat_hash') or a['first_combat_hash'] != b.get('first_combat_hash'):
            raise ValueError('Panel initial public states differ')
        if a.get('group') != item['group'] or b.get('group') != item['group']:
            raise ValueError('Panel group mismatch')
        if not isinstance(a.get('win'), bool) or not isinstance(b.get('win'), bool):
            raise ValueError('Panel lacks terminal outcome')
        paired += 1
        kind = ':unsalted' if key[2] == -1 else ':resampled'
        for tag in ('all'+kind, item['group']+kind):
            for metric in ('win','hp_lost'):
                groups[tag][metric][key[0]].append(_finite(a[metric])-_finite(b[metric]))
    complete = set(valid) == expected and bool(frozen.get('coverage_sufficient', True))
    return {'grid_complete': complete, 'confidence_claims_allowed': complete,
            'descriptive_dev_only': True, 'confirmation_closed': True,
            'expected_cells': len(expected), 'received_cells': len(records), 'paired_fights': paired,
            'excluded': counts, 'missing_cells': len(expected-set(indexed)),
            'weighting': 'equal_world', 'cluster_unit': 'source_world',
            'groups': {g: {m: clustered_mean(c, resamples=resamples, equal_world=True) for m,c in metrics.items()}
                       for g,metrics in groups.items()},
            'hp_lost_direction': 'negative difference favors student',
            'mde_note': '2.8 * bootstrap SE, normal approximation at 80% power.'}
