#!/usr/bin/env python3
"""Analyse the representation-controls study as its plan specified.

Reads the validated fight panels, label scores and full-game arms of run
representation_controls_20261016 and writes one JSON report plus the CSV data
behind the page's figures. Statistics reuse the study's own library code:
main effects come from sts_rl.controlled_factorial (each factor contrast
averaged over the other two, per training seed, then the mean of the seeds),
and intervals from sts_rl.fresh_statistics (bootstrap over whole source worlds,
10,000 resamples, seed 1919).

    analyse.py <out.json> <figures dir>

Run with the study's frozen runtime (it needs numpy and the study's src/).
"""
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

CONFIG = json.loads((Path(__file__).with_name('finish_games.json')).read_text())
ROOT = Path(CONFIG['root'])
BASE = ROOT / CONFIG['base']
sys.path.insert(0, str(ROOT / 'src'))
from sts_rl.controlled_factorial import FACTORS, LEVELS, SEEDS, _effect, _weights  # noqa: E402
from sts_rl.fresh_statistics import clustered_mean, exact_discordance  # noqa: E402

GROUPS = ('BOSS2', 'BOSS1', 'MONSTER2')
LR_NAME = {.0003: '3e4', .001: '1e3', .0001: '1e4'}
# Validation-selected learning rate of the old encoder at each seed (global_freeze_v1.json).
OLD = {1919: 'rc_old_lr1e3_s1919', 2020: 'rc_old_lr1e3_s2020'}


def model_name(feature, width, lr, seed):
    return f'rc_{"on" if feature else "off"}{width}_lr{LR_NAME[lr]}_s{seed}'


def panel_summary(model):
    """The one complete, fault-free panel for a model (re-runs live in recovery folders)."""
    found = []
    for path in [BASE / 'panels' / model / 'summary.json',
                 *map(Path, glob.glob(str(BASE / 'operational_recovery_*' / 'panels' / model / 'summary.json')))]:
        if path.exists():
            s = json.loads(path.read_text())
            if s['status'] == 'completed' and s['cohort_complete'] and s['faults'] == 0 and s['valid'] == 10800:
                found.append(s)
    if len(found) != 1:
        raise ValueError(f'{model}: expected one complete panel, found {len(found)}')
    return found[0]


def world_means(summary, policy='replacement'):
    """{group: {metric: {world: mean over the 8 resampled fights}}}."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in summary['records']:
        if r['policy'] == policy and r['salt'] >= 0:
            for metric in ('hp_lost', 'win'):
                rows[r['group']][metric][r['world_seed']].append(float(r[metric]))
    for group in GROUPS:
        if any(len(v) != 8 for v in rows[group]['hp_lost'].values()):
            raise ValueError('Expected 8 resampled fights per source world')
    return {g: {m: {w: statistics.mean(v) for w, v in ws.items()} for m, ws in ms.items()}
            for g, ms in rows.items()}


def interval(diffs):
    r = clustered_mean({w: [x] for w, x in diffs.items()}, equal_world=True)
    return {'point': r['point'], 'ci95': r['ci95'], 'se': r['se'], 'worlds': r['worlds']}


def game_records(arm_dir):
    merged = arm_dir / 'merged' / 'summary.json'
    paths = [merged] if merged.exists() else sorted(arm_dir.glob('shard_*/summary.json'))
    records = {}
    for path in paths:
        for r in json.loads(path.read_text())['records']:
            if r['world_seed'] in records:
                raise ValueError('Duplicate world')
            records[r['world_seed']] = r
    expected = set(range(CONFIG['world_start'], CONFIG['world_start'] + CONFIG['shard_worlds'] * CONFIG['shards']))
    if set(records) != expected:
        raise ValueError(f'{arm_dir.name}: incomplete world set')
    if any(r['status'] not in ('completed', 'censored') for r in records.values()):
        raise ValueError(f'{arm_dir.name}: faulted world present')
    return records


def won(r):
    # A world cut at the decision limit counts as a loss (Sid, 19 October 2026).
    return 1 if r['status'] == 'completed' and r['win'] else 0


def compare_games(a, b):
    seeds = sorted(a)
    d = {s: won(a[s]) - won(b[s]) for s in seeds}
    only_a, only_b = sum(x == 1 for x in d.values()), sum(x == -1 for x in d.values())
    # Mean floor needs a recorded floor: worlds cut at the decision limit are left out of it.
    floor = interval({s: a[s]['floor'] - b[s]['floor'] for s in seeds if 'floor' in a[s] and 'floor' in b[s]})
    return {'wins': [sum(won(a[s]) for s in seeds), sum(won(b[s]) for s in seeds)], 'worlds': len(seeds),
            'difference_points': {k: (v * 100 if isinstance(v, float) else [x * 100 for x in v] if isinstance(v, list) else v)
                                  for k, v in interval(d).items() if k in ('point', 'ci95', 'se')},
            'only_first': only_a, 'only_second': only_b, 'exact_p': exact_discordance(only_a, only_b),
            'mean_floor_difference': {k: floor[k] for k in ('point', 'ci95')}}


def main():
    out, figures = Path(sys.argv[1]), Path(sys.argv[2])
    freeze = json.loads((BASE / 'global_freeze_v1.json').read_text())
    models = sorted(freeze['conditions'])
    means = {m: world_means(panel_summary(m)) for m in models}
    champion = world_means(panel_summary(models[0]), policy='champion')
    report = {'run': BASE.name, 'models': {}}

    # Per-model descriptive table: panels relative to the champion, label scores, validation loss.
    for m in models:
        row = {'validation_soft_ce': freeze['conditions'][m]['best_validation_loss']}
        score = json.loads((BASE / 'scores' / m / 'summary.json').read_text())['test']
        row['test_agreement'] = score['primary_agreement']
        row['test_soft_ce'] = score['loss']
        for g in GROUPS:
            for metric in ('hp_lost', 'win'):
                v = means[m][g][metric]
                row[f'{g}_{metric}'] = statistics.mean(v.values())
                row[f'{g}_{metric}_vs_champion'] = interval({w: v[w] - champion[g][metric][w] for w in v})
        report['models'][m] = row
    report['champion_panel'] = {f'{g}_{k}': statistics.mean(champion[g][k].values())
                                for g in GROUPS for k in ('hp_lost', 'win')}

    # Planned check: does the original old-vs-plain-structured gap repeat on new fights?
    report['original_gap'] = {
        str(seed): {metric: interval({w: means[OLD[seed]]['BOSS2'][metric][w]
                                      - means[model_name(False, 128, .0003, seed)]['BOSS2'][metric][w]
                                      for w in means[OLD[seed]]['BOSS2'][metric]})
                    for metric in ('hp_lost', 'win')}
        for seed in SEEDS}

    # Primary: the three main effects on Act 2 boss resampled HP lost; wins and other groups descriptive.
    report['main_effects'] = {}
    for g, metric in product(GROUPS, ('hp_lost', 'win')):
        values = {(seed, *p): means[model_name(*p, seed)][g][metric]
                  for seed in SEEDS for p in product(*LEVELS)}
        report['main_effects'][f'{g}:{metric}'] = {
            f: _effect(values, _weights((f,)), 10_000) for f in FACTORS}
        report['main_effects'][f'{g}:{metric}']['role'] = 'primary' if (g, metric) == ('BOSS2', 'hp_lost') else 'descriptive'
        if g == 'BOSS2':
            # Secondary: each factor's effect with the other two held at each fixed level.
            fixed = {}
            for f in FACTORS:
                others = [x for x in FACTORS if x != f]
                for levels in product(*(LEVELS[FACTORS.index(x)] for x in others)):
                    context = dict(zip(others, levels))
                    name = f + '|' + ','.join(f'{x}={v}' for x, v in context.items())
                    fixed[name] = _effect(values, _weights((f,), context), 10_000)
            report['main_effects'][f'{g}:{metric}']['fixed_level'] = fixed

    # Full games: primary old vs plain structured (OFF/128) at each seed; the rest secondary.
    games = BASE / 'games'
    arms = {arm: game_records(games / CONFIG['out_dir'].get(arm, arm)) for arm in CONFIG['arm_order']}
    arms['champion'] = game_records(games / 'champion')
    report['games'] = {
        'primary_old_vs_plain': {'1919': compare_games(arms['old1919'], arms['off128_s1919']),
                                 '2020': compare_games(arms['old2020'], arms['off128_s2020'])},
        'vs_champion': {arm: compare_games(arms[arm], arms['champion']) for arm in CONFIG['arm_order']},
        'seed1919_structured': {
            'features_on_vs_off_w128': compare_games(arms['on128_s1919'], arms['off128_s1919']),
            'features_on_vs_off_w216': compare_games(arms['on216_s1919'], arms['off216_s1919']),
            'width216_vs_128_features_off': compare_games(arms['off216_s1919'], arms['off128_s1919']),
            'width216_vs_128_features_on': compare_games(arms['on216_s1919'], arms['on128_s1919']),
            'best_structured_vs_old': compare_games(arms['on128_s1919'], arms['old1919'])},
        'cut_worlds': {arm: [s for s, r in recs.items() if r['status'] == 'censored'] for arm, recs in arms.items()},
    }
    reach = {}
    for arm, recs in arms.items():
        reach[arm] = {f'act{a}_{k}': sum(r.get('bosses', {}).get(str(a), {}).get(k, False) for r in recs.values())
                      for a in (1, 2, 3) for k in ('reach', 'pass')}
    report['games']['boss_reach_and_pass'] = reach

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, allow_nan=False))

    figures.mkdir(parents=True, exist_ok=True)
    with open(figures / 'main_effects.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['outcome', 'factor', 'seed', 'point', 'lo', 'hi'])
        for outcome in ('BOSS2:hp_lost', 'BOSS2:win'):
            for factor in FACTORS:
                e = report['main_effects'][outcome][factor]
                for seed, r in [*e['seeds'].items(), ('mean', e['mean_of_seeds'])]:
                    w.writerow([outcome, factor, seed, r['point'], *r['ci95']])
    with open(figures / 'games_vs_champion.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'wins', 'champion_wins', 'point', 'lo', 'hi'])
        for arm, r in report['games']['vs_champion'].items():
            w.writerow([arm, *r['wins'], r['difference_points']['point'], *r['difference_points']['ci95']])
    print('wrote', out)


if __name__ == '__main__':
    main()
