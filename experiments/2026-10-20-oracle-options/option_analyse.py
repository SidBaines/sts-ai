#!/usr/bin/env python3
"""Analyse the oracle option test at Act 2 card rewards.

    option_analyse.py <options.jsonl> <out.json> <figures dir>

Needs numpy only.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SKIP = 'skip card reward'


def card_name(description):
    if description == SKIP:
        return SKIP
    return re.match(r'take card (.+?) \[', description)[1]


def bootstrap_mean(values, resamples=10_000, seed=1919):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    return [float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    out, figures = Path(sys.argv[2]), Path(sys.argv[3])
    errors = [r['seed'] for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r and all(f['win'] is not None for fs in r['scores'].values() for f in fs)]
    report = {'decisions': len(rows), 'errors': errors, 'by_score': {}}
    for score in ('boss_damage', 'win'):
        clear, stable_clear, stable_all, champion_best, oracle_skip, champion_skip, gains = 0, 0, 0, 0, 0, 0, []
        per_card = defaultdict(list)
        for r in rows:
            options = list(r['scores'])
            x = np.array([[float(f[score]) for f in r['scores'][o]] for o in options])  # options x salts
            c = options.index(r['champion'])
            means = x.mean(1)
            b = int(means.argmax())
            d = x[b] - x[c]
            se = d.std(ddof=1) / np.sqrt(len(d)) if b != c else 0.0
            is_clear = b != c and d.mean() > 2 * se and se > 0
            even, odd = x[:, 0::2], x[:, 1::2]
            b_even, b_odd = int(even.mean(1).argmax()), int(odd.mean(1).argmax())
            clear += is_clear
            stable_clear += is_clear and b_even == b_odd
            stable_all += b_even == b_odd
            champion_best += b == c
            oracle_skip += options[b] == SKIP
            champion_skip += r['champion'] == SKIP
            # Unbiased gain: choose on even seeds, score the choice on odd seeds.
            gains.append(float(odd[b_even].mean() - odd[c].mean()))
            skip_mean = means[options.index(SKIP)]
            for o, m in zip(options, means):
                if o != SKIP:
                    per_card[card_name(o)].append((float(m - skip_mean), o == r['champion']))
        n = len(rows)
        cards = {name: {'offers': len(v), 'mean_vs_skip': float(np.mean([g for g, _ in v])),
                        'champion_took': float(np.mean([t for _, t in v]))}
                 for name, v in per_card.items()}
        report['by_score'][score] = {
            'clear_better_than_champion': clear / n, 'stable_among_clear': stable_clear / max(1, clear),
            'stable_overall': stable_all / n, 'champion_picks_oracle_best': champion_best / n,
            'oracle_best_is_skip': oracle_skip / n, 'champion_skipped': champion_skip / n,
            'unbiased_gain_from_following_oracle': bootstrap_mean(gains),
            'cards': dict(sorted(cards.items(), key=lambda kv: -kv[1]['mean_vs_skip']))}
    s = report['by_score']['boss_damage']
    report['decision'] = bool(s['clear_better_than_champion'] >= 0.2 and s['stable_among_clear'] >= 0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    figures.mkdir(parents=True, exist_ok=True)
    with open(figures / 'cards_vs_skip.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['card', 'offers', 'boss_damage_vs_skip', 'win_share_vs_skip', 'champion_took'])
        for name, c in report['by_score']['boss_damage']['cards'].items():
            w.writerow([name, c['offers'], c['mean_vs_skip'], report['by_score']['win']['cards'][name]['mean_vs_skip'], c['champion_took']])
    print('decisions', n, 'errors', errors, 'decision rule met:', report['decision'])
    for score, v in report['by_score'].items():
        print(f'== {score}')
        for k, val in v.items():
            if k != 'cards':
                print(f'   {k:38s} {val}')
        common = [(k, c) for k, c in v['cards'].items() if c['offers'] >= 5]
        print('   best cards vs skip (>=5 offers):', [(k, c['offers'], round(c['mean_vs_skip'], 3), round(c['champion_took'], 2)) for k, c in common[:8]])
        print('   worst cards vs skip:', [(k, c['offers'], round(c['mean_vs_skip'], 3), round(c['champion_took'], 2)) for k, c in common[-8:]])


if __name__ == '__main__':
    main()
