#!/usr/bin/env python3
"""Exploratory: which deck features go with the oracle's deck-strength score?

    deck_features.py <decks_at_act2_boss.json> <oracle.jsonl>

The decks file comes from experiments/2026-10-19-brainstorming/deck_at_boss.py on
the same champion games. Prints correlations, per-card and per-relic differences,
and cross-validated R^2 of simple ridge models. Needs numpy.
"""
import json
import sys
from collections import Counter

import numpy as np


def cards(combat):
    out = []
    for pile in ('hand', 'draw', 'discard', 'exhaust'):
        for k in combat['cards'][pile]:
            out += [k] * k.get('count', 1)
    return out


def cv_r2(x, y, lam=1.0, folds=10):
    idx = np.random.default_rng(0).permutation(len(y))
    pred = np.empty(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        a = np.column_stack([np.ones(len(tr)), (x[tr] - mu) / sd])
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y[tr])
        pred[te] = np.column_stack([np.ones(len(te)), (x[te] - mu) / sd]) @ w
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    decks = {x['seed']: x for x in json.load(open(sys.argv[1])) if 'state' in x}
    oracle = {r['seed']: r for r in map(json.loads, open(sys.argv[2])) if 'error' not in r}
    rows, names, relics = [], Counter(), Counter()
    for seed in sorted(set(decks) & set(oracle)):
        state = decks[seed]['state']
        combat = state['combat']
        deck = cards(combat)
        base = [k['name'].split('=')[0].rstrip('+0123456789') for k in deck]
        rel = {r['name'] for r in combat['relics']}
        names.update(set(base))
        relics.update(rel)
        rows.append(dict(y=np.mean([f['win'] for f in oracle[seed]['before_boss']['full']]),
                         boss=next(e['name'] for e in combat['enemies'] if e['alive']), size=len(deck),
                         strikes=base.count('Strike'), defends=base.count('Defend'),
                         ups=sum(k['upgrade_count'] > 0 for k in deck), powers=sum(k['type'] == 'Power' for k in deck),
                         curses=sum(k['type'] == 'Curse' for k in deck), relics=len(rel),
                         potions=combat['potions']['occupied'], energy=combat['player_energy_per_turn'],
                         maxhp=state['max_hp'], base=set(base), rel=rel))
    y = np.array([r['y'] for r in rows])
    print(f'{len(rows)} decks; oracle win share mean {y.mean():.2f}, sd {y.std():.2f}')
    counts = ('size', 'strikes', 'defends', 'ups', 'powers', 'curses', 'relics', 'potions', 'energy', 'maxhp')
    for k in counts:
        x = np.array([r[k] for r in rows], float)
        print(f'  {k:8s} r = {np.corrcoef(x, y)[0, 1]:+.2f} (mean {x.mean():.1f})')
    for label, pool, key, low in (('cards', names, 'base', 20), ('relics', relics, 'rel', 15)):
        effects = sorted(((np.mean([r['y'] for r in rows if n in r[key]]) - np.mean([r['y'] for r in rows if n not in r[key]]), n, c)
                          for n, c in pool.items() if low <= c <= len(rows) - low), reverse=True)
        print(f'{label} (with minus without):', [(n, c, round(float(e), 2)) for e, n, c in effects])
    bosses = sorted({r['boss'] for r in rows})
    x = np.array([[r[k] for k in counts] + [r['boss'] == b for b in bosses[1:]] for r in rows], float)
    print(f'cross-validated R^2, counts and boss: {cv_r2(x, y):.2f}')
    common = [n for n, c in names.items() if c >= 15]
    common_relics = [n for n, c in relics.items() if c >= 15]
    x2 = np.column_stack([x, [[n in r['base'] for n in common] + [n in r['rel'] for n in common_relics] for r in rows]])
    print(f'cross-validated R^2, plus {len(common)} cards and {len(common_relics)} relics present: {cv_r2(x2, y, lam=10.0):.2f}')


if __name__ == '__main__':
    main()
