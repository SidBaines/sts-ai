#!/usr/bin/env python3
"""Analyse the oracle spike: does a simulated-boss score predict the real Act 2 boss outcome?

    oracle_analyse.py <oracle.jsonl> <out.json> <figures dir>

Needs numpy only (any Python with numpy).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

MOMENTS = ('act2_start', 'mid_act2', 'before_boss')
VARIANTS = ('full', 'actual')
SCORES = ('boss_damage', 'win', 'hp_lost')


def auc(score, label):
    """Probability a random passing world scores above a random failing one (ties count half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def fit_logistic(x, y, ridge=1e-3, iterations=50):
    x = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(x.shape[1])
    penalty = ridge * np.r_[1e-3, np.ones(len(w) - 1)]
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        gradient = x.T @ (p - y) + penalty * w
        hessian = (x * (p * (1 - p))[:, None]).T @ x + np.diag(penalty)
        w -= np.linalg.solve(hessian, gradient)
    return w


def cross_validated(x, y, folds=10, seed=0):
    """Out-of-fold predictions; returns (log loss, AUC)."""
    order = np.random.default_rng(seed).permutation(len(y))
    p = np.empty(len(y))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        mean, sd = x[train].mean(0), x[train].std(0) + 1e-9
        w = fit_logistic((x[train] - mean) / sd, y[train])
        z = np.column_stack([np.ones(len(test)), (x[test] - mean) / sd]) @ w
        p[test] = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), auc(p, y)


def bootstrap_difference(x_base, x_full, y, resamples=1000, seed=1):
    """Bootstrap the improvement in out-of-fold log loss from adding the oracle."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(cross_validated(x_base[i], y[i], seed=2)[0] - cross_validated(x_full[i], y[i], seed=2)[0])
    return [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))]


def reliability(fights, score):
    """Split-half reliability of the per-world mean (even vs odd salts), Spearman-Brown corrected."""
    a = np.array([np.mean([f[score] for f in w[0::2]]) for w in fights], float)
    b = np.array([np.mean([f[score] for f in w[1::2]]) for w in fights], float)
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return 2 * r / (1 + r)


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    out, figures = Path(sys.argv[2]), Path(sys.argv[3])
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    bad_replay = [r['seed'] for r in rows if not r['replay_matches']]
    invalid = sum(f['win'] is None for r in rows for m in MOMENTS if m in r for v in VARIANTS for f in r[m][v])
    rows = [r for r in rows if r['replay_matches'] and all(m in r for m in MOMENTS)]
    rows = [r for r in rows if all(f['win'] is not None for m in MOMENTS for v in VARIANTS for f in r[m][v])]
    y = np.array([r['pass2'] for r in rows], float)
    entry = np.array([r['entry_hp'] / r['entry_max_hp'] for r in rows])
    entry_abs = np.array([r['entry_hp'] for r in rows], float)
    base = np.column_stack([entry, entry_abs])
    report = {'worlds': len(rows), 'errors': [r['seed'] for r in errors], 'replay_mismatch': bad_replay,
              'invalid_fights': invalid, 'pass_rate': float(y.mean()),
              'seconds_per_world': float(np.mean([r['seconds'] for r in rows])),
              'entry_hp_only': dict(zip(('log_loss', 'auc'), cross_validated(base, y))),
              'entry_hp_auc': auc(entry, y), 'oracles': {}}
    for m in MOMENTS:
        for v in VARIANTS:
            fights = [r[m][v] for r in rows]
            for s in SCORES:
                score = np.array([np.mean([f[s] for f in w]) for w in fights], float)
                if s == 'hp_lost':
                    score = -score / np.array([w[0]['entry_hp'] for w in fights], float)
                combined = np.column_stack([base, score])
                loss, combined_auc = cross_validated(combined, y)
                entry_cv = report['entry_hp_only']
                key = f'{m}:{v}:{s}'
                report['oracles'][key] = {
                    'mean': float(score.mean()), 'sd_across_worlds': float(score.std()),
                    'auc_alone': auc(score, y), 'cv_log_loss_with_entry_hp': loss,
                    'cv_auc_with_entry_hp': combined_auc,
                    'log_loss_gain': entry_cv['log_loss'] - loss, 'auc_gain': combined_auc - entry_cv['auc'],
                    'split_half_reliability': reliability(fights, s if s != 'hp_lost' else 'hp_lost'),
                    'correlation_with_entry_hp': float(np.corrcoef(score, entry)[0, 1])}
    primary = 'before_boss:full:boss_damage'
    combined = np.column_stack([base, [np.mean([f['boss_damage'] for f in r['before_boss']['full']]) for r in rows]])
    report['primary'] = {'score': primary, **report['oracles'][primary],
                         'log_loss_gain_ci95': bootstrap_difference(base, combined, y)}
    p = report['primary']
    report['decision'] = bool(p['log_loss_gain'] > 0 and p['log_loss_gain_ci95'][0] > 0 and p['auc_gain'] >= 0.03
                              and (p['split_half_reliability'] or 0) >= 0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    figures.mkdir(parents=True, exist_ok=True)
    with open(figures / 'oracle_by_world.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'pass2', 'entry_hp_fraction', *[f'{m}_{v}_{s}' for m in MOMENTS for v in VARIANTS for s in ('boss_damage', 'win')]])
        for r, e in zip(rows, entry):
            w.writerow([r['seed'], int(r['pass2']), e, *[np.mean([x[s] for x in r[m][v]]) for m in MOMENTS for v in VARIANTS for s in ('boss_damage', 'win')]])
    print(json.dumps({k: report[k] for k in ('worlds', 'errors', 'replay_mismatch', 'invalid_fights', 'pass_rate', 'seconds_per_world', 'entry_hp_only', 'entry_hp_auc', 'decision')}, indent=1))
    print(f"{'oracle':34s} {'mean':>6s} {'sd':>5s} {'AUC':>5s} {'+HP AUC':>7s} {'gainAUC':>7s} {'gainLL':>7s} {'rel':>5s} {'r(HP)':>6s}")
    for k, o in report['oracles'].items():
        rel = o['split_half_reliability']
        print(f"{k:34s} {o['mean']:6.3f} {o['sd_across_worlds']:5.3f} {o['auc_alone']:5.3f} {o['cv_auc_with_entry_hp']:7.3f} {o['auc_gain']:+7.3f} {o['log_loss_gain']:+7.4f} {('%.2f' % rel) if rel is not None else '  n/a':>5s} {o['correlation_with_entry_hp']:+6.2f}")
    print('primary log-loss gain 95% CI', report['primary']['log_loss_gain_ci95'])


if __name__ == '__main__':
    main()
