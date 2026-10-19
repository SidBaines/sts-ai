#!/usr/bin/env python3
"""Figures for the representation-controls page, from analyse.py's report.

    plot.py <report.json> <figures dir>

Needs matplotlib, which the frozen study runtime lacks; any Python with
matplotlib works because this script only reads JSON.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {'old1919': 'Old encoder, seed 1919', 'old2020': 'Old encoder, seed 2020',
          'off128_s1919': 'Structured 128, seed 1919', 'off128_s2020': 'Structured 128, seed 2020',
          'on128_s1919': 'Structured 128 + rule features', 'off216_s1919': 'Structured 216',
          'on216_s1919': 'Structured 216 + rule features'}


def main():
    report = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1: full-game win difference against the champion, one row per arm.
    games = report['games']['vs_champion']
    arms = list(LABELS)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for y, arm in enumerate(reversed(arms)):
        d = games[arm]['difference_points']
        colour = '#1f6fb2' if arm.startswith('old') else '#c45a1c'
        ax.errorbar(d['point'], y, xerr=[[d['point'] - d['ci95'][0]], [d['ci95'][1] - d['point']]],
                    fmt='o', color=colour, capsize=3)
        ax.text(3.6, y, f"{games[arm]['wins'][0]} wins", va='center', fontsize=8)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in reversed(arms)], fontsize=8)
    ax.set_xlim(-3, 4.6)
    ax.set_xlabel('Win-rate difference from the champion (points, 95% interval)')
    ax.set_title(f"Full games on 4,096 new DEV worlds (champion {games['old1919']['wins'][1]} wins)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'games_vs_champion.png', dpi=110)

    # Figure 2: Act 2 boss fight win rate against agreement with the teacher, all 28 models.
    rows = []
    for name, m in report['models'].items():
        kind = 'old' if '_old_' in name else ('on' if '_on' in name else 'off') + name.split('_')[1][-3:]
        rows.append((name, kind, m['test_agreement'], m['BOSS2_win'], m['validation_soft_ce'], m['BOSS2_hp_lost']))
    with open(out / 'models.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'kind', 'test_agreement', 'act2_boss_fight_win', 'validation_soft_ce', 'act2_boss_hp_lost'])
        w.writerows(rows)
    style = {'old': ('#1f6fb2', 'o', 'Old encoder'), 'off128': ('#c45a1c', 's', 'Structured 128'),
             'on128': ('#e0a030', 's', 'Structured 128 + rules'), 'off216': ('#7a3b8f', '^', 'Structured 216'),
             'on216': ('#b07cc6', '^', 'Structured 216 + rules')}
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, (colour, marker, label) in style.items():
        pts = [(r[2] * 100, r[3] * 100) for r in rows if r[1] == kind]
        ax.scatter(*zip(*pts), color=colour, marker=marker, label=label, s=28)
    ax.axhline(report['champion_panel']['BOSS2_win'] * 100, color='grey', ls='--', lw=0.8)
    ax.text(min(r[2] for r in rows) * 100, report['champion_panel']['BOSS2_win'] * 100 - 0.6, 'champion', fontsize=8, color='grey')
    ax.set_xlabel('Agreement with the search teacher on held-out labels (%)')
    ax.set_ylabel('Act 2 boss fights won (%)')
    ax.set_title('Every student wins fewer Act 2 boss fights than the champion', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'agreement_vs_boss_wins.png', dpi=110)
    print('wrote figures to', out)


if __name__ == '__main__':
    main()
