#!/usr/bin/env python3
"""Real Act 2 boss pass rate by oracle-score quintile, from oracle_analyse.py's CSV.

    oracle_plot.py <figures dir>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    figures = Path(sys.argv[1])
    rows = list(csv.DictReader(open(figures / 'oracle_by_world.csv')))
    passed = np.array([int(r['pass2']) for r in rows])
    series = {'Entry HP (reference)': 'entry_hp_fraction',
              'Oracle at start of Act 2': 'act2_start_full_boss_damage',
              'Oracle at floor 25': 'mid_act2_full_boss_damage',
              'Oracle just before the boss': 'before_boss_full_boss_damage'}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for (label, column), style in zip(series.items(), ('k--', 'o-', 's-', '^-')):
        score = np.array([float(r[column]) for r in rows])
        order = np.argsort(score, kind='stable')
        bins = np.array_split(order, 5)
        ax.plot(range(1, 6), [100 * passed[b].mean() for b in bins], style, label=label)
    ax.set_xticks(range(1, 6), ['lowest', '2', '3', '4', 'highest'])
    ax.set_xlabel('Score quintile (60 worlds each)')
    ax.set_ylabel('Act 2 boss actually beaten (%)')
    ax.set_title('Full-HP oracle score against the real Act 2 boss outcome (300 champion games)', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / 'pass_rate_by_oracle_quintile.png', dpi=110)
    print('wrote', figures / 'pass_rate_by_oracle_quintile.png')


if __name__ == '__main__':
    main()
