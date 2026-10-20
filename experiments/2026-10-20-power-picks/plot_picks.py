#!/usr/bin/env python3
"""One figure: how often the champion takes each power when it is offered.

    plot_picks.py <analysis.json> <figures dir>

Needs matplotlib (the main repository runtime, not the frozen study runtime).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOURS = {'Uncommon': '#4c72b0', 'Rare': '#c44e52'}


def main():
    report = json.load(open(sys.argv[1]))
    figures = Path(sys.argv[2])
    powers = [(name, c) for name, c in report['per_card'].items()
              if c['type'] == 'Power' and not name.endswith('+') and c['offers'] >= 100]
    powers.sort(key=lambda kv: kv[1]['take_rate'][0])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [3, 2]})
    y = range(len(powers))
    for i, (name, c) in enumerate(powers):
        rate, low, high, _ = c['take_rate']
        left.barh(i, 100 * rate, color=COLOURS[c['rarity']], xerr=[[100 * (rate - low)], [100 * (high - rate)]],
                  error_kw={'elinewidth': 1, 'capsize': 2})
    left.set_yticks(list(y))
    left.set_yticklabels([f"{name} ({c['offers']})" for name, c in powers])
    for t, style in (('Attack', '--'), ('Skill', ':')):
        left.axvline(100 * report['take_rate'][t][0], color='grey', linestyle=style, linewidth=1,
                     label=f'all {t.lower()}s: {100 * report["take_rate"][t][0]:.0f}%')
    left.axvline(100 * report['take_rate']['Power'][0], color='black', linewidth=1,
                 label=f'all powers: {100 * report["take_rate"]["Power"][0]:.0f}%')
    left.set_xlabel('Taken when offered at a card reward, % (Wilson 95% interval)')
    left.set_xlim(0, 100)
    left.legend(loc='lower right', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS.values()]
    left.legend(handles + left.get_legend_handles_labels()[0], list(COLOURS) + left.get_legend_handles_labels()[1],
                loc='lower right', fontsize=8)
    left.set_title('Champion: take rate per power (offers in brackets)', fontsize=10)

    win = report['exploratory_win_rate_by_powers_acquired_in_acts_1_2']
    keys = [k for k in ('0', '1', '2', '3+') if k in win]
    rates = [win[k] for k in keys]
    right.bar(keys, [100 * r[0] for r in rates], color='#55a868',
              yerr=[[100 * (r[0] - r[1]) for r in rates], [100 * (r[2] - r[0]) for r in rates]], capsize=3)
    for i, r in enumerate(rates):
        right.text(i, 100 * r[2] + 0.3, f'n={r[3]}', ha='center', fontsize=8)
    right.set_xlabel('Powers acquired in Acts 1 and 2')
    right.set_ylabel('Games won, %')
    right.set_title(f"Exploratory: {win['games_reaching_act_2_boss']} games\nthat reached the Act 2 boss", fontsize=10)
    fig.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / 'power_take_rates.png', dpi=110)
    print('wrote', figures / 'power_take_rates.png')


if __name__ == '__main__':
    main()
