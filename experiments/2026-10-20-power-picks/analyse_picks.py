#!/usr/bin/env python3
"""Why do the champion's decks hold few powers? Offers, picks and preferences.

    analyse_picks.py <champion_rewards.jsonl> <cards_vs_skip.csv or -> <report.json> <figures dir>

Needs numpy only. Every rate is given with a Wilson 95% interval and its count.
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SKIP = 'skip card reward'
TYPES = ('Attack', 'Skill', 'Power')
RARITIES = ('Common', 'Uncommon', 'Rare')


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None, None, 0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h, n]


def base_name(name):
    return name.rstrip('+').strip()


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    oracle_csv, out, figures = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    rewards = [(r, w) for r in rows for w in r['rewards'] if w.get('chosen') is not None]
    report = {'worlds': len(rows), 'reward_screens': len(rewards)}
    card_type = {}
    for _, w in rewards:
        for o in w['options']:
            if o['type'] in TYPES:
                card_type[o['card']] = o['type']
    for r in rows:
        for s in r['shops']:
            for o in s['offers']:
                card_type[o['card']] = o['type']

    # A. What is offered, by act.
    offered = defaultdict(Counter)
    screens_with_power = Counter()
    for _, w in rewards:
        act = w['act']
        for o in w['options']:
            if o['type'] in TYPES:
                offered[act][(o['type'], o['rarity'])] += 1
                offered['all'][(o['type'], o['rarity'])] += 1
        if any(o['type'] == 'Power' for o in w['options']):
            screens_with_power[act] += 1
            screens_with_power['all'] += 1
    screens = Counter(w['act'] for _, w in rewards)
    screens['all'] = len(rewards)
    report['offers'] = {str(act): {
        'screens': screens[act],
        'screens_with_a_power': wilson(screens_with_power[act], screens[act]),
        'option_share_by_type': {t: sum(v for (tt, _), v in c.items() if tt == t) / max(1, sum(c.values())) for t in TYPES},
        'option_share_by_type_rarity': {f'{t} {ra}': c[(t, ra)] / max(1, sum(c.values())) for t in TYPES for ra in RARITIES}}
        for act, c in offered.items()}

    # B. Take rate per offered option, by type and rarity, by act.
    taken = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for _, w in rewards:
        for o in w['options']:
            if o['type'] not in TYPES:
                continue
            for key in ((o['type'],), (o['type'], o['rarity']), (w['act'], o['type']), (w['act'], o['type'], o['rarity'])):
                cell = taken[len(key)][key]
                cell[1] += 1
                cell[0] += w['chosen'] == o['card'] and w['chosen_type'] == o['type']
    report['take_rate'] = {' '.join(map(str, k)): wilson(*v) for n in taken for k, v in taken[n].items()}
    report['skip_rate'] = wilson(sum(w['chosen'] == SKIP for _, w in rewards), len(rewards))
    report['proceed_without_card_rate'] = wilson(sum(bool(w['proceeded']) for _, w in rewards), len(rewards))

    # Within-screen: when exactly one power is offered, what wins?
    head = Counter()
    for _, w in rewards:
        powers = [o for o in w['options'] if o['type'] == 'Power']
        if len(powers) == 1:
            head[w['chosen_type']] += 1
    report['screens_with_one_power_chosen_type'] = {k: wilson(v, sum(head.values())) for k, v in head.items()}

    # C. Per-card take rates and mean probability when offered.
    per_card = defaultdict(lambda: {'offers': 0, 'taken': 0, 'probs': [], 'type': None, 'rarity': None, 'acts': Counter()})
    for _, w in rewards:
        for o in w['options']:
            if o['type'] not in TYPES:
                continue
            c = per_card[o['card']]
            c['offers'] += 1
            c['taken'] += w['chosen'] == o['card']
            c['probs'].append(o['prob'] if o['prob'] is not None else float('nan'))
            c['type'], c['rarity'] = o['type'], o['rarity']
            c['acts'][w['act']] += 1
    report['per_card'] = {name: {'type': c['type'], 'rarity': c['rarity'], 'offers': c['offers'],
                                 'take_rate': wilson(c['taken'], c['offers']),
                                 'mean_prob': float(np.nanmean(c['probs'])),
                                 'median_prob': float(np.nanmedian(c['probs']))}
                          for name, c in sorted(per_card.items(), key=lambda kv: -kv[1]['offers'])}

    # D. Preference strength on options not taken: is a rejected power a near miss?
    strength = defaultdict(list)
    for _, w in rewards:
        for o in w['options']:
            if o['type'] in TYPES and w['chosen'] != o['card'] and o['prob'] is not None:
                strength[o['type']].append(o['prob'])
    report['rejected_option_probability'] = {
        t: {'n': len(v), 'median': float(np.median(v)), 'share_below_0.01': float(np.mean(np.array(v) < .01)),
            'share_above_0.1': float(np.mean(np.array(v) > .1)), 'share_above_0.3': float(np.mean(np.array(v) > .3))}
        for t, v in strength.items()}

    # E. Powers per game: taken at rewards, offered at rewards, bought, in the final deck.
    per_game = []
    for r in rows:
        ws = [w for w in r['rewards'] if w.get('chosen') is not None]
        taken_powers = sum(w['chosen_type'] == 'Power' for w in ws)
        screens_offering = sum(any(o['type'] == 'Power' for o in w['options']) for w in ws)
        bought = sum(card_type.get(c) == 'Power' for s in r['shops'] for c in s['bought'])
        final = [base_name(c) for c in (r['final_deck'] or [])]
        in_deck = sum(card_type.get(c) == 'Power' for c in final)
        unknown = sum(c not in card_type for c in final if c not in ('Strike', 'Defend', 'Bash'))
        per_game.append({'seed': r['seed'], 'win': r['win'], 'floor': r['floor'], 'screens': len(ws),
                         'screens_offering_power': screens_offering, 'powers_taken': taken_powers,
                         'powers_bought': bought, 'powers_in_final_deck': in_deck, 'final_deck_size': len(final),
                         'unknown_type_cards': unknown})
    g = per_game
    report['per_game'] = {
        'mean_reward_screens': float(np.mean([x['screens'] for x in g])),
        'mean_screens_offering_a_power': float(np.mean([x['screens_offering_power'] for x in g])),
        'mean_powers_taken_at_rewards': float(np.mean([x['powers_taken'] for x in g])),
        'mean_powers_bought': float(np.mean([x['powers_bought'] for x in g])),
        'mean_powers_in_final_deck': float(np.mean([x['powers_in_final_deck'] for x in g])),
        'mean_final_deck_size': float(np.mean([x['final_deck_size'] for x in g])),
        'final_deck_cards_of_unknown_type': int(sum(x['unknown_type_cards'] for x in g)),
        'share_of_games_with_no_power_in_final_deck': wilson(sum(x['powers_in_final_deck'] == 0 for x in g), len(g))}

    # F. Shops: power offers and purchases.
    shop_offers, shop_bought, shop_power_screens, shops = Counter(), Counter(), 0, 0
    affordable = Counter()
    for r in rows:
        for s in r['shops']:
            shops += 1
            if any(o['type'] == 'Power' for o in s['offers']):
                shop_power_screens += 1
            for o in s['offers']:
                shop_offers[o['type']] += 1
                shop_bought[o['type']] += o['card'] in s['bought']
                price = int(o['description'].rsplit('for ', 1)[1].rstrip('g'))
                if o['type'] == 'Power':
                    affordable['offers'] += 1
                    affordable['affordable'] += price <= s['gold']
                    affordable['bought_when_affordable'] += price <= s['gold'] and o['card'] in s['bought']
    report['shops'] = {'visits': shops, 'visits_offering_a_power': wilson(shop_power_screens, shops),
                       'buy_rate_by_type': {t: wilson(shop_bought[t], shop_offers[t]) for t in TYPES},
                       'power_offers_affordable': wilson(affordable['affordable'], affordable['offers']),
                       'power_bought_when_affordable': wilson(affordable['bought_when_affordable'], affordable['affordable'])}

    # G. Exploratory: outcome by powers taken in Acts 1 and 2, among games that reached the Act 2 boss.
    reached = [r for r in rows if (r['floor'] or 0) >= 33]
    by_count = defaultdict(lambda: [0, 0])
    for r in reached:
        k = sum(w.get('chosen_type') == 'Power' for w in r['rewards'] if w['act'] <= 2 and w.get('chosen') is not None)
        k += sum(card_type.get(c) == 'Power' for s in r['shops'] if s['act'] <= 2 for c in s['bought'])
        cell = by_count[min(k, 3)]
        cell[1] += 1
        cell[0] += bool(r['win'])
    report['exploratory_win_rate_by_powers_acquired_in_acts_1_2'] = {
        'games_reaching_act_2_boss': len(reached),
        **{('3+' if k == 3 else str(k)): wilson(*v) for k, v in sorted(by_count.items())}}

    # H. Boss rewards: screens whose cards are all rare. Which rares are taken, and how often nothing is.
    boss = [w for _, w in rewards if all(o['rarity'] == 'Rare' for o in w['options'] if o['type'] != 'Skip')]
    boss_offers, boss_taken = Counter(), Counter()
    for w in boss:
        for o in w['options']:
            if o['type'] != 'Skip':
                boss_offers[o['card']] += 1
                boss_taken[o['card']] += w['chosen'] == o['card']
    best_card_prob = [max(o['prob'] for o in w['options'] if o['type'] != 'Skip' and o['prob'] is not None)
                      for w in boss if w['chosen'] == SKIP and any(o['prob'] is not None for o in w['options'])]
    report['boss_rewards'] = {
        'screens': len(boss), 'rooms': dict(Counter(w['room'] for w in boss)),
        'skip_rate_by_act': {str(a): wilson(sum(w['chosen'] == SKIP for w in boss if w['act'] == a),
                                            sum(w['act'] == a for w in boss)) for a in (1, 2, 3)},
        'skip_rate_elsewhere': wilson(sum(w['chosen'] == SKIP for _, w in rewards if w not in boss),
                                      len(rewards) - len(boss)),
        'best_card_probability_when_skipping': {
            'median': float(np.median(best_card_prob)), 'share_above_0.3': float(np.mean(np.array(best_card_prob) > .3))},
        'take_rate_per_card': {card: wilson(boss_taken[card], n) for card, n in boss_offers.most_common() if n >= 30}}

    # I. Luck: the three powers the champion almost always takes.
    favourites = ('Juggernaut', 'Inflame', 'Metallicize')
    fav_offered = sum(any(base_name(o['card']) in favourites for w in r['rewards'] for o in w['options'])
                      or any(base_name(o['card']) in favourites for s in r['shops'] for o in s['offers']) for r in rows)
    fav_in_deck = sum(any(base_name(c) in favourites for c in (r['final_deck'] or [])) for r in rows)
    report['favourite_powers'] = {'cards': favourites,
                                  'games_offered_one_at_a_reward_or_shop': wilson(fav_offered, len(rows)),
                                  'games_with_one_in_the_final_deck': wilson(fav_in_deck, len(rows))}

    # J. Demon Form: what the champion does instead when it declines it.
    instead, instead_type, by_act = Counter(), Counter(), defaultdict(lambda: [0, 0])
    for _, w in rewards:
        if any(o['card'] == 'Demon Form' for o in w['options']):
            by_act[w['act']][1] += 1
            if w['chosen'] == 'Demon Form':
                by_act[w['act']][0] += 1
            else:
                instead[w['chosen']] += 1
                instead_type[w['chosen_type']] += 1
    declined = sum(instead.values())
    report['demon_form'] = {'take_rate_by_act': {str(a): wilson(*v) for a, v in sorted(by_act.items())},
                            'declined': declined,
                            'chosen_instead_by_type': {k: v / declined for k, v in instead_type.items()},
                            'chosen_instead_top': instead.most_common(8)}

    # K. The Act 2 oracle's view of powers (option test), where available.
    if oracle_csv != '-':
        oracle = {row['card']: row for row in csv.DictReader(open(oracle_csv))}
        report['oracle_act2_powers'] = {name: {'offers': int(row['offers']),
                                               'boss_damage_vs_skip': float(row['boss_damage_vs_skip']),
                                               'champion_took': float(row['champion_took'])}
                                        for name, row in oracle.items() if card_type.get(base_name(name)) == 'Power'}
        by_type = defaultdict(list)
        for name, row in oracle.items():
            t = card_type.get(base_name(name))
            if t:
                by_type[t].append((float(row['boss_damage_vs_skip']), int(row['offers'])))
        report['oracle_act2_mean_value_vs_skip_by_type'] = {
            t: {'offers': sum(n for _, n in v), 'mean_boss_damage_vs_skip': sum(x * n for x, n in v) / sum(n for _, n in v)}
            for t, v in by_type.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    figures.mkdir(parents=True, exist_ok=True)
    with open(figures / 'take_rates.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'take_rate', 'low', 'high', 'offers'])
        for k, v in report['take_rate'].items():
            w.writerow([k, *v])
    with open(figures / 'per_card.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['card', 'type', 'rarity', 'offers', 'take_rate', 'low', 'high', 'mean_prob', 'median_prob'])
        for name, c in report['per_card'].items():
            w.writerow([name, c['type'], c['rarity'], c['offers'], *c['take_rate'][:3], c['mean_prob'], c['median_prob']])

    def pct(v):
        return f'{100 * v[0]:.1f}% [{100 * v[1]:.1f}, {100 * v[2]:.1f}] (n={v[3]})' if v[0] is not None else 'n/a'
    print(f"{report['worlds']} worlds, {report['reward_screens']} card-reward decisions")
    for act in ('all', '1', '2', '3'):
        o = report['offers'].get(act)
        if o:
            print(f"act {act}: {o['screens']} screens; options by type", {t: round(s, 3) for t, s in o['option_share_by_type'].items()},
                  '; screens with a power', pct(o['screens_with_a_power']))
    print('take rate by type:', {t: pct(report['take_rate'][t]) for t in TYPES})
    for t in TYPES:
        print(f'  {t}:', {ra: pct(report['take_rate'][f'{t} {ra}']) for ra in RARITIES if f'{t} {ra}' in report['take_rate']})
    for act in (1, 2, 3):
        print(f'  act {act}:', {t: pct(report['take_rate'][f'{act} {t}']) for t in TYPES if f'{act} {t}' in report['take_rate']})
    print('skip rate', pct(report['skip_rate']), '; proceeded without a card', pct(report['proceed_without_card_rate']))
    print('one power offered, chosen type:', {k: pct(v) for k, v in report['screens_with_one_power_chosen_type'].items()})
    print('rejected option probability:', json.dumps(report['rejected_option_probability']))
    print('per game:', json.dumps(report['per_game']))
    print('shops:', json.dumps(report['shops']))
    print('powers:')
    for name, c in report['per_card'].items():
        if c['type'] == 'Power':
            print(f"  {name:16s} {c['rarity']:9s} offers {c['offers']:4d} take {pct(c['take_rate'])} mean prob {c['mean_prob']:.2f} median {c['median_prob']:.3f}")
    print('most offered non-powers:')
    for name, c in list((n, c) for n, c in report['per_card'].items() if c['type'] != 'Power')[:12]:
        print(f"  {name:16s} {c['type']:6s} {c['rarity']:9s} offers {c['offers']:4d} take {pct(c['take_rate'])}")
    print('exploratory win rate by powers acquired in acts 1-2:', json.dumps(report['exploratory_win_rate_by_powers_acquired_in_acts_1_2']))
    b = report['boss_rewards']
    print(f"boss rewards (all-rare screens): {b['screens']} {b['rooms']}; skip by act",
          {a: pct(v) for a, v in b['skip_rate_by_act'].items()}, '; elsewhere', pct(b['skip_rate_elsewhere']))
    print('  best card probability when skipping:', b['best_card_probability_when_skipping'])
    print('  take rate per rare:', [(c, v[3], round(100 * v[0])) for c, v in b['take_rate_per_card'].items()])
    print('favourite powers:', json.dumps(report['favourite_powers']))
    print('demon form:', json.dumps(report['demon_form']))
    if 'oracle_act2_powers' in report:
        print('oracle act 2 powers:', json.dumps(report['oracle_act2_powers']))
        print('oracle act 2 by type:', json.dumps(report['oracle_act2_mean_value_vs_skip_by_type']))


if __name__ == '__main__':
    main()
