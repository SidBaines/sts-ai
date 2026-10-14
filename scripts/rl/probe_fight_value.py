#!/usr/bin/env python3
"""World-grouped fixed-ridge value probe; diagnostic only, never a policy update."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


ENEMY_FIELDS = ('cur_hp', 'max_hp', 'block', 'intent_damage', 'intent_hits',
                'strength', 'vulnerable', 'weak', 'poison')
EXTRA_NAMES = tuple(f'enemy_{slot}_{key}' for slot in range(3) for key in ENEMY_FIELDS) + (
    'hand_free_once', 'hand_retained', 'hand_upgrade_sum', 'hand_base_cost_sum',
    'hand_turn_cost_sum', 'draw_free_once', 'discard_free_once', 'exhaust_free_once')


def public_extras(obs):
    """Explicit allowlist; no diagnostic info, native bits, world seed or draw order."""
    combat = obs.get('state', {}).get('combat', {})
    enemies = [e for e in combat.get('enemies', []) if e.get('alive', True)]
    values = [float(enemies[i].get(key, 0)) if i < len(enemies) else 0.
              for i in range(3) for key in ENEMY_FIELDS]
    piles = combat.get('cards', {})
    hand = piles.get('hand', [])
    values += [sum(bool(c.get('free_to_play_once')) for c in hand),
               sum(bool(c.get('retain')) for c in hand),
               sum(c.get('upgrade_count', 0) for c in hand),
               sum(c.get('cost', 0) for c in hand),
               sum(c.get('cost_for_turn', c.get('cost', 0)) for c in hand)]
    values += [sum(c.get('count', 1) * bool(c.get('free_to_play_once')) for c in piles.get(pile, []))
               for pile in ('draw', 'discard', 'exhaust')]
    return values


def main():
    import numpy as np
    from sts_policy.encoding import encode_observation
    from sts_rl.recording import begin_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--ridge', type=float, default=1.)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() or args.folds < 2 or args.ridge <= 0:
        parser.error('Need a fresh output, at least two folds and positive fixed ridge')
    files = sorted(Path(args.dataset).glob('shard_*.jsonl'))
    if not files:
        parser.error('No teacher-played label shards')
    record = begin_run(str(out) + '.record', vars(args), args.checkpoint, inputs=files)
    rows = [json.loads(line) for p in files for line in p.read_text().splitlines() if line.strip()]
    if any(not r.get('teacher_played') or 'teacher_fight_win' not in r for r in rows):
        raise ValueError('Every label needs a completed teacher-played fight outcome')
    old, extra, keys = [], [], []
    for r in rows:
        obs = r['obs']
        if obs.get('phase') != 'combat':
            raise ValueError('Combat observations required')
        e = encode_observation(obs)
        old.append(np.concatenate((e.state_features, e.action_features.mean(axis=0))))
        extra.append(public_extras(obs))
        keys.append((r['seed'], obs['state']['act'], obs['state']['floor']))
    old = np.asarray(old, dtype=np.float64)
    extra = np.asarray(extra, dtype=np.float64)
    if not np.isfinite(old).all() or not np.isfinite(extra).all():
        raise ValueError('Nonfinite public features')
    target = np.asarray([r['teacher_fight_win'] for r in rows], dtype=np.float64)
    counts = Counter(keys)
    worlds = sorted({r['seed'] for r in rows})
    fights_per_world = Counter(k[0] for k in counts)
    # Each world has equal total weight; fights and decisions share that weight.
    weight = np.asarray([1 / counts[k] / fights_per_world[k[0]] for k in keys])
    fold_by_world = {s: int.from_bytes(hashlib.sha256(f'fight-value-probe-v1:{s}'.encode()).digest()[:8], 'little') % args.folds
                     for s in worlds}
    fold = np.asarray([fold_by_world[r['seed']] for r in rows])
    predictions = {name: np.empty(len(rows)) for name in ('constant', 'legacy', 'legacy_plus_public')}
    fold_records = []
    for f in range(args.folds):
        train, test = fold != f, fold == f
        train_worlds = {rows[i]['seed'] for i in np.flatnonzero(train)}
        test_worlds = {rows[i]['seed'] for i in np.flatnonzero(test)}
        if not train_worlds or not test_worlds or train_worlds & test_worlds:
            raise ValueError('Empty or overlapping world split')
        mean = np.average(target[train], weights=weight[train])
        predictions['constant'][test] = mean
        for name, features in [('legacy', old), ('legacy_plus_public', np.concatenate((old, extra), axis=1))]:
            center = np.average(features[train], axis=0, weights=weight[train])
            scale = np.sqrt(np.average((features[train] - center) ** 2, axis=0, weights=weight[train]))
            scale = np.maximum(scale, 1e-6)
            x = np.clip((features - center) / scale, -10, 10)
            w = weight[train] / weight[train].sum()
            xtw = x[train].T * w
            beta = np.linalg.solve(xtw @ x[train] + args.ridge * np.eye(x.shape[1]),
                                   xtw @ (target[train] - mean))
            predictions[name][test] = np.clip(mean + x[test] @ beta, 0, 1)
        fold_records.append({'fold': f, 'train_worlds': sorted(train_worlds), 'test_worlds': sorted(test_worlds)})
    first_by_fight = {}
    for i, key in enumerate(keys):
        first_by_fight.setdefault(key, i)
    first = np.asarray(list(first_by_fight.values()))
    first_weight = np.asarray([1 / fights_per_world[keys[i][0]] for i in first])
    metrics = {name: {'world_balanced_brier': float(np.average((p-target)**2, weights=weight)),
                     'first_label_world_balanced_brier': float(np.average((p[first]-target[first])**2, weights=first_weight))}
               for name, p in predictions.items()}
    per_world = []
    seeds = np.asarray([r['seed'] for r in rows])
    for seed in worlds:
        mask = seeds == seed
        per_world.append({'seed': seed, 'fold': fold_by_world[seed], 'fights': fights_per_world[seed],
                          'brier': {name: float(np.average((p[mask]-target[mask])**2, weights=weight[mask]))
                                    for name, p in predictions.items()}})
    differences = np.asarray([r['brier']['legacy_plus_public'] - r['brier']['legacy'] for r in per_world])
    rng = np.random.default_rng(1729)
    bootstrap = differences[rng.integers(0, len(worlds), size=(4000, len(worlds)))].mean(axis=1)
    report = {'schema': 'sts_world_grouped_fight_value_probe_v1', 'run_record': record,
              'rows': len(rows), 'worlds': len(worlds), 'fights': len(counts), 'extra_features': EXTRA_NAMES,
              'metrics': metrics, 'extra_minus_legacy_brier': float(differences.mean()),
              'world_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
              'folds': fold_records, 'per_world': per_world,
              'scope': 'Fixed-ridge linear diagnostic on search-played boss-fight states. All decisions from a world stay in one fold. Normalization fitted on training worlds only. First-label metric uses the first labelled decision per fight, which can follow automatic single-choice decisions; it is not an exact reset-state metric. No actor update, search guidance or competence claim. Small selected reached-fight dataset; independent larger validation required.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('rows', 'worlds', 'fights', 'metrics', 'extra_minus_legacy_brier', 'world_bootstrap_ci95')}))


if __name__ == '__main__':
    main()
