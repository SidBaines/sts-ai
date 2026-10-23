#!/usr/bin/env python3
"""Paired value regret and pooled agreement between combat fits on the same TEST rows.

Usage: compare_regret.py BASE_TEST_DIR ARM_TEST_DIR... [--out FILE]
Rows pair on source_row_id; intervals are 95%, clustered by world.
"""
import argparse, gzip, json, math
from collections import defaultdict
from pathlib import Path

GROUPS = ('all', 'BOSS', 'BOSS1', 'BOSS2', 'BOSS3', 'ELITE', 'MONSTER')


def rows(directory):
    path = next(p for p in (Path(directory)/'test_rows.jsonl', Path(directory)/'test_rows.jsonl.gz') if p.exists())
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return {r['source_row_id']: r for r in map(json.loads, handle) if 'value_regret' in r}


def clustered(pairs):
    """Mean of per-row differences and a world-clustered 95% interval."""
    by_world = defaultdict(list)
    for world, value in pairs:
        by_world[world].append(value)
    n, total = sum(map(len, by_world.values())), sum(map(sum, by_world.values()))
    mean = total / n
    k = len(by_world)
    var = sum((sum(v) - mean*len(v))**2 for v in by_world.values()) * k / (k-1) / n**2
    half = 1.96 * math.sqrt(var)
    return mean, mean-half, mean+half, n


def in_group(row, group):
    g = row.get('group', '')
    return group == 'all' or g == group or (group in ('BOSS', 'ELITE', 'MONSTER') and g.startswith(group))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base'); p.add_argument('arms', nargs='+'); p.add_argument('--out')
    a = p.parse_args()
    base = rows(a.base)
    report = {}
    for arm_dir in a.arms:
        arm = rows(arm_dir)
        common = [k for k in base if k in arm]
        report[arm_dir] = {}
        for group in GROUPS:
            keys = [k for k in common if in_group(base[k], group)]
            if not keys:
                continue
            reg = clustered([(base[k]['world_seed'], arm[k]['value_regret']-base[k]['value_regret']) for k in keys])
            agr = clustered([(base[k]['world_seed'], float(arm[k]['pooled_agreement'])-float(base[k]['pooled_agreement'])) for k in keys])
            b_reg = sum(base[k]['value_regret'] for k in keys)/len(keys)
            b_agr = sum(base[k]['pooled_agreement'] for k in keys)/len(keys)
            report[arm_dir][group] = dict(rows=len(keys), base_regret=b_reg, regret_diff=reg[:3],
                                         base_agreement=b_agr, agreement_diff=agr[:3])
            print(f"{Path(arm_dir).name:>16} {group:>7} n={len(keys):6d}  regret {b_reg:6.1f} {reg[0]:+6.1f} [{reg[1]:+.1f}, {reg[2]:+.1f}]"
                  f"  agreement {100*b_agr:5.1f}% {100*agr[0]:+5.1f} [{100*agr[1]:+.1f}, {100*agr[2]:+.1f}]")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
