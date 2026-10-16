#!/usr/bin/env python3
"""Strict generic frozen LEFT-minus-RIGHT development comparisons; no verdict."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from sts_rl.controlled_comparison import (compare_games_runs, compare_labels_runs,
                                         compare_panel_runs, write_comparison)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('labels', 'panel', 'games'))
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--expectations', required=True)
    parser.add_argument('--cohort', help='Required frozen exact row-ID/seed list for labels/games')
    parser.add_argument('--grid', help='Required frozen panel grid')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    source = args.grid if args.mode == 'panel' else args.cohort
    if not source:
        parser.error('panel requires --grid; labels/games require --cohort')
    compare = dict(labels=compare_labels_runs, panel=compare_panel_runs, games=compare_games_runs)[args.mode]
    result = compare(args.left, args.right, source, args.expectations)
    write_comparison(args.out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
