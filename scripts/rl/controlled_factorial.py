#!/usr/bin/env python3
"""Strict analysis of the predeclared 16-arm representation-control fight panels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from sts_rl.controlled_factorial import compare_panel_runs, write_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('panels',))
    parser.add_argument('--expectations', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    result = compare_panel_runs(args.expectations)
    write_comparison(args.out, result)
    print(json.dumps({'schema': result['schema'], 'out': args.out,
                      'filesystem_validated': result['filesystem_validated'],
                      'primary': result['groups']['BOSS2:resampled']['hp_lost']['main_effects']}, allow_nan=False))


if __name__ == '__main__':
    main()
