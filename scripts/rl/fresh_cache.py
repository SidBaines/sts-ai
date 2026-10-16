#!/usr/bin/env python3
"""Create one optional lossless controlled combat FIT/VALIDATION cache (queue only)."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from sts_rl.controlled_cache import build_cache


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--manifest', required=True, help='Original frozen common manifest; never replaced')
    p.add_argument('--out', required=True, help='NEW directory; receipt.json is the final success marker')
    p.add_argument('--deadline-utc', required=True)
    return p


def main(argv=None):
    print(json.dumps(build_cache(vars(parser().parse_args(argv))), allow_nan=False))


if __name__ == '__main__':
    main()
