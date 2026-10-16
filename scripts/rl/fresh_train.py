#!/usr/bin/env python3
"""Supervised warm-up from scratch; old control matches champion architecture.

The old control is not parameter-matched to the entity transformer.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from sts_rl.fresh_training import run_training


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True,help='C1 source directories or immutable gzip shards')
    p.add_argument('--manifest',help='Require this exact frozen common manifest')
    p.add_argument('--probe',nargs='+',help='Required for production: immutable unlabelled 93e9 world shards; movement monitoring only')
    p.add_argument('--out',required=True)
    p.add_argument('--architecture',choices=('new','old'),required=True)
    p.add_argument('--phase',choices=('combat','ooc'),required=True)
    p.add_argument('--fit-size',choices=('5000','20000','full'),default='full',
                   help='Informative combat target prefix; OOC is full-only (optimizer excludes forced targets)')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--old-size',choices=('small','medium','large'),default='large',
                   help='Old candidate size (default: large, matching champion architecture; fresh weights)')
    p.add_argument('--old-context',choices=('none','mean'),default='none',
                   help='Old candidate menu context (default: none, matching champion architecture)')
    p.add_argument('--width',type=int,default=128,help='Entity width (default: 128)')
    p.add_argument('--layers',type=int,default=2,help='Entity transformer layers')
    p.add_argument('--heads',type=int,default=4,help='Entity attention heads')
    p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--min-steps',type=int,default=2000)
    p.add_argument('--batch',type=int,help='Minibatch size (default: 128 for either architecture)')
    p.add_argument('--defer-test',action='store_true',
                   help='Save validation-selected checkpoint without encoding/scoring TEST; finalize with fresh_score.py')
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--seed',type=int,default=1919)
    p.add_argument('--device',choices=('cpu','mps'),default='cpu')
    p.add_argument('--deadline-utc',required=True)
    return p


def main(argv=None):
    print(json.dumps(run_training(vars(parser().parse_args(argv))),allow_nan=False))

if __name__=='__main__':
    main()
