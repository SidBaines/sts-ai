#!/usr/bin/env python3
"""Finalize deferred TEST after freezing checkpoint/manifest selection identities.

CPU singleton inference only; no training or hyperparameter selection.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from sts_rl.fresh_scoring import score_deferred


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True,help='Frozen JSON: selected_checkpoint, manifest, selection_criterion; controlled also requires architecture and training_condition')
    p.add_argument('--out',required=True,help='Fresh scoring directory')
    p.add_argument('--deadline-utc',required=True)
    a = p.parse_args(argv)
    print(json.dumps(score_deferred(a.selection,a.out,deadline_utc=a.deadline_utc),allow_nan=False))


if __name__=='__main__':
    main()
