#!/usr/bin/env python3
"""Create a trainable composite checkpoint from combat and OOC candidate policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts_policy.composition import compose_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combat-checkpoint", type=Path, required=True)
    parser.add_argument("--ooc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compose_checkpoint(args.combat_checkpoint, args.ooc_checkpoint, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
