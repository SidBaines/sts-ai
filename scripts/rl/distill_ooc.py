#!/usr/bin/env python3
"""CLI for sts_rl.distill: every DistillConfig field is a --flag."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.distill import DistillConfig, distill  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for f in fields(DistillConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name in ("records", "out"):
            parser.add_argument(flag, required=True)
        else:
            kind = {"int": int, "float": float}.get(str(f.type), str)
            parser.add_argument(flag, type=kind, default=f.default)
    distill(DistillConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
