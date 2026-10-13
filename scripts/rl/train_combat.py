#!/usr/bin/env python3
"""CLI for sts_rl.combat_trainer: every CombatTrainConfig field is a --flag."""
from __future__ import annotations

import argparse
from dataclasses import fields
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.combat_trainer import CombatTrainConfig, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for f in fields(CombatTrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.name in ("out", "bank", "checkpoint"):
            parser.add_argument(flag, required=True)
        else:
            kind = {"int": int, "float": float}.get(str(f.type), str)
            parser.add_argument(flag, type=kind, default=f.default)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    train(CombatTrainConfig(**args), resume=resume)


if __name__ == "__main__":
    main()
