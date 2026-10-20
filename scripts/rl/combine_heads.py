#!/usr/bin/env python3
"""Splice the OOC head of one composite checkpoint with the combat head of another."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.training import load_checkpoint  # noqa: E402
from sts_rl.common import load_policy, save_policy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ooc", required=True)
    p.add_argument("--combat", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model, template = load_policy(args.ooc)
    combat_state = load_checkpoint(args.combat)["model_state"]
    # A composite stores the combat head under "combat."; a trained phase head
    # (for example a distilled combat student) stores it bare.
    prefix = "combat." if any(key.startswith("combat.") for key in combat_state) else ""
    state = model.state_dict()
    for key in state:
        if key.startswith("combat."):
            state[key] = combat_state[prefix + key[len("combat."):]].clone()
    model.load_state_dict(state)
    save_policy(args.output, model, template, metadata={"stage": "sts_rl_combined", "ooc_source": args.ooc,
                                                        "combat_source": args.combat})
    print("wrote", args.output)


if __name__ == "__main__":
    main()
