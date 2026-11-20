"""Fit a single fighter with the original study settings."""

import argparse
from pathlib import Path
from sts_rl.combat_distillation import fit as fit_menus
from study import CFG, CHAMP, OUT, check_parent, note


def fit(fraction, out):
    check_parent()
    return fit_menus(
        fraction, out, config=CFG, teacher_path=CHAMP, directory=OUT / "demonstrations", note=note
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boss-share", required=True, type=float)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    fit(a.boss_share, a.out)
