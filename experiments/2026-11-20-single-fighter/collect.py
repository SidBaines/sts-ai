"""Collect champion decisions on the originally assigned training and validation worlds."""

import argparse
from itertools import count, islice
from pathlib import Path
from sts_policy.curriculum import world_split
from sts_rl.distillation_collection import collect_menus
from study import CFG, CHAMP, OUT, check_parent, note


def main(out, jobs):
    check_parent()
    roster = {}
    for split, tag in (("train", "train"), ("validation", "dev")):
        start, n = CFG[split]
        roster[split] = list(islice((i for i in count(start) if world_split(i) == tag), n))
    collect_menus(
        out,
        jobs,
        teacher_path=CHAMP,
        acting_path=CHAMP,
        roster=roster,
        limits=(CFG["ordinary_samples_per_world"], CFG["boss_samples_per_world"]),
        note=note,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT / "demonstrations")
    p.add_argument("--processes", type=int, default=6)
    a = p.parse_args()
    main(a.out, a.processes)
