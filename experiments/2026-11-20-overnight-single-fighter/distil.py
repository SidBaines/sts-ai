"""Run the fixed longer fit, or one conditional round of student-played teacher labels."""

import argparse
from itertools import count, islice
from pathlib import Path

from sts_policy.curriculum import world_split
from sts_rl.combat_distillation import fit
from sts_rl.distillation_collection import collect_menus
from study import BASE, CFG, CHAMP, DEMO, OLD, OUT, check_inputs, models, note, read


def main(mode, out):
    check_inputs()
    if mode == "long":
        return fit(
            0.25,
            out,
            config=CFG,
            teacher_path=CHAMP,
            directory=DEMO,
            note=note,
            fixed_steps=True,
            snapshots=CFG["snapshots"],
            replay_reference=(3200, BASE / "head.pt"),
        )
    selection = read(OUT / "first-selection.json")
    if not selection["feedback"]:
        raise ValueError("The frozen selection rule did not call for student-played feedback")
    actor = models()[selection["feedback_actor"]] / "player.pt"
    if mode == "collect":
        start, n = CFG["feedback"]
        seeds = list(islice((i for i in count(start) if world_split(i) == "train"), n))
        return collect_menus(
            out,
            6,
            teacher_path=CHAMP,
            acting_path=actor,
            roster={"train": seeds},
            limits=(48, 96),
            note=note,
        )
    settings = {**CFG, "steps": CFG["feedback_steps"]}
    return fit(
        0.25,
        out,
        config=settings,
        teacher_path=CHAMP,
        directory=DEMO,
        note=note,
        initial_player=actor,
        extra=(OUT / "feedback-data",),
        scales=read(OLD / "mix25/summary.json")["temperatures"],
        fixed_steps=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("long", "collect", "feedback"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.mode, args.out)
