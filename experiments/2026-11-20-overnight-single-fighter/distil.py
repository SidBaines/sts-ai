"""Run the fixed longer fit, or one conditional round of student-played teacher labels."""

import argparse
from itertools import count, islice
from pathlib import Path
import numpy as np

from sts_policy.curriculum import world_split
from sts_rl.combat_distillation import fit
from sts_rl.distillation_collection import collect_menus
from sts_rl.fresh_study import atomic_json
from study import BASE, CFG, CHAMP, DEMO, OLD, OUT, check_inputs, models, note, read


def main(mode, out):
    check_inputs()
    if mode == "verify-feedback":

        def quiet(stage, message, *, log=True):
            print(message, flush=True)

        start = CFG["smoke_train_base"] + 10_000_000
        seeds = list(islice((i for i in count(start) if world_split(i) == "train"), 16))
        folder = out / "menus"
        collect_menus(
            folder,
            1,
            teacher_path=CHAMP,
            acting_path=BASE / "player.pt",
            roster={"train": seeds},
            limits=(48, 96),
            note=quiet,
        )
        differences = 0
        with np.load(folder / "train-000.npz") as data:
            for role in ("ordinary", "boss"):
                scores = data[f"{role}_scores"]
                offsets = np.r_[0, np.cumsum(data[f"{role}_lengths"])[:-1]]
                if len(scores):
                    actual = scores[offsets + data[f"{role}_chosen"]]
                    differences += int((np.maximum.reduceat(scores, offsets) - actual > 1e-5).sum())
        fit(
            0.25,
            out / "fit",
            config={**CFG, "steps": 2, "check_every": 1},
            teacher_path=CHAMP,
            directory=DEMO,
            extra=(folder,),
            initial_player=BASE / "player.pt",
            scales=read(OLD / "mix25/summary.json")["temperatures"],
            fixed_steps=True,
            note=quiet,
        )
        atomic_json(
            out / "finished.json",
            dict(
                worlds=len(seeds),
                learner_moves_below_teacher_top=differences,
                trained_steps=read(out / "fit/summary.json")["actual_steps"],
            ),
        )
        return
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
            replay_reference=(3200, BASE / "player.pt"),
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
    parser.add_argument("mode", choices=("long", "collect", "feedback", "verify-feedback"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.mode, args.out)
