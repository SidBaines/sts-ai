"""One combat PPO update for each pilot setting on exactly the same sampled games."""

from __future__ import annotations
import argparse
import gzip
from pathlib import Path

from pilot import (
    BASE,
    CHAMP,
    ROLES,
    action_diagnostics,
    atomic_json,
    begin_run,
    load_policy,
    torch,
)
from sts_rl.common import JsonlLog, save_policy
from sts_rl.online_combat import combat_optimizers, update_fighters, viable_combat_fit
import json


def sample(directory):
    result = {r: [] for r in ROLES}
    config = json.loads((directory / "config.json").read_text())
    files = sorted(directory.glob("rows-*.pt.gz"))
    if len(files) != config["processes"]:
        raise ValueError("missing sharded combat samples")
    for path in files:
        with gzip.open(path, "rb") as f:
            part = torch.load(f, map_location="cpu", weights_only=False)
        for role in ROLES:
            result[role].extend(part[role])
    return result


def fit(out, settings):
    torch.set_num_threads(6)
    old, template = load_policy(str(CHAMP))
    temps = json.loads((BASE / "fit-data" / "summary.json").read_text())["config"][
        "temperatures"
    ]
    rows = sample(BASE / "fit-data")
    held = sample(BASE / "calibration")
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "progress.jsonl")
    record = begin_run(
        out,
        {
            "temperatures": temps,
            "settings": settings,
            "train_worlds": 3072,
            "eval_worlds": 192,
        },
        CHAMP,
    )
    if not torch.backends.mps.is_available():
        raise RuntimeError("the joint pilot and both production arms require MPS")
    for rate, epochs in settings:
        name = f"lr{rate:g}-e{epochs}"
        directory = out / name
        directory.mkdir(exist_ok=True)
        if (directory / "summary.json").exists():
            continue
        model, _ = load_policy(str(CHAMP))
        opt = combat_optimizers(model, rate)
        stats = update_fighters(
            model, rows, opt, temps, rate=rate, epochs=epochs, device="mps", seed=11001
        )
        movement = {
            r: action_diagnostics(
                getattr(old, r), held[r], [temps[r]], getattr(model, r)
            )
            for r in ROLES
        }
        if not all(
            torch.equal(a, b)
            for a, b in zip(old.ooc.parameters(), model.ooc.parameters())
        ):
            raise RuntimeError("pilot changed the strategy weights")
        save_policy(
            directory / "player.pt",
            model,
            template,
            metadata={
                "stage": "combat_pilot",
                "run_record": record,
                "config": {
                    "checkpoint": str(CHAMP),
                    "combat_lr": rate,
                    "combat_epochs": epochs,
                },
            },
        )
        result = {
            "name": name,
            "rate": rate,
            "epochs": epochs,
            "steps": stats,
            "movement": movement,
            "eligible": viable_combat_fit(movement),
        }
        atomic_json(directory / "summary.json", result)
        log.emit(
            {
                "event": "fit",
                "name": name,
                "eligible": result["eligible"],
                "steps": {r: stats[r]["steps"] for r in ROLES},
            }
        )
    log.emit({"event": "done"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fallback", action="store_true")
    a = p.parse_args()
    settings = [(1e-5, 1), (3e-5, 1), (1e-4, 1), (3e-5, 2)]
    if a.fallback:
        settings = [(1e-6, 1), (3e-6, 1), (3e-4, 1)]
    fit(a.out, settings)
