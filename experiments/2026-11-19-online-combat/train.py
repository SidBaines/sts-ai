"""Run one arm using the pilot-frozen settings or a bounded wiring check."""

import argparse
import fcntl
import time
import json
from pathlib import Path

from pilot import BASE, CHAMP, atomic_json, temperatures, torch
from sts_rl.ooc_trainer import OOCTrainConfig, train
from sts_rl.common import load_policy


def launch_settings(base=BASE):
    """Freeze one choice for both production arms at their first actual launch."""
    lock = base / "launch-settings.lock"
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        path = base / "launch-settings.json"
        if path.exists():
            return json.loads(path.read_text())["selected"]
        original = json.loads((base / "fit-selection.json").read_text())["selected"]
        source, selected = "first-pilot", original
        expanded = base / "expanded" / "verdict.json"
        started = any((base / r / "config.json").exists() for r in ("control", "joint"))
        if expanded.exists() and not started:
            verdict = json.loads(expanded.read_text())
            if verdict["qualified"]:
                source, selected = "expanded-pilot", verdict["challenger"]
        atomic_json(path, {"source": source, "selected": selected, "time": time.time()})
        return selected


def config(out, role, smoke=False, *, setting=None):
    raw = json.loads(Path(__file__).with_name("ppo.json").read_text())
    if smoke:
        setting = {"target": 0.12, "rate": 0.00003, "epochs": 1}
        raw.update(
            iterations=1,
            games=48,
            workers=16,
            collectors=2,
            collect_chunk=8,
            eval_n=16,
            eval_every=1,
            minibatch=512,
            micro_batch=128,
            train_world_start=235_000_000_000,
            eval_world_start=237_000_000_000,
            health_every=1,
        )
    elif setting is None:
        path = BASE / "launch-settings.json"
        setting = json.loads(
            (path if path.exists() else BASE / "fit-selection.json").read_text()
        )["selected"]
    temps = temperatures(setting["target"])
    raw.update(
        out=str(out),
        checkpoint=str(CHAMP),
        combat_temperature=temps["combat"],
        boss_temperature=temps["boss_specialist"],
        train_combat=role == "joint",
        combat_lr=setting["rate"],
        combat_epochs=setting["epochs"],
    )
    return OOCTrainConfig(**raw)


def run(out, role, smoke=False):
    cfg = config(out, role, smoke, setting=None if smoke else launch_settings())
    out.mkdir(parents=True, exist_ok=True)
    # The dashboard uses the physical parent and iteration count to extend the training line.
    atomic_json(
        out / "training-plan.json",
        dict(
            iterations=cfg.iterations,
            checkpoint=str(CHAMP),
            label=f"48-game verification · {role}"
            if smoke
            else f"Online combat · {role}",
        ),
    )
    resume = (out / "latest.pt").is_file()
    train(cfg, resume=resume)


def smoke(out):
    for role in ("control", "joint"):
        folder = out / role
        if not (folder / "iter_0001.pt").is_file():
            run(folder, role, True)
    import gzip

    def records(role):
        with gzip.open(out / role / "collection_0001.json.gz", "rt") as f:
            return {
                e["world_seed"]: e["action_trace_sha256"]
                for e in json.load(f)["episodes"]
            }

    if records("control") != records("joint") or len(records("joint")) != 48:
        raise RuntimeError(
            "the first matched collections must have identical complete trajectories"
        )
    players = [
        load_policy(str(path))[0]
        for path in (
            CHAMP,
            out / "control" / "iter_0001.pt",
            out / "joint" / "iter_0001.pt",
        )
    ]

    def same(a, b, role):
        return all(
            torch.equal(x, y)
            for x, y in zip(
                getattr(a, role).parameters(), getattr(b, role).parameters()
            )
        )

    for role in ("combat", "boss_specialist"):
        if not same(*players[:2], role) or same(players[0], players[2], role):
            raise RuntimeError(f"frozen or updating arm failed weight check for {role}")
    if not same(*players[1:], "ooc") or same(*players[:2], "ooc"):
        raise RuntimeError("both matched strategy updates must move identically")
    atomic_json(
        out / "summary.json",
        dict(
            identical_worlds=48,
            strategy_updates_identical=True,
            both_joint_fighters_changed=True,
            both_control_fighters_frozen=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("smoke", "control", "joint"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.role == "smoke":
        smoke(a.out)
    else:
        run(a.out, a.role)
