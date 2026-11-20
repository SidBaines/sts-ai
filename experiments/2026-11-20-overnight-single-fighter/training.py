"""Calibrate exploration on fixed menus and run a restartable matched joint-training arm."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch

from sts_policy.model_agent import CompositeActorCritic, Act3BossActorCritic
from sts_rl.combat_distillation import Menus
from sts_rl.common import load_policy, train_tasks
from sts_rl.fresh_study import atomic_json
from sts_rl.online_combat import temperature_for_exploration
from sts_rl.ooc_trainer import OOCTrainConfig, train
from sts_rl.paired_endpoints import digest
from study import BASE, CFG, CHAMP, DEMO, HERE, OUT, check_inputs, note, read


@torch.inference_mode()
def calibration(checkpoint):
    data = Menus(DEMO, "validation")
    model, _ = load_policy(checkpoint)
    rng = np.random.default_rng(1120)
    result = {}
    for i, role in enumerate(("combat", "boss_specialist")):
        ids = rng.choice(data.groups[i], min(8192, len(data.groups[i])), replace=False)
        head = getattr(model, role, model.combat).eval()
        scores = []
        for start in range(0, len(ids), 256):
            s, a, m, _, _ = data.batch(ids[start : start + 256], "cpu")
            logits = head.actor_logits(s, a, m)
            scores.extend(row[valid] for row, valid in zip(logits, m))
        padded = torch.nn.utils.rnn.pad_sequence(scores, batch_first=True, padding_value=-torch.inf)
        result[role] = temperature_for_exploration(padded, torch.isfinite(padded), CFG["exploration"])
    return dict(checkpoint_sha256=digest(checkpoint), target=CFG["exploration"], groups=result)


def run_arm(arm, out, *, smoke=False):
    check_parent = BASE / "player.pt" if smoke else OUT / "finalist/player.pt"
    parent = CHAMP if arm == "two" else check_parent
    if not smoke and arm == "one" and digest(parent) != read(OUT / "frozen.json")["player_sha256"]:
        raise ValueError("Single-fighter parent changed after selection")
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    cal_path = out / "calibration.json"
    if not cal_path.exists():
        atomic_json(cal_path, calibration(parent))
    cal = read(cal_path)
    if cal["checkpoint_sha256"] != digest(parent):
        raise ValueError("Exploration calibration belongs to a different parent")
    settings = read(HERE / "ppo.json")
    settings.update(
        out=str(out),
        checkpoint=str(parent),
        iterations=CFG["rl_iterations"],
        games=CFG["rl_games"],
        train_world_start=CFG["rl_train_base"],
        seed=CFG["rl_seed"],
        eval_world_start=CFG["development"][0],
        eval_n=CFG["development"][1],
        train_combat=True,
        combat_lr=CFG["combat_lr"],
        combat_epochs=CFG["combat_epochs"],
        combat_temperature=cal["groups"]["combat"]["temperature"],
        boss_temperature=cal["groups"]["boss_specialist"]["temperature"],
    )
    if smoke:
        settings.update(
            iterations=1,
            games=48,
            workers=16,
            collectors=2,
            collect_chunk=8,
            eval_every=1,
            eval_n=16,
            minibatch=512,
            micro_batch=128,
            train_world_start=CFG["smoke_train_base"],
            seed=0,
            eval_world_start=CFG["smoke_eval"],
        )
    cfg = OOCTrainConfig(**settings)
    atomic_json(
        out / "training-plan.json",
        dict(
            iterations=cfg.iterations,
            checkpoint=str(parent),
            label=f"{'Verification' if smoke else 'Overnight joint learning'} · {arm}-fighter player",
        ),
    )
    train(cfg, resume=(out / "latest.pt").exists())
    endpoint = out / f"iter_{cfg.iterations:04d}.pt"
    final, _ = load_policy(endpoint)
    original, _ = load_policy(parent)
    required = ["ooc", "combat"] + (["boss_specialist"] if arm == "two" else [])
    expected_type = Act3BossActorCritic if arm == "two" else CompositeActorCritic
    if type(final) is not expected_type:
        raise ValueError("Endpoint has the wrong fighting architecture")
    if any(
        all(
            torch.equal(v, getattr(final, role).state_dict()[k])
            for k, v in getattr(original, role).state_dict().items()
        )
        for role in required
    ):
        raise ValueError("One of the required networks did not learn")
    roster = {} if smoke else read(OUT / "rl-worlds.json")
    world_count = faults = cuts = 0
    logs = {
        r["iteration"]: r
        for r in map(json.loads, (out / "log.jsonl").read_text().splitlines())
        if r["event"] == "train"
    }
    for i in range(1, cfg.iterations + 1):
        expected = (
            [t.seed for t in train_tasks(cfg.train_world_start + 10_000_000, 48)]
            if smoke
            else roster[f"rl-{i:02}"]
        )
        with gzip.open(out / f"collection_{i:04d}.json.gz", "rt") as f:
            collection = json.load(f)
        assigned = [t["seed"] for t in collection["tasks"]]
        if assigned != expected:
            raise ValueError(f"Training world roster changed at update {i}")
        completed_or_cut = [e["world_seed"] for e in collection["episodes"]]
        faulted = [f["world_seed"] for f in collection["faults"]]
        if sorted(completed_or_cut + faulted) != sorted(expected):
            raise ValueError(f"Unaccounted training games at update {i}")
        for role in ("combat", "boss_specialist"):
            diag = logs[i][role + "_exploration"]
            if not diag["choices"] or diag["max_log_prob_error"] > 0.002:
                raise ValueError(
                    f"Sampled and reconstructed {role} move probabilities disagree at update {i}"
                )
        world_count += len(assigned)
        faults += len(faulted)
        cuts += logs[i].get("cuts", 0)
    atomic_json(
        out / "finished.json",
        dict(
            parent=str(parent),
            parent_sha256=digest(parent),
            endpoint=str(endpoint),
            sha256=digest(endpoint),
            updates=cfg.iterations,
            worlds=world_count,
            faults=faults,
            cuts=cuts,
            trained=required,
            temperatures=cal["groups"],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("one", "two", "smoke"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check_inputs()
    if args.arm == "smoke":
        for arm in ("one", "two"):
            run_arm(arm, args.out / arm, smoke=True)
        atomic_json(args.out / "finished.json", dict(verified=["one", "two"], worlds_per_arm=48))
        note(
            "training verified",
            "Both architectures completed a matched 48-game wiring check with the intended networks learning.",
        )
    else:
        run_arm(args.arm, args.out)
