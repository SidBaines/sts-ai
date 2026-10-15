#!/usr/bin/env python3
"""Warm only independent critics on archived public TRAIN features."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import torch

    from sts_rl.common import load_policy, save_atomic
    from sts_rl.critic_training import fit, health, held_world, metrics
    from sts_rl.joint_sparse import WinCritics, identity_state
    from sts_rl.recording import begin_run, capture_rng, file_identity

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--initial", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if (out / "warm.pt").exists():
        p.error("Fresh warm-up output required")
    record = begin_run(out, vars(a), a.checkpoint, inputs=[a.features, a.initial])
    torch.set_num_threads(2)
    torch.manual_seed(73)
    actor, _ = load_policy(a.checkpoint)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor_hash = identity_state(actor)
    raw = np.load(a.features)
    data = {k: raw[k] for k in raw.files}
    data["x"] = torch.from_numpy(data["x"])
    data["y"] = torch.from_numpy(data["y"])
    data["w"] = torch.from_numpy(data["w"])
    data["held"] = np.array([held_world(s) for s in data["seed"]])
    assert set(data["seed"][data["held"]]).isdisjoint(set(data["seed"][~data["held"]]))
    critics = WinCritics()
    initial = torch.load(a.initial, map_location="cpu", weights_only=False)
    critics.load_state_dict(initial["critic_state"])
    optimizer = torch.optim.AdamW(critics.parameters(), lr=3e-4, weight_decay=0.01)
    before = metrics(critics, data)
    start = time.monotonic()
    update = fit(critics, optimizer, data, 256, 73)
    fitted = metrics(critics, data, "fit")
    held = metrics(critics, data)
    assessment = health(held, fitted)
    assert not assessment["urgent"], assessment
    assert identity_state(actor) == actor_hash
    report = {
        "schema": "joint_critic_warm_v2",
        "record": record,
        "actor_checkpoint": file_identity(a.checkpoint),
        "actor_hash_unchanged": actor_hash,
        "features": file_identity(a.features),
        "initial_critic_checkpoint": file_identity(a.initial),
        "learning_rate": 3e-4,
        "minibatch": 256,
        "passes": 1,
        "update": update,
        "before_held": before,
        "fit": fitted,
        "held": held,
        "health": assessment,
        "seconds": time.monotonic() - start,
        "fit_worlds": len(set(data["seed"][~data["held"]])),
        "held_worlds": len(set(data["seed"][data["held"]])),
    }
    save_atomic(
        {
            "critic_state": critics.state_dict(),
            "critic_optimizer_state": optimizer.state_dict(),
            "warm_record": report,
            "rng_state": capture_rng(),
        },
        out / "warm.pt",
    )
    report["checkpoint"] = file_identity(out / "warm.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "update": update,
                "health": assessment,
                "held": {k: v for k, v in held.items() if k.endswith("act0")},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
