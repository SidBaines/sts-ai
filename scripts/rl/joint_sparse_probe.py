#!/usr/bin/env python3
"""Bounded sampling and CPU/MPS comparisons, with recorded conditions."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch

    from sts_rl.common import load_policy, train_tasks
    from sts_rl.joint_sparse import (
        JointConfig,
        SparseCollectors,
        WinCritics,
        actor_update,
        freeze_legacy_critics,
        identity_state,
        summary_episodes,
    )
    from sts_rl.recording import begin_run, capture_rng, restore_rng, write_collection

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--cpu-only", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    record = begin_run(out, vars(a), a.checkpoint)
    torch.set_num_threads(2)
    torch.manual_seed(19)
    actor, _ = load_policy(a.checkpoint)
    critics = WinCritics()
    report = {"record": record, "sampling": {}, "devices": {}}
    for name, kind, temp, eps, ot, oe, n in (
        []
        if a.cpu_only
        else [
            ("greedy", "eval", 1, 0, 1, 0, 128),
            ("conservative", "train", 0.25, 0.001, 0.5, 0.002, 256),
        ]
    ):
        cfg = JointConfig(
            out=str(out),
            checkpoint=a.checkpoint,
            combat_temperature=temp,
            combat_epsilon=eps,
            ooc_temperature=ot,
            ooc_epsilon=oe,
            collectors=2,
            workers=4,
        )
        pool = SparseCollectors(cfg, 4 if kind == "eval" else 2, 4)
        try:
            r = pool.collect(
                actor,
                critics,
                train_tasks(65_000_000_000, n),
                kind,
                19,
                lambda x, name=name: print(
                    json.dumps({"condition": name, "progress": x}), flush=True
                ),
            )
        finally:
            pool.close()
        write_collection(
            out / f"{name}.json.gz",
            {"config": cfg.__dict__, "episodes": r["episodes"], "faults": r["faults"]},
        )
        report["sampling"][name] = summary_episodes(r["episodes"], r["faults"], n)
    fixture = torch.load(Path(a.fixture) / "buffer.pt", weights_only=False)
    rows = []
    for phase in ("combat", "ooc"):
        rows += [r for r in fixture["rows"] if r["phase"] == phase][:1024]
    snapshot = capture_rng(random.Random(19))
    hashes = {}
    for device in ["cpu"] + (
        ["mps"] if not a.cpu_only and torch.backends.mps.is_available() else []
    ):
        m = copy.deepcopy(actor)
        ref = copy.deepcopy(actor)
        freeze_legacy_critics(m)
        rng = random.Random(19)
        restore_rng({"rng_state": snapshot}, rng)
        cfg = JointConfig(**{**fixture["config"], "device": device})
        opts = {
            p: torch.optim.AdamW(
                [v for v in getattr(m, p).parameters() if v.requires_grad],
                lr=1e-5,
                weight_decay=0,
            )
            for p in ("combat", "ooc")
        }
        start = time.monotonic()
        update = actor_update(m, ref, opts, rows, fixture["anchors"], cfg, rng, 0.1)
        elapsed = time.monotonic() - start
        report["devices"][device] = {
            "seconds": elapsed,
            "accepted": update["accepted"],
            "phases": update.get("phases"),
            "drift": update["drift"],
        }
        hashes[device] = identity_state(m)
        if device == "cpu":
            baseline = copy.deepcopy(m)
        else:
            report["devices"][device]["max_parameter_delta_cpu"] = max(
                float((x - y).abs().max())
                for x, y in zip(m.parameters(), baseline.parameters())
            )
    report["accelerator_note"] = (
        "Previous separate device probe exited SIGABRT (-6); CPU selected."
        if a.cpu_only
        else None
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
