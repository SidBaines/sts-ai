#!/usr/bin/env python3
"""Native receipt, likelihood, update, rollback and fresh resume checks."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch

    from sts_rl.common import load_policy, save_atomic, train_tasks
    from sts_rl.joint_sparse import (
        JointConfig,
        SparseCollectors,
        WinCritics,
        actor_update,
        critic_update,
        freeze_legacy_critics,
        identity_state,
        likelihood_error,
        save_buffer,
    )
    from sts_rl.recording import begin_run, capture_rng, restore_rng

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    if (out / "acceptance.json").exists() and not a.verify:
        p.error("Fresh output required")
    cfg = JointConfig(
        out=str(out),
        checkpoint=a.checkpoint,
        games=a.games,
        anchor_games=a.games,
        collectors=2,
        workers=4,
        max_games=a.games,
        eval_n=16,
        anchor_per_bucket=32,
        device="cpu",
    )
    if a.verify:
        buffer = torch.load(out / "buffer.pt", weights_only=False)
        cfg = JointConfig(**buffer["config"])
        fixture = torch.load(out / "resume_fixture.pt", weights_only=False)
        expected = torch.load(out / "expected_resume.pt", weights_only=False)
        torch.set_num_threads(2)
        actor, _ = load_policy(a.checkpoint)
        ref, _ = load_policy(a.checkpoint)
        freeze_legacy_critics(actor)
        critics = WinCritics()
        rng = random.Random()
        actor.load_state_dict(fixture["actor"])
        critics.load_state_dict(fixture["critics"])
        opts = {
            phase: torch.optim.AdamW(
                [x for x in getattr(actor, phase).parameters() if x.requires_grad],
                lr=1e-5,
                weight_decay=0.0,
            )
            for phase in ("combat", "ooc")
        }
        for phase in opts:
            opts[phase].load_state_dict(fixture["optimizers"][phase])
        co = torch.optim.AdamW(critics.parameters(), lr=0.0003)
        co.load_state_dict(fixture["critic_optimizer"])
        restore_rng(fixture, rng)
        update = actor_update(
            actor, ref, opts, buffer["rows"], buffer["anchors"], cfg, rng, 0.1
        )
        critic_update(critics, co, buffer["rows"], cfg)
        assert identity_state(actor) == expected["actor_hash"]
        assert identity_state(critics) == expected["critic_hash"]
        assert rng.getstate() == expected["local_rng"]
        for phase in opts:
            for key, st in expected["optimizers"][phase]["state"].items():
                for name, v in st.items():
                    assert torch.equal(
                        v, opts[phase].state_dict()["state"][key][name]
                    ), (phase, key, name)
        report = json.loads((out / "acceptance.json").read_text())
        report["fresh_process_resume_exact"] = True
        save_buffer(buffer, out / "optimizer_buffer.pt.gz")
        report["compressed_buffer_bytes"] = (
            (out / "optimizer_buffer.pt.gz").stat().st_size
        )
        (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"fresh_process_resume_exact": True}), flush=True)
        return
    record = begin_run(out, vars(a), a.checkpoint)
    torch.set_num_threads(2)
    torch.manual_seed(19)
    actor, _ = load_policy(a.checkpoint)
    ref, _ = load_policy(a.checkpoint)
    freeze_legacy_critics(actor)
    critics = WinCritics()
    rng = random.Random(19)
    pool = SparseCollectors(cfg)
    try:
        r = pool.collect(
            actor,
            critics,
            train_tasks(64_000_000_000, a.games),
            "train",
            19,
            lambda x: print(json.dumps({"progress": x}), flush=True),
        )
    finally:
        pool.close()
    assert (
        not r["faults"]
        and len(r["episodes"]) == a.games
        and all(e["classification"] in ("win", "loss") for e in r["episodes"])
    )
    assert all(e["reward"] in (0.0, 1.0) for e in r["episodes"])
    error = likelihood_error(actor, r["rows"], cfg)
    assert error < 1e-3, error
    anchors = []
    for phase in ("combat", "ooc"):
        anchors += [x for x in r["rows"] if x["phase"] == phase][
            :: max(1, len([x for x in r["rows"] if x["phase"] == phase]) // 64)
        ][:64]
    opts = {
        phase: torch.optim.AdamW(
            [x for x in getattr(actor, phase).parameters() if x.requires_grad],
            lr=1e-5,
            weight_decay=0.0,
        )
        for phase in ("combat", "ooc")
    }
    # This is a pipeline check. Real outcome targets are preserved even if the tiny
    # smoke has no wins; the production minimum-success guard is independently tested.
    before = {
        phase: identity_state(getattr(actor, phase)) for phase in ("combat", "ooc")
    }
    legacy = {
        phase: identity_state(getattr(actor, phase).critic)
        for phase in ("combat", "ooc")
    }
    update = actor_update(actor, ref, opts, r["rows"], anchors, cfg, rng, 0.1)
    assert update["accepted"], update
    assert all(legacy[p] == identity_state(getattr(actor, p).critic) for p in legacy)
    changed = {
        phase: before[phase] != identity_state(getattr(actor, phase))
        for phase in before
    }
    assert all(changed.values()), changed
    co = torch.optim.AdamW(critics.parameters(), lr=0.0003)
    critic = critic_update(critics, co, r["rows"], cfg)
    save_atomic(
        {
            "actor": actor.state_dict(),
            "critics": critics.state_dict(),
            "optimizers": {p: o.state_dict() for p, o in opts.items()},
            "critic_optimizer": co.state_dict(),
            "rng_state": capture_rng(rng),
        },
        out / "resume_fixture.pt",
    )
    save_atomic(
        {"rows": r["rows"], "anchors": anchors, "config": cfg.__dict__},
        out / "buffer.pt",
    )
    # Same next step is replayed in a fresh Python process by --verify.
    second = actor_update(actor, ref, opts, r["rows"], anchors, cfg, rng, 0.1)
    assert second["accepted"], second
    critic_update(critics, co, r["rows"], cfg)
    save_atomic(
        {
            "actor_hash": identity_state(actor),
            "critic_hash": identity_state(critics),
            "local_rng": rng.getstate(),
            "optimizers": {p: o.state_dict() for p, o in opts.items()},
        },
        out / "expected_resume.pt",
    )
    report = {
        "schema": "joint_sparse_preflight_v1",
        "record": record,
        "games": a.games,
        "wins": sum(e["classification"] == "win" for e in r["episodes"]),
        "faults": 0,
        "reward_binary_exact": True,
        "likelihood_max_error": error,
        "both_heads_changed": changed,
        "actor_update": update,
        "critic": critic,
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
