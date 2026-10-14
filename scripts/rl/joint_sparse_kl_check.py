#!/usr/bin/env python3
"""Offline replay of historical rejected updates; never changes study outputs."""

import argparse
import gc
import gzip
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch

    import sts_rl.joint_sparse as joint
    from sts_rl.common import load_policy
    from sts_rl.recording import begin_run, file_identity, restore_rng

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    study = Path(a.study)
    out = Path(a.out)
    if (out / "results.json").exists():
        p.error("Fresh diagnostic output required")
    inputs = [study / "anchors.pt", study / "latest.pt"] + [
        next(study.glob(f"{kind}_{i:04d}_*{suffix}"))
        for i in [14, 16, 19, 20]
        for kind, suffix in [("before", ".pt"), ("buffer", ".pt.gz")]
    ]
    record = begin_run(out, vars(a), a.checkpoint, inputs=inputs)
    before_latest = file_identity(study / "latest.pt")
    torch.set_num_threads(2)
    anchors = torch.load(study / "anchors.pt", map_location="cpu", weights_only=False)
    report = {"record": record, "diagnostic_only": True, "replays": []}
    stable = joint.masked_kl_divergence
    for iteration, legacy in [
        (20, True),
        (14, False),
        (16, False),
        (19, False),
        (20, False),
    ]:
        before = next(study.glob(f"before_{iteration:04d}_*.pt"))
        bufferpath = next(study.glob(f"buffer_{iteration:04d}_*.pt.gz"))
        payload = torch.load(before, map_location="cpu", weights_only=False)
        with gzip.open(bufferpath, "rb") as f:
            buffer = torch.load(f, map_location="cpu", weights_only=False)
        cfg = joint.JointConfig(**buffer["config"])
        actor, _ = load_policy(before)
        reference, _ = load_policy(a.checkpoint)
        joint.freeze_legacy_critics(actor)
        opts = {
            p: torch.optim.AdamW(
                [v for v in getattr(actor, p).parameters() if v.requires_grad],
                lr=getattr(cfg, p + "_lr"),
                weight_decay=0.0,
            )
            for p in joint.PHASES
        }
        for phase in opts:
            opts[phase].load_state_dict(payload["optimizer_state"][phase])
        rng = random.Random()
        restore_rng({"rng_state": buffer["rng_state"]}, rng)
        start = time.monotonic()
        hash_before = joint.identity_state(actor)
        legacy_critics = {
            p: joint.identity_state(getattr(actor, p).critic) for p in joint.PHASES
        }
        if legacy:
            joint.masked_kl_divergence = lambda p, q, mask: (
                torch.distributions.kl_divergence(
                    joint.masked_distribution(p, mask),
                    joint.masked_distribution(q, mask),
                )
            )
        try:
            result = joint.actor_update(
                actor,
                reference,
                opts,
                buffer["rows"],
                anchors,
                cfg,
                rng,
                buffer["beta"],
            )
        finally:
            joint.masked_kl_divergence = stable
        assert not result["accepted"] if legacy else result["accepted"], result
        if legacy:
            assert joint.identity_state(actor) == hash_before
        else:
            assert all(result["phases"][p]["actor_changed"] for p in joint.PHASES)
        assert all(
            legacy_critics[p] == joint.identity_state(getattr(actor, p).critic)
            for p in joint.PHASES
        )
        item = {
            "iteration": iteration,
            "implementation": "historical_categorical" if legacy else "masked_logspace",
            "before": file_identity(before),
            "buffer": file_identity(bufferpath),
            "seconds": time.monotonic() - start,
            "result": result,
            "legacy_critics_unchanged": True,
        }
        report["replays"].append(item)
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "legacy": legacy,
                    "accepted": result["accepted"],
                    "steps": result["retained_steps"],
                    "seconds": item["seconds"],
                }
            ),
            flush=True,
        )
        del actor, reference, opts, payload, buffer
        gc.collect()
    assert before_latest == file_identity(study / "latest.pt")
    report["study_latest_unchanged"] = True
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
