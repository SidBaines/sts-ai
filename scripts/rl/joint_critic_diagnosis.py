#!/usr/bin/env python3
"""Bounded offline critic audits; no actor learning, rollouts or pilot resume."""

import argparse
import gc
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def held(seed):
    return int(hashlib.sha256(str(int(seed)).encode()).hexdigest()[:8], 16) % 5 == 0


def scores(y, p, w):
    import numpy as np

    w = w / w.sum()
    mu = float((w * y).sum())
    res = y - p
    variance = float((w * (y - mu) ** 2).sum())
    resmean = float((w * res).sum())
    order = np.argsort(p)
    sortp = p[order]
    start = np.r_[0, np.flatnonzero(np.diff(sortp)) + 1]
    pw = np.add.reduceat((w * y)[order], start)
    nw = np.add.reduceat((w * (1 - y))[order], start)
    den = float(pw.sum() * nw.sum())
    auc = float((pw * (np.cumsum(nw) - nw + 0.5 * nw)).sum() / den) if den else None
    return {
        "n": len(y),
        "return_mean": mu,
        "predicted_mean": float((w * p).sum()),
        "brier": float((w * res**2).sum()),
        "bce": float(
            -(
                w
                * (
                    y * np.log(np.clip(p, 1e-12, 1))
                    + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))
                )
            ).sum()
        ),
        "explained_variance": 1 - float((w * (res - resmean) ** 2).sum()) / variance
        if variance
        else None,
        "auc": auc,
        "predicted_p05": float(np.quantile(p, 0.05)),
        "predicted_p95": float(np.quantile(p, 0.95)),
    }


def main():
    import numpy as np
    import torch

    from sts_rl.common import save_atomic
    from sts_rl.joint_sparse import (
        PHASES,
        JointConfig,
        WinCritics,
        batch_rows,
        critic_update,
        identity_state,
    )
    from sts_rl.recording import begin_run, file_identity

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--study", required=True)
    a = p.parse_args()
    out = Path(a.out)
    study = Path(a.study)
    if (out / "results.json").exists():
        p.error("Fresh diagnostic output required")
    buffers = sorted(study.glob("buffer_*.pt.gz"))
    initial_file = next(study.glob("before_0001_*.pt"))
    current_file = study / "latest.pt"
    record = begin_run(out, vars(a), inputs=[initial_file, current_file, *buffers])
    initial = torch.load(initial_file, map_location="cpu", weights_only=False)
    current = torch.load(current_file, map_location="cpu", weights_only=False)
    cfg = JointConfig(**initial["metadata"]["config"])
    torch.set_num_threads(2)
    torch.manual_seed(73)
    original = WinCritics()
    original.load_state_dict(initial["critic_state"])
    optimizer = torch.optim.AdamW(
        original.parameters(), lr=cfg.critic_lr, weight_decay=0.01
    )
    optimizer.load_state_dict(initial["critic_optimizer_state"])
    parts = []
    small = defaultdict(list)
    audit = {
        "buffers": len(buffers),
        "phase_mismatches": 0,
        "act_feature_error_max": 0.0,
        "nonfinite_features": 0,
        "nonbinary_labels": 0,
        "label_conflicts": 0,
    }
    world_labels = {}
    report = {
        "record": record,
        "diagnostic_only": True,
        "trials": {},
        "training_pilot_unchanged": True,
    }
    latest_identity = file_identity(current_file)
    for index, path in enumerate(buffers):
        with gzip.open(path, "rb") as f:
            buffer = torch.load(f, map_location="cpu", weights_only=False)
        rows = buffer["rows"]
        critic_update(original, optimizer, rows, cfg)
        selected = []
        for phaseid, phase in enumerate(PHASES):
            byworld = defaultdict(list)
            for r in rows:
                if r["phase"] == phase:
                    byworld[r["seed"]].append(r)
            for seed, rs in byworld.items():
                chosen = rs[:: max(1, len(rs) // 8)][:8]
                selected.extend((phaseid, r, 1 / len(chosen)) for r in chosen)
                last = rs[-1]
                key = (phase, int(last["return"]))
                if len(small[key]) < 64:
                    small[key].append(last)
        features = []
        for start in range(0, len(selected), 256):
            features.append(
                WinCritics.features(
                    batch_rows([r for _, r, _ in selected[start : start + 256]])
                ).numpy()
            )
        x = np.concatenate(features)
        y = np.array([r["return"] for _, r, _ in selected], np.float32)
        seeds = np.array([r["seed"] for _, r, _ in selected], np.int64)
        phase = np.array([p for p, _, _ in selected], np.int8)
        acts = np.array([r["act"] for _, r, _ in selected], np.int8)
        weights = np.array([w for _, _, w in selected], np.float32)
        audit["phase_mismatches"] += int(((x[:, 7] > 0.5) != (phase == 0)).sum())
        audit["act_feature_error_max"] = max(
            audit["act_feature_error_max"], float(np.max(np.abs(x[:, 4] - acts / 3)))
        )
        audit["nonfinite_features"] += int((~np.isfinite(x)).sum())
        audit["nonbinary_labels"] += int((~np.isin(y, [0, 1])).sum())
        for seed, label in zip(seeds, y):
            if seed in world_labels and world_labels[seed] != label:
                audit["label_conflicts"] += 1
            world_labels[seed] = label
        parts.append(
            {
                "x": x,
                "y": y,
                "seed": seeds,
                "phase": phase,
                "act": acts,
                "w": weights,
                "batch": np.full(len(y), index, dtype=np.int16),
            }
        )
        print(
            json.dumps(
                {
                    "event": "buffer_audit",
                    "iteration": index + 1,
                    "selected": len(y),
                    "critic_steps": index + 1,
                    "seconds": time.time(),
                }
            ),
            flush=True,
        )
        del rows, buffer, selected, byworld, rs, chosen, last, features
        gc.collect()
    data = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    del parts
    gc.collect()
    np.savez_compressed(out / "public_critic_features.npz", **data)
    audit["selected_states"] = len(data["y"])
    audit["worlds"] = len(world_labels)
    audit["wins"] = int(sum(world_labels.values()))
    audit["varying_features"] = int((data["x"].std(0) > 1e-7).sum())
    audit["max_feature_abs"] = float(np.abs(data["x"]).max())
    audit["initial_trained_critic_hash"] = identity_state(original)
    expected = WinCritics()
    expected.load_state_dict(current["critic_state"])
    audit["historical_20_steps_exact"] = identity_state(original) == identity_state(
        expected
    )
    audit["historical_max_parameter_error"] = max(
        float((p - q).abs().max())
        for p, q in zip(original.parameters(), expected.parameters())
    )
    assert audit["historical_20_steps_exact"], audit
    assert (
        audit["phase_mismatches"] == 0
        and audit["act_feature_error_max"] < 1e-6
        and not audit["nonfinite_features"]
        and not audit["label_conflicts"]
        and not audit["nonbinary_labels"]
    )
    duplicate = defaultdict(lambda: [0, 0, 0])
    for x, y, phase in zip(data["x"], data["y"], data["phase"]):
        key = (int(phase), hashlib.sha256(x.tobytes()).digest())
        duplicate[key][int(y)] += 1
        duplicate[key][2] += 1
    conflict = [v for v in duplicate.values() if v[0] and v[1]]
    audit["opposite_label_feature_groups"] = len(conflict)
    audit["states_in_opposite_label_groups"] = sum(v[2] for v in conflict)
    del duplicate
    hold = np.array([held(s) for s in data["seed"]], bool)
    fit = ~hold
    assert set(data["seed"][hold]).isdisjoint(set(data["seed"][fit]))
    audit["fit_worlds"] = len(set(data["seed"][fit]))
    audit["held_worlds"] = len(set(data["seed"][hold]))
    report["data_audit"] = audit
    X = torch.from_numpy(data["x"])
    Y = torch.from_numpy(data["y"])
    W = torch.from_numpy(data["w"])

    def evaluate(model=None, baseline=None):
        preds = np.zeros(len(Y), np.float64)
        for phaseid, phase in enumerate(PHASES):
            indices = np.flatnonzero(data["phase"] == phaseid)
            if baseline:
                train = indices[fit[indices]]
                mean = float(np.average(data["y"][train], weights=data["w"][train]))
                preds[indices] = mean
                if baseline == "act":
                    for act in [1, 2, 3]:
                        tr = train[data["act"][train] == act]
                        ids = indices[data["act"][indices] == act]
                        preds[ids] = np.average(data["y"][tr], weights=data["w"][tr])
            else:
                with torch.no_grad():
                    for start in range(0, len(indices), 1024):
                        ix = indices[start : start + 1024]
                        preds[ix] = (
                            model.heads[phase](X[ix]).squeeze(-1).sigmoid().numpy()
                        )
        result = {}
        for phaseid, phase in enumerate(PHASES):
            for split, mask in [("fit", fit), ("held", hold)]:
                for act in [0, 1, 2, 3]:
                    ids = (data["phase"] == phaseid) & mask
                    if act:
                        ids &= data["act"] == act
                    result[f"{phase}:{split}:act{act}"] = scores(
                        data["y"][ids], preds[ids], data["w"][ids]
                    )
        return result

    report["historical_metrics"] = evaluate(original)
    report["constant_baseline"] = evaluate(baseline="constant")
    report["act_baseline"] = evaluate(baseline="act")

    # Verify gradient wiring at initialization and after actual historical fitting.
    def gradient_audit(model):
        model.zero_grad(set_to_none=True)
        for phaseid, phase in enumerate(PHASES):
            ids = np.flatnonzero(fit & (data["phase"] == phaseid))
            ix = ids[:4096]
            logits = model.heads[phase](X[ix]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y[ix])
            loss.backward()
        return {
            name: {
                "norm": float(p.grad.norm()),
                "finite": bool(torch.isfinite(p.grad).all()),
            }
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    fresh = WinCritics()
    fresh.load_state_dict(initial["critic_state"])
    report["gradients_initial"] = gradient_audit(fresh)
    report["gradients_after20"] = gradient_audit(original)

    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def step(model, opt, phase, ix):
        opt.zero_grad(set_to_none=True)
        z = model.heads[phase](X[ix]).squeeze(-1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            z, Y[ix], reduction="none"
        )
        loss = (losses * W[ix]).sum() / W[ix].sum()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        assert math.isfinite(norm)
        opt.step()

    for name, lr, epochs, minibatch in [
        ("fullbatch_200_steps_same_lr", 0.0003, 10, None),
        ("fullbatch_20_steps_10x_lr", 0.003, 1, None),
        ("minibatch256_one_pass_same_lr", 0.0003, 1, 256),
    ]:
        model = WinCritics()
        model.load_state_dict(initial["critic_state"])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        start = time.monotonic()
        steps = 0
        trace = []
        gen = torch.Generator().manual_seed(73)
        for epoch in range(epochs):
            if minibatch:
                for phaseid, phase in enumerate(PHASES):
                    ids = torch.tensor(np.flatnonzero(fit & (data["phase"] == phaseid)))
                    ids = ids[torch.randperm(len(ids), generator=gen)]
                    for offset in range(0, len(ids), minibatch):
                        step(model, opt, phase, ids[offset : offset + minibatch])
                        steps += 1
            else:
                for batch in range(len(buffers)):
                    opt.zero_grad(set_to_none=True)
                    for phaseid, phase in enumerate(PHASES):
                        ids = np.flatnonzero(
                            fit & (data["phase"] == phaseid) & (data["batch"] == batch)
                        )
                        den = float(W[ids].sum())
                        for offset in range(0, len(ids), 256):
                            ix = ids[offset : offset + 256]
                            logits = model.heads[phase](X[ix]).squeeze(-1)
                            loss = (
                                torch.nn.functional.binary_cross_entropy_with_logits(
                                    logits, Y[ix], reduction="none"
                                )
                                * W[ix]
                            ).sum() / den
                            loss.backward()
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    )
                    assert math.isfinite(norm)
                    opt.step()
                    steps += 1
            if epoch in [0, 4, 9]:
                trace.append(
                    {"epoch": epoch + 1, "steps": steps, "metrics": evaluate(model)}
                )
            if time.monotonic() - start > 300:
                raise TimeoutError(
                    "Critic condition exceeded bounded five-minute budget"
                )
            print(
                json.dumps(
                    {
                        "event": "critic_epoch",
                        "condition": name,
                        "epoch": epoch + 1,
                        "steps": steps,
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
        metrics = evaluate(model)
        report["trials"][name] = {
            "learning_rate": lr,
            "epochs": epochs,
            "optimizer_steps": steps,
            "seconds": time.monotonic() - start,
            "metrics": metrics,
            "trace": trace,
        }
        save_atomic(
            {"critic_state": model.state_dict(), "condition": report["trials"][name]},
            out / (name + ".pt"),
        )
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        del model, opt
        gc.collect()
    # Deliberate same-state fit test, not evidence of held-world generalisation.
    model = WinCritics()
    model.load_state_dict(initial["critic_state"])
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    tiny = {}
    steps = 0
    start = time.monotonic()
    for phase in PHASES:
        rows = small[phase, 0][:32] + small[phase, 1][:32]
        tiny[phase] = (
            WinCritics.features(batch_rows(rows)),
            torch.tensor([r["return"] for r in rows]),
        )
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        for phase, (x, y) in tiny.items():
            torch.nn.functional.binary_cross_entropy_with_logits(
                model.heads[phase](x).squeeze(-1), y
            ).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
    report["memorization"] = {
        "steps": steps,
        "seconds": time.monotonic() - start,
        "phases": {},
    }
    with torch.no_grad():
        for phase, (x, y) in tiny.items():
            prob = model.heads[phase](x).squeeze(-1).sigmoid()
            report["memorization"]["phases"][phase] = {
                "n": len(y),
                "accuracy": float(((prob > 0.5) == y.bool()).float().mean()),
                "bce": float(torch.nn.functional.binary_cross_entropy(prob, y)),
            }
    assert file_identity(current_file) == latest_identity
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"audit": audit, "memorization": report["memorization"]}), flush=True
    )


if __name__ == "__main__":
    main()
