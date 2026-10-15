"""Public-feature critic fitting and actionable learning-health diagnostics."""

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np
import torch

PHASES = ("combat", "ooc")


def held_world(seed):
    return int(hashlib.sha256(str(int(seed)).encode()).hexdigest()[:8], 16) % 5 == 0


def select_rows(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["phase"], r["seed"]].append(r)
    selected = []
    for rs in by.values():
        selected.extend(rs[:: max(1, len(rs) // 8)][:8])
    return selected


def pack(critics, rows, collate):
    rows = select_rows(rows)
    counts = Counter((r["phase"], r["seed"]) for r in rows)
    x = []
    for i in range(0, len(rows), 256):
        x.append(critics.features(collate(rows[i : i + 256])).detach().cpu())
    if not rows:
        raise ValueError("No complete critic samples")
    return {
        "x": torch.cat(x),
        "y": torch.tensor([r["return"] for r in rows], dtype=torch.float32),
        "phase": np.array([PHASES.index(r["phase"]) for r in rows]),
        "seed": np.array([r["seed"] for r in rows], np.int64),
        "act": np.array([r["act"] for r in rows]),
        "w": torch.tensor([1 / counts[r["phase"], r["seed"]] for r in rows]),
        "held": np.array([held_world(r["seed"]) for r in rows]),
    }


def fit(critics, optimizer, data, minibatch=256, seed=19):
    if minibatch < 1:
        raise ValueError("Positive critic minibatch required")
    generator = torch.Generator().manual_seed(seed)
    phases = {}
    norms = []
    losses = []
    for phaseid, phase in enumerate(PHASES):
        ix = torch.tensor(np.flatnonzero((data["phase"] == phaseid) & ~data["held"]))
        if not len(ix):
            raise ValueError("Missing critic fitting worlds for " + phase)
        ix = ix[torch.randperm(len(ix), generator=generator)]
        steps = 0
        for start in range(0, len(ix), minibatch):
            part = ix[start : start + minibatch]
            optimizer.zero_grad(set_to_none=True)
            z = critics.heads[phase](data["x"][part]).squeeze(-1)
            w = data["w"][part]
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    z, data["y"][part], reduction="none"
                )
                * w
            ).sum() / w.sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite critic minibatch loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(critics.parameters(), 1.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite critic minibatch gradient")
            optimizer.step()
            steps += 1
            norms.append(norm)
            losses.append(float(loss.detach()))
        phases[phase] = {
            "steps": steps,
            "fit_states": len(ix),
            "fit_worlds": len(set(data["seed"][ix.numpy()].tolist())),
        }
    return {
        "loss": float(np.mean(losses)),
        "gradient_norm": max(norms),
        "optimizer_steps": sum(p["steps"] for p in phases.values()),
        "phases": phases,
        "fit_states": sum(p["fit_states"] for p in phases.values()),
    }


@torch.no_grad()
def metrics(critics, data, selection="held", reference_rate=0.046):
    out = {}
    for phaseid, phase in enumerate(PHASES):
        chosen = data["phase"] == phaseid
        if selection == "held":
            chosen &= data["held"]
        elif selection == "fit":
            chosen &= ~data["held"]
        elif selection != "all":
            raise ValueError(selection)
        ix = np.flatnonzero(chosen)
        if not len(ix):
            continue
        probabilities = []
        for start in range(0, len(ix), 1024):
            probabilities.extend(
                critics.heads[phase](data["x"][ix[start : start + 1024]])
                .squeeze(-1)
                .sigmoid()
                .tolist()
            )
        preds = np.array(probabilities, np.float64)
        for act in [0, 1, 2, 3]:
            keep = np.ones(len(ix), bool) if act == 0 else data["act"][ix] == act
            ids = ix[keep]
            if not len(ids):
                continue
            y = data["y"][ids].numpy().astype(np.float64)
            p = preds[keep]
            w = data["w"][ids].numpy().astype(np.float64)
            w /= w.sum()
            if not np.isfinite(p).all():
                raise FloatingPointError("Nonfinite critic predictions")
            mu = float((w * y).sum())
            error = y - p
            var = float((w * (y - mu) ** 2).sum())
            brier = float((w * error**2).sum())
            base = (y - reference_rate) ** 2
            delta = error**2 - base
            unique, inverse = np.unique(data["seed"][ids], return_inverse=True)
            worldmass = np.bincount(inverse, weights=w)
            worlddiff = np.bincount(inverse, weights=w * delta) / worldmass
            se = (
                float(np.std(worlddiff, ddof=1) / math.sqrt(len(worlddiff)))
                if len(worlddiff) > 1
                else None
            )
            out[f"{phase}:act{act}"] = {
                "n": len(ids),
                "worlds": len(worlddiff),
                "positive_worlds": len(set(data["seed"][ids][y == 1].tolist())),
                "predicted_mean": float((w * p).sum()),
                "return_mean": mu,
                "predicted_std": float(
                    np.sqrt((w * (p - float((w * p).sum())) ** 2).sum())
                ),
                "brier": brier,
                "constant_brier": float((w * base).sum()),
                "brier_delta": float((w * delta).sum()),
                "brier_delta_se_world": se,
                "bce": float(
                    -(
                        w
                        * (
                            y * np.log(np.clip(p, 1e-12, 1))
                            + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))
                        )
                    ).sum()
                ),
                "explained_variance": 1
                - float((w * (error - float((w * error).sum())) ** 2).sum()) / var
                if var > 0
                else None,
            }
    return out


def health(held, fit_metrics=None):
    """Sample-aware warnings; no requirement for noisy short-term game-win gains."""
    warnings = []
    urgent = []
    for key, m in held.items():
        if m["worlds"] < 64 or m["positive_worlds"] < 8:
            warnings.append(
                {"bucket": key, "reason": "insufficient_critic_validation_support"}
            )
            continue
        if m["predicted_std"] < 1e-4:
            item = {"bucket": key, "reason": "flat_critic_predictions"}
            warnings.append(item)
            urgent.append(item)
        tolerance = max(
            0.03,
            3
            * math.sqrt(
                max(m["return_mean"] * (1 - m["return_mean"]), 1e-6) / m["worlds"]
            ),
        )
        if abs(m["predicted_mean"] - m["return_mean"]) > tolerance:
            warnings.append({"bucket": key, "reason": "critic_calibration_gap"})
        se = m["brier_delta_se_world"]
        if se is not None and m["brier_delta"] > max(0.002, 3 * se):
            item = {"bucket": key, "reason": "critic_worse_than_constant"}
            warnings.append(item)
            urgent.append(item)
        if m["explained_variance"] is not None and m["explained_variance"] < 0.001:
            warnings.append({"bucket": key, "reason": "weak_critic_predictive_value"})
        train = (fit_metrics or {}).get(key)
        if train and m["bce"] - train["bce"] > 0.05 and m["bce"] > train["bce"] * 1.5:
            warnings.append({"bucket": key, "reason": "critic_generalisation_gap"})
    return {"warnings": warnings, "urgent": urgent}


def subset(data, selection):
    return {k: v[selection] for k, v in data.items()}


def combine(parts):
    return {
        k: torch.cat([p[k] for p in parts])
        if torch.is_tensor(parts[0][k])
        else np.concatenate([p[k] for p in parts])
        for k in parts[0]
    }
