from __future__ import annotations

import math
import random
import statistics
from typing import Mapping, Sequence


def sign_test(deltas: Sequence[float]) -> dict:
    n_pos = sum(1 for delta in deltas if delta > 0.0)
    n_neg = sum(1 for delta in deltas if delta < 0.0)
    n_zero = sum(1 for delta in deltas if delta == 0.0)
    n_effective = n_pos + n_neg

    if n_effective == 0:
        p_value = 1.0
    else:
        k = n_pos
        m = min(k, n_effective - k)
        tail = sum(
            math.comb(n_effective, i) * 0.5**n_effective
            for i in range(m + 1)
        )
        p_value = min(1.0, 2.0 * tail)

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_effective": n_effective,
        "p_value": p_value,
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")

    rng = random.Random(seed)
    observed = [float(value) for value in values]
    n = len(observed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = sum(observed[rng.randrange(n)] for _ in range(n))
        means.append(total / n)

    means.sort()
    return (
        _percentile(means, alpha / 2.0),
        _percentile(means, 1.0 - alpha / 2.0),
    )


def paired_floor_summary(
    base_by_seed: Mapping[int, float],
    trained_by_seed: Mapping[int, float],
    *,
    bootstrap_seed: int = 0,
    n_resamples: int = 10000,
) -> dict:
    shared_seeds = sorted(set(base_by_seed) & set(trained_by_seed))
    if not shared_seeds:
        return {
            "n_seeds": 0,
            "mean_delta": 0.0,
            "se_delta": 0.0,
            "sign_test": sign_test([]),
            "bootstrap_ci": [0.0, 0.0],
            "deltas_by_seed": {},
        }

    deltas_by_seed = {
        seed: float(trained_by_seed[seed] - base_by_seed[seed])
        for seed in shared_seeds
    }
    deltas = list(deltas_by_seed.values())
    n = len(deltas)
    mean_delta = float(statistics.mean(deltas))
    se_delta = statistics.stdev(deltas) / math.sqrt(n) if n >= 2 else 0.0
    lo, hi = bootstrap_ci(deltas, n_resamples=n_resamples, seed=bootstrap_seed)

    return {
        "n_seeds": n,
        "mean_delta": mean_delta,
        "se_delta": se_delta,
        "sign_test": sign_test(deltas),
        "bootstrap_ci": [lo, hi],
        "deltas_by_seed": deltas_by_seed,
    }
