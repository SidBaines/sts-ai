"""Paired saved-decision means, with bootstrap resampling of entire worlds.

Records contain nonempty string prompt_id/world_id and a finite numeric value.
The estimand weights decisions equally, not worlds equally. Repeated generations
must be averaged within a decision before calling this function.
"""
from __future__ import annotations

import math
import random


def _index(records):
    indexed = {}
    for record in records:
        prompt, world, value = (record[k] for k in ("prompt_id", "world_id", "value"))
        if any(not isinstance(x, str) or not x for x in (prompt, world)):
            raise ValueError("prompt_id and world_id must be nonempty strings")
        if prompt in indexed:
            raise ValueError(f"duplicate prompt_id: {prompt}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("values must be finite numbers")
        indexed[prompt] = (world, float(value))
    return indexed


def _quantile(ordered, q):
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def paired_cluster_bootstrap(left, right, *, seed=1729, replicates=10000):
    """Return right-minus-left means and a 95% world-cluster percentile interval.

    Pair only identical prompt IDs and require their world IDs to agree. Each
    bootstrap draws K worlds with replacement, where K is the number of paired
    worlds, then includes every paired decision in each drawn world. Therefore
    the denominator varies when world clusters contain different state counts.
    The interval describes this finite decision panel; it is not a win-rate CI.
    """
    if type(seed) is not int or type(replicates) is not int or replicates < 1:
        raise ValueError("seed must be an integer and replicates a positive integer")
    a, b = _index(left), _index(right)
    ids = sorted(a.keys() & b.keys())
    pairs, clusters = [], {}
    for prompt in ids:
        world, av = a[prompt]
        other_world, bv = b[prompt]
        if world != other_world:
            raise ValueError(f"world mismatch for {prompt}")
        delta = bv - av
        if not math.isfinite(delta):
            raise ValueError("paired differences must be finite")
        pairs.append({"prompt_id": prompt, "world_id": world, "left": av, "right": bv, "difference": delta})
        clusters.setdefault(world, []).append(delta)
    result = {
        "direction": "right_minus_left", "estimand": "equal_decision_weight_paired_mean",
        "method": "world_cluster_percentile_bootstrap", "confidence": 0.95,
        "seed": seed, "requested_replicates": replicates, "used_replicates": 0,
        "n_left": len(a), "n_right": len(b), "n_pairs": len(pairs), "n_worlds": len(clusters),
        "missing_left_prompt_ids": sorted(b.keys() - a.keys()),
        "missing_right_prompt_ids": sorted(a.keys() - b.keys()),
        "left_mean": None, "right_mean": None, "difference": None, "ci95": None, "pairs": pairs,
        "interval_caveat": "Fewer than two paired worlds; between-world variation cannot be estimated." if len(clusters) < 2 else None,
    }
    if not pairs:
        return result
    deltas = [pair["difference"] for pair in pairs]
    constant = all(delta == deltas[0] for delta in deltas)
    difference = deltas[0] if constant else math.fsum(deltas) / len(pairs)
    if constant:
        interval = [difference, difference]
    else:
        groups = [(math.fsum(clusters[w]), len(clusters[w])) for w in sorted(clusters)]
        rng, draws = random.Random(seed), []
        for _ in range(replicates):
            sampled = rng.choices(groups, k=len(groups))
            draws.append(math.fsum(total for total, _ in sampled) / sum(n for _, n in sampled))
        draws.sort()
        interval = [_quantile(draws, .025), _quantile(draws, .975)]
        result["used_replicates"] = replicates
    result.update(left_mean=math.fsum(pair["left"] for pair in pairs) / len(pairs),
                  right_mean=math.fsum(pair["right"] for pair in pairs) / len(pairs),
                  difference=difference, ci95=interval)
    return result
