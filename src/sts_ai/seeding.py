from __future__ import annotations

import hashlib
from typing import Iterable

_SEED_MASK_63_BITS = (1 << 63) - 1


def _digest_to_seed(digest: bytes) -> int:
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & _SEED_MASK_63_BITS


def derive_policy_seed(world_seed: int, rollout_index: int) -> int:
    """Derive a process-stable policy RNG seed from rollout identity."""
    payload = f"{world_seed}:{rollout_index}".encode("utf-8")
    return _digest_to_seed(hashlib.sha256(payload).digest())


def expand_specs(seeds: Iterable[int], rollouts_per_seed: int) -> list[tuple[int, int]]:
    """Expand world seeds into (world_seed, rollout_index) rollout identities."""
    if rollouts_per_seed < 1:
        raise ValueError("rollouts_per_seed must be >= 1")
    return [
        (seed, rollout_index)
        for seed in seeds
        for rollout_index in range(rollouts_per_seed)
    ]


def rollout_stem(world_seed: int, rollout_index: int) -> str:
    """run_rollout.py intentionally keeps its default rollout_{agent}_... prefix."""
    return f"seed_{world_seed}_r{rollout_index}"


# Used by the batched K-rollout path; keep seeding policy centralized here.
def derive_batch_seed(members: Iterable[tuple[int, int, int]]) -> int:
    """Derive an order-independent seed from (world, rollout, decision) members."""
    hasher = hashlib.sha256()
    for world_seed, rollout_index, decision_index in sorted(members):
        hasher.update(f"{world_seed}:{rollout_index}:{decision_index}\n".encode("utf-8"))
    return _digest_to_seed(hasher.digest())
