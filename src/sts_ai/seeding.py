from __future__ import annotations

import hashlib
from typing import Iterable

_SEED_MASK_63_BITS = (1 << 63) - 1


def _digest_to_seed(digest: bytes) -> int:
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & _SEED_MASK_63_BITS


def derive_policy_seed(world_seed: int, rollout_index: int, *, salt: int = 0) -> int:
    """Derive a process-stable policy RNG seed from rollout identity.

    ``salt=0`` (the default) preserves the historical unsalted stream, keeping
    frozen-seed trajectories byte-identical. A non-zero salt (e.g. the GRPO
    iteration index, plus a restart offset) yields an independent stream so
    repeated passes over the same (world_seed, rollout_index) grid sample
    fresh trajectories instead of deterministically replaying — required both
    for per-iteration exploration and to dodge state-dependent simulator
    hangs on supervised restarts.
    """
    suffix = "" if salt == 0 else f":s{salt}"
    payload = f"{world_seed}:{rollout_index}{suffix}".encode("utf-8")
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
def derive_batch_seed(
    members: Iterable[tuple[int, int, int]], *, salt: int = 0
) -> int:
    """Derive an order-independent seed from (world, rollout, decision) members.

    ``salt=0`` is byte-identical to the historical unsalted derivation; see
    ``derive_policy_seed`` for salt semantics. Both the lockstep round seed and
    the streaming per-request seed flow through here, so this is the single
    point where a salt actually changes sampled tokens.
    """
    hasher = hashlib.sha256()
    if salt != 0:
        hasher.update(f"salt:{salt}\n".encode("utf-8"))
    for world_seed, rollout_index, decision_index in sorted(members):
        hasher.update(f"{world_seed}:{rollout_index}:{decision_index}\n".encode("utf-8"))
    return _digest_to_seed(hasher.digest())


def derive_stage_seed(
    world_seed: int,
    rollout_index: int,
    decision_index: int,
    stage: str,
) -> int:
    """Derive a process-stable seed for a streaming hint sub-stage."""
    hasher = hashlib.sha256()
    hasher.update(
        f"{world_seed}:{rollout_index}:{decision_index}:{stage}\n".encode("utf-8")
    )
    return _digest_to_seed(hasher.digest())
