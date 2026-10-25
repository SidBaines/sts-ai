"""Per-decision quantities that decide results: advantages, decisiveness and scores.

All functions are pure and dependency-free, so both the PyTorch and the MLX
trainers, the evaluators and the analysis scripts share one definition.
"""
from __future__ import annotations

import math
import statistics
from typing import Sequence

from .generation import K2_REASONING_END_TOKEN_IDS
from .teacher import validate_teacher


def centered_advantages(rewards: Sequence[float]) -> list[float]:
    """Group-mean baseline without standard-deviation scaling.

    Near-tied rewards give near-zero advantages, instead of being stretched to
    unit scale as division by a small group standard deviation would.
    """
    if not rewards or any(isinstance(value, bool) or not math.isfinite(value) for value in rewards):
        raise ValueError('rewards must be a finite nonempty group')
    mean = statistics.fmean(rewards)
    return [value - mean for value in rewards]


def teacher_table(row: dict) -> dict[int, float]:
    """Teacher probability for every legal index; filtered actions get zero."""
    teacher = validate_teacher(row)
    table = {index: 0. for index in row['legal_indices']}
    table.update(zip(teacher['legal_indices'], map(float, teacher['probabilities'])))
    return table


def teacher_top(row: dict) -> int:
    table = teacher_table(row)
    return max(table, key=lambda index: (table[index], -index))


def decisive_margin(row: dict) -> float:
    """Top teacher probability minus the second highest (0 for a one-action menu)."""
    ranked = sorted(teacher_table(row).values(), reverse=True)
    return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.)


def normalized_teacher_score(row: dict, action_index: int | None) -> float | None:
    """(p(chosen) - mean legal p) / (p(top) - mean legal p).

    0 is a random legal action in expectation, 1 the teacher's own choice, and
    an invalid answer scores as if it chose probability zero. None when every
    legal action has the same teacher probability.
    """
    table = teacher_table(row)
    mean = statistics.fmean(table.values())
    spread = max(table.values()) - mean
    if spread <= 1e-12:
        return None
    chosen = table.get(action_index, 0.) if type(action_index) is int else 0.
    return (chosen - mean) / spread


def reasoning_token_count(output_token_ids: Sequence[int]) -> tuple[int, bool]:
    """Tokens before the first reasoning closer, and whether a closer appeared."""
    for position, token in enumerate(output_token_ids):
        if token in K2_REASONING_END_TOKEN_IDS:
            return position, True
    return len(output_token_ids), False
