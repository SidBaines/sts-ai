"""Detachable local-curriculum tasks.

Local tasks are small, replayable training/eval problems cut out of full
Slay-the-Spire traces. They intentionally live outside the main full-run
training path: a task can produce its own datasets, eval reports, and adapters,
while the rest of the harness only consumes the resulting adapter path.
"""
from __future__ import annotations

from sts_ai.local_tasks.elite_fights import LagavulinTask, SentriesTask
from sts_ai.local_tasks.gremlin_nob import GremlinNobTask

__all__ = [
    "GremlinNobTask",
    "LagavulinTask",
    "SentriesTask",
    "get_task",
    "task_ids",
]


_TASK_TYPES = (GremlinNobTask, LagavulinTask, SentriesTask)


def task_ids() -> tuple[str, ...]:
    return tuple(task_type.task_id for task_type in _TASK_TYPES)


def get_task(task_id: str):
    for task_type in _TASK_TYPES:
        if task_id == task_type.task_id:
            return task_type()
    raise ValueError(f"unknown local task: {task_id}")
