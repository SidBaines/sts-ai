"""Detachable local-curriculum tasks.

Local tasks are small, replayable training/eval problems cut out of full
Slay-the-Spire traces. They intentionally live outside the main full-run
training path: a task can produce its own datasets, eval reports, and adapters,
while the rest of the harness only consumes the resulting adapter path.
"""
from __future__ import annotations

from sts_ai.local_tasks.gremlin_nob import GremlinNobTask

__all__ = ["GremlinNobTask", "get_task"]


def get_task(task_id: str):
    if task_id == GremlinNobTask.task_id:
        return GremlinNobTask()
    raise ValueError(f"unknown local task: {task_id}")
