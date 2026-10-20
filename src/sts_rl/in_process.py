"""Synchronous game slots for batched collection without native RPC.

There is no per-operation timeout: a native hang stalls the collecting process.
The job queue's stall watchdog provides containment. In-process mode gives up
crash isolation: a native segfault or abort kills every game in that collector
process, and with --collectors 1 the trainer itself. A replacement collector
replays the identical request (same seed, tasks and slot count), so a
deterministic native crash recurs and ends the run with "failed twice", where
the subprocess path would have excluded one episode.

Every episode exception marks the slot finished and propagates unchanged;
only SimulatorError discards the episode. collect_rollouts records every game
slot exception as a fault and excludes that episode, so programming errors
differ only in naming the real exception type in the fault record. No fault
packets are written in-process.

CurriculumEpisode's own docstring predates this module: it says episodes are
only instantiated in a disposable process. The first reset here installs the
native guards permanently in the hosting process (the trainer process with
one collector). native_guard_scope remains for direct runners. Unlike
CurriculumWorker, this worker is not thread-safe and requires one caller
thread. Native reset and step time runs inside submit(), so collect_rollouts'
reset_wait_seconds and step_wait_seconds are near zero and no longer measure
native time.
"""
from __future__ import annotations

from concurrent.futures import Future
import operator
import os
from typing import Any

from sts_env._backend.lightspeed_import import import_lightspeed
from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec
from sts_policy.workers import CurriculumEpisode

from .fresh_native import install_native_guards


class InProcessWorker:
    """One lazily constructed episode, with already-resolved submit futures."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.config.setdefault("build_dir", os.environ.get("STS_LIGHTSPEED_BUILD"))
        self.config.pop("fault_dir", None)
        self.config.pop("run_record", None)
        self.last_fault_path: str | None = None
        self._episode: CurriculumEpisode | None = None
        self.observation: dict[str, Any] | None = None
        self.info: dict[str, Any] | None = None
        self.finished = True

    def _call(self, method: str, *args: Any) -> Any:
        try:
            if self._episode is None:
                if method != "reset":
                    raise RuntimeError("reset is required after a native fault")
                native = import_lightspeed(self.config.get("build_dir"))
                if not getattr(native, "_fresh_native_guards", False):
                    install_native_guards(native)
                self._episode = CurriculumEpisode(self.config)
            return getattr(self._episode, method)(*args)
        except Exception as exc:
            self.finished = True
            if isinstance(exc, SimulatorError):
                self._episode = None
            raise

    def reset(self, spec: TaskSpec | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.observation, self.info = None, None
        self.finished = True
        observation, info = self._call("reset", spec)
        self.observation, self.info = observation, info
        self.finished = False
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.finished or self.observation is None:
            raise RuntimeError("reset before step")
        if isinstance(action, bool) or not 0 <= operator.index(action) < len(self.observation["legal_actions"]):
            raise ValueError("action outside legal menu")
        result = self._call("step", int(action))
        self.observation, _, terminated, truncated, self.info = result
        self.finished = terminated or truncated
        return result

    def teacher_action(self, simulations: int = 50, search_seed: int | None = None) -> int:
        return int(self._call("teacher_action", simulations, search_seed))

    def submit(self, method: str, *args: Any) -> Future[Any]:
        if method not in {"reset", "step", "teacher_action"}:
            raise ValueError("unknown public RPC")
        future: Future[Any] = Future()
        try:
            future.set_result(getattr(self, method)(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def close(self) -> None:
        self._episode = None
        self.finished = True

    def __enter__(self) -> InProcessWorker:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
