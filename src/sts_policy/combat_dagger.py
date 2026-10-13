"""Privileged combat labels on unchanged public-policy trajectories."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Callable

from sts_env.env import SimulatorError
from .curriculum import world_split
from .rollout_training import task_identity

DAGGER_VERSION = "public_policy_combat_labels_v1"


class LabelingWorker:
    """Query before a combat play; quarantine all labels after any native fault.

    The wrapper never substitutes a teacher action or queries the scripted OOC
    controller. Labels travel to a separate journal, never policy observations.
    A matched no-query rollout must establish search-query trajectory purity.
    """
    def __init__(self, worker: Any, budgets: tuple[int, ...], *, stride: int = 1,
                 sink: Callable[[dict], None] | None = None, panel: bool = False) -> None:
        if not budgets or any(b < 1 for b in budgets) or len(set(budgets)) != len(budgets) or stride < 1:
            raise ValueError("distinct positive search budgets and positive stride required")
        self.worker, self.budgets, self.stride, self.sink = worker, budgets, stride, sink
        self.panel = panel
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sts-labels")
        self.worlds: list[dict] = []
        self.current: dict | None = None
        self.observation = None
        self.steps = self.choices = 0

    def _publish(self, status: str, error: Exception | None = None) -> None:
        if self.current is None:
            return
        row = self.current
        row.update(status=status, steps=self.steps)
        if status != "completed":
            row["discarded_records"] = len(row["records"])
            row["records"] = []
        if error is not None:
            row.update(error_type=type(error).__name__, error=str(error))
        if self.sink:
            self.sink(row)
        # Journals can be large; the summary keeps no second copy of observations.
        self.worlds.append({k:v for k,v in row.items() if k != "records"} | {"record_count":len(row["records"])})
        self.current = None

    def _call(self, method: str, *args: Any) -> Any:
        try:
            if method == "reset":
                self._publish("incomplete")
                task = args[0]
                if world_split(task.seed) != "train":
                    raise ValueError("learner-induced supervision is TRAIN-only")
                self.steps = self.choices = 0
                self.current = {"data_version":DAGGER_VERSION,"task_id":task_identity(task),
                    "world_seed":task.seed,"task":task.to_dict(),"records":[],"query_seconds":{},"query_counts":{}}
                result = self.worker.reset(task)
                self.observation, info = result
                self.current["provenance"] = {k:info[k] for k in ("native_provenance","python_interface_sha256",
                    "runtime_python_version","environment_version","observation_version") if k in info}
                return result
            if method != "step" or self.current is None:
                raise ValueError("reset before a public step")
            obs, action = self.observation, int(args[0])
            if obs["phase"] == "combat" and len(obs["legal_actions"]) > 1:
                if self.choices % self.stride == 0:
                    teachers, panels = {}, {}
                    legal = {int(a["index"]) for a in obs["legal_actions"]}
                    for budget in self.budgets:
                        stamp = time.monotonic()
                        if self.panel:
                            panel = self.worker.teacher_panel(budget)
                            teacher = panel["primary_index"]
                            panels[str(budget)] = panel
                        else:
                            teacher = self.worker.teacher_action(budget)
                        if teacher not in legal:
                            raise SimulatorError("Teacher selected an illegal public action")
                        key = str(budget)
                        teachers[key] = teacher
                        self.current["query_seconds"][key] = self.current["query_seconds"].get(key,0.)+time.monotonic()-stamp
                        self.current["query_counts"][key] = self.current["query_counts"].get(key,0)+1
                    self.current["records"].append({"data_version":DAGGER_VERSION,"observation":obs,
                        "world_seed":self.current["world_seed"],"task_id":self.current["task_id"],
                        "step":self.steps,"phase":"combat","split":"train","teacher_kind":"native_privileged_search",
                        "teacher_index":teachers[str(self.budgets[-1])],"teacher_by_budget":teachers,
                        "simulations":self.budgets[-1],"executed_index":action})
                    if panels:
                        self.current["records"][-1]["teacher_panels"] = panels
                self.choices += 1
            result = self.worker.step(action)
            self.observation, _, terminated, truncated, info = result
            self.steps += 1
            if terminated or truncated:
                self.current["outcome"] = info.get("task_outcome")
                self._publish("completed" if terminated and not truncated else "truncated")
            return result
        except SimulatorError as error:
            self._publish("simulator_fault",error)
            raise

    def submit(self, method: str, *args: Any):
        return self.executor.submit(self._call,method,*args)

    def close(self) -> None:
        self.executor.shutdown(wait=True,cancel_futures=True)
        self._publish("incomplete")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
