"""Data-parallel rollout collection: several collector processes, each with its
own game workers and a frozen copy of the current policy.

Profiling (2026-10-13, 48 games, 8 workers) showed single-process collection is
bound by the trainer's main thread: Python observation encoding ~44% and the
policy forward ~16% of wall time, while 14 game workers idled at ~13% CPU each.
K collectors divide that main-thread work K ways. Greedy (argmax) evaluation is
per-world deterministic, so splitting worlds across collectors leaves records
unchanged; sampled collection is stochastic either way (each collector seeds its
own torch RNG from the iteration seed and its index).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import traceback
from typing import Any, Sequence

import torch

from sts_policy.losses import PPOConfig

from .common import close_workers, load_policy, make_workers, phase_forward


def _collector_main(conn, checkpoint: str, workers: int, worker_options: dict[str, Any], threads: int) -> None:
    torch.set_num_threads(threads)
    model, _ = load_policy(checkpoint)
    model.eval()
    pool = make_workers(workers, **worker_options)
    try:
        while True:
            try:
                kind, payload = conn.recv()
            except EOFError:
                return
            if kind == "close":
                return
            try:
                model.load_state_dict(payload.pop("state"))
                torch.manual_seed(payload.pop("seed"))
                if kind == "ooc":
                    from .ooc_trainer import collect_ooc_batch
                    forward = phase_forward(combat=payload.pop("combat"), temperature=payload.pop("temperature"),
                                            strategic_temperature=payload.pop("strategic_temperature", 1.0),
                                            strategic_cards=payload.pop("strategic_cards", False),
                                            card_temperature=payload.pop('card_temperature',1.0))
                    out = collect_ooc_batch(model, pool, payload.pop("tasks"), forward=forward, **payload)
                elif kind == "eval":
                    from .evaluate import evaluate_tasks
                    out = evaluate_tasks(model, pool, payload.pop("tasks"), **payload)
                else:
                    raise ValueError(f"unknown request {kind}")
                conn.send(("ok", out))
                del out
            except Exception:  # report, keep serving
                conn.send(("error", traceback.format_exc()))
    finally:
        close_workers(pool)


class CollectorPool:
    """K persistent collector processes splitting ``workers`` game workers."""

    def __init__(self, collectors: int, *, workers: int, checkpoint: str, dead_action_filter: bool = True,
                 threads: int = 1, in_process: bool = False, progress_floor_weight: float | None = None,
                 **worker_options: Any) -> None:
        worker_options = {"dead_action_filter": dead_action_filter, "in_process": in_process,
                          "progress_floor_weight": progress_floor_weight, **worker_options}
        self._context = mp.get_context("spawn")
        per = max(1, math.ceil(workers / collectors))
        self._collector_args = (checkpoint, per, worker_options, threads)
        self.conns, self.procs = [None] * collectors, [None] * collectors
        self.restarts = 0
        try:
            for index in range(collectors):
                self._launch(index)
        except BaseException:
            for index in range(collectors):
                self._stop(index)
            raise

    def _launch(self, index: int) -> None:
        parent, child = self._context.Pipe()
        # Not daemonic: subprocess game workers need to spawn children.
        proc = self._context.Process(target=_collector_main, args=(child, *self._collector_args))
        try:
            proc.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self.conns[index], self.procs[index] = parent, proc

    def _stop(self, index: int) -> None:
        conn, proc = self.conns[index], self.procs[index]
        if conn is not None:
            conn.close()
        if proc is not None:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=.5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            proc.close()
        self.conns[index] = self.procs[index] = None

    def _scatter(self, kind: str, model: torch.nn.Module, tasks: Sequence[Any], seed: int,
                 **payload: Any) -> list[Any]:
        """Restart a dead collector once and resend the identical request.

        A second death or an error reply raises RuntimeError; the pool must
        then be closed, not reused.
        """
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        shares = [list(tasks[i::len(self.conns)]) for i in range(len(self.conns))]
        requests = [(kind, {"state": state, "seed": seed * 1009 + index, "tasks": share, **payload})
                    for index, share in enumerate(shares)]
        retried = [False] * len(requests)

        def recover(index: int, error: Exception) -> None:
            self._stop(index)
            if retried[index]:
                raise RuntimeError(f"collector {index} failed twice during {kind}") from error
            retried[index] = True
            self._launch(index)
            self.restarts += 1

        def send(index: int) -> None:
            while True:
                data = mp.reduction.ForkingPickler.dumps(requests[index])
                try:
                    if not self.procs[index].is_alive():
                        raise EOFError("collector is no longer alive")
                    self.conns[index].send_bytes(data)
                    return
                except (EOFError, BrokenPipeError, OSError) as exc:
                    recover(index, exc)

        for index in range(len(requests)):
            send(index)
        results = []
        for index in range(len(requests)):
            while True:
                try:
                    status, out = self.conns[index].recv()
                    break
                except (EOFError, BrokenPipeError, OSError) as exc:
                    recover(index, exc)
                    send(index)
            if status != "ok":
                raise RuntimeError(f"collector failed:\n{out}")
            results.append(out)
        return results

    def collect_ooc(self, model: torch.nn.Module, tasks: Sequence[Any], *, combat: str, temperature: float,
                    ppo: PPOConfig, credit: str, gae_lambda: float, chunk: int, seed: int,
                    strategic_temperature: float = 1.0, strategic_cards: bool = False,
                    card_temperature: float = 1.0, value_scale: float = 1.0) -> dict[str, Any]:
        restarts_before = self.restarts
        parts = self._scatter("ooc", model, tasks, seed, combat=combat, temperature=temperature, ppo=ppo,
                              credit=credit, gae_lambda=gae_lambda, chunk=chunk,
                              strategic_temperature=strategic_temperature, strategic_cards=strategic_cards,
                              card_temperature=card_temperature, value_scale=value_scale)
        merged: dict[str, Any] = {"kept": [], "game_episodes": [], "faults": 0, 'fault_records': [], "stats": {},
                                  "collector_restarts": self.restarts - restarts_before}
        for part in parts:
            # Unpickled observations are new objects: re-bind the encoding identity
            # that update_ppo checks against id(observation).
            for episode in part["kept"]:
                for t in episode:
                    if t.encoded_observation is not None:
                        t.encoding_observation_id = id(t.observation)
            merged["kept"] += part["kept"]
            merged["game_episodes"] += part["game_episodes"]
            merged["faults"] += part["faults"]
            merged['fault_records'] += part['fault_records']
            for key, values in part["stats"].items():
                merged["stats"].setdefault(key, []).extend(values)
        return merged

    def evaluate_tasks(self, model: torch.nn.Module, tasks: Sequence[Any], *, mode: str = "greedy",
                       max_seconds: float = 3600., chunk: int = 256) -> tuple[list[dict[str, Any]], int, float]:
        parts = self._scatter("eval", model, tasks, 0, mode=mode, max_seconds=max_seconds,
                              chunk=max(1, chunk // len(self.conns)))
        records = [r for records, _, _ in parts for r in records]
        return records, sum(f for _, f, _ in parts), max(e for _, _, e in parts)

    def close(self) -> None:
        for conn in self.conns:
            if conn is None:
                continue
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for index, proc in enumerate(self.procs):
            if proc is not None:
                proc.join(timeout=60)
            self._stop(index)
