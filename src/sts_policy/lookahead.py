"""Optional privileged OOC teacher via short common-world continuations.

An experimental partial-menu teacher, not a claim of strategic expertise.
Policies only receive the original public observation and selected menu index.
Each branch reconstructs the same seeded prefix in a disposable native process;
combat continuations use native search and OOC continuations its scripted policy.
"""
from __future__ import annotations
from concurrent.futures import Future
from dataclasses import replace
import hashlib
import multiprocessing
import random
from typing import Any

from sts_env.env import SimulatorError
from sts_policy.curriculum import TaskSpec, player_hp, public_hash
from sts_policy.workers import CurriculumEpisode, CurriculumWorker

LOOKAHEAD_VERSION = "native_ooc_partial_menu_lookahead_v1"


class _LookaheadEpisode(CurriculumEpisode):
    def score_ooc(self, spec: TaskSpec | dict[str, Any], candidates: list[int] | None = None,
                  branches: int = 4, horizon: int = 30, simulations: int = 50,
                  hp_weight: float = 0.25, floor_weight: float = 0.05) -> dict[str, Any]:
        spec = TaskSpec.from_dict(spec) if isinstance(spec, dict) else spec
        if spec.kind != "run":
            raise ValueError("Lookahead source prefix must use full-run semantics")
        if not 1 <= branches <= 8 or not 1 <= horizon <= 60 or simulations < 1:
            raise ValueError("Lookahead branch/horizon/simulation bounds exceeded")
        spec = replace(spec, horizon=horizon + 1)
        original, info = self.reset(spec)
        if original["phase"] != "out_of_combat":
            raise ValueError("Lookahead requires an OOC decision")
        teacher = self.teacher_action(simulations)
        indices = [action["index"] for action in original["legal_actions"]]
        if candidates is None:
            # Stable candidate sample includes the scripted baseline, and is
            # independent of branch evaluation order and future state.
            rng = random.Random(int(hashlib.sha256(f"{spec.seed}:{spec.prefix}".encode()).hexdigest()[:16], 16))
            remaining = [index for index in indices if index != teacher]
            rng.shuffle(remaining)
            candidates = [teacher, *remaining[:branches - 1]]
        else:
            candidates = list(dict.fromkeys(candidates))
            if not candidates or len(candidates) > branches or any(index not in indices for index in candidates):
                raise ValueError("Invalid evaluated candidate menu subset")
        results = []
        start_floor = int(original["state"].get("floor", 0))
        for index in candidates:
            observation, _ = self.reset(spec)
            if public_hash(observation) != public_hash(original):
                raise SimulatorError("Lookahead branch reset changed public state")
            # This is an adapter-only continuation intervention after exact full
            # replay. No private native object or action bits cross IPC.
            self.episode.env._env.combat_control = "search"
            self.episode.env._env.battle_agent.simulation_count_base = simulations
            self.teacher_action(simulations)  # reproduce conditional OOC RNG commit
            observation, _, terminated, truncated, branch_info = self.step(index)
            decisions = 1
            while not (terminated or truncated) and decisions < horizon:
                action = self.teacher_action(simulations)
                observation, _, terminated, truncated, branch_info = self.step(action)
                decisions += 1
            outcome = branch_info["task_outcome"]
            terminal = 1.0 if outcome in {"act_clear", "victory"} else -1.0 if outcome == "defeat" else 0.0
            hp_value = hp_weight * player_hp(observation) / max(1.0, float(observation["state"].get("max_hp", 80)))
            progress = floor_weight * max(0, int(observation["state"].get("floor", 0)) - start_floor)
            score = terminal + hp_value + progress
            results.append({"index": index, "score": score, "terminal": terminal,
                            "hp_value": hp_value, "floor_progress": progress,
                            "outcome": outcome, "decisions": decisions,
                            "terminated": terminated, "truncated": truncated,
                            "continuation_horizon_reached": decisions >= horizon and not terminated})
        best = max(results, key=lambda row: (row["score"], row["index"] == teacher, -row["index"]))
        baseline = next((row for row in results if row["index"] == teacher), None)
        return {"lookahead_version": LOOKAHEAD_VERSION, "observation": original,
                "teacher_index": best["index"], "scripted_index": teacher,
                "score_gain_over_scripted": best["score"] - baseline["score"] if baseline else None,
                "evaluated_candidates": results, "world_seed": spec.seed,
                "public_hash": public_hash(original), "split": info["split"],
                "settings": {"branches": branches, "horizon": horizon,
                             "simulations": simulations, "hp_weight": hp_weight,
                             "floor_weight": floor_weight},
                "native_provenance": info["native_provenance"]}


def _serve_lookahead(connection: Any, config: dict[str, Any]) -> None:
    episode = _LookaheadEpisode(config)
    try:
        while True:
            method, args = connection.recv()
            if method == "close":
                break
            try:
                connection.send(("ok", getattr(episode, method)(*args)))
            except Exception as exc:
                connection.send(("error", {"type": type(exc).__name__, "message": str(exc)}))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class CounterfactualOOCWorker(CurriculumWorker):
    def _launch(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve_lookahead, args=(child, self.config),
                                        name="sts-ooc-lookahead", daemon=True)
        self._process.start()
        child.close()

    def score_ooc(self, spec: TaskSpec | dict[str, Any], candidates: list[int] | None = None,
                  branches: int = 4, horizon: int = 30, simulations: int = 50,
                  hp_weight: float = 0.25, floor_weight: float = 0.05) -> dict[str, Any]:
        # Boot through public reset to retain fault recovery semantics.
        if self._process is None:
            self.reset(spec)
        return self._call("score_ooc", spec, candidates, branches, horizon, simulations,
                          hp_weight, floor_weight)

    def submit_score(self, *args: Any) -> Future[Any]:
        return self._executor.submit(self.score_ooc, *args)
