"""Persistent bounded native RPC workers; policy inference stays in the parent."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import math
import multiprocessing
import operator
import os
import threading
import time
from typing import Any

from sts_env.env import SimulatorError, SimulatorTimeout, _Episode
from sts_policy.curriculum import TASK_VERSION, REWARD_VERSION, TaskSpec, player_hp, public_hash, reward_components, validate_provenance, world_split


class CurriculumEpisode:
    """Only instantiated inside a disposable native process."""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        if config.get("fast_observations", False):
            from sts_policy.fast_observations import install_status_caches
            install_status_caches()
        self.episode: Any = None
        self.prefix: list[int] = []
        self.ooc_previews: list[bool] = []
        self.preview_pending = False
        self.finished = True

    def reset(self, spec: TaskSpec | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.spec = TaskSpec.from_dict(spec) if isinstance(spec, dict) else spec
        config = {"mode": "ooc" if self.spec.kind == "ooc" else "full",
                  "ascension": self.spec.ascension, "max_act": self.spec.max_act,
                  "max_steps": max(10000, len(self.spec.prefix) + self.spec.horizon + 1024),
                  "battle_simulations": self.spec.battle_simulations,
                  "build_dir": self.config.get("build_dir"), "ooc_seed": self.spec.seed}
        self.episode = _Episode(config, self.spec.seed)
        observation, info = self.episode.reset()
        if self.spec.expected_provenance is not None:
            try:
                validate_provenance(self.spec.expected_provenance, info)
            except ValueError as exc:
                raise SimulatorError(str(exc)) from exc
        self.prefix = []
        self.ooc_previews = []
        self.preview_pending = False
        fast_replay = bool(self.config.get("fast_replay", False))
        last_selected = None
        for i, action in enumerate(self.spec.prefix):
            preview = bool(self.spec.ooc_previews and self.spec.ooc_previews[i])
            if preview:
                if self.episode.env.phase() != "out_of_combat":
                    raise SimulatorError("OOC preview prefix marker at combat decision")
                self.episode.native_action()
            if fast_replay:
                # Preserve the v5 semantic action/history wrapper and native
                # advancement; skip only disposable intermediate text rendering.
                last_selected = self.episode.env.step(action)
                self.episode.steps += 1
                self.episode._advance()
                state = self.episode.env.summary()
                if state.get("undefined_behavior_evoked"):
                    raise SimulatorError("Simulator reported undefined behavior; trajectory is invalid")
                if self.episode.env.is_terminal():
                    raise SimulatorError("Task prefix reached an ended episode")
                if not self.episode.env.legal_actions():
                    raise SimulatorError("Undecided simulator has no legal actions")
            else:
                observation, _, terminated, truncated, info = self.episode.step(action)
                if terminated or truncated:
                    raise SimulatorError("Task prefix reached an ended episode")
            self.prefix.append(action)
            self.ooc_previews.append(preview)
        if fast_replay and self.spec.prefix:
            observation = self.episode._observe()
            terminated, truncated, reason = self.episode._status(observation)
            self.episode.finished = terminated or truncated
            info = self.episode._info(observation, reason)
            info["executed_action"] = asdict(last_selected)
        if self.spec.kind == "battle" and observation["phase"] != "combat":
            if not self.spec.seek_combat:
                raise SimulatorError("Battle prefix does not end at a combat decision")
            for _ in range(256):
                action = self.episode.native_action()
                observation, _, terminated, truncated, info = self.episode.step(action)
                self.prefix.append(action)
                self.ooc_previews.append(True)
                if terminated or truncated:
                    raise SimulatorError("World ended before requested combat task")
                if observation["phase"] == "combat":
                    break
            else:
                raise SimulatorError("No battle within 256 native OOC actions")
        actual = public_hash(observation)
        if self.spec.expected_public_hash is not None and actual != self.spec.expected_public_hash:
            raise SimulatorError("Replay public observation hash mismatch")
        self.start_prefix = tuple(self.prefix)
        self.start_ooc_previews = tuple(self.ooc_previews)
        self.steps = 0
        self.start_hp = player_hp(observation)
        self.observation = observation
        self.finished = bool(info.get("terminated") or info.get("truncated"))
        if self.finished:
            raise SimulatorError("Cannot reset an already-ended curriculum task")
        info = self._info(info, "")
        info.update(terminated=False, truncated=False, public_hash=actual)
        return observation, info

    def _info(self, info: dict[str, Any], outcome: str) -> dict[str, Any]:
        return {**info, "task_version": TASK_VERSION, "task_kind": self.spec.kind,
                "task_steps": self.steps, "task_outcome": outcome,
                "task_start_prefix": list(self.start_prefix),
                "task_start_ooc_previews": list(self.start_ooc_previews),
                "action_prefix": list(self.prefix), "ooc_previews": list(self.ooc_previews), "split": world_split(self.spec.seed),
                "hp_cost": self.start_hp - player_hp(self.observation)}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.finished:
            raise RuntimeError("Task ended; reset before step")
        if isinstance(action, bool):
            raise ValueError("action must be an integer menu index")
        action = operator.index(action)
        if not 0 <= action < len(self.observation["legal_actions"]):
            raise ValueError("action outside legal menu")
        previous = self.observation
        # Retain the old battle reference: the base clears its bc on exit. This
        # diagnostic remains behind IPC and never contributes policy features.
        old_battle = getattr(self.episode.env, "bc", None)
        observation, _, terminated, truncated, info = self.episode.step(action)
        self.prefix.append(action)
        self.ooc_previews.append(self.preview_pending)
        self.preview_pending = False
        self.steps += 1
        self.observation = observation
        outcome = info.get("termination_reason", "") if terminated else ""
        if self.spec.kind == "battle" and old_battle is not None and observation["phase"] != "combat":
            battle_outcome = str(old_battle.outcome).rsplit(".", 1)[-1]
            if battle_outcome not in {"PLAYER_VICTORY", "PLAYER_LOSS"}:
                raise SimulatorError(f"Unknown battle boundary outcome {battle_outcome}")
            outcome = "battle_victory" if battle_outcome == "PLAYER_VICTORY" else "battle_defeat"
            terminated, truncated = True, False
            info["battle_outcome"] = battle_outcome
            info["termination_reason"] = "battle_finished"
        if truncated and info.get("termination_reason") == "max_act" and self.config.get("goal_act_clear", True):
            terminated, truncated = True, False
            outcome = "act_clear"
            info["termination_reason"] = "curriculum_act_clear"
        if not terminated and self.steps >= self.spec.horizon:
            truncated = True
            info["termination_reason"] = "task_horizon"
        self.finished = terminated or truncated
        info = self._info(info, outcome)
        reward_settings = {"mode": self.config.get("reward_mode", "potential"),
                           "gamma": float(self.config.get("gamma", 1.0)),
                           "progress_floor_weight": float(self.config.get("progress_floor_weight", 0.05)),
                           "terminal_hp_weight": float(self.config.get("terminal_hp_weight", 0.0)),
                           "goal_act_clear": bool(self.config.get("goal_act_clear", True))}
        components = reward_components(previous, observation, task_outcome=outcome,
                                       terminated=terminated,
                                       **{k: v for k, v in reward_settings.items() if k != "goal_act_clear"})
        info.update(reward_version=REWARD_VERSION, reward_settings=reward_settings,
                    reward_components=components)
        return observation, sum(components.values()), terminated, truncated, info

    def teacher_action(self, simulations: int = 50, search_seed: int | None = None) -> int:
        if self.finished:
            raise RuntimeError("Task ended")
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if self.observation["phase"] == "out_of_combat":
            index = int(self.episode.env.native_ooc_action())
            self.preview_pending = True
        else:
            result = self.episode.env.search_best_combat_action(
                simulations, search_seed=self.spec.seed if search_seed is None else search_seed)
            index = int(self.episode.env.display_index_for_bits(int(result["bits"])))
        if not 0 <= index < len(self.observation["legal_actions"]):
            raise SimulatorError("Teacher selected an illegal public action")
        return index


def _serve(connection: Any, config: dict[str, Any]) -> None:
    episode = CurriculumEpisode(config)
    try:
        while True:
            method, args = connection.recv()
            if method == "close":
                break
            try:
                connection.send(("ok", getattr(episode, method)(*args)))
            except Exception as exc:
                connection.send(("error", {"type": type(exc).__name__, "message": str(exc)}))
                # Fail closed: native corruption cannot contaminate another task.
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class CurriculumWorker:
    """One serial RPC stream, asynchronously submit across independent workers.

    The process is persistent across resets. A timeout/error kills this process;
    the next reset creates a fresh one. Each submit Future has one operation
    deadline, including native teacher queries. Never turn its failure into loss.
    """
    def __init__(self, config: dict[str, Any] | None = None, timeout: float = 10.0) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.config = dict(config or {})
        self.config.setdefault("build_dir", os.environ.get("STS_LIGHTSPEED_BUILD"))
        self.timeout = timeout
        self._process: Any = None
        self._connection: Any = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sts-rpc")
        self.observation: dict[str, Any] | None = None
        self.info: dict[str, Any] | None = None
        self.finished = True

    def _launch(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child, self.config),
                                        name="sts-curriculum-native", daemon=True)
        self._process.start()
        child.close()

    def _kill(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
            self._process.close()
            self._process = None
        self.finished = True

    def _call(self, method: str, *args: Any) -> Any:
        with self._lock:
            if self._process is None:
                if method != "reset":
                    raise RuntimeError("reset is required after a native fault")
                self._launch()
            try:
                self._connection.send((method, args))
                if not self._connection.poll(self.timeout):
                    raise SimulatorTimeout(f"Curriculum {method} exceeded {self.timeout:g}s")
                status, value = self._connection.recv()
                if status != "ok":
                    raise SimulatorError(f"{value['type']}: {value['message']}")
                return value
            except (EOFError, BrokenPipeError, OSError) as exc:
                self._kill()
                raise SimulatorError(f"Native worker exited during {method}") from exc
            except SimulatorError:
                self._kill()
                raise

    def reset(self, spec: TaskSpec | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
        return self._executor.submit(getattr(self, method), *args)

    def close(self) -> None:
        with self._lock:
            self._kill()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> CurriculumWorker:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
