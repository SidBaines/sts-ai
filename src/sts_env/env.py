from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import multiprocessing
import operator
import os
from typing import Any, Callable

ENVIRONMENT_VERSION = "0.1.1"
OBSERVATION_VERSION = "sts_env_public_v1"


class SimulatorError(RuntimeError):
    """Native failure; it is not a policy loss or a valid terminal transition."""


class SimulatorTimeout(SimulatorError):
    """The simulator worker exceeded its bounded operation timeout."""


def _make_backend(config: dict[str, Any], seed: int) -> Any:
    from sts_env._backend.lightspeed import LightspeedHybridEnv
    from sts_env._backend.combat_public_v5 import CombatPublicV5Env

    native = LightspeedHybridEnv(
        world_seed=seed, ascension=config["ascension"],
        max_act=config["max_act"], battle_simulations=config["battle_simulations"],
        combat_control="search" if config["mode"] == "ooc" else "llm",
        combat_observation="combat_public_v3", build_dir=config["build_dir"],
    )
    return CombatPublicV5Env(native)


class _Episode:
    """Lives entirely inside one worker; no private native objects cross IPC."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        from sts_env.provenance import interface_identity
        self.interface_identity = interface_identity()
        self.seed = seed
        self.env = _make_backend(config, seed)
        self.steps = 0
        self.native_steps = 0
        self.finished = False
        self._seed_native = config["ooc_seed"] if config["ooc_seed"] is not None else seed
        self.env.seed_native_ooc(self._seed_native)

    def _advance(self) -> None:
        self.env.advance_to_decision()
        if self.config["mode"] == "combat":
            # Bound even a buggy automatic policy that cycles between screens.
            for _ in range(10000):
                if self.env.is_terminal() or self.env.phase() == "combat":
                    break
                self.env.advance_native_ooc()
                self.native_steps += 1
                self.env.advance_to_decision()
            else:
                raise SimulatorError("Native out-of-combat policy exceeded 10000 actions")

    def _observe(self) -> dict[str, Any]:
        state = deepcopy(self.env.summary())
        # Seeds identify episodes in info; they are not policy observation data.
        state.pop("world_seed", None)
        state.pop("combat_observation", None)
        state["observation_version"] = OBSERVATION_VERSION
        graph = self.env.map_graph()
        if graph is not None:
            state["map"] = deepcopy(graph)
        if state.get("undefined_behavior_evoked"):
            raise SimulatorError("Simulator reported undefined behavior; trajectory is invalid")
        actions = [] if self.env.is_terminal() else self.env.legal_actions()
        text = self.env.describe_state()
        from sts_env._backend.glossary_v5 import augment
        text = augment(text, [asdict(action) for action in actions], phase=self.env.phase(),
                       map_graph=graph,
                       turn_math_lines=self.env._turn_math_lines())
        if graph is not None:
            lines = [f"Visible act map: current row {graph['cur_y']}; coordinates (x,y)."]
            for node in sorted(graph["nodes"], key=lambda n: (n["y"], n["x"])):
                destinations = ", ".join(f"({x},{node['y'] + 1})" for x in node["edges"]) or "act boss"
                lines.append(f"({node['x']},{node['y']}) {node['room']} -> {destinations}")
            text += "\n\n-- VISIBLE ROUTES --\n" + "\n".join(lines)
        return {"state": state, "text": text, "phase": self.env.phase(),
                "legal_actions": [{"index": action.index, "description": action.description}
                                  for action in actions]}

    def _status(self, observation: dict[str, Any]) -> tuple[bool, bool, str]:
        state = observation["state"]
        outcome = str(state.get("outcome", "")).rsplit(".", 1)[-1]
        terminal = outcome in {"PLAYER_VICTORY", "PLAYER_LOSS"}
        if terminal:
            return True, False, "victory" if outcome == "PLAYER_VICTORY" else "defeat"
        if int(state.get("act", 0)) > self.config["max_act"]:
            return False, True, "max_act"
        if self.steps >= self.config["max_steps"]:
            return False, True, "max_steps"
        if not observation["legal_actions"]:
            raise SimulatorError("Undecided simulator has no legal actions")
        return False, False, ""

    def _info(self, observation: dict[str, Any], reason: str) -> dict[str, Any]:
        return {**self.interface_identity, "mode": self.config["mode"], "world_seed": self.seed,
                "environment_version": ENVIRONMENT_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "native_steps": self.native_steps, "steps": self.steps,
                "outcome": observation["state"].get("outcome", ""),
                "termination_reason": reason, "ooc_seed": self._seed_native,
                "ooc_effective_seed": self._seed_native & 0xFFFFFFFF,
                "native_provenance": deepcopy(getattr(self.env, "native_provenance", {}))}

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._advance()
        observation = self._observe()
        terminated, truncated, reason = self._status(observation)
        self.finished = terminated or truncated
        info = self._info(observation, reason)
        info.update(terminated=terminated, truncated=truncated)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.finished:
            raise RuntimeError("Episode ended; call reset before step")
        selected = self.env.step(action)
        self.steps += 1
        self._advance()
        observation = self._observe()
        terminated, truncated, reason = self._status(observation)
        self.finished = terminated or truncated
        info = self._info(observation, reason)
        info["executed_action"] = asdict(selected)
        reward = 1.0 if reason == "victory" else 0.0
        return observation, reward, terminated, truncated, info

    def native_action(self) -> int:
        if self.finished:
            raise RuntimeError("Episode ended; call reset before native_action")
        if self.env.phase() == "out_of_combat":
            return self.env.native_ooc_action()
        result = self.env.search_best_combat_action(
            self.config["battle_simulations"], search_seed=self.seed)
        return self.env.display_index_for_bits(int(result["bits"]))


def _worker(connection: Any, config: dict[str, Any], seed: int) -> None:
    try:
        episode = _Episode(config, seed)
        connection.send(("ok", episode.reset()))
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
    except Exception as exc:
        try:
            connection.send(("error", {"type": type(exc).__name__, "message": str(exc)}))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class SlayTheSpireEnv:
    """Gym-style, dependency-free API with bounded native worker isolation.

    full: external policy controls all decisions.
    ooc: external policy controls OOC; native search resolves combat.
    combat: external policy controls combat; native scripted policy handles OOC.

    Action indices refer to the current observation's legal-action menu.
    Call from a __main__ guard when using default spawn isolation in scripts.
    """

    def __init__(self, *, mode: str = "full", ascension: int = 0,
                 max_act: int = 3, max_steps: int = 1500,
                 battle_simulations: int = 50, step_timeout: float = 10.0,
                 isolated: bool = True, build_dir: str | os.PathLike[str] | None = None,
                 reward_fn: Callable[..., float] | None = None,
                 ooc_seed: int | None = None) -> None:
        if mode not in {"full", "ooc", "combat"}:
            raise ValueError("mode must be full, ooc, or combat")
        for name, value, low, high in (("ascension", ascension, 0, 20),
                                      ("max_act", max_act, 1, 4),
                                      ("max_steps", max_steps, 1, None),
                                      ("battle_simulations", battle_simulations, 1, None)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
                raise ValueError(f"Invalid {name}: {value!r}")
        if not math.isfinite(step_timeout) or step_timeout <= 0:
            raise ValueError("step_timeout must be finite and positive")
        if ooc_seed is not None:
            self._validate_seed(ooc_seed)
        self.config = {"mode": mode, "ascension": ascension, "max_act": max_act,
                       "max_steps": max_steps, "battle_simulations": battle_simulations,
                       "build_dir": str(build_dir) if build_dir is not None else None,
                       "ooc_seed": ooc_seed}
        self.step_timeout = float(step_timeout)
        self.isolated = bool(isolated)
        self.reward_fn = reward_fn
        self._process: Any = None
        self._connection: Any = None
        self._episode: _Episode | None = None
        self._observation: dict[str, Any] | None = None
        self._info: dict[str, Any] | None = None
        self._finished = False

    @staticmethod
    def _validate_seed(seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63-1]")

    @property
    def observation(self) -> dict[str, Any] | None:
        return deepcopy(self._observation)

    def _receive(self, operation: str) -> Any:
        try:
            if not self._connection.poll(self.step_timeout):
                raise SimulatorTimeout(f"Simulator {operation} exceeded {self.step_timeout:g}s")
            status, value = self._connection.recv()
            if status != "ok":
                raise SimulatorError(f"{value['type']}: {value['message']}")
            return value
        except SimulatorError:
            self.close()
            raise
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close()
            raise SimulatorError(f"Simulator worker exited during {operation}") from exc

    def _call(self, method: str, *args: Any) -> Any:
        if self.isolated:
            try:
                self._connection.send((method, args))
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.close()
                raise SimulatorError("Simulator worker disconnected") from exc
            return self._receive(method)
        try:
            return getattr(self._episode, method)(*args)
        except Exception as exc:
            self.close()
            if isinstance(exc, SimulatorError):
                raise
            raise SimulatorError(f"{type(exc).__name__}: {exc}") from exc

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_seed(seed)
        self.close()
        self._finished = False
        self._observation = None
        if self.isolated:
            context = multiprocessing.get_context("spawn")
            self._connection, child = context.Pipe()
            self._process = context.Process(target=_worker, args=(child, self.config, seed),
                                            name="sts-env-simulator", daemon=True)
            try:
                self._process.start()
                child.close()
                observation, info = self._receive("reset")
            except BaseException:
                self.close()
                raise
            finally:
                child.close()
        else:
            try:
                self._episode = _Episode(self.config, seed)
                observation, info = self._episode.reset()
            except Exception as exc:
                self.close()
                raise SimulatorError(f"{type(exc).__name__}: {exc}") from exc
        self._observation, self._info = deepcopy(observation), deepcopy(info)
        self._finished = bool(info["terminated"] or info["truncated"])
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._observation is None or (self._episode is None and self._connection is None):
            raise RuntimeError("Call reset before step")
        if self._finished:
            raise RuntimeError("Episode ended; call reset before step")
        if isinstance(action, bool):
            raise ValueError("Action must be an integer legal-action index")
        try:
            index = operator.index(action)
        except TypeError as exc:
            raise ValueError("Action must be an integer legal-action index") from exc
        if not 0 <= index < len(self._observation["legal_actions"]):
            raise ValueError(f"Action {index} is outside the current legal-action menu")
        previous = deepcopy(self._observation)
        observation, reward, terminated, truncated, info = self._call("step", index)
        self._observation, self._info = deepcopy(observation), deepcopy(info)
        self._finished = terminated or truncated
        if self.reward_fn is not None:
            reward = float(self.reward_fn(previous, index, deepcopy(observation), deepcopy(info)))
            if not math.isfinite(reward):
                raise ValueError("reward_fn must return a finite scalar")
        return observation, reward, terminated, truncated, info

    def native_action(self) -> int:
        if self._observation is None or self._finished or (self._episode is None and self._connection is None):
            raise RuntimeError("Call reset before native_action")
        return int(self._call("native_action"))

    def render(self) -> str:
        if self._observation is None:
            raise RuntimeError("Call reset before render")
        return self._observation["text"]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            if self._process.pid is not None:
                self._process.join(timeout=0.1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1)
            self._process.close()
            self._process = None
        self._episode = None

    def __enter__(self) -> SlayTheSpireEnv:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
