"""Small, model-independent agent interface and importable agent factories."""
from __future__ import annotations

import importlib
import operator
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol


class Agent(Protocol):
    def choose_action(self, observation: dict[str, Any]) -> int:
        """Return an index from observation['legal_actions']."""
        ...


class FirstAgent:
    def choose_action(self, observation: dict[str, Any]) -> int:
        return int(observation["legal_actions"][0]["index"])


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def choose_action(self, observation: dict[str, Any]) -> int:
        return int(self.rng.choice(observation["legal_actions"])["index"])


class NativeAgent:
    """Marker for the CLI's privileged env.native_action() baseline."""


def load_agent(spec: str, config: dict[str, Any] | None = None) -> Agent | NativeAgent:
    """Load a builtin or module:factory; factories receive JSON config as kwargs."""
    config = {} if config is None else config
    if spec in {"first", "random", "native"}:
        factories = {"first": FirstAgent, "random": RandomAgent, "native": NativeAgent}
        return factories[spec](**config)
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("agent must be first, random, native, or module:factory")
    # A console entry point starts on the executable directory, so explicitly
    # make user modules in the working directory importable during construction.
    working_directory = str(Path.cwd())
    inserted = working_directory not in sys.path
    if inserted:
        sys.path.insert(0, working_directory)
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
        if not callable(factory):
            raise TypeError(f"agent factory {spec!r} is not callable")
        agent = factory(**config)
    finally:
        if inserted:
            sys.path.remove(working_directory)
    if not callable(getattr(agent, "choose_action", None)):
        raise TypeError(f"agent factory {spec!r} must return an object with choose_action(observation)")
    return agent


def choose_action(agent: Agent | NativeAgent, observation: dict[str, Any], env: Any) -> int:
    action = env.native_action() if isinstance(agent, NativeAgent) else agent.choose_action(deepcopy(observation))
    if isinstance(action, bool):
        raise ValueError(f"agent returned {action!r}; expected an integer legal-action index")
    try:
        action = operator.index(action)
    except TypeError as exc:
        raise ValueError(f"agent returned {action!r}; expected an integer legal-action index") from exc
    legal = {item["index"] for item in observation["legal_actions"]}
    if action not in legal:
        raise ValueError(f"agent returned illegal action index {action}; legal indices: {sorted(legal)}")
    return action
