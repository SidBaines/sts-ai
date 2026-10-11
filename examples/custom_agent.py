"""Example: sts-env run --agent examples.custom_agent:create_agent --seed 3 --mode ooc.

Your agent can call any local model, API, or policy. Only the environment's
public observation is supplied; return an integer from its legal action list.
"""
from __future__ import annotations

import random
from typing import Any


class MyAgent:
    def __init__(self, prefer_rest: bool = True) -> None:
        self.prefer_rest = prefer_rest
        self.rng = random.Random()

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def choose_action(self, observation: dict[str, Any]) -> int:
        actions = observation["legal_actions"]
        if self.prefer_rest:
            for action in actions:
                if "rest" in action["description"].lower():
                    return action["index"]
        return self.rng.choice(actions)["index"]


def create_agent(prefer_rest: bool = True) -> MyAgent:
    return MyAgent(prefer_rest=prefer_rest)
