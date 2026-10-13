"""Explicit historical-policy projection onto the repaired public menu.

This bridge reproduces the old policy's information/action abstraction for a
baseline or frozen combat controller. It does not remove any core actions and
must not be confused with the richer trainable v3 representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from .training import file_identity


BRIDGE_VERSION = "legacy_candidate_v2_public_menu_bridge_v2"
_INSTANCE_SUFFIX = re.compile(r" \[hand \d+(?:; [^\]]+)*\]$")
_COST = re.compile(r"\(cost (?:-?\d+|X|unplayable)\)")


@dataclass(frozen=True)
class LegacyProjection:
    observation: dict[str, Any]
    current_indices: tuple[int, ...]
    # Aligned with every repaired candidate, including collapsed duplicates.
    legacy_positions: tuple[int, ...]


def project_legacy_menu(observation: dict[str, Any]) -> LegacyProjection:
    """Use public card mechanics to restore legacy labels and first-copy routing.

    World/RNG identity and native bits are neither read nor used. Only combat
    duplicates were collapsed in v2; OOC offers must remain one-to-one.
    """
    if observation.get("state", {}).get("observation_version") not in {"sts_env_public_v2","sts_env_public_v3"}:
        raise ValueError("legacy bridge requires a repaired public-v2/public-v3 observation")
    combat = observation.get("phase") == "combat"
    actions = []
    current = []
    seen: dict[str, int] = {}
    positions = []
    for action in observation.get("legal_actions", []):
        label = str(action["description"])
        if combat:
            label = _INSTANCE_SUFFIX.sub("", label)
            mechanics = action.get("public", {})
            if mechanics.get("kind") == "play_card":
                card = mechanics.get("card", {})
                if "cost_for_turn" not in card:
                    raise ValueError("typed play action is missing its public current cost")
                cost = int(card["cost_for_turn"])
                text = "X" if cost == -1 else "unplayable" if cost < -1 else str(cost)
                label = _COST.sub(lambda _: f"(cost {text})", label, count=1)
            if label in seen:
                positions.append(seen[label])
                continue
            seen[label] = len(actions)
        positions.append(len(actions))
        current.append(int(action["index"]))
        actions.append({"index": len(actions), "description": label})
    projected = dict(observation)
    if combat and "gold_at_battle_entry" in observation["state"]:
        projected["state"] = dict(observation["state"])
        for field,entry in (("gold","gold_at_battle_entry"),("cur_hp","hp_at_battle_entry"),
                            ("max_hp","max_hp_at_battle_entry")):
            if entry in observation["state"]:
                projected["state"][field] = observation["state"][entry]
    # The v2 encoder uses a public allowlist; inventory and typed extras are
    # deliberately ignored. Retain owned state/text and substitute only the menu.
    projected["legal_actions"] = actions
    return LegacyProjection(projected, tuple(current), tuple(positions))


class LegacyBridgeAgent:
    def __init__(self, checkpoint: str, device: str = "cpu", seed: int = 0,
                 deterministic: bool = True) -> None:
        from .model_agent import ModelAgent
        if not deterministic:
            raise ValueError("frozen legacy bridge currently requires deterministic control")
        self.agent = ModelAgent(checkpoint, device=device, seed=seed, deterministic=True)
        self.checkpoint_sha256 = file_identity(checkpoint)

    def reset(self, seed: int) -> None:
        self.agent.reset(seed)

    def choose_actions(self, observations: Sequence[dict[str, Any]]) -> list[int]:
        projected = [project_legacy_menu(observation) for observation in observations]
        choices = self.agent.choose_actions([row.observation for row in projected])
        return [row.current_indices[choice] for row, choice in zip(projected, choices)]

    def choose_action(self, observation: dict[str, Any]) -> int:
        return self.choose_actions([observation])[0]


def make_agent(checkpoint: str, device: str = "cpu", seed: int = 0) -> LegacyBridgeAgent:
    return LegacyBridgeAgent(checkpoint, device=device, seed=seed)
