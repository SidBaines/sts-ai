"""Versioned presentation adapter over the frozen v3 simulator interface.

Wrap only after replaying a v3 source prefix. The underlying environment retains
its original observation version, action enumeration, and native state; v4 changes
the public presentation only. Arithmetic reads the same public inputs as v3.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import re
from typing import Any

from sts_env._backend.schemas import LegalAction
from sts_env._backend.turn_math import (
    HAND_DEPENDENT_ATTACKS,
    HandAttack,
    TurnMathInputs,
)
from sts_env._backend.turn_math_v4 import turn_math_lines


_GAMBLE_NONE = "{ GAMBLE none"
_KEEP_CURRENT_HAND = "keep current hand (discard no cards)"
_DEAL_RE = re.compile(r"\(deal\s+(\d+)")


def _action_description(description: str) -> str:
    """Rename the audited no-discard action, without interpreting other masks."""
    return _KEEP_CURRENT_HAND if description == _GAMBLE_NONE else description


def _public_cards(cards: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(cards)
    selection = public.get("selection")
    if isinstance(selection, dict) and selection.get("type") == "GAMBLE":
        selection["candidates"] = [
            _action_description(candidate)
            for candidate in selection.get("candidates", [])
        ]
    return public


class CombatPublicV4Env:
    """Delegate simulation to a v3 environment and expose the v4 public view."""

    combat_observation = "combat_public_v4"

    def __init__(self, env: Any) -> None:
        if env.combat_observation != "combat_public_v3":
            raise ValueError("CombatPublicV4Env requires a combat_public_v3 environment")
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def legal_actions(self) -> list[LegalAction]:
        return [
            replace(action, description=_action_description(action.description))
            for action in self._env.legal_actions()
        ]

    def step(self, action_index: int) -> LegalAction:
        action = self._env.step(action_index)
        return replace(action, description=_action_description(action.description))

    def describe_state(self) -> str:
        lines = self._env.describe_state().split("\n")
        for index, line in enumerate(lines):
            if line == f"  - {_GAMBLE_NONE}":
                lines[index] = f"  - {_KEEP_CURRENT_HAND}"
            elif line.startswith("Recent actions this turn: "):
                prefix = "Recent actions this turn: "
                lines[index] = prefix + " -> ".join(
                    _action_description(part)
                    for part in line[len(prefix):].split(" -> ")
                )
        return "\n".join(lines)

    def public_combat_cards(self) -> dict[str, Any]:
        return _public_cards(self._env.public_combat_cards())

    def summary(self) -> dict[str, Any]:
        data = deepcopy(self._env.summary())
        data["combat_observation"] = self.combat_observation
        combat = data.get("combat")
        if isinstance(combat, dict):
            if "cards" in combat:
                combat["cards"] = _public_cards(combat["cards"])
            if "recent_actions" in combat:
                combat["recent_actions"] = [
                    _action_description(action) for action in combat["recent_actions"]
                ]
        return data

    def _turn_math_lines(self) -> list[str]:
        """Gather only public v3 arithmetic inputs, then use the v4 renderer.

        This mirrors v3's input assembly deliberately: editing its module would
        invalidate frozen source replay, and replacing its module globals would
        make simultaneous v3/v4 use unsafe. No hidden pile order or RNG is read.
        """
        if self.bc is None or self.bc.input_state != self.sts.InputState.PLAYER_NORMAL:
            return []
        cards = dict(self.bc.public_cards(include_card_type=True))
        player = dict(self.bc.public_player())
        living = [dict(enemy) for enemy in self.bc.enemies() if bool(enemy.get("alive"))]
        incoming = sum(
            int(enemy.get("intent_damage", 0)) * int(enemy.get("intent_hits", -1))
            for enemy in living
            if int(enemy.get("intent_hits", -1)) > 0
        )
        shown_damage: dict[tuple[str, int], int] = {}
        for action in self.bc.legal_actions(include_card_type=True):
            description = str(action.describe(
                self.bc, include_card_type=True, include_computed_damage=True,
            ))
            match = _DEAL_RE.search(description)
            if match is None:
                continue
            for card in cards.get("hand", []):
                if str(card.get("type")) != "Attack":
                    continue
                name = str(card.get("name", ""))
                if name in HAND_DEPENDENT_ATTACKS:
                    continue
                cost_for_turn = int(card.get("cost_for_turn", -1))
                cost_label = "X" if cost_for_turn < 0 else str(cost_for_turn)
                prefix = f"play {name} [Attack] (cost {cost_label}"
                if re.match(re.escape(prefix) + r"[,)]", description):
                    key = (name, cost_for_turn)
                    shown_damage[key] = max(shown_damage.get(key, 0), int(match.group(1)))
        grouped: Counter[tuple[str, int, bool]] = Counter()
        for card in cards.get("hand", []):
            if str(card.get("type")) == "Attack":
                grouped[(
                    str(card.get("name", "")),
                    int(card.get("cost_for_turn", -1)),
                    bool(card.get("free_to_play_once", False)),
                )] += 1
        attacks: list[HandAttack] = []
        for (name, raw_cost, free_once), copies in grouped.items():
            cost = None if raw_cost < 0 else (0 if free_once else raw_cost)
            attacks.append((name, cost, shown_damage.get((name, raw_cost)), copies))
        powers = player.get("powers", {})
        return turn_math_lines(TurnMathInputs(
            incoming_damage=incoming,
            player_block=int(player.get("block", 0)),
            player_metallicize=(int(powers.get("Metallicize", 0))
                                if isinstance(powers, dict) else 0),
            energy=int(player.get("energy", 0)),
            hand_attacks=attacks,
            living_enemies=[(
                str(enemy.get("name", "")),
                int(enemy.get("cur_hp", 0)),
                int(enemy.get("block", 0)),
            ) for enemy in living],
        ))
