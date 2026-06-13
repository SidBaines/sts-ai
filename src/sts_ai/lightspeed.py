from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sts_ai.lightspeed_import import import_lightspeed
from sts_ai.schemas import LegalAction


class LightspeedHybridEnv:
    """Python-controlled out-of-combat decisions with search-resolved combats."""

    def __init__(
        self,
        seed: int,
        ascension: int = 0,
        battle_simulations: int = 2_000,
        boss_simulation_multiplier: float = 2.0,
        max_act: int = 1,
        build_dir: str | None = None,
    ) -> None:
        self.sts = import_lightspeed(build_dir)
        self.seed = seed
        self.ascension = ascension
        self.max_act = max_act
        self.gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, seed, ascension)
        self.battle_agent = self.sts.Agent()
        self.battle_agent.simulation_count_base = battle_simulations
        self.battle_agent.boss_simulation_multiplier = boss_simulation_multiplier
        self.battle_agent.print_logs = False

    def is_terminal(self) -> bool:
        if self.gc.outcome != self.sts.GameOutcome.UNDECIDED:
            return True
        return self.gc.act > self.max_act

    def resolve_battle_if_needed(self) -> bool:
        if self.gc.screen_state != self.sts.ScreenState.BATTLE:
            return False
        return bool(self.sts.resolve_current_battle(self.gc, self.battle_agent))

    def advance_to_decision(self) -> int:
        resolved = 0
        while not self.is_terminal() and self.gc.screen_state == self.sts.ScreenState.BATTLE:
            if not self.resolve_battle_if_needed():
                break
            resolved += 1
        return resolved

    def raw_actions(self) -> list[Any]:
        self.advance_to_decision()
        return list(self.gc.legal_actions())

    def legal_actions(self) -> list[LegalAction]:
        actions = self.raw_actions()
        return [
            LegalAction(index=i, bits=int(action.bits), description=action.describe(self.gc))
            for i, action in enumerate(actions)
        ]

    def describe_state(self) -> str:
        return str(self.gc.describe_state())

    def step(self, action_index: int) -> LegalAction:
        actions = self.raw_actions()
        if action_index < 0 or action_index >= len(actions):
            raise IndexError(f"action_index {action_index} outside legal range 0..{len(actions) - 1}")

        selected = LegalAction(
            index=action_index,
            bits=int(actions[action_index].bits),
            description=actions[action_index].describe(self.gc),
        )
        actions[action_index].execute(self.gc)
        self.advance_to_decision()
        return selected

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "ascension": self.ascension,
            "act": int(self.gc.act),
            "floor": int(self.gc.floor_num),
            "screen_state": str(self.gc.screen_state),
            "room": str(self.gc.cur_room),
            "outcome": str(self.gc.outcome),
            "cur_hp": int(self.gc.cur_hp),
            "max_hp": int(self.gc.max_hp),
            "gold": int(self.gc.gold),
            "done": self.is_terminal(),
        }

    @staticmethod
    def action_dict(action: LegalAction) -> dict[str, Any]:
        return asdict(action)
