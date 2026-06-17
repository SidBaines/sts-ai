from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sts_ai.lightspeed_import import import_lightspeed
from sts_ai.schemas import LegalAction


class LightspeedHybridEnv:
    """Python-controlled out-of-combat decisions with search-resolved combats."""

    def __init__(
        self,
        world_seed: int,
        ascension: int = 0,
        battle_simulations: int = 2_000,
        boss_simulation_multiplier: float = 2.0,
        max_act: int = 3,
        combat_control: str = "search",
        build_dir: str | None = None,
    ) -> None:
        if combat_control not in ("search", "llm"):
            raise ValueError(f"combat_control must be 'search' or 'llm', got {combat_control!r}")
        self.sts = import_lightspeed(build_dir)
        self.world_seed = world_seed
        self.ascension = ascension
        self.max_act = max_act
        # "search": battles auto-resolved by the built-in C++ search agent (hybrid).
        # "llm": each in-combat decision is surfaced to the agent (full control).
        self.combat_control = combat_control
        # Live combat state when an in-combat decision is pending; None otherwise.
        # Its presence is what distinguishes a combat decision from an out-of-combat
        # one (see `phase`).
        self.bc: Any | None = None
        # Sticky: set if any combat evoked simulator UB. `exit_battle` clears
        # `self.bc`, so a UB flag raised by a battle-ending action would otherwise be
        # lost before the after-state is recorded; latch it here so `summary()`
        # surfaces it for the rest of the run.
        self._undefined_behavior_evoked = False
        self.gc = self.sts.GameContext(self.sts.CharacterClass.IRONCLAD, world_seed, ascension)
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
        if self.combat_control == "llm":
            return self._advance_to_decision_llm()
        resolved = 0
        while not self.is_terminal() and self.gc.screen_state == self.sts.ScreenState.BATTLE:
            if not self.resolve_battle_if_needed():
                break
            resolved += 1
        return resolved

    def _advance_to_decision_llm(self) -> int:
        """Full-control combat: drive the battle engine until an in-combat player
        decision is pending (``self.bc`` set) or the run reaches an out-of-combat
        decision / terminal state.

        ``BattleContext.init`` and ``BattleAction.execute`` each drain the engine to
        the next player decision or a decided outcome, so this never needs to call
        ``execute_actions`` itself. Idempotent when a combat decision is already
        pending. Any simulator error propagates (fail-closed; never hangs).
        """
        resolved = 0
        while not self.is_terminal() and self.gc.screen_state == self.sts.ScreenState.BATTLE:
            if self.bc is None:
                bc = self.sts.BattleContext()
                bc.init(self.gc)
                self.bc = bc
            if self.bc.outcome != self.sts.BattleOutcome.UNDECIDED:
                if bool(self.bc.undefined_behavior_evoked):
                    self._undefined_behavior_evoked = True
                self.bc.exit_battle(self.gc)
                self.bc = None
                resolved += 1
                continue
            # An in-combat player decision is pending; yield it to the agent.
            return resolved
        return resolved

    def phase(self) -> str:
        return "combat" if self.bc is not None else "out_of_combat"

    def _action_context(self) -> Any:
        return self.bc if self.bc is not None else self.gc

    def raw_actions(self) -> list[Any]:
        self.advance_to_decision()
        if self.bc is not None:
            return list(self.bc.legal_actions())
        return list(self.gc.legal_actions())

    def _action_views(self) -> tuple[list[Any], list[LegalAction], list[int]]:
        """Build the display action list and a map back to raw-action indices.

        In combat, collapse actions whose descriptions are byte-identical: two
        copies of the same card played from different hand slots (or otherwise
        equivalent actions) produce the same description and the same resulting
        state, so listing both only inflates and confuses the menu (~36% of combat
        decisions had a duplicate). The chosen display index maps to the FIRST raw
        action carrying that description. This is safe because same-named enemy
        targets are disambiguated in the action text (``-> NAME [enemy i]``), so an
        identical description really does mean an interchangeable action.

        Out of combat the list stays 1:1 (indices and order unchanged), so the
        out-of-combat trace shape and reproducibility are untouched.
        """
        raw = self.raw_actions()
        ctx = self._action_context()
        dedup = self.bc is not None  # combat only
        display: list[LegalAction] = []
        display_to_raw: list[int] = []
        seen: set[str] = set()
        for i, action in enumerate(raw):
            description = action.describe(ctx)
            if dedup and description in seen:
                continue
            seen.add(description)
            display.append(LegalAction(index=len(display), bits=int(action.bits), description=description))
            display_to_raw.append(i)
        return raw, display, display_to_raw

    def legal_actions(self) -> list[LegalAction]:
        _, display, _ = self._action_views()
        return display

    def describe_state(self) -> str:
        if self.bc is not None:
            return str(self.bc.describe_state())
        return str(self.gc.describe_state())

    def map_graph(self) -> dict[str, Any] | None:
        """Structured act map for the current MAP_SCREEN decision, else None.

        Returns ``{"cur_y": int, "nodes": [{"x", "y", "room", "edges"}]}`` from the
        binding (the DAG `sts_ai.glossary` renders into a neutral per-choice path
        summary). None during combat or on any non-map screen — the underlying
        ``GameContext.map`` is only populated on the map screen.
        """
        if self.bc is not None:
            return None
        if self.gc.screen_state != self.sts.ScreenState.MAP_SCREEN:
            return None
        return self.gc.map_graph()

    def step(self, action_index: int) -> LegalAction:
        # Resolve against the same display list the agent saw (deduped in combat),
        # then map the chosen display index back to the underlying raw action.
        raw, display, display_to_raw = self._action_views()
        if action_index < 0 or action_index >= len(display):
            raise IndexError(f"action_index {action_index} outside legal range 0..{len(display) - 1}")

        # Capture the action context (gc or bc) before executing, since executing a
        # combat action and advancing may end the battle and clear self.bc.
        ctx = self._action_context()
        selected = display[action_index]
        raw[display_to_raw[action_index]].execute(ctx)
        self.advance_to_decision()
        return selected

    def summary(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "world_seed": self.world_seed,
            "ascension": self.ascension,
            "act": int(self.gc.act),
            "floor": int(self.gc.floor_num),
            "screen_state": str(self.gc.screen_state),
            "room": str(self.gc.cur_room),
            "outcome": str(self.gc.outcome),
            "cur_hp": int(self.gc.cur_hp),
            "max_hp": int(self.gc.max_hp),
            "gold": int(self.gc.gold),
            "phase": self.phase(),
            # Latched across the run (see __init__); also OR in the live battle so an
            # in-progress combat that has evoked UB reports it immediately.
            "undefined_behavior_evoked": bool(
                self._undefined_behavior_evoked
                or (self.bc is not None and self.bc.undefined_behavior_evoked)
            ),
            "done": self.is_terminal(),
        }
        if self.bc is not None:
            data["combat"] = {
                "turn": int(self.bc.turn),
                "input_state": str(self.bc.input_state),
                "battle_outcome": str(self.bc.outcome),
                "player_cur_hp": int(self.bc.player_cur_hp),
                "player_max_hp": int(self.bc.player_max_hp),
                "player_block": int(self.bc.player_block),
                "player_energy": int(self.bc.player_energy),
                "undefined_behavior_evoked": bool(self.bc.undefined_behavior_evoked),
                "enemies": list(self.bc.enemies()),
            }
        return data

    @staticmethod
    def action_dict(action: LegalAction) -> dict[str, Any]:
        return asdict(action)
