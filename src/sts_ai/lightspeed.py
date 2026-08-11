from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import re
from typing import Any

from sts_ai.lightspeed_import import import_lightspeed
from sts_ai.schemas import LegalAction
from sts_ai.turn_math import HandAttack, TurnMathInputs, turn_math_lines


_PUBLIC_COMBAT_OBSERVATIONS = (
    "combat_public_v1",
    "combat_public_v2",
    "combat_public_v3",
)
_TYPED_COMBAT_OBSERVATIONS = ("combat_public_v2", "combat_public_v3")
_DEAL_RE = re.compile(r"\(deal\s+(\d+)")


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
        combat_observation: str = "legacy",
        public_combat_state: bool | None = None,
    ) -> None:
        if combat_control not in ("search", "llm"):
            raise ValueError(f"combat_control must be 'search' or 'llm', got {combat_control!r}")
        if public_combat_state is not None:
            alias_observation = "combat_public_v1" if public_combat_state else "legacy"
            if combat_observation != "legacy" and combat_observation != alias_observation:
                raise ValueError(
                    "public_combat_state conflicts with combat_observation="
                    f"{combat_observation!r}"
                )
            combat_observation = alias_observation
        if combat_observation not in ("legacy", *_PUBLIC_COMBAT_OBSERVATIONS):
            raise ValueError(
                "combat_observation must be 'legacy', 'combat_public_v1', "
                "'combat_public_v2', or 'combat_public_v3', "
                f"got {combat_observation!r}"
            )
        self.sts = import_lightspeed(build_dir)
        self.world_seed = world_seed
        self.ascension = ascension
        self.max_act = max_act
        # "search": battles auto-resolved by the built-in C++ search agent (hybrid).
        # "llm": each in-combat decision is surfaced to the agent (full control).
        self.combat_control = combat_control
        # Explicit versioned switch. v1 is retained for artifact replay only; v2
        # adds the human-visible card type, and v3 adds opt-in computed damage plus
        # arithmetic derived only from the displayed/structured public values.
        self.combat_observation = combat_observation
        # Live combat state when an in-combat decision is pending; None otherwise.
        # Its presence is what distinguishes a combat decision from an out-of-combat
        # one (see `phase`).
        self.bc: Any | None = None
        self._combat_history_turn: int | None = None
        self._combat_recent_actions: list[str] = []
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
                self._sync_combat_history()
            if self.bc.outcome != self.sts.BattleOutcome.UNDECIDED:
                if bool(self.bc.undefined_behavior_evoked):
                    self._undefined_behavior_evoked = True
                self.bc.exit_battle(self.gc)
                self.bc = None
                self._sync_combat_history()
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
            return list(
                self.bc.legal_actions(
                    include_card_type=getattr(
                        self, "combat_observation", "legacy"
                    ) in _TYPED_COMBAT_OBSERVATIONS
                )
            )
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
            if self.bc is not None:
                observation = getattr(self, "combat_observation", "legacy")
                if observation == "combat_public_v3":
                    description = action.describe(
                        ctx,
                        include_card_type=True,
                        include_computed_damage=True,
                    )
                else:
                    description = action.describe(
                        ctx,
                        include_card_type=observation in _TYPED_COMBAT_OBSERVATIONS,
                    )
            else:
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
            public = self.combat_observation in _PUBLIC_COMBAT_OBSERVATIONS
            state = str(
                self.bc.describe_state(
                    public_state=public,
                    include_card_type=(
                        self.combat_observation in _TYPED_COMBAT_OBSERVATIONS
                    ),
                )
            )
            if public:
                self._sync_combat_history()
                recent = " -> ".join(self._combat_recent_actions) or "none"
                state += f"\nRecent actions this turn: {recent}"
            return state
        return str(self.gc.describe_state())

    def _turn_math_lines(self) -> list[str]:
        """Compose v3 arithmetic inputs without reimplementing simulator rules."""
        if (
            self.bc is None
            or self.bc.input_state != self.sts.InputState.PLAYER_NORMAL
        ):
            return []

        cards = dict(self.bc.public_cards(include_card_type=True))
        player = dict(self.bc.public_player())
        enemies = [dict(enemy) for enemy in self.bc.enemies()]
        living = [enemy for enemy in enemies if bool(enemy.get("alive"))]
        incoming = sum(
            int(enemy.get("intent_damage", 0)) * int(enemy.get("intent_hits", -1))
            for enemy in living
            if int(enemy.get("intent_hits", -1)) > 0
        )

        shown_damage: dict[tuple[str, int], int] = {}
        for action in self.bc.legal_actions(include_card_type=True):
            description = str(
                action.describe(
                    self.bc,
                    include_card_type=True,
                    include_computed_damage=True,
                )
            )
            match = _DEAL_RE.search(description)
            if match is None:
                continue
            for card in cards.get("hand", []):
                if str(card.get("type")) != "Attack":
                    continue
                name = str(card.get("name", ""))
                cost_for_turn = int(card.get("cost_for_turn", -1))
                cost_label = "X" if cost_for_turn < 0 else str(cost_for_turn)
                prefix = f"play {name} [Attack] (cost {cost_label})"
                if description.startswith(prefix):
                    key = (name, cost_for_turn)
                    shown_damage[key] = max(
                        shown_damage.get(key, 0),
                        int(match.group(1)),
                    )

        grouped: Counter[tuple[str, int, bool]] = Counter()
        for card in cards.get("hand", []):
            if str(card.get("type")) == "Attack":
                grouped[
                    (
                        str(card.get("name", "")),
                        int(card.get("cost_for_turn", -1)),
                        bool(card.get("free_to_play_once", False)),
                    )
                ] += 1
        attacks: list[HandAttack] = []
        for (name, raw_cost, free_once), copies in grouped.items():
            cost = None if raw_cost < 0 else (0 if free_once else raw_cost)
            attacks.append(
                (name, cost, shown_damage.get((name, raw_cost)), copies)
            )

        powers = player.get("powers", {})
        metallicize = (
            int(powers.get("Metallicize", 0)) if isinstance(powers, dict) else 0
        )
        inputs = TurnMathInputs(
            incoming_damage=incoming,
            player_block=int(player.get("block", 0)),
            player_metallicize=metallicize,
            energy=int(player.get("energy", 0)),
            hand_attacks=attacks,
            living_enemies=[
                (
                    str(enemy.get("name", "")),
                    int(enemy.get("cur_hp", 0)),
                    int(enemy.get("block", 0)),
                )
                for enemy in living
            ],
        )
        return turn_math_lines(inputs)

    def _sync_combat_history(self) -> None:
        """Reset the public recent-action trace at combat/turn boundaries."""
        # A few pure unit tests construct the env through `object.__new__` with a
        # lightweight battle sentinel; tolerate that compatibility fixture.
        if not hasattr(self, "_combat_recent_actions"):
            self._combat_recent_actions = []
        if not hasattr(self, "_combat_history_turn"):
            self._combat_history_turn = None
        if self.bc is None:
            self._combat_history_turn = None
            self._combat_recent_actions.clear()
            return
        if not hasattr(self.bc, "turn"):
            return
        turn = int(self.bc.turn)
        if turn != self._combat_history_turn:
            self._combat_history_turn = turn
            self._combat_recent_actions.clear()

    def public_combat_cards(self) -> dict[str, Any]:
        """Return the structured public card view for the pending combat choice.

        Draw/discard/exhaust entries are sorted aggregates: no pile order or RNG
        state crosses this API boundary.
        """
        self.advance_to_decision()
        if self.bc is None:
            raise RuntimeError("public_combat_cards requires a pending combat decision")
        return dict(
            self.bc.public_cards(
                include_card_type=(
                    self.combat_observation in _TYPED_COMBAT_OBSERVATIONS
                )
            )
        )

    def public_combat_player(self) -> dict[str, Any]:
        """Return public player resources, named powers, stance, and counters."""
        self.advance_to_decision()
        if self.bc is None:
            raise RuntimeError("public_combat_player requires a pending combat decision")
        return dict(self.bc.public_player())

    def public_combat_relics(self) -> list[dict[str, Any]]:
        """Return relic ownership copied from GameContext plus public counters."""
        self.advance_to_decision()
        if self.bc is None:
            raise RuntimeError("public_combat_relics requires a pending combat decision")
        return [dict(relic) for relic in self.bc.public_relics()]

    def public_combat_potions(self) -> dict[str, Any]:
        """Return human-visible potion capacity and stable indexed slots."""
        self.advance_to_decision()
        if self.bc is None:
            raise RuntimeError("public_combat_potions requires a pending combat decision")
        return dict(self.bc.public_potions())

    def search_best_combat_action(
        self,
        simulations: int,
        *,
        search_seed: int | None = None,
        draw_order_seed: int | None = None,
    ) -> dict[str, Any]:
        """Run the privileged search teacher without mutating live combat state.

        ``draw_order_seed`` is a teacher-only intervention: it shuffles the cloned
        hidden draw order before search so label stability across public-equivalent
        states can be measured. It never changes or appears in the model state.
        """
        self.advance_to_decision()
        if self.bc is None:
            raise RuntimeError("search_best_combat_action requires a pending combat decision")
        return dict(
            self.bc.search_best_action(
                simulations,
                search_seed=search_seed,
                draw_order_seed=draw_order_seed,
                include_card_type=(
                    self.combat_observation in _TYPED_COMBAT_OBSERVATIONS
                ),
            )
        )

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
        if self.bc is not None:
            # Record only successfully executed actions. `_sync_combat_history`
            # below keeps the action if the next choice is in the same turn and
            # clears it when this action advances the turn or ends the battle.
            if not hasattr(self, "_combat_recent_actions"):
                self._combat_recent_actions = []
            self._combat_recent_actions.append(selected.description)
        self.advance_to_decision()
        self._sync_combat_history()
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
            "combat_observation": self.combat_observation,
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
                "player_energy_per_turn": int(self.bc.player_energy_per_turn),
                "undefined_behavior_evoked": bool(self.bc.undefined_behavior_evoked),
                "enemies": list(self.bc.enemies()),
            }
            if self.combat_observation in _PUBLIC_COMBAT_OBSERVATIONS:
                self._sync_combat_history()
                data["combat"].update(
                    {
                        "recent_actions": list(self._combat_recent_actions),
                        "player": dict(self.bc.public_player()),
                        "cards": dict(
                            self.bc.public_cards(
                                include_card_type=(
                                    self.combat_observation
                                    in _TYPED_COMBAT_OBSERVATIONS
                                )
                            )
                        ),
                        "relics": [dict(relic) for relic in self.bc.public_relics()],
                        "potions": dict(self.bc.public_potions()),
                    }
                )
        return data

    @staticmethod
    def action_dict(action: LegalAction) -> dict[str, Any]:
        return asdict(action)
