"""Display model for rollout traces.

Pure functions that turn canonical rollout JSONL records (see schemas.py) into
flat, render-friendly structures. No visualisation dependency lives here so the
parsing stays unit-testable and the Streamlit app (scripts/visualize_rollout.py)
can be a thin shell over it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ActionView:
    index: int
    description: str
    chosen: bool


@dataclass(frozen=True)
class CardView:
    """A card tile. `cost_text` is the raw cost token ("1", "X") for combat-hand
    cards and None for out-of-combat deck cards (the deck serialization carries no
    costs). `playable` is meaningful only in combat (a matching `play` action
    exists this instant); `chosen` marks the hand slot the decision played."""
    name: str
    cost_text: Optional[str] = None
    upgraded: bool = False
    playable: bool = True
    chosen: bool = False
    count: int = 1


@dataclass(frozen=True)
class EnemyView:
    index: int
    name: str           # raw engine id, e.g. ACID_SLIME_S
    display_name: str   # prettified, e.g. "Acid Slime (S)"
    cur_hp: int
    max_hp: int
    block: int
    intent: str         # raw move id, e.g. ACID_SLIME_S_LICK
    intent_label: str   # prettified, enemy prefix stripped, e.g. "Lick"
    alive: bool
    targeted: bool      # the chosen action targets this enemy


@dataclass
class CombatView:
    """Render-ready snapshot of an in-combat decision. Player stats and enemies
    come from the structured `state["combat"]` dict (authoritative; includes dead
    enemies); hand/piles/powers and energy-max are parsed from `state_text`, which
    is their only source. Degrades to empty pieces if a line is absent."""
    turn: int
    player_cur_hp: int
    player_max_hp: int
    player_block: int
    player_energy: int
    player_energy_max: int
    powers: list[str]
    enemies: list[EnemyView]
    hand: list[CardView]
    draw_count: int
    discard_count: int
    exhaust_count: int
    potions: list[str]
    # which hand card / enemy the chosen action acts on (for board highlighting)
    chosen_card_name: str = ""
    chosen_is_end_turn: bool = False
    chosen_target_index: Optional[int] = None


@dataclass
class DecisionView:
    decision_index: int
    seed: int
    # headline state (from the `state` dict, which is authoritative)
    act: int
    floor: int
    screen: str
    room: str
    boss: str
    cur_hp: int
    max_hp: int
    gold: int
    # parsed from state_text (not present in the state dict)
    deck: list[str]
    relics: list[str]
    potions: list[str]
    # the decision
    actions: list[ActionView]
    chosen_index: int
    chosen_description: str
    reasoning: str
    thinking: str
    valid: bool
    retries: int
    raw_response: str
    # consequence
    hp_after: Optional[int]
    floor_after: Optional[int]
    state_text: str = ""
    # present only for in-combat decisions (phase == "combat"); None otherwise.
    combat: Optional[CombatView] = None


def _clean_brace_list(segment: str) -> list[str]:
    """Parse `{a,b,c,}` (the serializer's trailing-comma list) into [a, b, c]."""
    match = re.search(r"\{(.*)\}", segment, flags=re.DOTALL)
    inner = match.group(1) if match else segment
    return [item.strip() for item in inner.split(",") if item.strip()]


def _parse_potions(state_text: str) -> list[str]:
    """Parse the `Potions: a, b` line (or `Potions: none`). Shared by the
    out-of-combat header and the combat serializer, which use the same format."""
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Potions:"):
            body = stripped[len("Potions:"):].strip()
            if body and body.lower() != "none":
                return [p.strip() for p in body.split(",") if p.strip()]
            return []
    return []


def parse_state_text(state_text: str) -> dict[str, Any]:
    """Pull deck/relics/potions/boss out of the serialized state header.

    These fields are only present in `state_text` (not the `state` dict). Returns
    empty values when a line is absent or unparseable, so a serializer change
    degrades gracefully rather than crashing the viewer.
    """
    deck: list[str] = []
    relics: list[str] = []
    boss = ""
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Deck:"):
            deck = _clean_brace_list(stripped)
        elif stripped.startswith("Relics:"):
            relics = _clean_brace_list(stripped)
        elif stripped.startswith("Potions:"):
            continue  # handled by _parse_potions; skip so it can't match the boss regex
        elif not boss:
            match = re.search(r"\bboss (.+?)\s*$", stripped)
            if match:
                boss = match.group(1).strip()
    return {"deck": deck, "relics": relics, "potions": _parse_potions(state_text), "boss": boss}


_SIZE_SUFFIXES = {"S", "M", "L"}
# "[i] Name (cost C)" hand line; cost token is non-greedy so "X" / "-1" survive.
_HAND_RE = re.compile(r"^\[(\d+)\]\s+(.*?)\s+\(cost\s+(\S+?)\)\s*$")
# "play Name (cost C)" prefix of a combat action; optional " -> Target" tail.
_PLAY_RE = re.compile(r"^play\s+(.*?)\s+\(cost\s+(\S+?)\)(?:\s*->\s*(.*?))?\s*$")
_PILES_RE = re.compile(r"Piles:\s*draw\s+(\d+),\s*discard\s+(\d+),\s*exhaust\s+(\d+)")
_ENERGY_RE = re.compile(r"energy:\s*(\d+)\s*/\s*(\d+)")


def prettify_enemy_name(raw: str) -> str:
    """ACID_SLIME_S -> 'Acid Slime (S)', JAW_WORM -> 'Jaw Worm'."""
    parts = [p for p in raw.split("_") if p]
    if not parts:
        return raw
    size = ""
    if len(parts) > 1 and parts[-1] in _SIZE_SUFFIXES:
        size = f" ({parts[-1]})"
        parts = parts[:-1]
    return " ".join(word.capitalize() for word in parts) + size


def prettify_intent(intent: str, enemy_raw: str) -> str:
    """ACID_SLIME_S_LICK (enemy ACID_SLIME_S) -> 'Lick'. No category/damage is
    inferred — records carry only the raw move id (see CLAUDE.md)."""
    body = intent
    prefix = f"{enemy_raw}_"
    if enemy_raw and body.startswith(prefix):
        body = body[len(prefix):]
    body = body.replace("_", " ").strip()
    if not body:
        body = intent.replace("_", " ").strip()
    return " ".join(word.capitalize() for word in body.split())


def _parse_hand_lines(state_text: str) -> list[tuple[str, str]]:
    """Pull (name, cost_text) pairs out of the `Hand:` section of a combat
    serialization. Returns [] for an empty hand or absent section."""
    pairs: list[tuple[str, str]] = []
    in_hand = False
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Hand:"):
            in_hand = stripped[len("Hand:"):].strip() != "empty"
            continue
        if in_hand:
            match = _HAND_RE.match(stripped)
            if match:
                pairs.append((match.group(2), match.group(3)))
            else:
                break  # next section (Piles:/Potions:) ends the hand block
    return pairs


def _parse_powers(state_text: str) -> list[str]:
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Player powers:"):
            body = stripped[len("Player powers:"):].strip()
            if body and body.lower() != "none":
                return [p.strip() for p in body.split(",") if p.strip()]
            return []
    return []


def to_combat_view(record: dict[str, Any]) -> Optional[CombatView]:
    """Build a CombatView from a combat decision record, or None if the record
    has no `state["combat"]` block (i.e. an out-of-combat decision)."""
    state = record.get("state", {})
    combat = state.get("combat")
    if not combat:
        return None
    state_text = record.get("state_text", "")

    # Which (name, cost) cards are playable right now, and what the decision did.
    play_keys = set()
    for action in record.get("legal_actions", []):
        match = _PLAY_RE.match(str(action.get("description", "")))
        if match:
            play_keys.add((match.group(1), match.group(2)))

    chosen_card_name = ""
    chosen_cost = None
    chosen_target_raw = None
    chosen_is_end_turn = False
    sel_desc = str(record.get("selected_action", {}).get("description", "")).strip()
    sel_match = _PLAY_RE.match(sel_desc)
    if sel_match:
        chosen_card_name = sel_match.group(1)
        chosen_cost = sel_match.group(2)
        chosen_target_raw = (sel_match.group(3) or "").strip() or None
    elif sel_desc.lower().startswith("end turn"):
        chosen_is_end_turn = True

    enemies_raw = combat.get("enemies", []) or []
    chosen_target_index: Optional[int] = None
    if chosen_target_raw:
        matches = [e for e in enemies_raw if e.get("name") == chosen_target_raw]
        alive = [e for e in matches if e.get("alive", True)]
        pick = alive or matches
        if pick:
            chosen_target_index = pick[0].get("index")

    enemies = [
        EnemyView(
            index=int(e.get("index", -1)),
            name=str(e.get("name", "")),
            display_name=prettify_enemy_name(str(e.get("name", ""))),
            cur_hp=int(e.get("cur_hp", 0)),
            max_hp=int(e.get("max_hp", 0)),
            block=int(e.get("block", 0)),
            intent=str(e.get("intent", "")),
            intent_label=prettify_intent(str(e.get("intent", "")), str(e.get("name", ""))),
            alive=bool(e.get("alive", True)),
            targeted=(chosen_target_index is not None and e.get("index") == chosen_target_index),
        )
        for e in enemies_raw
    ]

    hand: list[CardView] = []
    chosen_marked = False
    for name, cost_text in _parse_hand_lines(state_text):
        is_chosen = (
            not chosen_marked
            and bool(chosen_card_name)
            and name == chosen_card_name
            and cost_text == chosen_cost
        )
        if is_chosen:
            chosen_marked = True
        hand.append(
            CardView(
                name=name,
                cost_text=cost_text,
                upgraded=name.endswith("+"),
                playable=(name, cost_text) in play_keys,
                chosen=is_chosen,
            )
        )

    piles = _PILES_RE.search(state_text)
    energy = _ENERGY_RE.search(state_text)
    return CombatView(
        turn=int(combat.get("turn", 0)),
        player_cur_hp=int(combat.get("player_cur_hp", 0)),
        player_max_hp=int(combat.get("player_max_hp", 0)),
        player_block=int(combat.get("player_block", 0)),
        player_energy=int(combat.get("player_energy", 0)),
        player_energy_max=int(energy.group(2)) if energy else int(combat.get("player_energy", 0)),
        powers=_parse_powers(state_text),
        enemies=enemies,
        hand=hand,
        draw_count=int(piles.group(1)) if piles else 0,
        discard_count=int(piles.group(2)) if piles else 0,
        exhaust_count=int(piles.group(3)) if piles else 0,
        potions=_parse_potions(state_text),
        chosen_card_name=chosen_card_name,
        chosen_is_end_turn=chosen_is_end_turn,
        chosen_target_index=chosen_target_index,
    )


def to_view(record: dict[str, Any]) -> DecisionView:
    state = record.get("state", {})
    agent = record.get("agent", {})
    selected = record.get("selected_action", {})
    after = record.get("after_state", {})
    parsed = parse_state_text(record.get("state_text", ""))

    chosen_index = selected.get("index", agent.get("action_index", -1))
    actions = [
        ActionView(
            index=a.get("index", i),
            description=a.get("description", ""),
            chosen=(a.get("index", i) == chosen_index),
        )
        for i, a in enumerate(record.get("legal_actions", []))
    ]

    screen = str(state.get("screen_state", "")).split(".")[-1]
    room = str(state.get("room", "")).split(".")[-1]

    return DecisionView(
        decision_index=record.get("decision_index", -1),
        seed=record.get("seed", state.get("seed", -1)),
        act=state.get("act", -1),
        floor=state.get("floor", -1),
        screen=screen,
        room=room,
        boss=str(state.get("boss", "") or parsed.get("boss", "")) or "?",
        cur_hp=state.get("cur_hp", 0),
        max_hp=state.get("max_hp", 0),
        gold=state.get("gold", 0),
        deck=parsed["deck"],
        relics=parsed["relics"],
        potions=parsed["potions"],
        actions=actions,
        chosen_index=chosen_index,
        chosen_description=selected.get("description", ""),
        reasoning=str(agent.get("reasoning", "")),
        thinking=str(agent.get("thinking", "")),
        valid=bool(agent.get("valid", True)),
        retries=int(agent.get("retries", 0)),
        raw_response=str(agent.get("raw_response", "")),
        hp_after=after.get("cur_hp"),
        floor_after=after.get("floor"),
        state_text=record.get("state_text", ""),
        combat=to_combat_view(record),
    )


@dataclass
class RolloutView:
    path: str
    decisions: list[DecisionView]
    error: Optional[dict[str, Any]] = None

    @property
    def boss(self) -> str:
        return self.decisions[0].boss if self.decisions else "?"

    @property
    def seed(self) -> int:
        return self.decisions[0].seed if self.decisions else -1


def load_rollout(path: str | Path) -> RolloutView:
    """Load a rollout JSONL (and an adjacent `.error.json` sidecar if present)."""
    path = Path(path)
    decisions: list[DecisionView] = []
    if path.exists():
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    decisions.append(to_view(json.loads(line)))

    error = None
    sidecar = path.with_suffix(".error.json")
    if sidecar.exists():
        try:
            error = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            error = {"message": "unparseable error sidecar"}

    return RolloutView(path=str(path), decisions=decisions, error=error)
