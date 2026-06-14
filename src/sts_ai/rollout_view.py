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


def _clean_brace_list(segment: str) -> list[str]:
    """Parse `{a,b,c,}` (the serializer's trailing-comma list) into [a, b, c]."""
    match = re.search(r"\{(.*)\}", segment, flags=re.DOTALL)
    inner = match.group(1) if match else segment
    return [item.strip() for item in inner.split(",") if item.strip()]


def parse_state_text(state_text: str) -> dict[str, Any]:
    """Pull deck/relics/potions/boss out of the serialized state header.

    These fields are only present in `state_text` (not the `state` dict). Returns
    empty values when a line is absent or unparseable, so a serializer change
    degrades gracefully rather than crashing the viewer.
    """
    deck: list[str] = []
    relics: list[str] = []
    potions: list[str] = []
    boss = ""
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Deck:"):
            deck = _clean_brace_list(stripped)
        elif stripped.startswith("Relics:"):
            relics = _clean_brace_list(stripped)
        elif stripped.startswith("Potions:"):
            body = stripped[len("Potions:"):].strip()
            if body and body.lower() != "none":
                potions = [p.strip() for p in body.split(",") if p.strip()]
        elif not boss:
            match = re.search(r"\bboss (.+?)\s*$", stripped)
            if match:
                boss = match.group(1).strip()
    return {"deck": deck, "relics": relics, "potions": potions, "boss": boss}


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
