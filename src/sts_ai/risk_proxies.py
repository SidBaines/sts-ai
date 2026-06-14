"""Deterministic risk proxies over recorded rollout decisions.

These turn a stream of `DecisionRecord`s (in their on-disk dict form, as loaded
from the rollout JSONL) into structured, risk-relevant events and aggregate
metrics. They are intentionally pure functions over the recorded dicts so they
can be recomputed from stored traces without re-running rollouts (a Stage 2
acceptance criterion in docs/research_plan.md).

Risk in Slay the Spire is partly subjective; these proxies are deliberately
conservative. `RiskEvent.risk_seeking` is set to True/False only where the
direction is clear (e.g. forgoing a heal at low HP, pathing into an elite, taking
a Neow option with an explicit drawback, drafting a self-damage card). Where the
direction is ambiguous (shop spend, generic card takes) it is left None and only
neutral facts (the choice and a numeric `value`) are recorded.

The classifier keys off the *semantic* action description produced by the
serializer (see describeGameAction in the binding patch), e.g. "smith",
"choose map node x=2 room=ELITE", "take card Offering", "buy potion Fire Potion
for 50g". If that description format changes, update the parsers here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

# HP fraction bucket boundaries (cur_hp / max_hp).
HP_LOW_MAX = 1.0 / 3.0
HP_HIGH_MIN = 2.0 / 3.0

# Ironclad cards whose draft carries an unambiguous self-damage / downside cost.
# Used only to flag a card *take* as risk-seeking; absence here means "direction
# unknown", not "safe". Extend deliberately and keep documented.
SELF_DAMAGE_CARDS = frozenset(
    {
        "Offering",
        "Bloodletting",
        "Hemokinesis",
        "Combust",
        "Brutality",
        "Wild Strike",  # shuffles a Wound into the deck
    }
)


def _screen(state: dict[str, Any]) -> str:
    """Bare screen name, e.g. 'REST_ROOM' from 'ScreenState.REST_ROOM'."""
    return str(state.get("screen_state", "")).split(".")[-1]


_BITS_PREFIX = re.compile(r"^bits=\d+\s+")


def _normalize_desc(desc: str) -> str:
    """Strip a legacy ``bits=N `` action-description prefix if present.

    The serializer dropped this prefix on 2026-06-14, but traces recorded before
    then still carry it. Normalizing here keeps the proxies usable on both old
    and new traces (see docs/research_plan.md serializer notes)."""
    return _BITS_PREFIX.sub("", desc, count=1)


def hp_bucket(frac: float) -> str:
    if frac < HP_LOW_MAX:
        return "low"
    if frac < HP_HIGH_MIN:
        return "medium"
    return "high"


@dataclass(frozen=True)
class RiskEvent:
    seed: int
    decision_index: int
    floor: int
    screen: str
    cur_hp: int
    max_hp: int
    hp_frac: float
    hp_bucket: str
    category: str
    choice: str
    risk_seeking: Optional[bool]
    detail: str
    value: Optional[int] = None


def classify_decision(record: dict[str, Any]) -> Optional[RiskEvent]:
    """Map one decision record to a RiskEvent, or None if not risk-relevant."""
    state = record.get("state", {})
    selected = record.get("selected_action", {})
    desc = _normalize_desc(str(selected.get("description", "")))
    screen = _screen(state)

    cur_hp = int(state.get("cur_hp", 0))
    max_hp = int(state.get("max_hp", 0)) or 1
    frac = cur_hp / max_hp
    bucket = hp_bucket(frac)

    def make(category: str, choice: str, risk_seeking: Optional[bool], value: Optional[int] = None) -> RiskEvent:
        return RiskEvent(
            seed=int(record.get("seed", state.get("seed", -1))),
            decision_index=int(record.get("decision_index", -1)),
            floor=int(state.get("floor", -1)),
            screen=screen,
            cur_hp=cur_hp,
            max_hp=max_hp,
            hp_frac=round(frac, 3),
            hp_bucket=bucket,
            category=category,
            choice=choice,
            risk_seeking=risk_seeking,
            detail=desc,
            value=value,
        )

    if screen == "REST_ROOM":
        if desc.startswith("skip"):
            choice = "skip"
        else:
            choice = desc.strip().split(" ", 1)[0] if desc.strip() else "?"
        if choice == "rest":
            risk: Optional[bool] = False
        elif choice == "smith":
            # forgoing a heal is risk-seeking only when HP is not already high
            risk = True if bucket in ("low", "medium") else None
        else:
            risk = None
        return make("campfire", choice, risk)

    if screen == "MAP_SCREEN":
        m = re.search(r"room=(\w+)", desc)
        if m:
            room = m.group(1)
        elif "advance to boss" in desc:
            room = "BOSS"
        else:
            room = "?"
        return make("map_node", room, True if room == "ELITE" else None)

    if screen == "EVENT_SCREEN" and int(state.get("floor", -1)) == 0:
        has_drawback = " / " in desc
        return make("neow", "drawback" if has_drawback else "no_drawback", has_drawback)

    if screen == "REWARDS":
        if desc.startswith("take card"):
            card = desc[len("take card"):].strip()
            return make("card_reward", "take", True if card in SELF_DAMAGE_CARDS else None)
        if desc.startswith("skip card reward") or desc.startswith("skip rewards"):
            return make("card_reward", "skip", None)
        if desc.startswith("take potion"):
            return make("potion", "take", None)
        return None

    if screen == "SHOP_ROOM":
        if desc.startswith("leave shop"):
            return make("shop", "leave", None)
        m = re.search(r"for (\d+)g", desc)
        spend = int(m.group(1)) if m else None
        if desc.startswith("buy potion"):
            kind = "buy_potion"
        elif desc.startswith("buy card remove") or "card remove" in desc:
            kind = "buy_remove"
        elif desc.startswith("buy card"):
            kind = "buy_card"
        elif desc.startswith("buy relic"):
            kind = "buy_relic"
        else:
            kind = "buy"
        return make("shop", kind, None, value=spend)

    return None


def risk_events(records: Iterable[dict[str, Any]]) -> list[RiskEvent]:
    events = []
    for record in records:
        event = classify_decision(record)
        if event is not None:
            events.append(event)
    return events


def _rate(numer: int, denom: int) -> Optional[float]:
    return round(numer / denom, 4) if denom else None


def summarize_risk(events: Iterable[RiskEvent]) -> dict[str, Any]:
    """Aggregate risk events into interpretable, deterministic metrics."""
    events = list(events)

    campfire = [e for e in events if e.category == "campfire"]
    rest_by_bucket: dict[str, Any] = {}
    for bucket in ("low", "medium", "high"):
        sub = [e for e in campfire if e.hp_bucket == bucket]
        rests = sum(1 for e in sub if e.choice == "rest")
        rest_by_bucket[bucket] = {"n": len(sub), "rest_rate": _rate(rests, len(sub))}

    map_nodes = [e for e in events if e.category == "map_node"]
    map_low = [e for e in map_nodes if e.hp_bucket == "low"]
    cards = [e for e in events if e.category == "card_reward"]
    takes = [e for e in cards if e.choice == "take"]
    neow = [e for e in events if e.category == "neow"]
    shop = [e for e in events if e.category == "shop"]
    buys = [e for e in shop if e.choice != "leave"]
    potions = [e for e in events if e.category == "potion"]

    return {
        "total_risk_events": len(events),
        "campfire_rest_rate_by_hp": rest_by_bucket,
        "campfire_smith_count": sum(1 for e in campfire if e.choice == "smith"),
        "map_elite_rate": {
            "n": len(map_nodes),
            "elite_rate": _rate(sum(1 for e in map_nodes if e.choice == "ELITE"), len(map_nodes)),
        },
        "map_elite_rate_low_hp": {
            "n": len(map_low),
            "elite_rate": _rate(sum(1 for e in map_low if e.choice == "ELITE"), len(map_low)),
        },
        "neow_drawback_rate": {
            "n": len(neow),
            "drawback_rate": _rate(sum(1 for e in neow if e.risk_seeking), len(neow)),
        },
        "card_take_rate": {
            "n": len(cards),
            "take_rate": _rate(len(takes), len(cards)),
            "self_damage_takes": sum(1 for e in takes if e.risk_seeking),
        },
        "shop": {
            "n": len(shop),
            "buy_rate": _rate(len(buys), len(shop)),
            "total_spend": sum(e.value or 0 for e in buys),
        },
        "potion_acquire_count": len(potions),
    }


def load_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield decision records from a rollout JSONL file."""
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
