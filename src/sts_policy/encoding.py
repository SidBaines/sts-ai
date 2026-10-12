"""Versioned, public-only numeric and signed-hash features for candidate policies.

Indices in the current legal menu are routing metadata, never model inputs.
Draw piles are encoded as unordered bags, without manufacturing draw order.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import re
from typing import Any, Mapping

import numpy as np

ENCODER_VERSION = "public_candidate_v2"
STATE_DIM = 256
ACTION_DIM = 128
ACTION_FLAGS = ("play", "end turn", "drink", "discard", "exhaust", "select", "take card",
                "skip card", "take relic", "take potion", "take gold", "buy", "leave",
                "rest", "smith", "remove", "upgrade", "transform", "duplicate", "map node",
                "open chest", "skip", "heal", "lose", "gain", "damage", "draw", "strength",
                "vulnerable", "weak", "curse", "power")
ROOMS = ("MONSTER", "ELITE", "REST", "SHOP", "EVENT", "TREASURE", "BOSS")
PLAYER_POWERS = ("Strength", "Dexterity", "Weak", "Vulnerable", "Frail", "Artifact",
                 "Intangible", "Barricade", "Corruption", "Rage", "Metallicize", "Thorns")
_WORDS = re.compile(r"[a-zA-Z][a-zA-Z_+'-]*")
_PUBLIC_LINES = re.compile(r"^(Deck|Relics|Potions|Card select|Selection type|Event):\s*(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class EncodedObservation:
    state_features: np.ndarray
    action_features: np.ndarray
    action_indices: np.ndarray
    coverage: dict[str, int]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=65536)
def _hash(token: str, size: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:7], "little") % size, 1.0 if digest[7] & 1 else -1.0


# Public text keys retain no observations, arrays, weights or private metadata.
# The text length bound also limits each cached ordered term sequence.
_BAG_CACHE_SIZE = 2048
_BAG_CACHE_TEXT_LIMIT = 256
_BAG_CACHE_PREFIX_LIMIT = 64


def _uncached_bag_terms(prefix: str, text: str, size: int) -> tuple[tuple[int, float], ...]:
    words = _WORDS.findall(text)
    return tuple(_hash(prefix + ":" + word, size)
                 for word in [*words, *(a + ":" + b for a, b in zip(words, words[1:]))])


@lru_cache(maxsize=_BAG_CACHE_SIZE)
def _bag_terms(prefix: str, text: str, size: int) -> tuple[tuple[int, float], ...]:
    """Cache ordered immutable hashes of a short, normalized public string."""
    return _uncached_bag_terms(prefix, text, size)


def _bag(target: np.ndarray, prefix: str, text: Any, weight: float = 1.0) -> None:
    """Preserve sequential float32 collision arithmetic and dynamic weights."""
    normalized = str(text).lower()
    terms = (_bag_terms(prefix, normalized, len(target)) if len(normalized) <= _BAG_CACHE_TEXT_LIMIT and len(prefix) <= _BAG_CACHE_PREFIX_LIMIT
             else _uncached_bag_terms(prefix, normalized, len(target)))
    for index, sign in terms:
        target[index] += sign * weight


@lru_cache(maxsize=64)
def _room_label(name: str) -> str:
    name = name.upper().rsplit(".", 1)[-1]
    return {"M": "MONSTER", "E": "ELITE", "R": "REST", "$": "SHOP",
            "?": "EVENT", "T": "TREASURE", "B": "BOSS"}.get(name, name)


def _room(name: Any) -> str:
    return _room_label(str(name))


def _powers(target: np.ndarray, prefix: str, powers: Mapping[str, Any]) -> None:
    for name, amount in powers.items():
        _bag(target, prefix, name, np.clip(_number(amount), -100, 100) / 5.0)


def _text_line(text: str, label: str) -> str | None:
    match = re.search(r"^" + re.escape(label) + r":\s*(.*)$", text, re.MULTILINE)
    return match.group(1) if match else None


def _enemy_target(description: str, enemies: list[dict[str, Any]]) -> dict[str, Any]:
    match = re.search(r"\[enemy (\d+)\]", description)
    if match:
        return next((e for e in enemies if int(e.get("index", -1)) == int(match.group(1))), {})
    if " -> " in description:
        target_name = description.split(" -> ", 1)[1].split(" (", 1)[0].strip()
        matching = [e for e in enemies if str(e.get("name", "")) == target_name]
        if len(matching) == 1:
            return matching[0]
    return {}


@lru_cache(maxsize=32768)
def _card_rule(name: str) -> str:
    from sts_env._backend.glossary_v5 import card_definition
    return card_definition(name, []) or ""


@lru_cache(maxsize=256)
def _potion_rule(name: str) -> str:
    from sts_env._backend.glossary import potion_definition
    return potion_definition(name) or ""


def _definition(description: str) -> tuple[str, str]:
    """Public rules reference already rendered in the environment glossary.

    Cache only card/potion rule names, rather than mutable observation identity
    or entire world-conditioned action descriptions.
    """
    if description.startswith("buy card remove"):
        return "", ""
    match = re.match(r"(?:play |take card |buy card )(.+?)(?: \[| \(cost| for \d+g|$)", description)
    if not match and description.startswith(("select card index", "select card for")) and ":" in description:
        match = re.match(r"(.+?)(?: \[| \(cost|$)", description.split(":", 1)[1].strip())
    if match:
        name = match.group(1)
        return name, _card_rule(name)
    match = re.match(r"(?:drink (?:potion )?|take potion |buy potion )(.+?)(?: -> | for \d+g|$)", description)
    if match:
        name = match.group(1)
        return name, _potion_rule(name)
    return "", ""


def encode_observation(observation: Mapping[str, Any]) -> EncodedObservation:
    """Encode only allowlisted public fields, never arbitrary dict serialization.

    Extra diagnostics, info, seeds, native action bits and unknown structured keys
    are ignored. Text parsing accepts controlled public sections only; appended
    glossary and arbitrary diagnostic lines do not enter the representation.
    """
    state = observation.get("state", {})
    combat = state.get("combat", {}) or {}
    player = combat.get("player", {}) or {}
    cards = combat.get("cards", {}) or {}
    enemies = [e for e in combat.get("enemies", []) if e.get("alive", True)]
    text = str(observation.get("text", ""))
    # Regex follows exactly the former per-label extraction semantics. Retain
    # the first match, matching re.search rather than overwriting duplicates.
    lines = {}
    for match in _PUBLIC_LINES.finditer(text):
        lines.setdefault(match.group(1), match.group(2))
    legal = observation.get("legal_actions", [])
    features = np.zeros(STATE_DIM, dtype=np.float32)
    hashed = features[64:]
    coverage = {"ooc_deck_line": 0, "ooc_relic_line": 0, "ooc_potion_line": 0,
                "unknown_event_effects": 0, "unrecognized_actions": 0,
                "known_action_rules": 0, "unknown_action_rules": 0}
    hp = _number(combat.get("player_cur_hp", state.get("cur_hp")))
    max_hp = max(1.0, _number(combat.get("player_max_hp", state.get("max_hp")), 1))
    gold = _number(state.get("gold"))
    incoming = sum(_number(e.get("intent_damage")) * max(0, _number(e.get("intent_hits"))) for e in enemies)
    numeric = [hp / 100, max_hp / 100, hp / max_hp, gold / 300,
               _number(state.get("act")) / 3, _number(state.get("floor")) / 60,
               _number(state.get("ascension")) / 20, float(observation.get("phase") == "combat"),
               _number(combat.get("turn")) / 10, _number(combat.get("player_energy")) / 3,
               _number(combat.get("player_energy_per_turn")) / 3,
               _number(combat.get("player_block")) / 30, len(enemies) / 3,
               sum(_number(e.get("cur_hp", e.get("hp"))) for e in enemies) / 150,
               incoming / 30, max(0, incoming - _number(combat.get("player_block"))) / max_hp,
               len(legal) / 50]
    for pile in ("hand", "draw", "discard", "exhaust"):
        entries = cards.get(pile, [])
        entries = entries if isinstance(entries, list) else []
        numeric.append(sum(_number(c.get("count", 1)) for c in entries) / 20)
        for card in entries:
            count = _number(card.get("count", 1))
            _bag(hashed, pile, card.get("name", card.get("label", "unknown")), count / 5)
            _bag(hashed, pile + ":type", card.get("type", "unknown"), count / 5)
            _bag(hashed, pile + ":upgraded", card.get("name", ""), count * _number(card.get("upgrade_count")) / 5)
            _bag(hashed, pile + ":cost", card.get("name", ""), count * _number(card.get("cost_for_turn", card.get("cost"))) / 5)
    for key in ("card_draw_per_turn", "cards_played_this_turn", "attacks_played_this_turn",
                "skills_played_this_turn", "cards_discarded_this_turn"):
        numeric.append(_number(player.get(key, cards.get(key))) / 5)
    for power in PLAYER_POWERS:
        numeric.append(_number(player.get("powers", {}).get(power)) / 5)
    _powers(hashed, "player-power", player.get("powers", {}))
    _bag(hashed, "stance", player.get("stance", ""))
    for enemy in enemies:
        _bag(hashed, "enemy", enemy.get("name", ""))
        _bag(hashed, "enemy-hp", enemy.get("name", ""), _number(enemy.get("cur_hp", enemy.get("hp"))) / 100)
        _bag(hashed, "enemy-intent", enemy.get("intent", ""))
        _powers(hashed, "enemy-power", enemy.get("powers", {}))
        for key in ("strength", "vulnerable", "weak", "poison"):
            _bag(hashed, "enemy-status", key, _number(enemy.get(key)) / 5)
    for relic in combat.get("relics", []):
        _bag(hashed, "relic", relic.get("name", ""))
        _bag(hashed, "relic-counter", relic.get("name", ""), _number(relic.get("counter", relic.get("data"))) / 10)
    potions = combat.get("potions", {})
    numeric.extend([_number(potions.get("occupied")) / 3, _number(potions.get("capacity")) / 3])
    for potion in potions.get("slots", []):
        if not potion.get("empty", False):
            _bag(hashed, "potion", potion.get("name", ""))
    for label in ("Deck", "Relics", "Potions"):
        line = lines.get(label)
        if line is not None:
            coverage["ooc_" + {"Deck": "deck", "Relics": "relic", "Potions": "potion"}[label] + "_line"] = 1
            _bag(hashed, "ooc-" + label.lower(), line, 0.2)
            if label == "Relics":
                for name, counter in re.findall(r"([^,:{}]+):(-?\d+)", line):
                    _bag(hashed, "ooc-relic-counter", name.strip(), _number(counter) / 10)
    deck_line = lines.get("Deck") or ""
    deck_match = re.search(r"\((\d+)\)", deck_line)
    numeric.append(int(deck_match.group(1)) / 30 if deck_match else 0)
    _bag(hashed, "screen", state.get("screen_state", ""))
    _bag(hashed, "room", state.get("room", ""))
    _bag(hashed, "combat-input", combat.get("input_state", ""))
    _bag(hashed, "combat-selection", (cards.get("selection", {}) or {}).get("type", ""))
    for label in ("Card select", "Selection type", "Event"):
        _bag(hashed, label, lines.get(label) or "")
    purpose = re.search(r"^Choose a card[^\n]*", text, re.MULTILINE)
    _bag(hashed, "selection-purpose", purpose.group(0) if purpose else "")
    boss = re.search(r"^Act [^\n]*end-of-act boss ([^\n]+)", text, re.MULTILINE)
    _bag(hashed, "act-boss", boss.group(1) if boss else "")
    event = state.get("event", {})
    for key in ("event", "falling_skill_card", "falling_power_card", "falling_attack_card", "relic0", "relic1", "card", "potion"):
        if key in event:
            _bag(hashed, "event:" + key, event[key])
    for key in ("phase", "event_data", "hp_amount0", "hp_amount1", "hp_amount2", "gold", "gold_loss"):
        numeric.append(_number(event.get(key)) / 100)
    graph = state.get("map", {}) or {}
    nodes = graph.get("nodes", [])
    row = _number(graph.get("cur_y"), -1)
    numeric.append(row / 15)
    future_counts = dict.fromkeys(ROOMS, 0)
    for node in nodes:
        if _number(node.get("y")) > row:
            room = _room(node.get("room"))
            if room in future_counts:
                future_counts[room] += 1
    numeric.extend(future_counts[room] / 30 for room in ROOMS)
    for node in nodes:
        if _number(node.get("y")) > row:
            _bag(hashed, "future-map", _room(node.get("room")), 1 / max(1, _number(node.get("y")) - row))
    features[:min(64, len(numeric))] = np.clip(numeric[:64], -10, 10)
    # Only menu semantics enter the state; sum pooling makes menu permutation harmless.
    for action in legal:
        _bag(hashed, "offer", action.get("description", ""), 1 / max(1, len(legal)))
    hashed[:] = np.clip(hashed, -10, 10)
    actions = np.zeros((len(legal), ACTION_DIM), dtype=np.float32)
    for i, action in enumerate(legal):
        description = str(action.get("description", ""))
        lower = description.lower()
        name, definition = _definition(description)
        semantics = (description + " " + definition).lower()
        actions[i, :32] = [float(flag in semantics) for flag in ACTION_FLAGS]
        if name:
            coverage["known_action_rules" if definition else "unknown_action_rules"] += 1
        if not any(actions[i, :32]):
            coverage["unrecognized_actions"] += 1
        if "unavailable" in lower:
            coverage["unknown_event_effects"] += 1
        _bag(actions[i, 64:], "action", description, 0.5)
        _bag(actions[i, 64:], "rules", definition, 0.2)
        # Public multi-selection labels include every selected card. Encoding
        # their rules as a bag permits arbitrary masks without a 2**hand head.
        for selected_name in re.findall(r"hand \[\d+\] (.+?)(?: \[| \(cost|;|$)", description):
            _bag(actions[i, 64:], "selected-rules", _card_rule(selected_name), 0.2)
        rule_lower = definition.lower()
        def rule_amount(pattern: str) -> float:
            match = re.search(pattern, rule_lower)
            return _number(match.group(1)) if match else 0.0
        def amount(pattern: str, default: float = 0.0) -> float:
            match = re.search(pattern, lower)
            return _number(match.group(1)) if match else default
        cost = amount(r"cost (-?\d+)")
        damage = amount(r"deal (\d+)")
        target = _enemy_target(description, enemies)
        target_hp = _number(target.get("cur_hp", target.get("hp")))
        target_block = _number(target.get("block"))
        _powers(actions[i, 64:], "target-power", target.get("powers", {}))
        _bag(actions[i, 64:], "target-intent", target.get("intent", ""), 0.5)
        hand_card = next((c for c in cards.get("hand", []) if c.get("name") == name), {})
        card_type = str(hand_card.get("type", "")).lower()
        from sts_env._backend.glossary_v5 import calculated_block
        computed_block = calculated_block(name, player.get("powers", {})) if name else None
        block = computed_block if computed_block is not None else rule_amount(r"gain (\d+) (?:base )?block")
        price = amount(r"(?:for |pay )(\d+)(?:g| gold)")
        values = [cost / 3, damage / 30, block / 30,
                  price / 300, price / max(1, gold),
                  amount(r"(?:lose|take) (\d+) (?:hp|damage)") / max_hp,
                  amount(r"heal (\d+)") / max_hp, amount(r"(\d+)%") / 100,
                  float("+" in description), float("[attack" in lower or card_type == "attack"), float("[skill" in lower or card_type == "skill"),
                  float("[power" in lower or card_type == "power"), float("[curse" in lower or card_type == "curse"), float("rare" in lower),
                  rule_amount(r"draw (\d+)") / 5, rule_amount(r"gain (\d+) energy") / 3, target_hp / 100,
                  target_block / 30, _number(target.get("intent_damage")) * max(0, _number(target.get("intent_hits"))) / 30,
                  _number(target.get("vulnerable")) / 5, _number(target.get("weak")) / 5,
                  float(bool(target) and damage >= target_hp + target_block),
                  min(damage, target_hp + target_block) / 30,
                  len(re.findall(r"hand \[\d+\]", lower)) / 10,
                  float("none" in lower or "no cards" in lower)]
        node_match = re.search(r"map node x=(\d+)", lower)
        node = next((n for n in nodes if node_match and int(n.get("x", -1)) == int(node_match.group(1))
                     and _number(n.get("y")) == row + 1), {})
        values.extend(float(_room(node.get("room")) == room) for room in ROOMS)
        # Room counts across only this choice's reachable routes supply a
        # topological path feature. Coordinates route the graph traversal but do
        # not become arbitrary position IDs in the policy input.
        if node:
            by_coordinate = {(int(n.get("x", -1)), int(n.get("y", -1))): n for n in nodes}
            pending = [node]
            visited = set()
            while pending:
                reachable = pending.pop()
                coordinate = (int(reachable.get("x", -1)), int(reachable.get("y", -1)))
                if coordinate in visited:
                    continue
                visited.add(coordinate)
                distance = max(1, coordinate[1] - row)
                band = "next" if distance == 1 else "near" if distance <= 4 else "far"
                _bag(actions[i, 64:], "route:" + band, _room(reachable.get("room")), 1 / distance)
                for destination_x in reachable.get("edges", []):
                    destination = by_coordinate.get((int(destination_x), coordinate[1] + 1))
                    if destination is not None:
                        pending.append(destination)
        actions[i, 32:64] = np.clip(values, -10, 10)
    return EncodedObservation(features, actions, np.asarray([int(a["index"]) for a in legal], dtype=np.int64), coverage)


def collate_observations(observations: list[EncodedObservation], device: Any = "cpu"):
    """Return torch state/actions/mask tensors; retain at least one padded slot."""
    import torch
    if not observations:
        raise ValueError("Cannot collate an empty batch")
    width = max(1, max(len(o.action_indices) for o in observations))
    state = np.stack([o.state_features for o in observations])
    actions = np.zeros((len(observations), width, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((len(observations), width), dtype=np.bool_)
    for i, observation in enumerate(observations):
        size = len(observation.action_indices)
        actions[i, :size] = observation.action_features
        mask[i, :size] = True
    return (torch.as_tensor(state, device=device), torch.as_tensor(actions, device=device),
            torch.as_tensor(mask, device=device))
