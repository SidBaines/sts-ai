"""Typed public entities and candidate relations for the v3 residual learner.

Indices only join a source/target to its public entity. They never become neural
features. Unordered piles remain bags; no entity or legal-candidate caps apply.
The unchanged legacy projection supplies an explicitly versioned behavior prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .encoding import encode_observation as encode_legacy, _room, _card_rule
from .legacy_bridge import project_legacy_menu

ENCODER_VERSION = "public_entities_v3"
VOCABULARY_PATH = Path(__file__).with_name("public_vocabulary_v3.json")
VOCABULARY_SHA256 = hashlib.sha256(VOCABULARY_PATH.read_bytes()).hexdigest()
_VOCABULARY = json.loads(VOCABULARY_PATH.read_text())
TOKENS = tuple(_VOCABULARY["tokens"])
TOKEN_IDS = {name: i for i, name in enumerate(TOKENS)}
STATUS_NAMES = tuple(_VOCABULARY["status_names"])
STATUS_IDS = {name: i for i, name in enumerate(STATUS_NAMES)}
ENTITY_DIM = 32 + len(STATUS_NAMES)
GLOBAL_DIM = 48
ACTION_DIM = 96
ROOMS = ("MONSTER", "ELITE", "REST", "SHOP", "EVENT", "TREASURE", "BOSS")
_TIMER = re.compile(r"\s*\((\d+) turns?\)\s*$", re.I)


def canonical(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


@lru_cache(maxsize=8192)
def token(namespace: str, name: str) -> int:
    return TOKEN_IDS.get(namespace + ":" + canonical(name), 1) if name else 0


def number(value: Any, default: float = 0.) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def available(row: Mapping[str, Any], name: str) -> float:
    return float(name in row and row[name] is not None)


def card_values(card: Mapping[str, Any]) -> np.ndarray:
    values = np.zeros(32, np.float32)
    cost = number(card.get("cost"))
    current = number(card.get("cost_for_turn", card.get("cost")))
    values[:20] = [cost / 3, current / 3, number(card.get("upgrade_count")) / 5,
        number(card.get("special_data")) / 30, available(card, "special_data"),
        float(card.get("free_to_play_once", False)), float(card.get("retain", False)),
        float(card.get("self_retain", False)), float(card.get("ethereal", False)),
        float(card.get("exhaust", False)), float(card.get("bottled", False)),
        float(card.get("x_cost", current == -1)), float(card.get("requires_target", False)),
        available(card, "cost"), available(card, "cost_for_turn"),
        available(card, "upgrade_count"), available(card, "free_to_play_once"),
        available(card, "retain"), available(card, "exhaust"),
        number(card.get("count", 1)) / 20]
    return values


def card_ids(card: Mapping[str, Any], role: str) -> tuple[int, ...]:
    name = str(card.get("base_name", re.sub(r"\+\d*$", "", str(card.get("name", "")))))
    return (token("role", role), token("card", name),
            token("rarity", str(card.get("rarity", ""))), token("type", str(card.get("type", ""))))


def powers_into(values: np.ndarray, powers: Mapping[str, Any]) -> int:
    unknown = 0
    for name, amount in powers.items():
        key = canonical(_TIMER.sub("", str(name)))
        index = STATUS_IDS.get(key)
        if index is None:
            unknown += 1
            values[30] += number(amount) / 5
        else:
            values[32 + index] += number(amount) / 5
    values[31] = unknown / 5
    return unknown


def enemy_values(enemy: Mapping[str, Any]) -> np.ndarray:
    values = np.zeros(ENTITY_DIM, np.float32)
    hp, maximum = number(enemy.get("cur_hp", enemy.get("hp"))), number(enemy.get("max_hp"))
    values[:14] = [hp / 100, maximum / 100, hp / max(1, maximum),
        number(enemy.get("block")) / 30, number(enemy.get("intent_damage")) / 30,
        number(enemy.get("intent_hits")) / 5, float(enemy.get("alive", True)),
        available(enemy, "max_hp"), available(enemy, "block"), available(enemy, "intent_damage"),
        available(enemy, "intent_hits"), number(enemy.get("strength")) / 5,
        number(enemy.get("weak")) / 5, number(enemy.get("vulnerable")) / 5]
    powers_into(values, enemy.get("powers", {}))
    return values


def route_features(graph: Mapping[str, Any]) -> dict[tuple[int, int], np.ndarray]:
    """Exact feasible-path extrema and nearest rooms over the visible act DAG.

    Counts include the chosen node. A leaf is the last visible node; the implicit
    act boss is not invented. A missing continuation is reported in slot 21.
    Coordinates perform joins only. Sum of route counts differs from the former
    reachable-set histogram when all paths must pass through an elite.
    """
    nodes = {(int(node["x"]), int(node["y"])): node for node in graph.get("nodes", [])}
    result: dict[tuple[int, int], np.ndarray] = {}
    for coordinate in sorted(nodes, key=lambda c: (-c[1], c[0])):
        node = nodes[coordinate]
        children = [result[(int(x), coordinate[1] + 1)] for x in node.get("edges", [])
                    if (int(x), coordinate[1] + 1) in result]
        value = np.zeros(22, np.float32)
        here = np.asarray([float(_room(node.get("room")) == room) for room in ROOMS])
        value[:7] = here + (np.min([child[:7] for child in children], axis=0) if children else 0)
        value[7:14] = here + (np.max([child[7:14] for child in children], axis=0) if children else 0)
        # Minimum distance to rest/shop/elite; 16 explicitly means absent.
        for i, room in enumerate(("REST", "SHOP", "ELITE")):
            value[14 + i] = (0 if _room(node.get("room")) == room else
                             min(16, 1 + min(child[14 + i] for child in children)) if children else 16)
        paths = sum(math.expm1(float(child[17])) for child in children) if children else 1
        value[17] = math.log1p(paths)
        value[18] = 1 + (min(child[18] for child in children) if children else 0)
        value[19] = 1 + (max(child[19] for child in children) if children else 0)
        value[20] = len(children)
        value[21] = float(len(children) != len(node.get("edges", [])))
        result[coordinate] = value
    return result


def _effects(description: str, max_hp: float) -> np.ndarray:
    """Conservative amounts from displayed effects; keep availability flags.

    These supplement typed mechanics while event effects are still prose. An
    event-option/menu/card-instance index is never parsed as an effect amount.
    Complex descriptions remain available to the explicit legacy prior. This
    parser does not claim to compute complete action outcomes or expected value.
    """
    text = description.lower()
    patterns = ((r"(?:gain|obtain|receive) (\d+) gold", 300),
                (r"(?:pay|lose|give up) (\d+) gold", 300),
                (r"(?:lose|take) (\d+) (?:hp|damage)", max_hp),
                (r"heal (\d+) (?:hp|health)", max_hp),
                (r"gain (\d+) (?:maximum|max) hp", max_hp),
                (r"lose (\d+) (?:maximum|max) hp", max_hp))
    values = []
    for pattern, scale in patterns:
        matches = [float(match) for match in re.findall(pattern, text)]
        values.extend((sum(matches) / max(1, scale), float(bool(matches))))
    return np.asarray(values, np.float32)


@dataclass(frozen=True)
class EncodedEntities:
    legacy_state: np.ndarray
    legacy_actions: np.ndarray
    legacy_representatives: np.ndarray
    global_values: np.ndarray
    entity_ids: np.ndarray
    entity_values: np.ndarray
    entity_weights: np.ndarray
    action_ids: np.ndarray
    action_values: np.ndarray
    # [candidate, source/target, entity], routing memberships only.
    action_links: np.ndarray
    action_indices: np.ndarray
    coverage: dict[str, int]


def encode_observation(observation: Mapping[str, Any]) -> EncodedEntities:
    state = observation.get("state", {})
    if state.get("observation_version") != "sts_env_public_v2":
        raise ValueError("v3 encoder requires typed sts_env_public_v2 observations")
    if not isinstance(state.get("inventory"), dict):
        raise ValueError("typed public inventory is missing")
    legal = observation.get("legal_actions", [])
    if any(not isinstance(action.get("public"), dict) for action in legal):
        raise ValueError("typed legal action metadata is missing")
    projection = project_legacy_menu(dict(observation))
    legacy = encode_legacy(projection.observation)
    combat = state.get("combat", {}) or {}
    inventory = state["inventory"]
    player = combat.get("player", {}) or {}
    cards = combat.get("cards", {}) or {}
    entities, values, weights = [], [], []
    sources: dict[tuple[str, int], int] = {}
    targets: dict[int, int] = {}
    coverage = {"unknown_entity_identity": 0, "unknown_status": 0,
                "unresolved_source": 0, "unresolved_target": 0,
                "unknown_action_kind": 0, "unknown_damage_targets": 0}

    def add(ids: tuple[int, ...], row: np.ndarray, weight: float = 1.) -> int:
        entities.append(ids)
        padded = np.zeros(ENTITY_DIM, np.float32)
        padded[:len(row)] = row
        values.append(padded)
        weights.append(weight)
        coverage["unknown_entity_identity"] += int(ids[1] == 1)
        return len(entities) - 1

    for role, pile in [("deck", inventory.get("deck", [])), *[(name, cards.get(name, []))
                       for name in ("hand", "draw", "discard", "exhaust")]]:
        for card in pile:
            entity = add(card_ids(card, role), card_values(card), number(card.get("count", 1)))
            if role == "hand" and "index" in card:
                sources[(role, int(card["index"]))] = entity
    for enemy in combat.get("enemies", []):
        if not enemy.get("alive", True):
            continue
        row = enemy_values(enemy)
        coverage["unknown_status"] += round(float(row[31]) * 5)
        entity = add((token("role", "enemy"), token("enemy", str(enemy.get("name", ""))),
                      token("intent", str(enemy.get("intent", ""))), 0), row)
        targets[int(enemy["index"])] = entity
    for name, amount in player.get("powers", {}).items():
        timer = _TIMER.search(str(name))
        base = _TIMER.sub("", str(name))
        row = np.zeros(32, np.float32)
        row[:4] = [number(amount) / 5, int(timer.group(1)) / 3 if timer else 0,
                   float(timer is not None), 1]
        add((token("role", "player"), token("status", base), 0, 0), row)
    for relic in inventory.get("relics", []):
        row = np.zeros(32, np.float32)
        row[:3] = [number(relic.get("counter")) / 10, available(relic, "counter"),
                   float(relic.get("counter_is_live", False))]
        add((token("role", "relic"), token("relic", str(relic.get("name", ""))), 0, 0), row)
    potions = inventory.get("potions", {})
    for potion in potions.get("slots", []):
        if potion.get("empty", False):
            continue
        entity = add((token("role", "potion"), token("potion", str(potion.get("name", ""))), 0, 0),
                     np.zeros(32, np.float32))
        sources[("potion", int(potion["index"]))] = entity
    boss = str(inventory.get("boss", ""))
    if boss:
        add((token("role", "boss"), token("boss", boss), 0, 0), np.zeros(32, np.float32))
    event = state.get("event", {}) or {}
    if event.get("event"):
        row = np.zeros(32, np.float32)
        for i, key in enumerate(("phase", "event_data", "hp_amount0", "hp_amount1", "hp_amount2", "gold", "gold_loss")):
            row[2*i:2*i+2] = [number(event.get(key)) / 100, available(event, key)]
        add((token("role", "event"), token("event", str(event["event"])), 0, 0), row)
    routes = route_features(state.get("map", {}) or {})
    # Future map nodes preserve room/path associations; absolute x is absent.
    cur_y = int(state.get("map", {}).get("cur_y", -1))
    node_by_coordinate = {(int(n["x"]), int(n["y"])): n for n in state.get("map", {}).get("nodes", [])}
    for coordinate, route in routes.items():
        if coordinate[1] > cur_y:
            row = np.zeros(32, np.float32)
            row[:22] = route / 15
            row[22] = (coordinate[1] - cur_y) / 15
            add((token("role", "map"), token("room", _room(node_by_coordinate[coordinate].get("room"))), 0, 0), row)
    count = len(legal)
    action_ids = np.zeros((count, 4), np.int64)
    action_values = np.zeros((count, ACTION_DIM), np.float32)
    links = np.zeros((count, 2, len(entities)), np.float32)
    hp = number(combat.get("player_cur_hp", state.get("cur_hp")))
    maximum = max(1, number(combat.get("player_max_hp", state.get("max_hp")), 80))
    for i, action in enumerate(legal):
        public = action["public"]
        card = public.get("card", {})
        action_ids[i] = (token("kind", str(public.get("kind", ""))),
            card_ids(card, "offer")[1] if card else token("potion" if "potion" in public else "relic",
                                                        str(public.get("potion", public.get("relic", "")))),
            token("rarity", str(card.get("rarity", ""))), token("type", str(card.get("type", ""))))
        coverage["unknown_action_kind"] += int(action_ids[i, 0] == 1)
        action_values[i, :32] = card_values(card) if card else 0
        action_values[i, 32:37] = [number(public.get("energy_cost")) / 3,
            available(public, "energy_cost"), float(public.get("free_to_play", False)),
            number(public.get("x_energy")) / 3, float(public.get("targets_all", False))]
        typed_targets = public.get("targets", [])
        known = [t for t in typed_targets if t.get("damage_known", False)]
        action_values[i, 37:43] = [len(typed_targets) / 3, len(known) / max(1, len(typed_targets)),
            sum(number(t.get("damage_per_hit")) for t in known) / 30,
            sum(number(t.get("hits")) for t in known) / 5,
            sum(number(t.get("damage_total")) for t in known) / 100, float(bool(typed_targets))]
        coverage["unknown_damage_targets"] += len(typed_targets) - len(known)
        target_indices = [int(t["index"]) for t in typed_targets]
        if "target_index" in public and int(public["target_index"]) not in target_indices:
            target_indices.append(int(public["target_index"]))
        for target in target_indices:
            if target in targets:
                links[i, 1, targets[target]] = 1
            else:
                coverage["unresolved_target"] += 1
        if "source_index" in public:
            source = (str(public.get("source_kind", "")), int(public["source_index"]))
            if source in sources:
                links[i, 0, sources[source]] = 1
            else:
                coverage["unresolved_source"] += 1
        for selected in public.get("selected_cards", []):
            source = ("hand", int(selected["index"]))
            if source not in sources:
                raise ValueError("selected public card is absent from the hand")
            links[i, 0, sources[source]] = 1
        action_values[i, 43:48] = [number(public.get("price")) / 300, available(public, "price"),
            number(public.get("gold_gain")) / 300, available(public, "gold_gain"),
            float("unavailable" in str(action["description"]).lower())]
        action_values[i, 48:60] = _effects(str(action["description"]), maximum)
        action_values[i, 60:64] = [number(public.get("heal")) / maximum,
            available(public, "heal"), links[i, 0].sum() / 10, links[i, 1].sum() / 3]
        if "map_x" in public and "map_y" in public:
            coordinate = (int(public["map_x"]), int(public["map_y"]))
            if coordinate not in routes:
                raise ValueError("offered public map node is absent from the visible DAG")
            action_values[i, 64:86] = routes[coordinate] / 15
            action_values[i, 86] = 1
        name = str(card.get("name", ""))
        rule = _card_rule(name) if name else ""
        action_values[i, 87:90] = [float(bool(rule)), float(bool(name) and not rule), float(bool(card))]
    global_values = np.zeros(GLOBAL_DIM, np.float32)
    global_values[:17] = [float(observation.get("phase") == "combat"), number(state.get("act")) / 3,
        number(state.get("floor")) / 60, number(state.get("ascension")) / 20, hp / 100,
        maximum / 100, hp / maximum, number(state.get("gold")) / 300,
        number(combat.get("player_energy")) / 3, number(combat.get("player_block")) / 30,
        number(combat.get("turn")) / 10, number(combat.get("player_energy_per_turn")) / 3,
        number(potions.get("occupied")) / 3, number(potions.get("capacity")) / 3,
        len(legal) / 50, len(entities) / 100, float(bool(legal) and len(legal) > 1)]
    for i, role in enumerate(("deck", "hand", "draw", "discard", "exhaust")):
        pile = inventory.get("deck", []) if role == "deck" else cards.get(role, [])
        global_values[17 + i] = sum(number(card.get("count", 1)) for card in pile) / 20
    for i, key in enumerate(("cards_played_this_turn", "attacks_played_this_turn", "skills_played_this_turn",
                             "cards_discarded_this_turn", "card_draw_per_turn")):
        global_values[22 + i] = number(player.get(key)) / 5
    first_positions = {}
    for position, legacy_position in enumerate(projection.legacy_positions):
        first_positions.setdefault(legacy_position, position)
    representatives = np.asarray([first_positions[p] for p in projection.legacy_positions], np.int64)
    return EncodedEntities(legacy.state_features, legacy.action_features[list(projection.legacy_positions)], representatives,
        global_values, np.asarray(entities, np.int64).reshape(-1, 4),
        np.asarray(values, np.float32).reshape(-1, ENTITY_DIM), np.asarray(weights, np.float32),
        action_ids, action_values, links, np.asarray([int(a["index"]) for a in legal], np.int64), coverage)


def collate_observations(observations: Sequence[EncodedEntities], device: Any = "cpu"):
    """Padded tensors plus masks; retain every candidate/entity in the batch."""
    import torch
    if not observations:
        raise ValueError("Cannot collate an empty batch")
    b, n, e = len(observations), max(1, max(len(o.action_indices) for o in observations)), max(1, max(len(o.entity_ids) for o in observations))
    actions = np.zeros((b, n, 128), np.float32)
    mask = np.zeros((b, n), np.bool_)
    entity_ids = np.zeros((b, e, 4), np.int64)
    entity_values = np.zeros((b, e, ENTITY_DIM), np.float32)
    weights = np.zeros((b, e), np.float32)
    action_ids = np.zeros((b, n, 4), np.int64)
    action_values = np.zeros((b, n, ACTION_DIM), np.float32)
    links = np.zeros((b, n, 2, e), np.float32)
    representatives = np.zeros((b, n), np.int64)
    prior_mask = np.zeros((b, n), np.bool_)
    for i, obs in enumerate(observations):
        size, entities = len(obs.action_indices), len(obs.entity_ids)
        actions[i, :size], mask[i, :size] = obs.legacy_actions, True
        representatives[i, :size] = obs.legacy_representatives
        prior_mask[i, :size] = obs.legacy_representatives == np.arange(size)
        entity_ids[i, :entities], entity_values[i, :entities], weights[i, :entities] = obs.entity_ids, obs.entity_values, obs.entity_weights
        action_ids[i, :size], action_values[i, :size], links[i, :size, :, :entities] = obs.action_ids, obs.action_values, obs.action_links
    arrays = (np.stack([o.legacy_state for o in observations]), actions, mask,
              np.stack([o.global_values for o in observations]), entity_ids, entity_values,
              weights, action_ids, action_values, links, representatives, prior_mask)
    return tuple(torch.as_tensor(array, device=device) for array in arrays)


def collate_public(observations: Sequence[dict[str, Any]], device: Any = "cpu"):
    return collate_observations([encode_observation(observation) for observation in observations], device)
