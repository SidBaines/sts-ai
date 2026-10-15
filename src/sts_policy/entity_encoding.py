"""B1 public entities. See docs/fresh_start_entity_design.md for the tensor ABI.

Only explicitly selected public fields are read. Menu indices are routing metadata,
never neural IDs. No native imports, seeds, bits, diagnostics or search values.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

ENCODER_VERSION = 'public_entities_fresh_v1'
# Exact installed source identity, including semantic entity/action index tables.
ENCODER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_VOCAB_BYTES = Path(__file__).with_name('entity_vocabulary.json').read_bytes()
VOCABULARY_SHA256 = hashlib.sha256(_VOCAB_BYTES).hexdigest()
VOCABULARY = json.loads(_VOCAB_BYTES)
TOKEN_TO_ID = {name: i for i, name in enumerate(VOCABULARY['tokens'])}
VOCAB_SIZE = len(TOKEN_TO_ID)
PAD, UNK = 0, 1
NUMERIC_DIM = TEXT_DIM = 32
ENTITY_TYPES = {name: i + 1 for i, name in enumerate((
    'run', 'player', 'enemy', 'hand', 'draw', 'discard', 'exhaust', 'deck',
    'relic', 'potion', 'map', 'context', 'offer'))}
ACTION_TYPES = {name: i + 1 for i, name in enumerate((
    'unknown play end_turn select_card select_many drink_potion discard_potion '
    'take_card take_relic take_potion take_gold take_key skip_card skip_rewards '
    'skip_boss_relic take_boss_relic buy_card buy_relic buy_potion remove_card '
    'leave_shop map rest smith recall lift toke dig skip_campfire open_chest '
    'skip_chest event singing_bowl card_reward card_remove').split())}


def canonical(value: Any) -> str:
    text = re.sub(r'^(?:ScreenState|Room|Stance|CardType|InputState|Phase)\.', '', str(value), flags=re.I)
    return re.sub(r'[^a-z0-9]', '', text.lower())


def vocabulary_id(namespace: str, name: Any) -> int:
    if namespace == 'card':
        name = _card_name(name)[0]
    key = namespace + ':' + canonical(name)
    return TOKEN_TO_ID.get(VOCABULARY['aliases'].get(key, key), UNK)


@dataclass(frozen=True)
class EncodedEntityObservation:
    entity_ids: np.ndarray
    entity_types: np.ndarray
    entity_aux: np.ndarray
    entity_numeric: np.ndarray
    entity_mask: np.ndarray
    power_owner: np.ndarray
    power_ids: np.ndarray
    power_amounts: np.ndarray
    power_amount_available: np.ndarray
    power_mask: np.ndarray
    recent_ids: np.ndarray
    recent_types: np.ndarray
    recent_numeric: np.ndarray
    recent_mask: np.ndarray
    action_ids: np.ndarray
    action_types: np.ndarray
    action_aux: np.ndarray
    action_numeric: np.ndarray
    action_text: np.ndarray
    action_source: np.ndarray
    action_target: np.ndarray
    action_source_members: np.ndarray
    action_source_mask: np.ndarray
    action_mask: np.ndarray
    action_indices: np.ndarray
    coverage: dict


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _numeric(pairs: list[tuple[Any, float]]) -> np.ndarray:
    """16 clipped values + 16 availability flags; nonfinite is unavailable."""
    if len(pairs) > 16:
        raise ValueError('numeric schema exceeds 16 fields')
    result = np.zeros(32, np.float32)
    for i, (value, scale) in enumerate(pairs):
        if _finite(value):
            result[i] = np.clip(float(value) / scale, -10, 10)
            result[i + 16] = 1
    return result


def _card_name(name: Any) -> tuple[str, int, int | None]:
    """Native publicCardName/Card output: name, optional =amount, then +upgrade."""
    text = str(name).strip()
    match = re.fullmatch(r'(.*?)(?:=(-?\d+))?(?:\+(\d*))?', text)
    assert match
    return (match[1].strip(), int(match[3] or 1) if match[3] is not None else 0,
            int(match[2]) if match[2] is not None else None)


def _normalized_cost(value: Any) -> Any:
    # publicCardDict keeps curses=-3/statuses=-2, but labels render both as
    # unplayable. Card type stays separate; X=-1 must remain distinct.
    return -2 if _finite(value) and float(value) < -1 else value


def _card_text(label: str, *, base_cost_visible: bool = False) -> dict:
    name = re.split(r' \[| \(cost', label, maxsplit=1)[0].strip()
    base, upgrade, _ = _card_name(name)
    if not base:
        raise ValueError(f'Malformed card label: {label!r}')
    card: dict[str, Any] = {'name': name, 'upgrade_count': upgrade}
    tag = re.search(r'\[([^]]+)\]', label)
    if tag:
        parts = [p.strip() for p in tag[1].split(',')]
        card['type'] = parts[0]
        if len(parts) > 1:
            card['rarity'] = parts[1]
    cost = re.search(r'\(cost (X|unplayable|-?\d+)([^)]*)\)', label)
    if ' (cost' in label and not cost:
        raise ValueError(f'Malformed public card cost: {label!r}')
    if cost:
        parse = lambda x: _normalized_cost({'X': -1, 'unplayable': -2}.get(x, int(x) if x.lstrip('-').isdigit() else -2))
        card['cost_for_turn'] = parse(cost[1])
        base_cost = re.search(r', base (X|unplayable|-?\d+)', cost[2])
        if base_cost:
            card['cost'] = parse(base_cost[1])
        elif base_cost_visible:
            # publicCardLabel(includeBase=true) prints a base only when it
            # differs. Play/hand text omits it even when it differs.
            card['cost'] = card['cost_for_turn']
        card['free_to_play_once'] = 'free once' in cost[2]
        card['retain'] = 'retained' in cost[2]
    return card


def _card_key(card: Mapping) -> tuple:
    name, upgrade, amount = _card_name(card.get('name', ''))
    return (canonical(name), card.get('upgrade_count', upgrade), amount,
            *(_normalized_cost(card.get(k)) for k in ('cost', 'cost_for_turn')),
            *(card.get(k) for k in ('free_to_play_once', 'retain', 'type', 'rarity')))


def _card_numeric(card: Mapping, count: Any = 1, *, slot: Any = None) -> np.ndarray:
    _, upgrade, amount = _card_name(card.get('name', ''))
    return _numeric([(count, 100), (card.get('upgrade_count', upgrade), 10),
                     (_normalized_cost(card.get('cost')), 5), (_normalized_cost(card.get('cost_for_turn')), 5),
                     (card.get('free_to_play_once'), 1), (card.get('retain'), 1), (slot, 10),
                     (None, 1), (None, 1), (None, 1), (None, 1), (amount, 100)])


def _inventory_line(text: str, label: str) -> str | None:
    match = re.search(r'^' + label + r':[^\S\n]*(.*)$', text, re.M)
    return match[1].strip() if match else None


def _braced_inventory(line: str, label: str) -> tuple[list[str], int | None]:
    if line.lower() in {'none', 'empty', ''}:
        return [], 0
    match = re.fullmatch(r'(?:\((\d+)\):\s*)?\{([^{}]*)\}', line)
    if not match:
        raise ValueError(f'Malformed public {label} inventory: {line!r}')
    items = [v.strip() for v in match[2].split(',')]
    if items and not items[-1]:
        items.pop()
    if any(not x for x in items):
        raise ValueError(f'Malformed public {label} inventory separator')
    declared = int(match[1]) if match[1] is not None else None
    if declared is not None and declared != len(items):
        raise ValueError(f'Public {label} count mismatch: {declared} vs {len(items)}')
    return items, declared


def _hash_bag(text: str) -> np.ndarray:
    """Action wording only; stable signed BLAKE2 word/bigram bag, L1 normalized."""
    words = re.findall(r'[a-z]+', text.lower())
    result = np.zeros(TEXT_DIM, np.float32)
    terms = words + [a + ':' + b for a, b in zip(words, words[1:])]
    for term in terms:
        digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
        result[int.from_bytes(digest[:7], 'little') % TEXT_DIM] += 1 if digest[7] & 1 else -1
    return result / max(len(terms), 1)


class _Encoder:
    def __init__(self, obs: Mapping):
        self.obs = obs
        self.state = obs.get('state', {})
        self.combat = self.state.get('combat') or {}
        self.text = str(obs.get('text', ''))
        self.menu = obs.get('legal_actions', [])
        if not self.menu:
            raise ValueError('public entity encoder requires a nonempty legal menu')
        self.coverage: dict[str, Any] = {'unk_by_namespace': {}, 'lookups_by_namespace': {},
            'parser_fallbacks': 0, 'ambiguous_sources': 0, 'unresolved_sources': 0,
            'unresolved_targets': 0, 'recent_total': 0, 'recent_truncated': 0,
            'recent_unparsed': 0}
        self.entities: list[tuple] = []
        self.powers: list[tuple] = []
        self.cards: dict[str, list[tuple[int, Mapping]]] = {k: [] for k in ('hand', 'draw', 'discard', 'exhaust', 'deck')}
        self.enemies: dict[int, tuple[int, Mapping]] = {}
        self.potions: list[tuple[int, Mapping]] = []
        self.destinations: dict[str, int] = {}
        self.selection_purpose: str | None = None

    def token(self, ns: str, name: Any) -> int:
        if name is None or name == '':
            return PAD
        result = vocabulary_id(ns, name)
        lookups = self.coverage['lookups_by_namespace']
        lookups[ns] = lookups.get(ns, 0) + 1
        if result == UNK:
            unknown = self.coverage['unk_by_namespace']
            unknown[ns] = unknown.get(ns, 0) + 1
        return result

    def add(self, kind: str, ns: str, name: Any, numeric=None, aux=()) -> int:
        index = len(self.entities)
        self.entities.append((self.token(ns, name), ENTITY_TYPES[kind],
                              [*aux, *([PAD] * (3 - len(aux)))],
                              numeric if numeric is not None else _numeric([])))
        return index

    def add_powers(self, owner: int, data: Mapping) -> None:
        # No iteration over surrounding dictionaries; this dict is a public name->amount surface.
        for name, amount in sorted(data.items()):
            available = _finite(amount)
            value = float(np.clip(float(amount) / 100, -10, 10)) if available else 0.0
            self.powers.append((owner, self.token('power', name), value, available))
            if not available:
                self.coverage['parser_fallbacks'] += 1

    def add_cards(self, pile: str, cards: list[Mapping]) -> None:
        if pile == 'hand':
            ordered = sorted(cards, key=lambda c: int(c['index']))
            if len({c['index'] for c in ordered}) != len(ordered):
                raise ValueError('duplicate public hand slot')
        else:
            groups: dict[tuple, dict] = {}
            for card in cards:
                key = _card_key(card)
                count = card.get('count', 1)
                if not _finite(count) or int(count) != count or count < 1:
                    raise ValueError(f'invalid {pile} count: {count!r}')
                if key not in groups:
                    # Whitelist card fields; metadata cannot enter sort/group keys.
                    groups[key] = {k: card[k] for k in ('name', 'upgrade_count', 'cost', 'cost_for_turn',
                        'free_to_play_once', 'retain', 'type', 'rarity') if k in card}
                    groups[key]['count'] = 0
                groups[key]['count'] += count
            ordered = [groups[k] for k in sorted(groups, key=lambda k: tuple('' if v is None else str(v) for v in k))]
        self.coverage[pile + '_count'] = sum(c.get('count', 1) for c in ordered)
        self.add(pile if pile != 'hand' else 'context', 'role', pile,
                 _numeric([(self.coverage[pile + '_count'], 100), (len(ordered), 100)]))
        for card in ordered:
            base, _, _ = _card_name(card.get('name', ''))
            idx = self.add(pile, 'card', base,
                           _card_numeric(card, card.get('count', 1), slot=card.get('index') if pile == 'hand' else None),
                           [self.token('card_type', card.get('type')), self.token('rarity', card.get('rarity'))])
            self.cards[pile].append((idx, card))

    def inventories(self) -> None:
        cards = self.combat.get('cards', {})
        for pile in ('hand', 'draw', 'discard', 'exhaust'):
            self.coverage[pile + '_available'] = int(pile in cards)
            if pile in cards:
                self.add_cards(pile, cards[pile])
        deck = _inventory_line(self.text, 'Deck')
        self.coverage['deck_available'] = int(deck is not None)
        if deck is not None:
            items, _ = _braced_inventory(deck, 'Deck')
            self.add_cards('deck', [_card_text(x) for x in items])
        if 'relics' in self.combat:
            relics = self.combat['relics']
            self.coverage['relics_available'] = 1
        else:
            line = _inventory_line(self.text, 'Relics')
            self.coverage['relics_available'] = int(line is not None)
            relics = []
            if line is not None:
                items, _ = _braced_inventory(line, 'Relics')
                for item in items:
                    match = re.fullmatch(r'(.+):\s*(-?\d+)', item)
                    if not match:
                        raise ValueError(f'Malformed public Relics entry: {item!r}')
                    relics.append({'name': match[1].strip(), 'counter': int(match[2])})
        for relic in sorted(relics, key=lambda x: (canonical(x.get('name', '')), str(x.get('counter')))):
            self.add('relic', 'relic', relic.get('name'),
                     _numeric([(relic.get('counter'), 100), (relic.get('counter_is_live'), 1)]))
        if 'potions' in self.combat:
            potions = self.combat['potions']
            slots = potions.get('slots', [])
            self.coverage['potions_available'] = 1
            self.coverage['potion_capacity'] = potions.get('capacity')
            if 'capacity' in potions and potions['capacity'] != len(slots):
                raise ValueError('public potion capacity/slot count mismatch')
            slots = [{**s, 'slot_known': True} for s in slots]
        else:
            line = _inventory_line(self.text, 'Potions')
            self.coverage['potion_capacity'] = None
            slots = self.ooc_potion_slots(line)
            self.coverage['potions_available'] = int(line is not None or
                'adapter_public_ooc_potions' in self.obs or bool(slots))
        known_slots = [s['index'] for s in slots if s.get('slot_known')]
        if len(set(known_slots)) != len(known_slots):
            raise ValueError('duplicate public potion slot')
        for potion in sorted(slots, key=lambda x: (x.get('index') is None, x.get('index') or 0, canonical(x.get('name')))):
            idx = self.add('potion', 'potion', 'empty' if potion.get('empty') else potion.get('name'),
                           _numeric([(potion.get('index'), 10), (potion.get('empty'), 1)]))
            self.potions.append((idx, potion))
        self.coverage['potion_count'] = sum(not s.get('empty', False) for s in slots)
        self.coverage['potion_slots_unresolved'] = sum(not s.get('slot_known') for s in slots)

    def ooc_potion_slots(self, line: str | None) -> list[dict]:
        # describeGameState prints positions [0, potionCount), NOT the occupied
        # subsequence or capacity. A hole may hide an occupied higher slot.
        items = [] if line is None or line.lower() in {'none', 'empty', ''} else line.split(',')
        slots: dict[int, dict] = {}
        for slot, item in enumerate(items):
            name = item.strip()
            if not re.fullmatch(r'[^{}\[\]:]+', name):
                raise ValueError(f'Malformed public Potions entry: {item!r}')
            empty = canonical(name) == 'emptypotionslot'
            slots[slot] = {'name': 'empty' if empty else name, 'index': slot,
                           'empty': empty, 'slot_known': True}

        public_slots = []
        if 'adapter_public_ooc_potions' in self.obs:
            # Adapter-only allowlist, extracted from the ORIGINAL public menu
            # before choices are filtered. Ignore every other field/metadata.
            records = self.obs['adapter_public_ooc_potions']
            if not isinstance(records, list):
                raise ValueError('Malformed adapter_public_ooc_potions list')
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError('Malformed public potion slot record')
                public_slots.append((record.get('slot'), record.get('name')))
        else:
            for action in self.menu:
                match = re.fullmatch(r'(?:drink|discard) potion (.+?) \(slot (\d+)\)(?:: .*)?',
                                     str(action['description']))
                if match:
                    public_slots.append((int(match[2]), match[1]))
        for slot, name in public_slots:
            if (not isinstance(slot, int) or isinstance(slot, bool) or slot < 0 or
                    not isinstance(name, str) or not name.strip() or
                    canonical(name) in {'empty', 'emptypotionslot'}):
                raise ValueError('Malformed occupied public potion slot')
            name = name.strip()
            if slot in slots and (slots[slot]['empty'] or canonical(slots[slot]['name']) != canonical(name)):
                raise ValueError(f'conflicting public potion slot {slot}')
            slots[slot] = {'name': name, 'index': slot, 'empty': False, 'slot_known': True}

        result = list(slots.values())
        if line is not None:
            missing = len(items) - sum(not s['empty'] for s in result)
            if missing < 0:
                raise ValueError('Public Potions occupied count mismatch')
            # WE_MEET_AGAIN suppresses the recovery menu. Keep the known occupied
            # count, with absent identity/slot, instead of inventing empty slots.
            result += [{'name': None, 'index': None, 'empty': False, 'slot_known': False}
                       for _ in range(missing)]
        return result

    def map_entities(self) -> None:
        graph = self.state.get('map')
        self.coverage['map_available'] = int(isinstance(graph, Mapping))
        if not isinstance(graph, Mapping):
            return
        nodes = {(int(n['x']), int(n['y'])): n for n in graph.get('nodes', [])}
        rooms = ('MONSTER', 'ELITE', 'REST', 'SHOP', 'EVENT', 'TREASURE', 'BOSS')
        memo: dict[tuple, tuple] = {}

        def paths(key):
            if key in memo:
                return memo[key]
            if key == (3, 15) and key not in nodes:
                # Map::createPathsIteration adds this implicit terminal boss;
                # map_graph() only serializes the 15 ordinary rows (0..14).
                boss = np.array([0] * 6 + [1], np.float64)
                memo[key] = 1, boss, boss
                return memo[key]
            if key not in nodes:
                raise ValueError('public map edge has no destination')
            node = nodes[key]
            children = [(int(x), key[1] + 1) for x in sorted(set(node.get('edges', [])))]
            own = np.array([int(canonical(node.get('room')) == canonical(r)) for r in rooms], np.float64)
            if children:
                descendants = [paths(c) for c in children]
                count = sum(d[0] for d in descendants)
                low = own + np.min([d[1] for d in descendants], axis=0)
                high = own + np.max([d[2] for d in descendants], axis=0)
            else:
                count, low, high = 1, own, own
            memo[key] = count, low, high
            return memo[key]

        row = int(graph.get('cur_y', -1)) + 1
        destinations: dict[tuple[int, int], list[str]] = {}
        for action in self.menu:
            description = action['description']
            if match := re.match(r'choose map node x=(\d+)', description):
                # The boss transition's x is a native action parameter, not a
                # spatial destination. Public top-row edges end at (3,15).
                key = ((3, 15) if description[match.end():] == ' advance to boss'
                       else (int(match[1]), row))
                destinations.setdefault(key, []).append(description)
        for key, descriptions in sorted(destinations.items()):
            x, y = key
            node = nodes.get(key)
            if node is None and key != (3, 15):
                raise ValueError(f'legal map destination {key} absent from public graph')
            count, low, high = paths(key)
            room = node.get('room') if node else 'BOSS'
            burning = (node.get('burning_elite') if node and 'burning_elite' in node else
                       (graph.get('burning_elite') == {'x': x, 'y': y} if 'burning_elite' in graph else None))
            # Min/max counts are feasible-path room bounds, not unions across forks.
            pairs = [(y, 15), (burning, 1), (math.log1p(count), 10)]
            pairs += [(v, 15) for v in low[:6]] + [(v, 15) for v in high[:6]]
            entity = self.add('map', 'room', room, _numeric(pairs))
            for description in descriptions:
                self.destinations[description] = entity

    def state_entities(self) -> None:
        state, combat = self.state, self.combat
        keys = state.get('keys', {})
        boss = re.search(r'end-of-act boss ([^\n]+)', self.text)
        self.add('run', 'role', 'run', _numeric([
            (state.get('act'), 4), (state.get('floor'), 60), (state.get('cur_hp'), 100),
            (state.get('max_hp'), 100), (state.get('gold'), 500), (state.get('ascension'), 20),
            (keys.get('ruby'), 1), (keys.get('emerald'), 1), (keys.get('sapphire'), 1)]),
            [self.token('boss', boss[1].strip() if boss else None), self.token('phase', self.obs.get('phase')),
             self.token('room', state.get('room'))])
        player = combat.get('player', {})
        if combat:
            names = ('cur_hp', 'max_hp', 'block', 'energy', 'energy_per_turn', 'card_draw_per_turn',
                     'orb_slots', 'gold', 'turn', 'cards_played_this_turn', 'attacks_played_this_turn',
                     'skills_played_this_turn', 'cards_discarded_this_turn')
            scales = (100, 100, 100, 10, 10, 10, 10, 500, 20, 20, 20, 20, 20)
            fallback = {'cur_hp': 'player_cur_hp', 'max_hp': 'player_max_hp', 'block': 'player_block',
                        'energy': 'player_energy', 'energy_per_turn': 'player_energy_per_turn', 'turn': 'turn'}
            values = [(player.get(k, combat.get(fallback.get(k, ''), combat.get('cards', {}).get(k))), s)
                      for k, s in zip(names, scales)]
            values.append((len(player['powers']) if 'powers' in player else None, 100))
            p = self.add('player', 'role', 'player', _numeric(values), [self.token('stance', player.get('stance'))])
            self.add_powers(p, player.get('powers', {}))
            for enemy in sorted(combat.get('enemies', []), key=lambda e: int(e['index'])):
                slot = int(enemy['index'])
                if slot in self.enemies:
                    raise ValueError('duplicate public enemy slot')
                idx = self.add('enemy', 'enemy', enemy.get('name'), _numeric([
                    (slot, 5), (enemy.get('alive'), 1), (enemy.get('cur_hp'), 100),
                    (enemy.get('max_hp'), 100), (enemy.get('block'), 100),
                    (enemy.get('intent_damage'), 50), (enemy.get('intent_hits'), 10),
                    (len(enemy['powers']) if 'powers' in enemy else None, 100)]),
                    [self.token('intent', enemy.get('intent'))])
                self.enemies[slot] = idx, enemy
                powers = dict(enemy.get('powers', {}))
                for key in ('strength', 'vulnerable', 'weak', 'poison'):
                    if key in enemy and not any(canonical(n) == key for n in powers):
                        powers[key] = enemy[key]
                self.add_powers(idx, powers)
        self.inventories()
        self.map_entities()
        event = state.get('event') or {}
        purpose = re.search(r'^Choose a card to (.+?)(?: \(|\.|$)', self.text, re.M)
        purpose_name = None
        if purpose:
            label = purpose[1].upper()
            purpose_name = ('TRANSFORM_UPGRADE' if 'TRANSFORM' in label and 'UPGRADE' in label else
                            next((p for p in ('TRANSFORM', 'UPGRADE', 'REMOVE', 'DUPLICATE', 'OBTAIN', 'BOTTLE') if p in label),
                                 'BONFIRE_SPIRITS' if 'SACRIFICE' in label else None))
        self.selection_purpose = purpose_name
        chosen = re.search(r'\(chosen (\d+) of (\d+)\)', self.text)
        # The event numeric allowlist is exactly publicEvent() from env-v0.1.5.
        event_fields = ('event_data', 'phase', 'upgrade_one', 'cleanup_is_remove', 'hp_amount0', 'hp_amount1', 'hp_amount2', 'gold_loss', 'gold')
        pairs = [(event.get(k), 1 if k in {'upgrade_one', 'cleanup_is_remove'} else 100) for k in event_fields]
        pairs += [(int(chosen[1]) if chosen else None, 10), (int(chosen[2]) if chosen else None, 10)]
        self.add('context', 'event' if event.get('event') else 'role', event.get('event') or 'context', _numeric(pairs),
                 [self.token('screen', state.get('screen_state')), self.token('purpose', purpose_name),
                  self.token('task', combat.get('cards', {}).get('selection', {}).get('type'))])
        for key, ns in (('falling_skill_card', 'card'), ('falling_power_card', 'card'),
                        ('falling_attack_card', 'card'), ('card', 'card'), ('relic0', 'relic'), ('relic1', 'relic'), ('potion', 'potion')):
            if event.get(key):
                if ns == 'card':
                    card = _card_text(event[key])
                    self.add('offer', ns, _card_name(card['name'])[0], _card_numeric(card),
                             [self.token('card_type', key.split('_')[1]) if key.startswith('falling_') else PAD])
                else:
                    self.add('offer', ns, event[key])
        # Availability/counts are neural inputs as well as audit diagnostics.
        self.add('context', 'role', 'context', _numeric([
            (self.coverage.get(k + '_count') if self.coverage.get(k + '_available') else None, 100)
            for k in ('hand', 'draw', 'discard', 'exhaust', 'deck')
        ] + [(len(self.potions) if self.coverage['potions_available'] else None, 10),
             (self.coverage['potion_capacity'], 10),
             (len([e for e in self.entities if e[1] == ENTITY_TYPES['relic']]) if self.coverage['relics_available'] else None, 100),
             (int('map' in state), 1), (int('recent_actions' in combat), 1)]))

    def card_source(self, label: str, pile: str, *, base_cost_visible=False) -> tuple[int, dict]:
        parsed = _card_text(label, base_cost_visible=base_cost_visible)
        key, upgrade, amount = _card_name(parsed['name'])
        candidates = []
        for idx, card in self.cards[pile]:
            base, up, counter = _card_name(card.get('name', ''))
            if (canonical(base) != canonical(key) or card.get('upgrade_count', up) != upgrade or
                    counter != amount):
                continue
            # OOC deck text has names/upgrades only; the offered type is a
            # public static property, not evidence for a different copy.
            if 'type' in parsed and 'type' in card and card['type'] != parsed['type']:
                continue
            if any(k in parsed and _normalized_cost(card.get(k)) != parsed[k] for k in ('cost', 'cost_for_turn')):
                continue
            if any(k in parsed and card.get(k) != parsed[k] for k in ('free_to_play_once', 'retain')):
                continue
            candidates.append((idx, card))
        if not candidates:
            self.coverage['unresolved_sources'] += 1
            return -1, parsed
        if len({_card_key(c) for _, c in candidates}) > 1:
            # Public description omitted a distinguishing attribute (notably base
            # cost): retain offered identity but don't invent a source reference.
            self.coverage['ambiguous_sources'] += 1
            return -1, parsed
        return candidates[0][0], parsed

    def target(self, description: str) -> int:
        match = re.search(r' -> (.+?)(?= \(deal|$)', description)
        if not match:
            return -1
        slot = re.search(r'\[enemy (\d+)\]', match[1])
        if slot:
            result = self.enemies.get(int(slot[1]))
            if result:
                return result[0]
        else:
            matches = [idx for idx, e in self.enemies.values()
                       if canonical(e.get('name')) == canonical(match[1]) and e.get('alive', True)]
            if len(matches) == 1:
                return matches[0]
        self.coverage['unresolved_targets'] += 1
        raise ValueError(f'cannot resolve public enemy target: {match[1]!r}')

    def action(self, description: str) -> tuple:
        kind, ns, identity, source, target = 'unknown', 'action', 'unknown', -1, -1
        aux = [PAD] * 3
        numeric = _numeric([])
        members: list[int] = []
        card = None
        fallback_text = ''
        if description.startswith('play '):
            kind = 'play'
            source, card = self.card_source(description[5:].split(' -> ')[0].split(' (deal')[0], 'hand')
            target = self.target(description)
        elif description.startswith(('select card for ', 'select card index ')):
            kind = 'select_card'
            if ': ' not in description:
                self.coverage['parser_fallbacks'] += 1
            else:
                prefix, label = description.split(': ', 1)
                task = prefix.removeprefix('select card for ')
                aux[2] = self.token('task', task) if prefix.startswith('select card for ') else PAD
                pile = {'HEADBUTT': 'discard', 'HOLOGRAM': 'discard', 'LIQUID_MEMORIES_POTION': 'discard',
                        'SECRET_TECHNIQUE': 'draw', 'SECRET_WEAPON': 'draw', 'EXHUME': 'exhaust'}.get(task, 'hand')
                if prefix.startswith('select card index '):
                    pile = 'deck'
                if label == 'skip':
                    kind = 'skip_card'
                elif (task in {'CODEX', 'DISCOVERY'} or
                      (prefix.startswith('select card index ') and self.selection_purpose == 'OBTAIN')):
                    card = _card_text(label, base_cost_visible=True)
                else:
                    source, card = self.card_source(label, pile, base_cost_visible=True)
        elif (description.startswith(('exhaust:', 'exhaust: ', 'discard and redraw:')) or
              description in {'exhaust no cards', 'keep current hand (discard no cards)'} or
              re.match(r'^\{ (?:EXHAUST_MANY|GAMBLE)\b', description)):
            kind = 'select_many'
            task = 'EXHAUST_MANY' if description.startswith(('exhaust', '{ EXHAUST')) else 'GAMBLE'
            aux[2] = self.token('task', task)
            if description.startswith('{ '):
                # Legacy native formatter prints wrong nonempty mask indices. Never
                # read bits to repair it. v5 emits explicit correct public slots.
                if 'none' not in description:
                    self.coverage['parser_fallbacks'] += 1
                    fallback_text = description
            else:
                slots = re.findall(r'hand \[(\d+)\]', description)
                if not slots and description not in {'exhaust no cards', 'keep current hand (discard no cards)'}:
                    self.coverage['parser_fallbacks'] += 1
                    fallback_text = description
                for slot in slots:
                    found = [idx for idx, c in self.cards['hand'] if c['index'] == int(slot)]
                    if not found:
                        raise ValueError(f'multi-select refers to absent hand slot {slot}')
                    members.append(found[0])
            source = members[0] if members else -1
            numeric = _numeric([(len(members), 10)])
        elif description.startswith(('drink potion ', 'discard potion ')):
            kind = 'drink_potion' if description.startswith('drink') else 'discard_potion'
            name = re.split(r' -> | \(slot |: ', description.split('potion ', 1)[1])[0]
            ns, identity = 'potion', name
            slot = re.search(r'\(slot (\d+)\)', description)
            matches = [idx for idx, p in self.potions if canonical(p.get('name')) == canonical(name)
                       and (slot is None or p.get('index') == int(slot[1]))]
            source = matches[0] if matches else -1
            if source < 0:
                self.coverage['unresolved_sources'] += 1
            target = self.target(description)
        elif description.startswith('choose map node x='):
            kind = 'map'
            source = self.destinations.get(description, -1)
            room = re.search(r'room=(\w+)', description)
            ns, identity = 'room', room[1] if room else 'BOSS'
            numeric = _numeric([('burning:' in description, 1)])
        elif description.startswith('event option '):
            kind = 'event'
            # Option ordinal is screen routing; effects enter only this small bag.
            fallback_text = re.sub(r'^event option \d+:?\s*', '', description)
        else:
            for prefix, candidate, namespace in (
                ('buy card remove', 'remove_card', 'action'), ('take boss relic ', 'take_boss_relic', 'relic'),
                ('take card ', 'take_card', 'card'), ('buy card ', 'buy_card', 'card'),
                ('take relic ', 'take_relic', 'relic'), ('buy relic ', 'buy_relic', 'relic'),
                ('take potion ', 'take_potion', 'potion'), ('buy potion ', 'buy_potion', 'potion')):
                if description.startswith(prefix):
                    kind, ns = candidate, namespace
                    label = re.split(r' for \d+g', description[len(prefix):])[0]
                    if ns == 'card':
                        card = _card_text(label)
                    elif ns != 'action':
                        identity = label
                    break
            else:
                simple = [('gain +2 max hp', 'singing_bowl'), ('skip boss relic', 'skip_boss_relic'),
                          ('skip card', 'skip_card'), ('skip rewards', 'skip_rewards'),
                          ('card reward choice', 'card_reward'), ('card remove', 'card_remove'),
                          ('take gold', 'take_gold'), ('take key', 'take_key'), ('end turn', 'end_turn'),
                          ('leave shop', 'leave_shop'), ('skip campfire', 'skip_campfire'),
                          ('open chest', 'open_chest'), ('skip chest', 'skip_chest')]
                simple += [(k, k) for k in ('rest', 'smith', 'recall', 'lift', 'toke', 'dig')]
                kind = next((k for prefix, k in simple if description == prefix or description.startswith(prefix + ' ') or description.startswith(prefix + ' (')), 'unknown')
        if card is not None:
            ns, identity = 'card', _card_name(card['name'])[0]
            numeric = _card_numeric(card)
            aux[:2] = [self.token('card_type', card.get('type')), self.token('rarity', card.get('rarity'))]
        if ns == 'action':
            identity = kind
        # Keep price/amount numeric rather than making names or ordinals into IDs.
        price = re.search(r'\bfor (\d+)g\b', description)
        gold = re.search(r'^take gold (\d+)g', description)
        if price or gold:
            numeric[7] = np.clip(int((price or gold)[1]) / 500, -10, 10)
            numeric[23] = 1
        damage = re.search(r'\(deal (\d+)(?: = (\d+) x(\d+))?\)', description)
        if damage:
            for column, value, scale in ((8, int(damage[1]), 100),
                                         (9, int(damage[2]) if damage[2] else int(damage[1]), 100),
                                         (10, int(damage[3]) if damage[3] else 1, 10)):
                numeric[column], numeric[column + 16] = np.clip(value / scale, -10, 10), 1
        if kind == 'unknown':
            self.coverage['parser_fallbacks'] += 1
            fallback_text = description
        # Known mechanics have categorical/numeric features; only long-tail event
        # and unknown wording is hashed. No full observation text bag exists.
        text = _hash_bag(fallback_text)
        if not members and source >= 0:
            members = [source]
        return self.token(ns, identity), ACTION_TYPES[kind], aux, numeric, text, source, target, members

    def encode(self) -> EncodedEntityObservation:
        self.state_entities()
        actions = [self.action(str(a['description'])) for a in self.menu]
        recent = []
        history = self.combat.get('recent_actions', [])
        self.coverage['recent_total'] = len(history)
        for label in history:
            if label.startswith('play '):
                card = _card_text(label[5:].split(' -> ')[0].split(' (deal')[0])
                recent.append((self.token('card', _card_name(card['name'])[0]), ACTION_TYPES['play'], _card_numeric(card)))
            else:
                # Preserve sequence position for all publicly recorded actions.
                kind = 'end_turn' if label == 'end turn' else ('discard_potion' if label.startswith('discard potion') else
                       'drink_potion' if label.startswith('drink potion') else 'select_many' if 'hand [' in label else 'unknown')
                if kind == 'unknown':
                    self.coverage['recent_unparsed'] += 1
                recent.append((self.token('action', kind), ACTION_TYPES[kind], _numeric([])))
        self.coverage['entity_count'] = len(self.entities)
        self.coverage['action_count'] = len(actions)
        ids = lambda v: np.asarray(v, dtype=np.int64)
        floats = lambda v, shape: np.asarray(v, dtype=np.float32).reshape(shape)
        membership = np.full((len(actions), max(1, max(len(a[7]) for a in actions))), -1, np.int64)
        for i, a in enumerate(actions):
            membership[i, :len(a[7])] = a[7]
        return EncodedEntityObservation(
            ids([e[0] for e in self.entities]), ids([e[1] for e in self.entities]), ids([e[2] for e in self.entities]),
            floats([e[3] for e in self.entities], (-1, 32)), np.ones(len(self.entities), bool),
            ids([p[0] for p in self.powers]), ids([p[1] for p in self.powers]),
            floats([p[2] for p in self.powers], (-1,)), np.asarray([p[3] for p in self.powers], bool),
            np.ones(len(self.powers), bool),
            ids([r[0] for r in recent]), ids([r[1] for r in recent]), floats([r[2] for r in recent], (-1, 32)), np.ones(len(recent), bool),
            ids([a[0] for a in actions]), ids([a[1] for a in actions]), ids([a[2] for a in actions]),
            floats([a[3] for a in actions], (-1, 32)), floats([a[4] for a in actions], (-1, 32)),
            ids([a[5] for a in actions]), ids([a[6] for a in actions]), membership, membership >= 0,
            np.ones(len(actions), bool), ids([a['index'] for a in self.menu]), self.coverage)


def encode_observation(obs: Mapping[str, Any]) -> EncodedEntityObservation:
    """Encode public state/menu, optionally with adapter_public_ooc_potions.

    That adapter-only list contains only public {'slot': int, 'name': str}
    extracted before hiding OOC potion choices. No other metadata is consulted.
    """
    return _Encoder(obs).encode()


def collate_observations(observations: list[EncodedEntityObservation], device='cpu') -> dict:
    """Pad without caps. True masks denote real tokens; -1 denotes no reference.

    action_indices and coverage intentionally stay outside the neural batch.
    Empty sparse/recent dimensions get one masked PAD element.
    """
    import torch
    if not observations:
        raise ValueError('cannot collate an empty observation list')
    batch = {}
    for field in fields(EncodedEntityObservation):
        name = field.name
        if name in {'action_indices', 'coverage'}:
            continue
        arrays = [getattr(o, name) for o in observations]
        shape = tuple(max(1, max(a.shape[d] for a in arrays)) for d in range(arrays[0].ndim))
        fill = -1 if name in {'power_owner', 'action_source', 'action_target', 'action_source_members'} else 0
        padded = np.full((len(arrays), *shape), fill, arrays[0].dtype)
        for i, a in enumerate(arrays):
            padded[(i, *(slice(0, d) for d in a.shape))] = a
        batch[name] = torch.as_tensor(padded, device=device)
    for prefix in ('entity', 'power', 'recent', 'action'):
        batch[prefix + '_counts'] = batch[prefix + '_mask'].sum(1)
    batch['action_source_counts'] = batch['action_source_mask'].sum(-1)
    return batch
