"""Freeze B1 public names; never fit IDs to trajectories or evaluation data.

Default regeneration uses the public_names table embedded in the committed JSON.
--bootstrap-json accepts a static public vocabulary, not observations. --native-source
reads only simulator include/constants name tables and needs no simulator import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

DEFAULT = Path(__file__).resolve().parents[2] / 'src/sts_policy/entity_vocabulary.json'
TABLES = {
    'card': ('Cards.h', 'cardNames'), 'enemy': ('MonsterIds.h', 'monsterIdStrings'),
    'intent': ('MonsterMoves.h', 'monsterMoveStrings'), 'relic': ('Relics.h', 'relicNames'),
    'potion': ('Potions.h', 'potionNames'), 'power': ('PlayerStatusEffects.h', 'playerStatusStrings'),
    'boss': ('MonsterEncounters.h', 'monsterEncounterStrings'), 'event': ('Events.h', 'eventIdStrings'),
    'room': ('Rooms.h', 'roomStrings'), 'card_type': ('Cards.h', 'cardTypeStrings'),
}
ACTIONS = ('unknown play end_turn select_card select_many drink_potion discard_potion '
           'take_card take_relic take_potion take_gold take_key skip_card skip_rewards '
           'skip_boss_relic take_boss_relic buy_card buy_relic buy_potion remove_card '
           'leave_shop map rest smith recall lift toke dig skip_campfire open_chest '
           'skip_chest event singing_bowl card_reward card_remove').split()
TASKS = ('CODEX DISCOVERY ARMAMENTS DUAL_WIELD EXHAUST_ONE FORETHOUGHT WARCRY HEADBUTT '
         'HOLOGRAM LIQUID_MEMORIES_POTION SECRET_TECHNIQUE SECRET_WEAPON EXHUME EXHAUST_MANY GAMBLE').split()


def canonical(text: str) -> str:
    text = re.sub(r'^(?:ScreenState|Room|Stance|CardType|InputState|Phase)\.', '', text, flags=re.I)
    return re.sub(r'[^a-z0-9]', '', text.lower())


def names_in_array(path: Path, array: str) -> list[str]:
    match = re.search(r'\b' + array + r'\s*\[\]\s*(?:=\s*)?\{(.*?)\};', path.read_text(), re.S)
    if not match:
        raise ValueError(f'Missing public array {array} in {path}')
    return [json.loads('"' + v + '"') for v in re.findall(r'"((?:\\.|[^"\\])*)"', match[1])]


def build(names: dict, source: dict) -> dict:
    names = {k: sorted(set(v)) for k, v in names.items()}
    additions = {
        'action': ACTIONS, 'task': TASKS,
        'role': ['run', 'player', 'context', 'hand', 'deck', 'draw', 'discard', 'exhaust'],
        'stance': ['NEUTRAL', 'CALM', 'WRATH', 'DIVINITY'],
        'screen': ['EVENT_SCREEN', 'REWARDS', 'BOSS_RELIC_REWARDS', 'CARD_SELECT', 'MAP_SCREEN',
                   'TREASURE_ROOM', 'REST_ROOM', 'SHOP_ROOM', 'BATTLE', 'INVALID'],
        'phase': ['combat', 'out_of_combat'],
        'purpose': ['TRANSFORM', 'TRANSFORM_UPGRADE', 'UPGRADE', 'REMOVE', 'DUPLICATE',
                    'OBTAIN', 'BOTTLE', 'BONFIRE_SPIRITS'],
        'power': ['The Bomb (1 turn)', 'The Bomb (2 turns)', 'The Bomb (3 turns)'],
        'potion': ['empty'],
        'rarity': ['Basic', 'Common', 'Uncommon', 'Rare', 'Curse', 'Special'],
        'room': ['MONSTER', 'ELITE', 'REST', 'SHOP', 'EVENT', 'TREASURE', 'BOSS'],
    }
    for k, v in additions.items():
        names[k] = sorted(set(names.get(k, []) + v))
    tokens = sorted({ns + ':' + canonical(name) for ns, values in names.items() for name in values if name})
    aliases = {ns + ':' + canonical(name): ns + ':' + canonical(name)
               for ns, values in names.items() for name in values if name}
    # Actual display aliases, not learned synonym matches.
    for name in ('Strike', 'Defend'):
        if 'card:' + canonical(name) not in tokens and 'card:' + canonical(name + '_R') in tokens:
            aliases['card:' + canonical(name)] = 'card:' + canonical(name + '_R')
    for public, display in (('BONFIRE_SPIRITS', 'Bonfire Elementals'), ('DESIGNER_IN_SPIRE', 'Designer')):
        target = 'event:' + canonical(display)
        if target not in tokens:
            raise ValueError(f'Missing static event alias destination: {display}')
        aliases['event:' + canonical(public)] = target
    return {'version': 'public_entities_fresh_v1_vocabulary_v1', 'environment_contract': 'env-v0.1.5',
            'source': source, 'normalization': 'lowercase; strip namespace enum prefix, spaces, underscores and punctuation; card =N suffix is numeric public amount, followed by optional +N numeric upgrade_count',
            'special_ids': {'PAD': 0, 'UNK': 1}, 'public_names': dict(sorted(names.items())),
            'aliases': dict(sorted(aliases.items())), 'tokens': ['<pad>', '<unknown>', *tokens]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--bootstrap-json', type=Path)
    parser.add_argument('--native-source', type=Path)
    args = parser.parse_args()
    if args.bootstrap_json and args.native_source:
        parser.error('choose one static name source')
    if args.bootstrap_json:
        raw = args.bootstrap_json.read_bytes()
        old = json.loads(raw)
        names: dict[str, list[str]] = {}
        rename = {'status': 'power', 'type': 'card_type'}
        for token in old['tokens']:
            if ':' in token:
                ns, name = token.split(':', 1)
                if ns not in {'kind', 'role'}:
                    names.setdefault(rename.get(ns, ns), []).append(name)
        source = {'kind': 'static_public_names_bootstrap',
                  'reference': 'archive/nonllm/critique:src/sts_policy/public_vocabulary_v3.json',
                  'bootstrap_sha256': hashlib.sha256(raw).hexdigest(),
                  'original_header_sha256': old.get('source_sha256', {}),
                  'note': 'Public names only; no weights, trajectories, labels or behavior prior. Embedded for standalone regeneration.'}
    elif args.native_source:
        base = args.native_source / 'include/constants'
        names = {ns: names_in_array(base / f, a) for ns, (f, a) in TABLES.items()}
        names['power'] += names_in_array(base / 'MonsterStatusEffects.h', 'enemyStatusStrings')
        # Static glossary keys only, never observations.
        from sts_env._backend.glossary import STATUS_DB
        names['power'] += list(STATUS_DB)
        source = {'kind': 'static_simulator_headers', 'header_sha256': {
            f: hashlib.sha256((base / f).read_bytes()).hexdigest()
            for f in sorted({f for f, _ in TABLES.values()} | {'MonsterStatusEffects.h'})}}
    else:
        current = json.loads(DEFAULT.read_text())
        names, source = current['public_names'], current['source']
    content = json.dumps(build(names, source), indent=2, sort_keys=True) + '\n'
    args.output.write_text(content)
    print(json.dumps({'tokens': len(json.loads(content)['tokens']),
                      'sha256': hashlib.sha256(content.encode()).hexdigest(), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
