"""Freeze categorical IDs from public static game names, not sampled futures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from sts_env._backend.glossary import STATUS_DB
from sts_env._backend.ooc import _STATIC


def names_in_array(path: Path, array: str) -> list[str]:
    match = re.search(r"\b" + re.escape(array) + r"\s*\[\]\s*(?:=\s*)?\{(.*?)\};",
                      path.read_text(), flags=re.S)
    if match is None:
        raise ValueError(f"Missing public name array {array} in {path}")
    return [json.loads('"' + value + '"')
            for value in re.findall(r'"((?:\\.|[^"\\])*)"', match[1])]


def canonical(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.native_source / "include/constants"
    tables = {
        "card": ("Cards.h", "cardNames"),
        "enemy": ("MonsterIds.h", "monsterIdStrings"),
        "intent": ("MonsterMoves.h", "monsterMoveStrings"),
        "relic": ("Relics.h", "relicNames"),
        "potion": ("Potions.h", "potionNames"),
        "status": ("PlayerStatusEffects.h", "playerStatusStrings"),
        "boss": ("MonsterEncounters.h", "monsterEncounterStrings"),
        "event": ("Events.h", "eventIdStrings"),
        "room": ("Rooms.h", "roomStrings"),
        "type": ("Cards.h", "cardTypeStrings"),
    }
    namespaces = {key: names_in_array(base / filename, array)
                  for key, (filename, array) in tables.items()}
    namespaces["status"] += list(STATUS_DB)
    namespaces["status"] += names_in_array(base / "MonsterStatusEffects.h", "enemyStatusStrings")
    namespaces["status"] += ["The Bomb"]
    namespaces["event"] += sorted(_STATIC)
    namespaces["role"] = ["deck", "hand", "draw", "discard", "exhaust", "enemy", "player",
                          "relic", "potion", "boss", "event", "map", "offer"]
    namespaces["kind"] = ["card", "enemy", "relic", "potion", "player_power", "map_node",
        "boss", "event", "play_card", "end_turn", "select_card", "select_cards", "skip_card",
        "drink_potion", "discard_potion", "take_card", "take_relic", "take_potion", "take_gold",
        "buy_card", "buy_relic", "buy_potion", "remove_card", "leave", "map", "rest", "smith",
        "out_of_combat"]
    namespaces["rarity"] = ["Basic", "Common", "Uncommon", "Rare", "Curse", "Special"]
    tokens = sorted({key + ":" + canonical(name) for key, names in namespaces.items()
                     for name in names if name})
    status_names = sorted({canonical(name) for name in namespaces["status"]})
    content = {"version": "public_entities_v3_vocabulary_v1", "tokens": ["<pad>", "<unknown>", *tokens],
               "status_names": status_names,
               "source": "Static human-visible names from the pinned native headers and public glossary",
               "source_sha256": {filename: hashlib.sha256((base / filename).read_bytes()).hexdigest()
                                 for filename, _ in tables.values()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(content, indent=2) + "\n")
    print(json.dumps({"tokens": len(content["tokens"]), "statuses": len(status_names),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
