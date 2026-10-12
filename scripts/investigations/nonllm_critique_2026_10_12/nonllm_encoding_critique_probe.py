"""Read-only CPU probes of public_candidate_v2 information loss.

Synthetic observations demonstrate encoder collisions, not live failure rates.
The optional artifact scan reads TRAIN rows only and counts public distinctions.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
from sts_policy.encoding import encode_observation


def compare(label, first, second, changed_public_field):
    a, b = encode_observation(first), encode_observation(second)
    return {"case": label, "changed_public_field": changed_public_field,
            "state_equal": bool(np.array_equal(a.state_features, b.state_features)),
            "all_action_features_equal": bool(np.array_equal(a.action_features, b.action_features)),
            "state_changed_indices": np.flatnonzero(a.state_features != b.state_features).tolist(),
            "action_changed_coordinates": np.argwhere(a.action_features != b.action_features).tolist(),
            "first_public_observation": first, "second_public_observation": second}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--max-train-rows", type=int, default=30000)
    args = parser.parse_args()
    combat = {"phase": "combat", "text": "", "state": {
        "cur_hp": 50, "max_hp": 80, "gold": 100, "act": 2, "floor": 20,
        "combat": {"player_cur_hp": 50, "player_max_hp": 80, "turn": 1,
            "player_energy": 3, "player_energy_per_turn": 3, "player_block": 0,
            "player": {"powers": {}}, "cards": {"hand": [
                {"name": "Whirlwind", "type": "Attack", "cost_for_turn": -1},
                {"name": "Defend", "type": "Skill", "cost_for_turn": 1}]},
            "enemies": [
                {"index": 0, "name": "SPIKE_SLIME_M", "cur_hp": 30, "max_hp": 35,
                 "block": 0, "intent": "SPIKE_SLIME_M_FLAME_TACKLE", "intent_damage": 8,
                 "intent_hits": 1, "alive": True},
                {"index": 1, "name": "ACID_SLIME_M", "cur_hp": 30, "max_hp": 35,
                 "block": 0, "intent": "ACID_SLIME_M_TACKLE", "intent_damage": 8,
                 "intent_hits": 1, "alive": True}]}},
        "legal_actions": [{"index": 0, "description": "play Whirlwind [Attack] (cost X)"},
                          {"index": 1, "description": "play Defend [Skill] (cost 1)"},
                          {"index": 2, "description": "end turn"}]}
    changed = deepcopy(combat)
    for enemy in changed["state"]["combat"]["enemies"]:
        enemy["block"] = 20
    probes = [compare("untargeted_menu_enemy_block", combat, changed,
                      "state.combat.enemies[*].block: 0 -> 20")]

    first = deepcopy(combat)
    first["state"]["combat"]["player"]["powers"] = {"The Bomb (2 turns)": 40}
    changed = deepcopy(first)
    changed["state"]["combat"]["player"]["powers"] = {"The Bomb (3 turns)": 40}
    probes.append(compare("bomb_countdown", first, changed,
                          "state.combat.player.powers: The Bomb (2 turns) -> (3 turns)"))

    first = deepcopy(combat)
    first["state"]["combat"]["cards"]["hand"] = [
        {"name": "Heavy Blade", "type": "Attack", "cost_for_turn": 2}]
    first["state"]["combat"]["enemies"] = [{"index": 0, "name": "ACID_SLIME_L",
        "cur_hp": 50, "max_hp": 65, "block": 0, "intent": "ACID_SLIME_L_TACKLE",
        "intent_damage": 16, "intent_hits": 1, "alive": True}]
    first["legal_actions"] = [{"index": 0, "description":
        "play Heavy Blade [Attack] (cost 2) -> ACID_SLIME_L (deal 16)"},
        {"index": 1, "description": "end turn"}]
    changed = deepcopy(first)
    changed["state"]["combat"]["enemies"][0]["max_hp"] = 69
    probes.append(compare("slime_split_threshold", first, changed,
                          "state.combat.enemies[0].max_hp: 65 -> 69; after damage 16, HP=34"))

    ooc = {"phase": "out_of_combat", "text":
        "Deck: (1): {Searing Blow+1,}\nRelics: {Burning Blood:0,}\nPotions: none",
        "state": {"cur_hp": 50, "max_hp": 80, "gold": 100, "act": 2, "floor": 20},
        "legal_actions": [{"index": 0, "description": "smith"},
                          {"index": 1, "description": "rest heal 24 hp"}]}
    changed = deepcopy(ooc)
    changed["text"] = changed["text"].replace("Blow+1", "Blow+5")
    probes.append(compare("ooc_card_upgrade_value", ooc, changed,
                          "text Deck: Searing Blow+1 -> Searing Blow+5"))

    first = deepcopy(ooc)
    first["legal_actions"] = [{"index": 0, "description": "event option 0: Obtain 100 gold."},
                              {"index": 1, "description": "leave"}]
    changed = deepcopy(first)
    changed["legal_actions"][0]["description"] = "event option 0: Obtain 250 gold."
    probes.append(compare("unparsed_gold_gain_amount", first, changed,
                          "legal_actions[0].description: Obtain 100 gold -> Obtain 250 gold"))

    first = deepcopy(ooc)
    first["legal_actions"] = [{"index": 0, "description": "take card Searing Blow+1 [Attack, Uncommon]"},
                              {"index": 1, "description": "take card Searing Blow+5 [Attack, Uncommon]"}]
    encoded = encode_observation(first)
    candidate_probes = [{"case": "offered_upgrade_candidates", "public_observation": first,
                        "candidate_features_equal": bool(np.array_equal(encoded.action_features[0], encoded.action_features[1]))}]

    nodes = []
    for x in (0, 4):
        nodes.extend([{"x": x, "y": 0, "room": "MONSTER", "edges": [x, x + 1]},
                      {"x": x, "y": 1, "room": "REST", "edges": [x]},
                      {"x": x + 1, "y": 1, "room": "ELITE", "edges": [x + 1]}])
    nodes.extend([{"x": 0, "y": 2, "room": "REST", "edges": []},
                  {"x": 1, "y": 2, "room": "ELITE", "edges": []},
                  {"x": 4, "y": 2, "room": "ELITE", "edges": []},
                  {"x": 5, "y": 2, "room": "REST", "edges": []}])
    first = deepcopy(ooc)
    first["state"]["map"] = {"cur_y": -1, "nodes": nodes}
    first["legal_actions"] = [{"index": 0, "description": "choose map node x=0 room=MONSTER"},
                              {"index": 1, "description": "choose map node x=4 room=MONSTER"}]
    encoded = encode_observation(first)
    candidate_probes.append({"case": "map_optional_versus_inevitable_elite", "public_observation": first,
        "candidate_features_equal": bool(np.array_equal(encoded.action_features[0], encoded.action_features[1])),
        "explanation": "Route x=0 allows MONSTER->REST->REST; route x=4 requires an ELITE on either available two-room path. Reachable room counts by depth match. This is a synthetic mini-map, not a recorded native map."})

    scan = {"teacher": args.teacher, "max_train_rows": args.max_train_rows}
    if args.teacher:
        counts, examples = Counter(), {}
        with Path(args.teacher).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != "train":
                    continue
                observation = row["observation"]
                counts["train_rows"] += 1
                counts["phase_" + observation["phase"]] += 1
                state = observation["state"].get("combat", {})
                enemies = [e for e in state.get("enemies", []) if e.get("alive", True)]
                if any(e.get("block", 0) for e in enemies):
                    counts["rows_positive_enemy_block"] += 1
                for card in state.get("cards", {}).get("hand", []):
                    for key in ("retain", "free_to_play_once"):
                        if card.get(key):
                            counts["rows_with_" + key] += 1
                            break
                encoded = encode_observation(observation)
                seen = {}
                for action, features in zip(observation["legal_actions"], encoded.action_features):
                    key = features.tobytes()
                    other = seen.get(key)
                    if other is not None and other != action["description"]:
                        counts["different_labels_identical_candidate_vector"] += 1
                        if len(examples) < 12:
                            examples.setdefault((other, action["description"]), {
                                "world_seed": row.get("world_seed"), "step": row.get("step"),
                                "first": other, "second": action["description"]})
                    else:
                        seen[key] = action["description"]
                if counts["train_rows"] >= args.max_train_rows:
                    break
        scan.update(counts=dict(counts), examples=list(examples.values()))
    result = {"encoder": "public_candidate_v2", "synthetic_counterexamples": probes,
              "synthetic_candidate_counterexamples": candidate_probes,
              "real_train_artifact_scan": scan,
              "scope": "No model fitting, native simulation, DEV/final observation reading, or paused-source editing."}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": args.output,
        "aliases": [{"case": p["case"], "state_equal": p["state_equal"],
                     "all_action_features_equal": p["all_action_features_equal"]} for p in probes],
        "candidate_aliases": [{"case": p["case"], "equal": p["candidate_features_equal"]} for p in candidate_probes],
        "train_scan": scan}, indent=2))


if __name__ == "__main__":
    main()
