"""Create a frozen, public-information-only observation intervention.

Only state_text changes. Card costs come from a tiny stateless export of the
simulator's getEnergyCost/CardNames constants; no game instance is created.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from sts_ai.glossary import POTION_DB, RELIC_DB, _ooc_card_names


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_card_costs(include_dir: Path) -> dict[str, tuple[int, int]]:
    """Evaluate the actual public card constants, not a second hand-coded table."""
    source = r'''
#include <iostream>
#include "constants/Cards.h"
int main() {
    for (unsigned i = 1; i < sizeof(sts::cardNames)/sizeof(sts::cardNames[0]); ++i) {
        auto id = static_cast<sts::CardId>(i);
        std::cout << sts::cardNames[i] << "\t" << sts::getEnergyCost(id, false)
                  << "\t" << sts::getEnergyCost(id, true) << "\n";
    }
}
'''
    with tempfile.TemporaryDirectory(prefix="astra_public_costs_") as directory:
        cpp = Path(directory) / "costs.cpp"
        exe = Path(directory) / "costs"
        cpp.write_text(source)
        subprocess.run(["c++", "-std=c++17", "-I", str(include_dir), str(cpp), "-o", str(exe)],
                       check=True, capture_output=True, text=True, timeout=30)
        result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=5)
    costs = {}
    for line in result.stdout.splitlines():
        name, base, upgraded = line.split("\t")
        pair = (int(base), int(upgraded))
        if name in costs and costs[name] != pair:
            raise ValueError(f"Ambiguous public card name: {name}")
        costs[name] = pair
    return costs


def offered_effects(actions: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    lines, unavailable = [], []
    for action in actions:
        match = re.match(r"(?:take boss|take|buy) (relic|potion) (.*?)(?: for \d+g)?$", action["description"])
        if not match:
            continue
        kind, name = match.groups()
        definition = (RELIC_DB if kind == "relic" else POTION_DB).get(name)
        if definition is None:
            unavailable.append(f"{kind}: {name}")
            continue
        # These existing reference strings have ambiguous or omitted timing.
        # Clarify only newly appended references; preserve original text verbatim.
        definition = definition.replace("Gain 1 extra energy each combat", "At the start of each turn, gain 1 extra energy")
        if name == "Letter Opener":
            definition = "Every time you play 3 Skills in a single turn, deal 5 damage to ALL enemies."
        line = f"{name}: {definition}"
        if line not in lines:
            lines.append(line)
    return lines, unavailable


def cost_label(value: int) -> str:
    return "X" if value == -1 else "unplayable" if value < -1 else str(value)


def card_cost_lines(actions: list[dict[str, Any]], costs: dict[str, tuple[int, int]]) -> tuple[list[str], list[str]]:
    lines, unavailable = [], []
    for display_name in _ooc_card_names(actions):
        upgraded = bool(re.search(r"\+\d*$", display_name))
        base_name = re.sub(r"\+\d*$", "", display_name)
        base_name = re.sub(r"=-?\d+$", "", base_name)
        if base_name not in costs:
            unavailable.append(display_name)
            continue
        base, plus = costs[base_name]
        line = f"{display_name}: printed energy cost {cost_label(plus if upgraded else base)}."
        if not upgraded and base != plus:
            line += f" Upgraded printed energy cost: {cost_label(plus)}."
        lines.append(line)
    return lines, unavailable


def map_lines(graph: dict[str, Any]) -> list[str]:
    nodes = sorted(graph["nodes"], key=lambda n: (int(n["y"]), int(n["x"])))
    room_names = {"MONSTER": "Monster", "ELITE": "Elite", "EVENT": "Unknown (?)",
                  "REST": "Rest site", "SHOP": "Shop", "TREASURE": "Treasure"}
    keys = {(int(n["x"]), int(n["y"])) for n in nodes}
    lines = [f"Full visible act map; current row {graph['cur_y']}. Each arrow connects to the next row.",
             "Choose only the next-room actions already listed in LEGAL ACTIONS. Unknown-room contents remain unknown."]
    for node in nodes:
        x, y = int(node["x"]), int(node["y"])
        destinations = [f"({int(child)},{y + 1})" if (int(child), y + 1) in keys else "act boss"
                        for child in node["edges"]]
        lines.append(f"({x},{y}) {room_names.get(node['room'], node['room'])} -> {', '.join(destinations) or 'act boss'}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/experiments/astra_5sep/policy/cohort.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/astra_5sep/policy/cohort_enriched.jsonl"))
    parser.add_argument("--include-dir", type=Path, default=Path("external/sts_lightspeed/include"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Frozen output already exists: {args.output}")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    costs = export_card_costs(args.include_dir.resolve())
    assert costs["Bash"] == (2, 2) and costs["Whirlwind"] == (-1, -1)
    assert costs["Body Slam"] == (1, 0)
    counts: Counter[str] = Counter()
    missing_effects: Counter[str] = Counter()
    missing_costs: Counter[str] = Counter()
    audit_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    additions = []
    enriched = []
    for row in rows:
        new = dict(row)
        lines = []
        effects, unavailable = offered_effects(row["legal_actions"])
        # Avoid duplicating an identical tooltip already present for owned items.
        effects = [line for line in effects if line not in row["state_text"]]
        if effects:
            lines += ["Offered item effects:", *effects]
            counts["rows_with_offered_item_definitions"] += 1
            counts["offered_item_definitions_added"] += len(effects)
        missing_effects.update(unavailable)
        card_lines, unavailable = card_cost_lines(row["legal_actions"], costs)
        if card_lines:
            lines += ["Offered/selected card printed costs (combat effects can modify these):", *card_lines]
            counts["rows_with_card_costs"] += 1
            counts["card_cost_definitions_added"] += len(card_lines)
        missing_costs.update(unavailable)
        if row["probe_screen"] == "MAP_SCREEN":
            source = Path(row["source_trace"])
            audit_path = source.with_name(source.stem + ".audit.jsonl")
            if audit_path not in audit_cache:
                audit_cache[audit_path] = {int(r["decision_index"]): r for r in map(json.loads, audit_path.read_text().splitlines())}
            audit = audit_cache[audit_path][int(row["decision_index"])]
            if audit["state_text"] != row["state_text"] or audit["legal_actions"] != row["legal_actions"]:
                raise ValueError(f"Map source does not match frozen row {row['probe_id']}")
            if int(audit["state"]["world_seed"]) != int(row["world_seed"]):
                raise ValueError("Map source world mismatch")
            lines += map_lines(audit["map_graph"])
            counts["rows_with_full_map"] += 1
        if lines:
            new["state_text"] = row["state_text"] + "\n\n-- ADDITIONAL PUBLIC REFERENCE --\n" + "\n".join(lines)
            counts["rows_changed"] += 1
        assert {k: v for k, v in new.items() if k != "state_text"} == {k: v for k, v in row.items() if k != "state_text"}
        assert new["state_text"].startswith(row["state_text"])
        additions.append({"probe_id": row["probe_id"], "probe_screen": row["probe_screen"],
                          "added_lines": lines, "added_characters": len(new["state_text"]) - len(row["state_text"])})
        enriched.append(new)
    counts["rows"] = len(rows)
    counts["ordinary_event_rows_unresolved"] = sum(row["probe_screen"] == "EVENT_SCREEN" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in enriched))
    manifest = {
        "kind": "astra_public_reference_intervention_v1", "input": str(args.input), "input_sha256": digest(args.input),
        "output": str(args.output), "output_sha256": digest(args.output), "counts": dict(counts),
        "script_sha256": digest(Path(__file__)), "glossary_sha256": digest(Path("src/sts_ai/glossary.py")),
        "card_constants_sha256": digest(args.include_dir / "constants/Cards.h"),
        "map_source_sha256": {str(path): digest(path) for path in audit_cache},
        "missing_offered_item_definitions": dict(missing_effects), "missing_card_costs": dict(missing_costs),
        "changes": ["state_text append only; every other field and row order preserved", "existing offered-item glossary references, clarifying energy-per-turn and Letter Opener timing in appended definitions only", "card current/upgrade printed energy costs evaluated from simulator static getEnergyCost constants", "full public map DAG from exact matching source audit rows on map screens only"],
        "limitations": ["ordinary event identity/options remain anonymous", "no map added to non-map screens", "unavailable glossary entries omitted and counted", "original observations, including any pre-existing reference inaccuracies, are unchanged", "no teacher labels, after-state outcomes, RNG state, rollout values, or seeds enter appended text"],
        "per_row_additions": additions,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": dict(counts), "missing_effects": dict(missing_effects), "missing_costs": dict(missing_costs)}, indent=2))


if __name__ == "__main__":
    main()
