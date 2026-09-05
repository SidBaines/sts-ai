"""Describe lookahead route changes against heuristic choices at the same states."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def action_type(description: str) -> str:
    room = re.search(r"room=(\w+)", description)
    return room.group(1) if room else description.split()[0].lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    attempts = json.loads((args.directory / "attempts.json").read_text())
    report = {}
    for mode in ("map_only", "map_rest"):
        completed = {r["world_seed"] for r in attempts if r["mode"] == mode and r["status"] == "completed"}
        records = []
        for path in (args.directory / mode).glob("*.choices.jsonl"):
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        analyses = {}
        for scope in ("all_observed_choices", "completed_games_only"):
            included = [r for r in records if "selected_action_index" in r and
                        (scope == "all_observed_choices" or r["world_seed"] in completed)]
            screens = {}
            for screen in ("MAP_SCREEN", "REST_ROOM"):
                chosen, heuristic, changes = Counter(), Counter(), Counter()
                n, n_changed = 0, 0
                for row in included:
                    if row["screen"] != screen:
                        continue
                    actions = {a["index"]: a["description"] for a in row["legal_actions"]}
                    actual = action_type(actions[row["selected_action_index"]])
                    control = action_type(actions[row["heuristic_action_index"]])
                    n += 1; chosen[actual] += 1; heuristic[control] += 1
                    if row["changed_from_heuristic"]:
                        changes[f"{control}->{actual}"] += 1; n_changed += 1
                screens[screen] = {"n_choices": n, "n_changed": n_changed,
                    "chosen_types": dict(chosen), "heuristic_types_same_states": dict(heuristic),
                    "net_type_change_same_states": {k: chosen[k]-heuristic[k] for k in sorted(set(chosen)|set(heuristic))},
                    "changed_action_type_transitions": dict(changes)}
            analyses[scope] = screens
        report[mode] = analyses
    report["interpretation"] = "The counterfactual heuristic type is evaluated at each lookahead-visited state. These counts do not reconstruct a full heuristic trajectory or establish risk preference; same room-type switches can choose different downstream paths. Partial failed games are included only in all_observed_choices."
    (args.directory / "route_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
