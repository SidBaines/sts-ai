"""Observe the public interface without changing policy inputs or actions.

The baseline experiment imports AuditedAgent to capture live map graphs. This
script then audits those sidecars; collection and analysis never load a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any


@dataclass
class AuditedAgent:
    """Transparent serial-policy wrapper; sidecar writes precede policy calls."""

    env: Any
    agent: Any
    audit_path: Path
    decision_index: int = 0

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def config(self) -> dict[str, Any]:
        return getattr(self.agent, "config", {})

    def reseed(self, policy_seed: int) -> None:
        self.agent.reseed(policy_seed)

    def choose_action(self, state_text: str, legal_actions: list[Any], *,
                      phase: str | None = None) -> Any:
        row = {
            "decision_index": self.decision_index,
            "state": self.env.summary(),
            "state_text": state_text,
            "legal_actions": [self.env.action_dict(action) for action in legal_actions],
            "map_graph": self.env.map_graph(),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.decision_index += 1
        return self.agent.choose_action(state_text, legal_actions, phase=phase)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed writer may leave one partial final line.
                continue
    return rows


def map_audit(graph: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the actual room-sequence possibilities with displayed counts.

    Different room-sequence sets are a sufficient proof of lost route structure.
    Identical sets are not a proof that the underlying DAGs are equivalent.
    """
    from sts_ai.glossary import _reachable_counts

    nodes = {(int(n["x"]), int(n["y"])): n for n in graph["nodes"]}
    rooms = {key: str(node["room"]) for key, node in nodes.items()}
    edges = {key: list(node["edges"]) for key, node in nodes.items()}

    @lru_cache(maxsize=None)
    def paths(key: tuple[int, int]) -> frozenset[tuple[str, ...]]:
        # Last-row edges point at a boss sentinel absent from the node list.
        children = [(int(x), key[1] + 1) for x in edges.get(key, [])
                    if (int(x), key[1] + 1) in rooms]
        if not children:
            return frozenset({(rooms[key],)})
        return frozenset((rooms[key], *tail) for child in children for tail in paths(child))

    choices = []
    sequences_by_x = {}
    for action in actions:
        match = re.search(r"choose map node x=(\d+)", action["description"])
        if not match or "advance to boss" in action["description"]:
            continue
        x = int(match.group(1))
        key = (x, int(graph["cur_y"]) + 1)
        if key not in rooms:
            continue
        counts = _reachable_counts(key, edges, rooms)
        sequences = paths(key)
        sequences_by_x[x] = sequences
        # Rendered aggregate counts exclude the immediate room, so do the metrics.
        tails = [seq[1:] for seq in sequences]
        both = [seq for seq in tails if "ELITE" in seq and "REST" in seq]
        choices.append({
            "x": x,
            "immediate_room": rooms[key],
            "reachable_counts": counts,
            "distinct_room_sequences": len(sequences),
            "min_elites_on_one_route": min(seq.count("ELITE") for seq in tails),
            "max_elites_on_one_route": max(seq.count("ELITE") for seq in tails),
            "aggregate_elites_exceed_any_single_route": counts.get("ELITE", 0) > max(seq.count("ELITE") for seq in tails),
            "elite_and_rest_reachable_but_never_together": bool(counts.get("ELITE") and counts.get("REST") and not both),
            "both_elite_before_rest_and_rest_before_elite_possible": bool(
                any(seq.index("ELITE") < seq.index("REST") for seq in both)
                and any(seq.index("REST") < seq.index("ELITE") for seq in both)
            ),
            "sample_room_sequences": [list(seq) for seq in sorted(sequences)[:3]],
        })
    aliases = []
    witnesses = []
    for left, right in combinations(choices, 2):
        if (left["immediate_room"] == right["immediate_room"]
                and left["reachable_counts"] == right["reachable_counts"]
                and sequences_by_x[left["x"]] != sequences_by_x[right["x"]]):
            aliases.append([left["x"], right["x"]])
            left_only = sorted(sequences_by_x[left["x"]] - sequences_by_x[right["x"]])
            right_only = sorted(sequences_by_x[right["x"]] - sequences_by_x[left["x"]])
            witnesses.append({"left_x": left["x"], "right_x": right["x"],
                              "left_only_sequence": list(left_only[0]) if left_only else None,
                              "right_only_sequence": list(right_only[0]) if right_only else None})
    return {"choices": choices, "same_summary_different_routes_pairs": aliases,
            "alias_witnesses": witnesses}


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    from sts_ai.glossary import _card_definition, _ooc_card_names

    state = row["state"]
    text = row["state_text"]
    actions = row["legal_actions"]
    descriptions = [action["description"] for action in actions]
    screen = str(state.get("screen_state", "")).split(".")[-1]
    offers = []
    for description in descriptions:
        for kind, prefix in (("relic", "take boss relic "), ("relic", "take relic "),
                             ("relic", "buy relic "), ("potion", "take potion "),
                             ("potion", "buy potion ")):
            if description.startswith(prefix):
                name = re.sub(r" for \d+g$", "", description[len(prefix):])
                offers.append({"kind": kind, "name": name,
                               "tooltip_present": f"\n  {name}:" in text})
                break
    cards = _ooc_card_names(actions)
    result = {
        "decision_index": row["decision_index"],
        "world_seed": state["world_seed"],
        "floor": state["floor"],
        "screen": screen,
        "ordinary_event": screen == "EVENT_SCREEN" and int(state["floor"]) > 0,
        "blind_event": screen == "EVENT_SCREEN" and int(state["floor"]) > 0
        and bool(descriptions) and all(re.fullmatch(r"event option \d+", d) for d in descriptions),
        "offers": offers,
        "offered_cards": cards,
        "offered_cards_without_cost_mention": [name for name in cards
            if not re.search(r"\bcost", _card_definition(name) or "", re.IGNORECASE)],
        "map_available": row.get("map_graph") is not None,
    }
    if row.get("map_graph"):
        result["map"] = map_audit(row["map_graph"], actions)
    return result


def add_counts(counts: Counter[str], row: dict[str, Any]) -> None:
    counts["decisions"] += 1
    counts[f"screen_{row['screen']}"] += 1
    counts["ordinary_events"] += row["ordinary_event"]
    counts["blind_events"] += row["blind_event"]
    counts["card_offer_instances"] += len(row["offered_cards"])
    counts["card_offer_instances_without_cost_mention"] += len(row["offered_cards_without_cost_mention"])
    for offer in row["offers"]:
        counts[f"{offer['kind']}_offer_instances"] += 1
        counts[f"{offer['kind']}_offer_instances_without_tooltip"] += not offer["tooltip_present"]
    if row["screen"] != "MAP_SCREEN":
        counts["nonmap_screen_decisions"] += 1
        counts["nonmap_screen_decisions_without_map"] += not row["map_available"]
    if "map" in row:
        counts["map_decisions"] += 1
        choices = row["map"]["choices"]
        counts["map_choices"] += len(choices)
        counts["map_decisions_with_multiple_choices"] += len(choices) > 1
        aliases = row["map"]["same_summary_different_routes_pairs"]
        counts["map_decisions_with_proven_aliases"] += bool(aliases)
        counts["map_choice_pairs_with_proven_aliases"] += len(aliases)
        for choice in choices:
            for key in ("aggregate_elites_exceed_any_single_route",
                        "elite_and_rest_reachable_but_never_together",
                        "both_elite_before_rest_and_rest_before_elite_possible"):
                counts[f"map_choices_{key}"] += choice[key]


def analyze(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, Counter[str]] = {}
    episodes = []
    examples: dict[str, list[dict[str, Any]]] = {"blind_event": [], "missing_tooltip": [], "map_alias": []}
    alias_worlds: set[int] = set()
    with (output_dir / "decision_audit.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(input_dir.glob("*/*.audit.jsonl")):
            meta_path = path.with_name(path.name.replace(".audit.jsonl", ".meta.json"))
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            status = str(meta.get("stopped_reason", "partial_no_metadata"))
            arm = path.parent.name
            group = f"{arm}/{status}"
            counts = by_group.setdefault(group, Counter())
            counts["episodes"] += 1
            rows = read_jsonl(path)
            had_blind_event = had_alias = False
            for source in rows:
                audited = audit_row(source)
                audited.update({"arm": arm, "episode_status": status, "source": str(path)})
                add_counts(counts, audited)
                had_blind_event |= audited["blind_event"]
                if audited.get("map", {}).get("same_summary_different_routes_pairs"):
                    had_alias = True
                    alias_worlds.add(audited["world_seed"])
                handle.write(json.dumps(audited, sort_keys=True) + "\n")
                for key, condition in (("blind_event", audited["blind_event"]),
                    ("missing_tooltip", any(not o["tooltip_present"] for o in audited["offers"])),
                    ("map_alias", bool(audited.get("map", {}).get("same_summary_different_routes_pairs")))):
                    if condition and len(examples[key]) < 3:
                        examples[key].append({"audit": audited, "raw_observation": source})
            episodes.append({"arm": arm, "status": status, "source": str(path),
                             "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "n_decisions_observed": len(rows), "final_floor": meta.get("final_floor")})
            counts["episodes_with_blind_events"] += had_blind_event
            counts["episodes_with_map_aliases"] += had_alias
    aggregate = Counter()
    terminal_aggregate = Counter()
    for group, counts in by_group.items():
        aggregate.update(counts)
        if group.endswith("/terminal"):
            terminal_aggregate.update(counts)
    report = {
        "kind": "astra_public_information_audit_v1",
        "input_dir": str(input_dir),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "alias_world_seeds": sorted(alias_worlds),
        "definitions": {
            "population": "Policy-independent observation availability under random/heuristic visitation; frequencies are not estimates for an LLM policy.",
            "partial": "All observed decisions include episodes killed before metadata. Terminal-only counts are supplied separately; sidecars precede execution, so a final observed decision may be unexecuted.",
            "offer_instances": "Each offered item per visited decision, including repeated offers after taking another reward; not unique items or independent samples.",
            "cost_mention": "No substring beginning 'cost' in the supplied card glossary entry. A cost mention is weaker than a complete/correct energy-cost description.",
            "map_alias": "Two simultaneous choices have equal immediate room and downstream union-count dictionary but different complete room-sequence sets; ignores x labels.",
            "route_metrics": "Downstream metrics exclude the immediate choice to match the displayed aggregate. Repeated map states are dependent; no hypothesis-test p-values claimed.",
        },
        "all_observed": dict(aggregate),
        "terminal_episodes_only": dict(terminal_aggregate),
        "by_arm_and_status": {key: dict(value) for key, value in by_group.items()},
        "episodes": episodes,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True) + "\n")
    metrics = [
        ("Anonymous ordinary event decisions", "blind_events", "ordinary_events"),
        ("Offered relics without supplied tooltip", "relic_offer_instances_without_tooltip", "relic_offer_instances"),
        ("Offered potions without supplied tooltip", "potion_offer_instances_without_tooltip", "potion_offer_instances"),
        ("Offered cards without any cost mention", "card_offer_instances_without_cost_mention", "card_offer_instances"),
        ("Non-map screen decisions without map", "nonmap_screen_decisions_without_map", "nonmap_screen_decisions"),
        ("Multi-choice map decisions with proven route aliases", "map_decisions_with_proven_aliases", "map_decisions_with_multiple_choices"),
        ("Map choices whose aggregate elite count exceeds every realizable route", "map_choices_aggregate_elites_exceed_any_single_route", "map_choices"),
    ]
    lines = ["# Fresh diagnostic public-information audit", "",
             f"Observed {aggregate['decisions']:,} decisions in {aggregate['episodes']} episodes from random/heuristic policies, each with search budgets 50 and 500, worlds 1000–1031. Terminal episodes: {terminal_aggregate['episodes']}.", "",
             "| Finding | Count | Rate |", "| --- | ---: | ---: |"]
    for label, numerator, denominator in metrics:
        n, d = aggregate[numerator], aggregate[denominator]
        lines.append(f"| {label} | {n:,}/{d:,} | {n / d:.1%} |" if d else f"| {label} | {n}/0 | n/a |")
    lines += ["", f"Proven route aliases appeared in {len(alias_worlds)} distinct worlds: {sorted(alias_worlds)}.", "",
              "These are availability measurements under baseline visitation, not an LLM outcome experiment. Offers repeat across reward micro-decisions, and map states repeat across arms. Exact route aliasing proves missing distinctions but does not quantify their effect on survival. Tooltips may also be memorized by a model; these counts concern supplied information. See report.json for per-arm totals and definitions; examples.json preserves prompts, full graphs, and route witnesses.", "",
              "Reproduce after collecting baselines with the AuditedAgent wrapper:", "", "```sh",
              "PYTHONPATH=src .venv/bin/python scripts/experiments/astra_information_audit.py --self-check",
              "PYTHONPATH=src .venv/bin/python scripts/experiments/astra_information_audit.py", "```", "",
              "Audited versus unwrapped policy traces matched exactly for all eight independently repeated checks (both policies × both search budgets × worlds 1000 and 1020); evidence: ../baselines_verification/verification.json."]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return report


def self_check() -> None:
    """Check a minimal indistinguishable-summary / different-order counterexample."""
    graph = {"cur_y": -1, "nodes": [
        {"x": 0, "y": 0, "room": "MONSTER", "edges": [0]},
        {"x": 1, "y": 0, "room": "MONSTER", "edges": [1]},
        {"x": 0, "y": 1, "room": "REST", "edges": [0]},
        {"x": 1, "y": 1, "room": "ELITE", "edges": [1]},
        {"x": 0, "y": 2, "room": "ELITE", "edges": []},
        {"x": 1, "y": 2, "room": "REST", "edges": []},
    ]}
    result = map_audit(graph, [{"description": f"choose map node x={x} room=MONSTER"} for x in (0, 1)])
    assert result["same_summary_different_routes_pairs"] == [[0, 1]]
    # Identical routes must not produce an alias finding.
    graph["nodes"][3]["room"], graph["nodes"][5]["room"] = "REST", "ELITE"
    assert not map_audit(graph, [{"description": f"choose map node x={x} room=MONSTER"} for x in (0, 1)])["same_summary_different_routes_pairs"]
    graph["nodes"][4]["edges"] = [3]
    assert not map_audit(graph, [{"description": f"choose map node x={x} room=MONSTER"} for x in (0, 1)])["same_summary_different_routes_pairs"]
    row = {"decision_index": 0, "state": {"world_seed": 1000, "floor": 7, "screen_state": "ScreenState.EVENT_SCREEN"},
           "state_text": "Potions: none", "legal_actions": [{"description": "event option 0"}], "map_graph": None}
    assert audit_row(row)["blind_event"]
    row["state"]["floor"] = 0
    assert not audit_row(row)["blind_event"]
    print("information audit self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/experiments/astra_5sep/baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/astra_5sep/information"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        report = analyze(args.input_dir, args.output_dir)
        print(json.dumps(report["all_observed"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
