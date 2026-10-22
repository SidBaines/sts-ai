#!/usr/bin/env python3
"""Teacher benchmark: champion games in which a teacher plays the boss fights, against the champion's own games.

    analyse_benchmark.py CHAMPION_SUMMARY.json BENCH_DIR [BENCH_DIR...] [--out report.json]

CHAMPION_SUMMARY is a fast_eval summary.json on the same worlds; each BENCH_DIR holds fresh_collect
--teacher-plays --record-search shards (shard_*/worlds/*.json.gz). Trajectories agree until the teacher's
first move, which is checked (the Act 1 boss is entered from identical states). Reports paired wins, boss
pass rates by act, and search diagnostics over the recorded teacher queries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_statistics import clustered_mean, exact_discordance  # noqa: E402
from sts_rl.teacher_votes import RULES, votes_from_details  # noqa: E402


def champion(path):
    return {r["world_seed"]: r for r in json.load(open(path))["records"] if r["status"] == "completed"}


def bench(directory):
    worlds = {}
    for file in sorted(Path(directory).glob("shard_*/worlds/*.json.gz")):
        r = json.load(gzip.open(file))
        if r.get("status") == "completed":
            worlds[r["world_seed"]] = r
    return worlds


def boss(record, act):
    b = record.get("bosses", {}).get(act)
    return None if not b or not b.get("reach") else bool(b.get("pass"))


def first_teacher_step(record):
    steps = [len(s["prefix"]) for s in record["labels"] if s.get("played_by") == "teacher"]
    return min(steps) if steps else None


def interval(diffs):
    m = clustered_mean({i: [d] for i, d in enumerate(diffs)}, equal_world=True)
    return m["point"], m["ci95"]


def compare(base, arm):
    common = sorted(set(base) & set(arm))
    mismatched = 0
    for s in common:
        step = first_teacher_step(arm[s])
        cut = len(arm[s]["action_prefix"]) if step is None else step
        if arm[s]["action_prefix"][:cut] != base[s]["action_prefix"][:cut]:
            mismatched += 1
    only_arm = sum(bool(arm[s]["win"]) and not base[s]["win"] for s in common)
    only_base = sum(bool(base[s]["win"]) and not arm[s]["win"] for s in common)
    report = {"worlds": len(common), "prefix_mismatches_before_teacher": mismatched,
              "wins": [sum(bool(base[s]["win"]) for s in common), sum(bool(arm[s]["win"]) for s in common)],
              "only_champion": only_base, "only_teacher": only_arm, "mcnemar_p": exact_discordance(only_arm, only_base),
              "bosses": {}}
    for act in "123":
        both = [s for s in common if boss(base[s], act) is not None and boss(arm[s], act) is not None]
        if not both:
            continue
        point, ci = interval([float(boss(arm[s], act)) - float(boss(base[s], act)) for s in both])
        report["bosses"][act] = {"reached_both": len(both),
                                 "pass_rate": [sum(boss(base[s], act) for s in both) / len(both),
                                               sum(boss(arm[s], act) for s in both) / len(both)],
                                 "difference": point, "ci95": ci,
                                 "discordant": [sum(boss(base[s], act) and not boss(arm[s], act) for s in both),
                                                sum(boss(arm[s], act) and not boss(base[s], act) for s in both)]}
    return report


def entropy(visits):
    total = sum(visits)
    if total <= 0 or len(visits) < 2:
        return None
    return -sum(v / total * math.log(v / total) for v in visits if v) / math.log(len(visits))


def diagnostics(arm):
    by_act = defaultdict(lambda: {"queries": 0, "abstain": 0, "searches": 0, "winning_line": 0,
                                  "top_share": [], "unanimous": 0, "visit_entropy": [], "top_visit_share": [],
                                  "agree_with_best_line": Counter()})
    for record in arm.values():
        for s in record["labels"]:
            if "search_details" not in s:
                continue
            d = by_act[s["group"][-1]]
            d["queries"] += 1
            details = s["search_details"]
            if not details:
                d["abstain"] += 1
                continue
            d["searches"] += len(details)
            d["winning_line"] += sum(bool(x["winning_sequence_found"]) for x in details)
            votes = votes_from_details(details, "best_line")
            top = max(votes.values())
            d["top_share"].append(top / len(details))
            d["unanimous"] += top == len(details)
            for x in details:
                merged = Counter()
                for e in x["edges"]:
                    merged[e["index"]] += e["visits"]
                visits = list(merged.values())
                if entropy(visits) is not None:
                    d["visit_entropy"].append(entropy(visits))
                    d["top_visit_share"].append(max(visits) / sum(visits))
            best = max(sorted(votes), key=lambda k: votes[k])
            for rule in RULES[1:]:
                other = votes_from_details(details, rule)
                d["agree_with_best_line"][rule] += bool(other) and max(sorted(other), key=lambda k: other[k]) == best
    out = {}
    for act, d in sorted(by_act.items()):
        answered = d["queries"] - d["abstain"]
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        out[act] = {"queries": d["queries"], "abstain": d["abstain"],
                    "winning_line_rate": d["winning_line"] / max(1, d["searches"]),
                    "mean_top_vote_share": mean(d["top_share"]), "unanimous_rate": d["unanimous"] / max(1, answered),
                    "mean_root_visit_entropy": mean(d["visit_entropy"]),
                    "mean_top_visit_share": mean(d["top_visit_share"]),
                    "plurality_agreement_with_best_line": {r: n / max(1, answered)
                                                           for r, n in d["agree_with_best_line"].items()}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("champion")
    p.add_argument("bench", nargs="+")
    p.add_argument("--out")
    args = p.parse_args()
    base = champion(args.champion)
    report = {}
    for directory in args.bench:
        arm = bench(directory)
        r = report[Path(directory).name] = {**compare(base, arm), "search": diagnostics(arm)}
        print(f"{Path(directory).name}: {r['worlds']} worlds (prefix mismatches before the teacher's first move: "
              f"{r['prefix_mismatches_before_teacher']}); wins {r['wins'][0]} -> {r['wins'][1]} "
              f"(discordant {r['only_champion']}/{r['only_teacher']}, McNemar p={r['mcnemar_p']:.3f})")
        for act, b in r["bosses"].items():
            print(f"  Act {act} boss pass {100*b['pass_rate'][0]:.1f}% -> {100*b['pass_rate'][1]:.1f}% "
                  f"({100*b['difference']:+.1f} points [{100*b['ci95'][0]:+.1f}, {100*b['ci95'][1]:+.1f}], "
                  f"n={b['reached_both']}, discordant {b['discordant'][0]}/{b['discordant'][1]})")
        for act, d in r["search"].items():
            agree = ", ".join(f"{k} {100*v:.0f}%" for k, v in d["plurality_agreement_with_best_line"].items())
            print(f"  Act {act} search: {d['queries']} queries, {d['abstain']} abstained; winning playout found in "
                  f"{100*d['winning_line_rate']:.0f}% of searches; top vote share {d['mean_top_vote_share']:.2f}, "
                  f"unanimous {100*d['unanimous_rate']:.0f}%; root visit entropy {d['mean_root_visit_entropy']:.2f} "
                  f"(1 = uniform), most-visited share {d['mean_top_visit_share']:.2f}; same choice as best line: {agree}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
