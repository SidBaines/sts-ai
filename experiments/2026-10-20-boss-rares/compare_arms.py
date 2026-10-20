#!/usr/bin/env python3
"""Paired comparison of fast_eval arms against a baseline arm, world by world.

    compare_arms.py <baseline summary.json> <arm summary.json>... [--out report.json]

Wins: counts, discordant worlds and the exact McNemar p. Floors: the paired
difference with a bootstrap 95% interval over worlds. Act 2 boss: pass rate
among worlds that reached it, with the paired difference. Faulted or cut
worlds drop out of every comparison.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_statistics import clustered_mean, exact_discordance  # noqa: E402


def records(path):
    return {r["world_seed"]: r for r in json.load(open(path))["records"] if r["status"] == "completed"}


def boss_pass(record, act="2"):
    boss = record.get("bosses", {}).get(act)
    if not boss or not boss.get("reach"):
        return None
    return bool(boss.get("pass"))


def compare(base, arm):
    common = sorted(set(base) & set(arm))
    only_arm = sum(bool(arm[s]["win"]) and not base[s]["win"] for s in common)
    only_base = sum(bool(base[s]["win"]) and not arm[s]["win"] for s in common)
    floors = clustered_mean({s: [arm[s]["floor"] - base[s]["floor"]] for s in common}, equal_world=True)
    reached = [s for s in common if boss_pass(base[s]) is not None and boss_pass(arm[s]) is not None]
    passes = clustered_mean({s: [float(boss_pass(arm[s])) - float(boss_pass(base[s]))] for s in reached},
                            equal_world=True) if reached else None
    return {"worlds": len(common),
            "wins": [sum(bool(base[s]["win"]) for s in common), sum(bool(arm[s]["win"]) for s in common)],
            "win_rate": [sum(bool(base[s]["win"]) for s in common) / len(common),
                         sum(bool(arm[s]["win"]) for s in common) / len(common)],
            "only_baseline": only_base, "only_arm": only_arm,
            "mcnemar_p": exact_discordance(only_arm, only_base),
            "floor": {"baseline": sum(base[s]["floor"] for s in common) / len(common),
                      "arm": sum(arm[s]["floor"] for s in common) / len(common),
                      "difference": floors["point"], "ci95": floors["ci95"]},
            "act2_boss": {"reached_both": len(reached),
                          "pass_rate": [sum(boss_pass(base[s]) for s in reached) / max(1, len(reached)),
                                        sum(boss_pass(arm[s]) for s in reached) / max(1, len(reached))],
                          "difference": passes["point"] if passes else None,
                          "ci95": passes["ci95"] if passes else None}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("arms", nargs="+")
    p.add_argument("--out")
    args = p.parse_args()
    base = records(args.baseline)
    report = {}
    for path in args.arms:
        name = Path(path).parent.name
        report[name] = result = compare(base, records(path))
        f, b = result["floor"], result["act2_boss"]
        print(f"{name}: {result['worlds']} worlds; wins {result['wins'][0]} -> {result['wins'][1]} "
              f"(discordant {result['only_baseline']}/{result['only_arm']}, McNemar p={result['mcnemar_p']:.3f}); "
              f"floor {f['baseline']:.2f} -> {f['arm']:.2f}, {f['difference']:+.2f} [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}]; "
              f"Act 2 boss pass {100 * b['pass_rate'][0]:.1f}% -> {100 * b['pass_rate'][1]:.1f}% "
              f"({100 * b['difference']:+.1f} points [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}], n={b['reached_both']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
