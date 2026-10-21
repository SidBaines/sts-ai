#!/usr/bin/env python3
"""Take rate of each card when offered, from measure_takes.py output.

    analyse_takes.py TAKES.jsonl [OUT.csv]

A card-reward decision is a logged menu whose chosen action takes a card, skips
the card reward, or proceeds past it; earlier steps of the same screen (taking
gold or a potion first) are ignored. Prints, per card, offers, takes, take rate,
and the policy's mean probability of taking it, sorted by offers.
"""
import collections
import csv
import json
import sys

DECIDE = ("take card ", "skip card reward", "skip rewards / proceed")


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    decisions = [r for r in rows if r["chosen"].startswith(DECIDE)]
    games = len({r["seed"] for r in rows})
    stats = collections.defaultdict(lambda: {"offers": 0, "takes": 0, "p": 0.0, "type": "", "rarity": ""})
    skips = 0
    for r in decisions:
        taken = r["chosen"][len("take card "):].split(" [")[0] if r["chosen"].startswith("take card ") else None
        skips += taken is None
        for o in r["offers"]:
            s = stats[o["name"]]
            s["offers"] += 1
            s["takes"] += o["name"] == taken
            s["p"] += o["p"]
            s["type"], s["rarity"] = o["type"], o["rarity"]
    print(f"{games} games, {len(decisions)} card-reward decisions, skipped {skips} ({skips / max(1, len(decisions)):.1%})")
    table = sorted(((name, s) for name, s in stats.items()), key=lambda x: -x[1]["offers"])
    print(f"{'card':24s} {'type':7s} {'rarity':9s} {'offers':>6s} {'takes':>6s} {'rate':>6s} {'mean p':>7s}")
    for name, s in table:
        print(f"{name:24s} {s['type']:7s} {s['rarity']:9s} {s['offers']:6d} {s['takes']:6d} {s['takes'] / s['offers']:6.1%} {s['p'] / s['offers']:7.3f}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["card", "type", "rarity", "offers", "takes", "take_rate", "mean_policy_probability"])
            for name, s in table:
                writer.writerow([name, s["type"], s["rarity"], s["offers"], s["takes"],
                                 round(s["takes"] / s["offers"], 4), round(s["p"] / s["offers"], 4)])


if __name__ == "__main__":
    main()
