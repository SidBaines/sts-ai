#!/usr/bin/env python3
"""Summarise combat_diff.py rows: disagreement by act and fight type, and what kind of choice changed.

    summarise_diff.py rows.jsonl[.gz] [--top 25]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import re


def kind(desc):
    if desc.startswith("end turn"):
        return "end turn", None
    m = re.match(r"(drink|discard) potion (.+?)(?: \(slot| ->|$)", desc)
    if m:
        return m[1], m[2].strip()
    m = re.match(r"play (.+?) \[", desc)
    if m:
        return "play", m[1]
    return "other", desc.split(" (")[0][:40]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rows")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    opener = gzip.open if a.rows.endswith(".gz") else open
    by_group, change, drink_states, potion_new, potion_old = defaultdict(Counter), Counter(), Counter(), Counter(), Counter()
    cards_new, cards_old, worlds = Counter(), Counter(), set()
    for line in opener(a.rows, "rt"):
        r = json.loads(line)
        worlds.add(r["world"])
        room = {"MONSTER": "hallway", "ELITE": "elite", "BOSS": "boss"}.get(r["room"], "other")
        for g in (f"act {r['act']} {room}", f"all {room}", "all"):
            by_group[g]["n"] += 1
            by_group[g]["differ"] += r["old"] != r["new"]
        o, n = r["legal"][r["old"]], r["legal"][r["new"]]
        ko, kn = kind(o), kind(n)
        can_drink = any(d.startswith("drink potion") for d in r["legal"])
        if can_drink:
            drink_states["states"] += 1
            drink_states["old drinks"] += ko[0] == "drink"
            drink_states["new drinks"] += kn[0] == "drink"
            for g in (f"act {r['act']} {room}",):
                drink_states[g + " states"] += 1
                drink_states[g + " old"] += ko[0] == "drink"
                drink_states[g + " new"] += kn[0] == "drink"
        if r["old"] == r["new"]:
            continue
        if ko[0] != kn[0]:
            label = f"{ko[0]} -> {kn[0]}"
        elif ko[0] == "play":
            label = "same card, other target" if ko[1] == kn[1] else "other card"
        elif ko[0] in ("drink", "discard"):
            label = f"{ko[0]}: same potion, other target" if ko[1] == kn[1] else f"{ko[0]}: other potion"
        else:
            label = "other choice"
        change[label] += 1
        change[f"{room}: {label}"] += 1
        if kn[0] == "drink" and ko[0] != "drink":
            potion_new[kn[1]] += 1
        if ko[0] == "drink" and kn[0] != "drink":
            potion_old[ko[1]] += 1
        if ko[0] == "play" and kn[0] == "play" and ko[1] != kn[1]:
            cards_old[ko[1]] += 1
            cards_new[kn[1]] += 1
    print(f"{len(worlds)} worlds")
    print("\nDisagreement rate by group (states with >1 legal action)")
    for g in sorted(by_group):
        c = by_group[g]
        print(f"  {g:22s} n={c['n']:8d}  differ {100*c['differ']/c['n']:5.2f}%")
    total = by_group["all"]["differ"]
    print(f"\nKinds of change ({total} disagreements)")
    for k, v in change.most_common():
        if ":" not in k or k.startswith(("drink:", "discard:")):
            print(f"  {k:40s} {v:7d} {100*v/total:5.1f}%")
    print("\nKinds of change by fight type")
    for k, v in sorted(change.items()):
        if k.startswith(("hallway:", "elite:", "boss:", "other:")):
            print(f"  {k:48s} {v:7d}")
    print("\nDrink rate at states where a drink is legal (old -> new)")
    print(f"  all: {drink_states['states']} states, old {100*drink_states['old drinks']/drink_states['states']:.2f}%"
          f" new {100*drink_states['new drinks']/drink_states['states']:.2f}%")
    for g in sorted(k[:-7] for k in drink_states if k.endswith(" states") and k != "states"):
        s = drink_states[g + " states"]
        print(f"  {g:18s} {s:7d} states  old {100*drink_states[g+' old']/s:5.2f}%  new {100*drink_states[g+' new']/s:5.2f}%")
    print(f"\nPotions the new head drinks where the old head does something else (top {a.top})")
    for k, v in potion_new.most_common(a.top):
        print(f"  {k:28s} {v:6d}   (reverse: {potion_old[k]})")
    print(f"\nPlay-vs-play card swaps: cards the new head plays instead (top {a.top}), and cards it stops playing")
    for (k, v), (k2, v2) in zip(cards_new.most_common(a.top), cards_old.most_common(a.top)):
        print(f"  +{k:24s} {v:6d}    -{k2:24s} {v2:6d}")


if __name__ == "__main__":
    main()
