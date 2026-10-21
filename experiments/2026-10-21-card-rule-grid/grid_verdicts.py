#!/usr/bin/env python3
"""Holm-corrected verdicts for the card rule grid, from compare_arms.py output.

    grid_verdicts.py COMPARE.json

A rule is Better if it wins more than the champion and its exact McNemar p is
below 0.05 after Holm correction across every card in the file; Worse if it
wins fewer with corrected p below 0.05; otherwise no clear difference.
"""
import json
import sys


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    m, running, adjusted = len(order), 0.0, {}
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    arms = json.load(open(sys.argv[1]))
    adjusted = holm({name: arm["mcnemar_p"] for name, arm in arms.items()})
    print(f"{'card':16s} {'wins':>11s} {'discordant':>11s} {'p':>8s} {'Holm p':>8s} {'floor diff':>22s}  verdict")
    for name, arm in sorted(arms.items(), key=lambda kv: adjusted[kv[0]]):
        base, rule = arm["wins"]
        floor = arm["floor"]
        verdict = ("Better" if rule > base else "Worse") if adjusted[name] < 0.05 else "No clear difference"
        print(f"{name:16s} {base:5d} -> {rule:4d} {arm['only_baseline']:5d}/{arm['only_arm']:<5d} "
              f"{arm['mcnemar_p']:8.4f} {adjusted[name]:8.4f} {floor['difference']:+6.2f} [{floor['ci95'][0]:+.2f}, {floor['ci95'][1]:+.2f}]  {verdict}")


if __name__ == "__main__":
    main()
