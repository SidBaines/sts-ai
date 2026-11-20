"""Describe the boss outcomes on the same complete worlds after the primary verdict."""
from collections import Counter
import csv
import json

from sts_rl.fresh_statistics import paired_win_normal_interval


def boss_table(run, folder, expected_pairs):
    players = {}
    for arm in ("early", "joint"):
        records = json.loads((run / arm / "summary.json").read_bytes())["records"]
        players[arm] = {r["world_seed"]: r["bosses"] for r in records if r["status"] == "completed"}
        del records
    common = players["early"].keys() & players["joint"].keys()
    if len(common) != expected_pairs:
        raise ValueError("Boss outcomes and the full-game comparison have different complete pairs")
    lines = ["### Where the games diverged", "",
             "This exploratory breakdown was added after the championship verdict. Counts and percentage-point changes use all complete pairs.",
             "", "| Boss | Reached, joint vs early | Paired change [95% interval] | Cleared, joint vs early | Paired change [95% interval] |",
             "|---|---:|---:|---:|---:|"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "boss-outcomes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["act", "outcome", "pairs", "joint", "early", "joint_only", "early_only", "difference_pp", "lower95", "upper95"])
        for act in ("1", "2", "3"):
            row = [f"Act {act}"]
            for metric in ("reach", "pass"):
                counts = Counter(tuple(bool(players[arm][seed].get(act, {}).get(metric))
                                       for arm in ("early", "joint")) for seed in common)
                up, down, both = counts[False, True], counts[True, False], counts[True, True]
                delta = 100 * (up - down) / len(common)
                lo, hi = (100 * x for x in paired_win_normal_interval(len(common), up, down))
                writer.writerow([act, metric, len(common), both + up, both + down, up, down, delta, lo, hi])
                row += [f"{both + up:,} vs {both + down:,}", f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]"]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Each player fought with the decks, health and potions it brought to those encounters."]
    return "\n".join(lines) + "\n\n"
