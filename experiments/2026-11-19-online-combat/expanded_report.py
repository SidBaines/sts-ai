"""Tabulate every fit and the independent short-trajectory test."""

import csv
from datetime import datetime, timezone
import re

from report import ROOT, BASE, describe, read, replace_note

FIGURE = ROOT / "research/figures/online-combat-pilot"
TITLE = "### Expanded pilot results"


def rows_from_disk():
    return {f.stem: read(f) for f in (BASE / "expanded" / "comparisons").glob("*.json")}


def interval(stat):
    return (
        stat.get("difference_pp", stat.get("complete_pair_difference_pp")),
        stat.get("ci95_pp", stat.get("complete_pair_ci95_pp")),
    )


def section(results, verdict=None):
    original = next(
        k
        for k, v in results.items()
        if v["setting"]["target"] == 0.12
        and v["setting"]["rate"] == 3e-5
        and v["setting"]["epochs"] == 2
    )
    baseline = results[original]["comparison"]["first"]["wins"]
    text = [
        TITLE,
        "",
        f"All 27 settings played the same 768 screening worlds. The original recipe won {baseline}. "
        "Each cell gives wins and the paired difference in percentage points with a 95% interval against that recipe.",
        "",
        "| Combat learning rate | Passes | 5% exploration | 12% exploration | 22% exploration |",
        "|---:|---:|---|---|---|",
    ]
    by_setting = {
        (v["setting"]["rate"], v["setting"]["epochs"], v["setting"]["target"]): (k, v)
        for k, v in results.items()
    }
    for rate, passes in sorted({(r, e) for r, e, _ in by_setting}):
        cells = []
        for target in (0.05, 0.12, 0.22):
            k, v = by_setting[rate, passes, target]
            s = v["comparison"]
            change, ci = interval(s)
            counts = s["second"]["wins"]
            label = (
                f"{counts}, reference"
                if k == original
                else f"{counts}, {change:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                if ci
                else f"{counts}, incomplete"
            )
            cells.append(label + ("" if v["eligible"] else " †"))
        text.append(f"| {rate:.5f} | {passes} | " + " | ".join(cells) + " |")
    rejected = sum(not v["eligible"] for v in results.values())
    text += [
        "",
        f"{rejected} fits failed the movement screen"
        + (" (†)." if rejected else ".")
        + " The screen selected among these fits after one combat update with strategy held fixed. "
        "[Full screening data](../figures/online-combat-pilot/screen.csv).",
    ]
    parity = BASE / "expanded" / "cpu-parity.json"
    if parity.exists():
        text += [
            "",
            f"All {read(parity)['identical_worlds']} reference trajectories reproduced with three active evaluator processes and the same seven world partitions.",
        ]
    if verdict is None and (BASE / "expanded" / "shortlist.json").exists():
        candidate = read(BASE / "expanded" / "shortlist.json")["challenger"]
        text += [
            "",
            f"The screen nominated {candidate['target']:.0%} exploration, learning rate "
            f"{candidate['rate']:.5f} and {candidate['epochs']} passes for independent testing.",
        ]
    if (FIGURE / "screen.png").is_file():
        text += [
            "",
            "![Screening game wins by exploration, learning rate and training passes](../figures/online-combat-pilot/screen.png)",
        ]
    if verdict is not None:
        comp = verdict["comparison"]
        sel = verdict["challenger"]
        text += [
            "",
            f"The challenger used {sel['target']:.0%} exploration, learning rate {sel['rate']:.5f} and {sel['epochs']} pass(es). "
            f"On 8,192 separate games after three joint updates, challenger against original was {describe(comp)}. "
            f"Unresolved games were {comp['first']['faults']} faults and {comp['first']['cuts']} cuts for the original, "
            f"{comp['second']['faults']} faults and {comp['second']['cuts']} cuts for the challenger.",
            "",
            (
                "The challenger passed the replacement rule."
                if verdict["qualified"]
                else "The original recipe retained its place under the replacement rule."
            ),
            "",
            "| Short joint-training update | Original wins / 1,536 training worlds | Challenger wins / 1,536 training worlds |",
            "|---:|---:|---:|",
        ]
        for row in verdict["training"]:
            text.append(
                f"| {row['update']} | {row['reference']['wins']} | {row['challenger']['wins']} |"
            )
        diag = {}
        for role in ("reference", "challenger"):
            folder = BASE / "expanded" / ("short-" + role)
            diag[role] = [
                (read(folder / f"eval_{i:04d}.json")["summary"]["wins"]) for i in (0, 3)
            ]
        text += [
            "",
            f"Descriptive checks on the same 128 worlds at updates 0 and 3 gave {diag['reference'][0]} and {diag['reference'][1]} wins for the original, "
            f"{diag['challenger'][0]} and {diag['challenger'][1]} for the challenger. These checks had no role in selection.",
        ]
    return text


def ledger(results, verdict):
    label = "[Expanded combat pilot](experiments/2026-11-19-online-combat.md#expanded-pilot-results)"
    date = datetime.now(timezone.utc).date().isoformat()
    comparisons = [
        (
            "Three joint updates, selected challenger vs original",
            verdict["comparison"],
            "Confirmed",
        )
    ]
    for row in sorted(
        results.values(),
        key=lambda x: (
            x["setting"]["target"],
            x["setting"]["rate"],
            x["setting"]["epochs"],
        ),
    ):
        stat, cfg = row["comparison"], row["setting"]
        if stat["a"] != stat["b"]:
            comparisons.append(
                (
                    f"One fit at {cfg['target']:.0%} exploration, rate {cfg['rate']:.5f}, {cfg['epochs']} passes vs original fit",
                    stat,
                    "Exploratory",
                )
            )
    entries = []
    for test, stat, evidence in comparisons:
        low, high = stat["worst_case_ci95_pp"]
        outcome = (
            "Better"
            if low > 0
            else "Worse"
            if stat["complete"] and high < 0
            else "Inconclusive"
        )
        entries.append(
            f"| {date} | {label} | {test}, env 0.1.5.2 | {describe(stat)} | {evidence} | {outcome} |\n"
        )
    path = ROOT / "research/results.md"
    lines = [line for line in path.read_text().splitlines(True) if label not in line]
    start = next(i + 1 for i, line in enumerate(lines) if line.startswith("|---"))
    replace_note(path, "".join(lines[:start] + entries + lines[start:]))


def write_results(results, verdict=None):
    FIGURE.mkdir(parents=True, exist_ok=True)
    records = []
    for row in results.values():
        s = row["comparison"]
        d, ci = interval(s)
        m = row["movement"]
        cfg = row["setting"]
        records.append(
            {
                "exploration_target": cfg["target"],
                "combat_lr": cfg["rate"],
                "passes": cfg["epochs"],
                "worlds": s["n"],
                "wins": s["second"]["wins"],
                "reference_wins": s["first"]["wins"],
                "paired_gain_pp": d,
                "ci95_low_pp": ci[0] if ci else None,
                "ci95_high_pp": ci[1] if ci else None,
                "challenger_only": s["b_only"],
                "reference_only": s["a_only"],
                "paired_complete": s["paired_completed"],
                "faults": s["second"]["faults"],
                "cuts": s["second"]["cuts"],
                "eligible": row["eligible"],
                "ordinary_changed": m["combat"]["changed_rate"],
                "boss_changed": m["boss_specialist"]["changed_rate"],
                "ordinary_steps": row["steps"]["combat"]["steps"],
                "boss_steps": row["steps"]["boss_specialist"]["steps"],
            }
        )
    with (FIGURE / "screen.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(
            sorted(
                records,
                key=lambda x: (x["exploration_target"], x["combat_lr"], x["passes"]),
            )
        )
    p = ROOT / "research/experiments/2026-11-19-online-combat.md"
    old = p.read_text()
    if TITLE in old:
        start = old.index(TITLE)
        remainder = old[start + len(TITLE) :]
        following = re.search(r"^#{2,3} ", remainder, re.MULTILINE)
        old = old[:start] + (remainder[following.start() :] if following else "")
    following = re.search(r"^## (Conclusion|Log)\b", old, re.MULTILINE)
    if not following:
        raise ValueError(
            "experiment page has no insertion point for the expanded pilot"
        )
    start = following.start()
    replace_note(
        p,
        old[:start].rstrip()
        + "\n\n"
        + "\n".join(section(results, verdict))
        + "\n\n"
        + old[start:],
    )
    if verdict is not None:
        ledger(results, verdict)
