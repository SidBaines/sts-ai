"""Tabulate every fit and the independent short-trajectory test."""

import csv
from datetime import datetime, timezone
import re
import math
import statistics
import shutil
import subprocess
import sys

from report import ROOT, BASE, describe, read, replace_note
from sts_rl.fresh_study import atomic_json

FIGURE = ROOT / "research/figures/online-combat-pilot"
TITLE = "### Expanded pilot results"


def rows_from_disk():
    return {f.stem: read(f) for f in (BASE / "expanded" / "comparisons").glob("*.json")}


def interval(stat):
    return (
        stat.get("difference_pp", stat.get("complete_pair_difference_pp")),
        stat.get("ci95_pp", stat.get("complete_pair_ci95_pp")),
    )


def marginal_effects(results):
    """Average across other settings within a world before estimating sampling error."""
    outcomes = {}
    for label in results:
        d = read(BASE / "expanded" / "screen" / label / "summary.json")
        world_results = {
            r["world_seed"]: r["win"] if r["status"] == "completed" else None
            for r in d["records"]
        }
        if len(world_results) != 768 or len(d["records"]) != len(world_results):
            raise ValueError("bad marginal-effect world roster")
        outcomes[label] = world_results
    worlds = sorted(next(iter(outcomes.values())))
    if any(sorted(values) != worlds for values in outcomes.values()):
        raise ValueError("unpaired marginal-effect world rosters")
    table = []
    for factor, reference, levels in (
        ("target", 0.12, (0.05, 0.22)),
        ("rate", 0.00003, (0.00001, 0.00006)),
        ("epochs", 2, (1, 4)),
    ):
        control = [k for k, r in results.items() if r["setting"][factor] == reference]
        for level in levels:
            test = [k for k, r in results.items() if r["setting"][factor] == level]
            if len(control) != 9 or len(test) != 9:
                raise ValueError("incomplete factorial design")
            paired = []
            low = high = 0
            for w in worlds:
                a = [outcomes[k][w] for k in control]
                b = [outcomes[k][w] for k in test]
                low += sum(v is True for v in b) - sum(v is not False for v in a)
                high += sum(v is not False for v in b) - sum(v is True for v in a)
                if None not in a and None not in b:
                    paired.append(100 * (sum(b) - sum(a)) / 9)
            value = statistics.mean(paired) if paired else None
            half = (
                1.96 * statistics.stdev(paired) / math.sqrt(len(paired))
                if len(paired) > 1
                else None
            )
            table.append(
                dict(
                    factor=factor,
                    level=level,
                    reference=reference,
                    worlds=len(worlds),
                    complete=len(paired),
                    mean_pp=value,
                    ci95_pp=[value - half, value + half] if half is not None else None,
                    bounds_pp=[
                        100 * low / (9 * len(worlds)),
                        100 * high / (9 * len(worlds)),
                    ],
                )
            )
    return table


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
    identities = BASE / "expanded" / "weight-identities.json"
    if identities.exists():
        weights = read(identities)
        if weights["unique"] < weights["settings"]:
            text += [
                "",
                f"The {weights['settings']} settings produced {weights['unique']} distinct players. "
                "At 5% exploration and learning rate 0.00006, the per-update limit stopped both fighters early, "
                "giving the two- and four-pass settings identical weights.",
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
    path = BASE / "expanded" / "marginal-effects.json"
    if path.exists():
        text += [
            "",
            "Averaging over the other two settings gave these exploratory differences. Each interval uses independent worlds as its unit. A single training collection per exploration level leaves training-run variability unmeasured.",
            "",
            "| Factor | Compared with the original level | Mean win-rate change in points, 95% interval | Complete worlds / 768 |",
            "|---|---|---|---:|",
        ]
        for m in read(path):
            factor = m["factor"]
            if factor == "target":
                label, levels = (
                    "Exploration",
                    f"{m['level']:.0%} vs {m['reference']:.0%}",
                )
            elif factor == "rate":
                label, levels = (
                    "Learning rate",
                    f"{m['level']:.5f} vs {m['reference']:.5f}",
                )
            else:
                label, levels = "Passes", f"{m['level']} vs {m['reference']}"
            ci = m["ci95_pp"]
            effect = (
                f"{m['mean_pp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                if ci
                else "Insufficient complete worlds"
            )
            text.append(f"| {label} | {levels} | {effect} | {m['complete']} |")
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
    # A presentation or plotting failure must never postpone a verified scientific decision.
    folder = BASE / "expanded"
    identities = folder / "weight-identities.json"
    if identities.exists():
        for labels in read(identities)["identical"]:
            saved = []
            for label in labels:
                rows = read(folder / "screen" / label / "summary.json")["records"]
                saved.append(
                    {
                        r["world_seed"]: (
                            r["action_hash"],
                            r["terminal_hash"],
                            r["status"],
                            r["win"],
                        )
                        for r in rows
                    }
                )
            if any(scores != saved[0] for scores in saved[1:]):
                raise ValueError(
                    "identical trained players produced different screening trajectories"
                )
    if verdict is not None:
        record = folder / "verdict.json"
        if record.exists() and read(record) != verdict:
            raise ValueError("independent verdict has changed")
        if not record.exists():
            atomic_json(record, verdict)
    marginal_path = folder / "marginal-effects.json"
    if not marginal_path.exists():
        atomic_json(marginal_path, marginal_effects(results))
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
    if verdict is not None and not (FIGURE / "screen.png").exists():
        uv = shutil.which("uv")
        if uv is not None:
            with (folder / "plot.log").open("w") as log:
                plot = subprocess.run(
                    [
                        uv,
                        "run",
                        "--offline",
                        "--no-project",
                        "--with",
                        "matplotlib==3.11.2",
                        "--python",
                        sys.executable,
                        "python",
                        str(ROOT / "experiments/2026-11-19-online-combat/plot_grid.py"),
                        str(FIGURE),
                        str(FIGURE / "screen.png"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            atomic_json(
                folder / "plot-status.json",
                {"exit": plot.returncode, "image": (FIGURE / "screen.png").is_file()},
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
