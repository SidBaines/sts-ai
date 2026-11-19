"""File the pilot, every frozen comparison and the live handoff."""

from datetime import datetime, timezone
from pathlib import Path
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "data" / "rl_runs" / "online_combat_from_9700"


def read(p):
    return json.loads(p.read_text())


def replace_note(path, contents):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(contents)
    os.replace(temporary, path)


def describe(x):
    d = x.get("difference_pp", x.get("complete_pair_difference_pp"))
    ci = x.get("ci95_pp", x.get("complete_pair_ci95_pp"))
    diff = (
        f"{d:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "no paired interval"
    )
    return (
        f"{x['second']['wins']:,} vs {x['first']['wins']:,} on {x['n']:,} assigned worlds, {diff}, "
        f"{x['paired_completed']:,} complete pairs"
    )


def finish(final):
    first = final["primary"]
    confirmation = final["confirmation"]
    selected = read(BASE / "fit-selection.json")
    exploration = read(BASE / "exploration-selection.json")
    text = [
        "## Results",
        "",
        f"The chosen exploration level was {exploration['choice']} and the selected combat learning rate was "
        f"{selected['selected']['rate']:g} with {selected['selected']['epochs']} pass(es).",
        "",
        "| Frozen full-game comparison | Paired wins and 95% interval |",
        "|---|---|",
        "| Joint against frozen, primary | " + describe(first["comparison"]) + " |",
    ]
    for key, x in confirmation.items():
        text.append(
            f"| {x['b'].title()} against {x['a']}, independent confirmation | {describe(x)} |"
        )
    text += [
        "",
        "### Exploration pilot",
        "",
        "| Target | Wins / 768 | Reached Act 3 / 768 | Lower-ranked move, ordinary / boss | Unknown |",
        "|---|---:|---:|---|---:|",
    ]
    for f in sorted(BASE.glob("explore-*/summary.json")):
        d = read(f)
        c = d["counts"]
        e = d["exploration"]
        text.append(
            f"| {d['config']['target']:.1%} | {c['win']} | {c['act3']} | {e['combat']['actual_rate']:.1%} / "
            f"{e['boss_specialist']['actual_rate']:.1%} | {c['unknown']} |"
        )
    text += [
        "",
        "### Combat fitting pilot",
        "",
        "| Rate | Passes | Greedy games against frozen player | Changed choices, ordinary / boss | Passed movement screen |",
        "|---:|---:|---|---|---|",
    ]
    for entry in selected["results"].values():
        f = entry["fit"]
        m = f["movement"]
        text.append(
            f"| {f['rate']:g} | {f['epochs']} | {describe(entry['comparison'])} | {m['combat']['changed_rate']:.1%} / "
            f"{m['boss_specialist']['changed_rate']:.1%} | {'yes' if f['eligible'] else 'no'} |"
        )
    expanded = BASE / "expanded" / "verdict.json"
    if expanded.exists():
        from expanded_report import rows_from_disk, section

        text.extend(["", *section(rows_from_disk(), read(expanded))])
    binding = BASE / "launch-settings.json"
    if binding.exists():
        selected_main = read(binding)
        cfg = selected_main["selected"]
        text.extend(
            [
                "",
                f"The actual 24-update comparison used {cfg['target']:.0%} combat exploration, "
                f"combat learning rate {cfg['rate']:.5f}, and {cfg['epochs']} pass(es). "
                f"The selection was fixed from the {selected_main['source']} at first launch.",
            ]
        )
    nominee = first["nominee"]
    verdict = (
        f"{nominee.title()} won the independent schedule comparison."
        if final["schedule_advantage"]
        else "The full-game evidence does not establish a schedule advantage."
    )
    parent = (
        "The selected endpoint also beat its original starting champion."
        if final["beats_start"]
        else "The selected endpoint did not establish a gain over its starting champion."
    )
    text += [
        "",
        "## Conclusion",
        "",
        verdict
        + " "
        + parent
        + " This was one matched training seed. The pilot selected among a few settings.",
        "",
        "## Next",
        "",
        "Check replication on a new training seed before generalising the schedule result.",
    ]
    p = ROOT / "research/experiments/2026-11-19-online-combat.md"
    old = p.read_text()
    old = (
        old.replace("**Status:** planned 19 November.", "**Status:** finished.")
        if "**Status:** planned" in old
        else old.replace("**Status:** running.", "**Status:** finished.")
    )
    before, tail = old.split("## Results", 1)
    _, log = tail.split("## Log", 1)
    replace_note(p, before + "\n".join(text) + "\n\n## Log" + log)
    p = ROOT / "research/results.md"
    s = p.read_text()
    lines = [
        line
        for line in s.splitlines(True)
        if "| [Online combat](experiments/2026-11-19-online-combat.md)" not in line
    ]
    index = next(i + 1 for i, line in enumerate(lines) if line.startswith("|---"))
    date = datetime.now(timezone.utc).date().isoformat()
    results = [("Joint vs frozen, primary", first["comparison"], "Planned")] + [
        (k.replace("_vs_", " compared with "), x, "Confirmed")
        for k, x in confirmation.items()
    ]
    for name, x, evidence in reversed(results):
        ci = x["worst_case_ci95_pp"]
        best = (
            "Better"
            if ci[0] > 0
            else "Worse"
            if x["complete"] and ci[1] < 0
            else "No clear difference"
        )
        lines.insert(
            index,
            f"| {date} | [Online combat](experiments/2026-11-19-online-combat.md) | {x['b']} vs {x['a']}, {name}, env 0.1.5.2 | {describe(x)} | {evidence} | {best} |\n",
        )
    replace_note(p, "".join(lines))
    p = ROOT / "research/strategies.md"
    s = p.read_text()
    heading = "## Joint sparse-win PPO"
    if heading in s:
        section = "## Learning combat on each strategy update"
        if section in s:
            start, rest = s.split(section, 1)
            s = start + heading + rest.split(heading, 1)[1]
        s = s.replace(
            heading,
            "## Learning combat on each strategy update\n\n"
            + verdict
            + " "
            + parent
            + " [Matched online combat study](experiments/2026-11-19-online-combat.md).\n\n"
            + heading,
        )
        replace_note(p, s)
    p = ROOT / "research/README.md"
    s = p.read_text()
    a, b = s.split("## Now", 1)
    _, b = b.split("## Next", 1)
    replace_note(
        p,
        a
        + "## Now\n\nThe [online combat comparison](experiments/2026-11-19-online-combat.md) has finished. "
        + verdict
        + " "
        + parent
        + "\n\n## Next"
        + b,
    )
    replace_note(
        ROOT / "research/status.md",
        "# Status\n\nNothing running in the online combat study. "
        + verdict
        + " "
        + parent
        + " [Experiment and full results](experiments/2026-11-19-online-combat.md).\n",
    )
