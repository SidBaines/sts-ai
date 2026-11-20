"""File the selection, retention and joint-learning results, including null comparisons."""

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from decisions import evidence, estimate
from sts_rl.fresh_study import atomic_json
from study import CFG, HERE, OUT, ROOT, RUNS, note, read

RESEARCH = ROOT / "research"
LINK = f"[Overnight single-fighter study](experiments/{HERE.name}.md)"


def put(path, body):
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(body)
    tmp.replace(path)


def status(message):
    path = RESEARCH / "status.md"
    if HERE.name in path.read_text():
        put(
            path,
            f"# Status\n\n{message} {LINK}. [Live dashboard](http://127.0.0.1:8795/) · "
            "[Progress log](http://127.0.0.1:8795/progress).\n",
        )


def front(champion, message):
    path = RESEARCH / "README.md"
    body = path.read_text()
    name = str(Path(champion).relative_to(RUNS))
    if name not in body:
        rows = body.splitlines()
        previous = next(
            i for i, line in enumerate(rows) if "| Confirmed" in line and "current default" in line
        )
        rows[previous] = rows[previous].replace("current default", "previous champion")
        record = read(Path(champion).with_suffix(".json"))
        rows.insert(
            previous,
            f"| **Overnight champion** (`{name}`) | {record['evidence']} | Confirmed, current default |",
        )
        body = "\n".join(rows) + "\n"
    old = body.split("## Now\n", 1)[1].split("\n## Next", 1)[0]
    if HERE.name in old:
        body = body.replace(old, f"\n{message} {LINK}.\n")
    put(path, body)


def record_retention():
    marker = OUT / "retention-filed.json"
    if marker.exists():
        return
    r = read(OUT / "retention-verdict.json")
    front(
        r["incumbent"],
        f"Matched joint learning follows the 49,152-world test. Single fighter vs champion: {evidence(r['full'])}.",
    )
    atomic_json(marker, {"incumbent": r["incumbent"]})


def labelled_comparisons():
    first = read(OUT / "first-selection.json")
    frozen = read(OUT / "frozen.json")
    ret = read(OUT / "retention-verdict.json")
    rl = read(OUT / "rl-selection.json")
    final = read(OUT / "verdict.json")
    rows = []

    def add(label, result, level):
        rows.append((label, result, level))

    for name, r in first["comparisons"].items():
        add(f"Development, {name} vs original", r, "Planned")
    if frozen["feedback_screen"]:
        add(
            "Development, student-played feedback vs first nominee",
            frozen["feedback_screen"],
            "Planned",
        )
    add("Full-game retention, single fighter vs early-refit champion", ret["full"], "Confirmed")
    for k, b in ret["bosses"].items():
        add(
            f"Boss retention, {k} starts",
            dict(**b, first={"wins": b["champion_wins"]}, second={"wins": b["student_wins"]}),
            "Confirmed" if k == "original" else "Planned",
        )
    labels = {
        "one-vs-two": "Update 24, one fighter vs two",
        "one-vs-one-parent": "Trained single fighter vs its parent",
        "two-vs-two-parent": "Trained two-fighter player vs its parent",
    }
    for k, label in labels.items():
        add(label, rl["comparisons"][k], "Confirmed")
    add("Independent confirmation, nominated endpoint vs incumbent", final["paired"], "Confirmed")
    return rows


def publish():
    result = read(OUT / "verdict.json")
    retention = read(OUT / "retention-verdict.json")
    first = read(OUT / "first-selection.json")
    frozen = read(OUT / "frozen.json")
    folder = RESEARCH / "figures" / HERE.name
    folder.mkdir(parents=True, exist_ok=True)
    comparisons = labelled_comparisons()
    table = [
        "| Comparison | Wins (candidate vs reference) | Complete / assigned | Paired change in points [95% interval] |",
        "|---|---:|---:|---:|",
    ]
    entries = []
    day = datetime.now(timezone.utc).date().isoformat()
    with (folder / "paired.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "comparison",
                "assigned",
                "complete_pairs",
                "reference_wins",
                "candidate_wins",
                "gain_pp",
                "lower95",
                "upper95",
            ]
        )
        for label, r, level in comparisons:
            value, bounds = estimate(r)
            a, b = r["first"]["wins"], r["second"]["wins"]
            interval = f"{value:+.2f} [{bounds[0]:+.2f}, {bounds[1]:+.2f}]" if bounds else "Unavailable"
            table.append(
                f"| {label} | {b:,} vs {a:,} | {r['paired_completed']:,} / {r['n']:,} | {interval} |"
            )
            writer.writerow(
                [label, r["n"], r["paired_completed"], a, b, value, *(bounds or (None, None))]
            )
            verdict = (
                "Better"
                if r["worst_case_ci95_pp"][0] > 0
                else "Worse"
                if r["complete"] and bounds and bounds[1] < 0
                else "Inconclusive"
            )
            entries.append(
                f"| {day} | {LINK} | {label}, env 0.1.5.2 | {evidence(r)} | {level} | {verdict} |"
            )
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    fig = folder / "paired.png"
    try:
        subprocess.run(
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
                str(HERE / "plot.py"),
                str(folder),
            ],
            check=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Plot export needs attention: {error}", flush=True)
    page = RESEARCH / "experiments" / (HERE.name + ".md")
    body = page.read_text().split("## Results\n", 1)[0].split("## Next\n", 1)[0].rstrip()
    lines = body.splitlines()
    lines[2] = f"**Status** — finished {day}. **Verdict** — {result['message']}"
    summary = (
        f"The selected single fighter was {frozen['chosen']}. Student-played feedback "
        f"{'ran' if first['feedback'] else 'was not triggered'}. Both retention margins met: {retention['retained']}."
    )
    records = [
        f"The {arm}-fighter player completed {read(OUT / ('rl-' + arm) / 'finished.json')['worlds']:,} assigned training games."
        for arm in ("one", "two")
    ]
    image = (
        f"\n\n![Paired win-rate differences](../figures/{HERE.name}/paired.png)"
        if fig.is_file()
        else ""
    )
    body = (
        "\n".join(lines) + "\n\n## Results\n\n" + result["message"] + "\n\n" + "\n".join(table) + image
    )
    body += (
        "\n\n## Conclusion\n\n"
        + summary
        + " "
        + " ".join(records)
        + " These estimates compare frozen players after one matched training seed."
    )
    body += f"\n\n## Next\n\nReview the difference in learning gains with Sid before choosing further training.\n\nCode in `experiments/{HERE.name}/`. Local run `{CFG['run']}`.\n"
    put(page, body)
    ledger = RESEARCH / "results.md"
    ls = [l for l in ledger.read_text().splitlines() if LINK not in l]
    i = ls.index("|---|---|---|---|---|---|") + 1
    ls[i:i] = reversed(entries)
    put(ledger, "\n".join(ls) + "\n")
    front(result["champion"], "Nothing running. " + result["message"])
    page_front = RESEARCH / "README.md"
    put(
        page_front,
        page_front.read_text().replace(
            "Compare both update-24 players and their starting parents, then independently confirm the nominee.",
            "Discuss the confirmed overnight results before selecting another run.",
        ),
    )
    strategies = RESEARCH / "strategies.md"
    body = strategies.read_text()
    if LINK not in body:
        body = body.replace(
            "\n## Joint sparse-win PPO",
            f"\n{summary} {result['message']} {LINK}.\n\n## Joint sparse-win PPO",
        )
        put(strategies, body)
    status("Nothing running. " + result["message"])
    atomic_json(OUT / "status.json", dict(stage="finished", champion=result["champion"], published=day))
    note("finished", result["message"], log=False)
