"""File the planned comparisons, all six gate outcomes, and any confirmed promotion."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from study import CHAMPION, CHAMPION_SHA, HERE, ROOT, RUN, SCHEDULES
from sts_rl.boss_refits import confirmed_schedule_advantage, read
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import digest

PAGE = "2026-11-19-refit-timing.md"
RESEARCH = ROOT / "research"
LINK = f"[Early versus spaced refits](experiments/{PAGE})"


def put(path, text):
    tmp = path.with_name(path.name + ".finishing")
    tmp.write_text(text)
    tmp.replace(path)


def values(result):
    if result["complete"]:
        return result["difference_pp"], result["ci95_pp"]
    return result["complete_pair_difference_pp"], result["complete_pair_ci95_pp"]


def result_text(r):
    diff, ci = values(r)
    desc = f"{r['second']['wins']:,} vs {r['first']['wins']:,} wins, {r['paired_completed']:,}/{r['n']:,} complete pairs"
    if ci:
        desc += f", {diff:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    if not r["complete"]:
        lo, hi = r["full_cohort_gain_bounds_pp"]
        desc += f" on complete pairs. Full-cohort gain bounds [{lo:+.2f}, {hi:+.2f}]"
    return desc


def verdict(ci):
    if not ci:
        return "Inconclusive"
    if ci[0] > 0:
        return "Better"
    if ci[1] < 0:
        return "Worse"
    return "No clear difference"


def finish(primary, confirmation, choice):
    arm = choice["arm"]
    other = "spaced" if arm == "early" else "early"
    direct = confirmation[f"{arm}_vs_{other}"]
    against_parent = confirmation[f"{arm}_vs_champion"]
    confirmed = confirmed_schedule_advantage(primary, direct)
    promoted = against_parent["worst_case_ci95_pp"][0] > 0
    if digest(CHAMPION) != CHAMPION_SHA or digest(Path(choice["checkpoint"])) != choice["sha256"]:
        raise ValueError("Frozen policy changed before filing")
    date = datetime.now(ZoneInfo("Europe/London")).date().isoformat()
    decisions = []
    table = [
        "| Schedule and update | Original fights | Reseeded fights | Full-game gate | Deployed |",
        "|---|---|---|---|---|",
    ]
    full = [
        ("Endpoint selection", primary, "Planned"),
        ("Independent schedule comparison", direct, "Confirmed"),
        ("Early vs starting champion", confirmation["early_vs_champion"], "Confirmed"),
        ("Spaced vs starting champion", confirmation["spaced_vs_champion"], "Confirmed"),
    ]
    ledger = [
        (f"{label}, {r['b']} vs {r['a']}", result_text(r), e, verdict(values(r)[1]))
        for label, r, e in full
    ]
    for name, schedule in SCHEDULES.items():
        for i in schedule:
            gate = read(RUN / name / f"refit{i}/decision.json")
            summary = read(Path(gate["fight_panel"]) / "summary.json")
            cells = []
            for kind, label in (("unsalted", "Original fights"), ("resampled", "Reseeded fights")):
                m = summary["groups"][f"BOSS3:{kind}"]["win"]
                n = m["worlds"]
                gain = 100 * m["paired_difference"]
                margin = 196 * m["se"]
                ci = [gain - margin, gain + margin]
                desc = (
                    f"{round(n * m['replacement_mean'])} vs {round(n * m['champion_mean'])} of {n}, "
                    f"{gain:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                )
                cells.append(desc)
                ledger.append(
                    (f"{name.capitalize()} update {i}, {label.lower()}", desc, "Planned", verdict(ci))
                )
            game = "Did not advance"
            if gate["full_game_bounds_pp"] is not None:
                r = read(RUN / name / f"gate{i}.json")
                game = result_text(r)
                ledger.append(
                    (
                        f"{name.capitalize()} update {i}, whole games",
                        game,
                        "Planned",
                        verdict(values(r)[1]),
                    )
                )
            decisions.append(dict(arm=name, iteration=i, adopted=gate["adopted"]))
            table.append(
                f"| {name.capitalize()} {i} | {cells[0]} | {cells[1]} | {game} | {'Yes' if gate['adopted'] else 'No'} |"
            )
    shared = choice["endpoint_sha256"]["early"] == choice["endpoint_sha256"]["spaced"]
    if shared:
        finding = "Both schedules finished with the same checkpoint."
    elif confirmed:
        finding = (
            f"The {arm} schedule beat the {other} schedule on selection and independent confirmation."
        )
    else:
        finding = "The two panels did not establish a repeatable schedule advantage."
    outcome = (
        f"The {arm} endpoint became champion."
        if promoted
        else "The starting champion retained the title."
    )
    final = dict(
        **choice,
        same_endpoint=shared,
        confirmed_schedule_advantage=confirmed,
        promoted=promoted,
        decisions=decisions,
    )
    atomic_json(RUN / "verdict.json", final)
    if promoted:
        destination = CHAMPION.with_name("champion_" + choice["sha256"][:12] + ".pt")
        if not destination.exists():
            temp = destination.with_suffix(".copying")
            shutil.copyfile(choice["checkpoint"], temp)
            if digest(temp) != choice["sha256"]:
                raise ValueError("Incomplete exported champion")
            temp.replace(destination)
        if digest(destination) != choice["sha256"]:
            raise ValueError("Champion destination conflict")
        atomic_json(
            destination.with_suffix(".json"),
            dict(
                promoted=date,
                by="Equal-budget timing confirmation rule",
                source=str(Path(choice["checkpoint"]).relative_to(RUN.parent)),
                sha256=choice["sha256"],
                parent_sha256=CHAMPION_SHA,
                environment="env-v0.1.5.2",
                evidence=result_text(against_parent),
                baseline_games=dict(
                    zip((against_parent["a"], against_parent["b"]), against_parent["folders"])
                ),
            ),
        )
    figure = RESEARCH / "figures/refit-timing"
    figure.mkdir(parents=True, exist_ok=True)
    with (figure / "full-games.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "panel",
                "candidate",
                "baseline",
                "pairs",
                "assigned",
                "candidate_wins",
                "baseline_wins",
                "gain_pp",
                "lower95",
                "upper95",
            ]
        )
        for label, r, _ in full:
            gain, ci = values(r)
            writer.writerow(
                [
                    label,
                    r["b"],
                    r["a"],
                    r["paired_completed"],
                    r["n"],
                    r["second"]["wins"],
                    r["first"]["wins"],
                    gain,
                    *(ci or ["", ""]),
                ]
            )
    subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--with",
            "matplotlib==3.11.2",
            "--python",
            sys.executable,
            "python",
            str(HERE / "plot.py"),
            str(figure),
        ],
        check=True,
        cwd=ROOT,
    )
    page = RESEARCH / "experiments" / PAGE
    doc = page.read_text()
    first = doc[: doc.index("## Results")].strip().splitlines()
    first[2] = f"**Status:** finished {date}. **Verdict:** {finding} {outcome}"
    results = [
        "## Results",
        "",
        finding,
        "",
        "| Panel | Candidate vs baseline | Paired full-game result |",
        "|---|---|---|",
    ]
    results.extend(
        f"| {label} | {r['b'].capitalize()} vs {r['a']} | {result_text(r)} |" for label, r, _ in full
    )
    results.extend(
        [
            "",
            "![Paired full-game comparisons](../figures/refit-timing/full-games.png)",
            "",
            "Each gate compares the proposed specialist with the combat policy active at that update.",
            "",
            *table,
        ]
    )
    conclusion = (
        f"## Conclusion\n\n{finding} {outcome} Both schedules received 1,536 teacher worlds and the same 73,728 strategy training worlds. "
        "The comparison covers one training seed from this parent.\n\n## Next\n\n"
        "Use the confirmed comparisons and the accepted head changes to choose the next training step.\n\n"
    )
    put(
        page,
        "\n".join(first)
        + "\n\n"
        + "\n".join(results)
        + "\n\n"
        + conclusion
        + doc[doc.index("## Log") :],
    )
    target = RESEARCH / "results.md"
    doc = target.read_text()
    if LINK not in doc:
        header = "|---|---|---|---|---|---|\n"
        put(
            target,
            doc.replace(
                header,
                header
                + "\n".join(
                    f"| {date} | {LINK} | {a}, env 0.1.5.2 | {v} | {e} | {j} |" for a, v, e, j in ledger
                )
                + "\n",
                1,
            ),
        )
    target = RESEARCH / "strategies.md"
    doc = target.read_text()
    if LINK not in doc:
        place = "## Distilling strategy rules"
        put(
            target,
            doc.replace(
                place,
                f"- {LINK} compared equally funded schedules from the November champion. {finding} {result_text(direct)}.\n\n"
                + place,
                1,
            ),
        )
    target = RESEARCH / "README.md"
    doc = target.read_text()
    now = f"Nothing running. {LINK} finished. {finding} {outcome}"
    if promoted and destination.name not in doc:
        doc = doc.replace("Confirmed; current default", "Confirmed; previous champion")
        h = "|---|---|---|\n"
        doc = doc.replace(
            h,
            h
            + f"| **Champion from {date}:** {arm} refits (`champions/{destination.name}`) | {result_text(against_parent)} | Confirmed; current default |\n",
            1,
        )
    if PAGE in doc[doc.index("## Now") : doc.index("## Next")]:
        put(
            target,
            doc[: doc.index("## Now")]
            + f"## Now\n\n{now}\n\n## Next\n\nReview the [matched timing results](experiments/{PAGE}#results) before the next run.\n\n"
            + doc[doc.index("## Open questions") :],
        )
    put(
        RESEARCH / "status.md",
        f"# Status\n\n{now} The [progress log](log.md) records the completed stages.\n",
    )
    # End retired training lines at their actual last PPO update.
    for path in RUN.glob("*/training-plan.json"):
        logs = (
            [json.loads(line) for line in (path.parent / "log.jsonl").read_bytes().splitlines()]
            if (path.parent / "log.jsonl").exists()
            else []
        )
        end = max((row["iteration"] for row in logs if row["event"] == "train"), default=0)
        if end:
            atomic_json(path, {**read(path), "iterations": end})
