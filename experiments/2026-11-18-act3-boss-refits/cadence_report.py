"""File the matched cadence comparison and every gate, including ties and rejections."""
import json
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from finish import PAGE, RESEARCH, describe, save, statistical_verdict
from heartbeat import EVERY_TWO
from monitor import digest, lines, read
from study import CHAMPION, RUN

LINK = f"[Act 3 refit cadence](experiments/{PAGE}#every-two-updates)"
TITLE = "Every two updates"


def current_activity(now, next_action):
    readme = RESEARCH / "README.md"
    text = readme.read_text()
    text = (text[:text.index("## Now")] + f"## Now\n\n{now}\n\n## Next\n\n{next_action}\n\n"
            + text[text.index("## Open questions"):])
    save(readme, text)
    save(RESEARCH / "status.md", f"# Status\n\n{now}\n\n{next_action}\n\n"
         "[Training dashboard](http://127.0.0.1:8795/) and [live progress and heartbeat](http://127.0.0.1:8795/progress).\n")


def starting():
    current_activity(f"{LINK} is running from the original November champion. Strategy trains for 24 updates at "
                     "0.0001. Each even update through 22 can admit a boss specialist after the two gates.",
                     "Compare the two schedules at updates 8 and 24, and compare both endpoints with frozen combat. "
                     "The 15-minute heartbeat covers both studies. The active run is `joint_a3_every2_from_0f52`.")


def fight_comparison(metric):
    n = metric["worlds"]
    new, old = [round(n*metric[key]) for key in ("replacement_mean", "champion_mean")]
    diff, margin = 100*metric["paired_difference"], 196*metric["se"]
    interval = (diff-margin, diff+margin)
    result = f"{new} vs {old} of {n}, {diff:+.2f} points [{interval[0]:+.2f}, {interval[1]:+.2f}]"
    return result, statistical_verdict(*interval)


def finish(monitor, early, endpoints, confirmation, selection):
    date = datetime.now(ZoneInfo("Europe/London")).date().isoformat()
    r = endpoints["every2_vs_every8"]
    result = describe(r)
    records = []
    def game(name, item, evidence):
        interval = item.get("ci95_pp", item.get("complete_pair_ci95_pp"))
        records.append((name, describe(item), evidence, statistical_verdict(*interval) if interval else "Inconclusive"))
    for key,item in endpoints.items():
        game(f"Update 24, {item['b']} vs {item['a']}", item, "Planned")
    for item in early.values():
        game("Update 8, every two vs every eight", item, "Planned")
    for item in (confirmation or {}).values():
        game(f"Independent confirmation, {item['b']} vs {item['a']}", item, "Confirmed")

    table = ["| Update | Original boss entries | Resalted entries | Full-game gate | Decision |", "|---|---|---|---|---|"]
    adopted = []
    for i in range(2,24,2):
        decision = read(EVERY_TWO / f"refit{i}/decision.json")
        groups = read(Path(decision["fight_panel"]) / "summary.json")["groups"]
        cells = []
        for group, label in (("BOSS3:unsalted","original entries"), ("BOSS3:resampled","resalted entries")):
            text, rating = fight_comparison(groups[group]["win"])
            records.append((f"Update {i} specialist, {label}", text, "Planned", rating))
            cells.append(text)
        full = "Did not advance"
        if decision["full_game_bounds_pp"] is not None:
            item = read(EVERY_TWO / f"gate{i}.json")
            full = describe(item)
            game(f"Update {i} whole-game gate", item, "Planned")
        if decision["adopted"]:
            adopted.append(str(i))
        table.append(f"| {i} | {cells[0]} | {cells[1]} | {full} | {'Adopted' if decision['adopted'] else 'Kept active head'} |")

    ck = Path(selection["checkpoint"])
    if selection["promotion_qualified"]:
        destination = CHAMPION.with_name("champion_" + selection["sha256"][:12] + ".pt")
        if not destination.exists():
            tmp = destination.with_suffix(".copying")
            shutil.copyfile(ck, tmp)
            if digest(tmp) != selection["sha256"]:
                raise ValueError("Incomplete confirmed champion export")
            tmp.replace(destination)
        if digest(destination) != selection["sha256"]:
            raise ValueError("Champion path already contains a different checkpoint")
        c = confirmation["every2_vs_incumbent"]
        receipt = dict(promoted=date, by="Independent cadence confirmation rule", sha256=selection["sha256"],
                       parent_sha256=digest(CHAMPION), source=str(ck.relative_to(EVERY_TWO.parent)),
                       environment="env-v0.1.5.2", evidence=describe(c), baseline_games=dict(zip((c["a"],c["b"]),c["folders"])))
        save(destination.with_suffix(".json"), json.dumps(receipt, indent=2)+"\n")
        conclusion = "The every-two endpoint passed independent confirmation and became champion."
        readme = RESEARCH / "README.md"
        doc = readme.read_text()
        if destination.name not in doc:
            doc = doc.replace("Confirmed; current default", "Confirmed; previous champion")
            header = "|---|---|---|\n"
            save(readme, doc.replace(header, header + f"| **Champion from {date}:** every-two cadence "
                                     f"(`champions/{destination.name}`) | {describe(c)} | Confirmed; current default |\n", 1))
    else:
        conclusion = ("The incumbent retained the title after independent confirmation." if confirmation else
                      "The every-two endpoint did not clear both comparisons required for independent confirmation.")
    page = RESEARCH / "experiments" / PAGE
    doc = page.read_text()
    section = "### Every-two results"
    headline = (f"The every-two endpoint recorded {result} against every eight. "
                f"Refits admitted at {', '.join(adopted) if adopted else 'none of the eleven opportunities'}. {conclusion}")
    if section not in doc:
        show = [f"- {name}: {value}." for name,value,_,_ in records if name.startswith(("Update 24, every","Update 8, every","Independent confirmation"))]
        content = f"{section}\n\n{headline}\n\n" + "\n".join(show) + "\n\n" + "\n".join(table)
        sparse_events = lines(RUN / "progress.jsonl")
        sparse_hours = (max(x["t"] for x in sparse_events) - min(x["t"] for x in sparse_events)) / 3600
        dense_hours = ((EVERY_TWO / "verdict.json").stat().st_mtime - read(EVERY_TWO / "started.json")["t"]) / 3600
        content += (f"\n\nEleven opportunities used 5,632 scheduled teacher worlds, compared with 1,024 for two opportunities. "
                    f"Elapsed wall time including evaluations and any power pauses was {dense_hours:.2f} hours for every two "
                    f"and {sparse_hours:.2f} hours for the original study including its frozen control. This is one matched training seed.\n\n")
        doc = doc.replace("## Conclusion", content+"## Conclusion", 1)
        save(page, doc)
    ledger = RESEARCH / "results.md"
    doc = ledger.read_text()
    if LINK not in doc:
        rows = [f"| {date} | {LINK} | {name}, env 0.1.5.2 | {value} | {evidence} | {verdict} |"
                for name,value,evidence,verdict in records]
        marker = "|---|---|---|---|---|---|\n"
        save(ledger, doc.replace(marker, marker+"\n".join(rows)+"\n", 1))
    strategies = RESEARCH / "strategies.md"
    doc = strategies.read_text()
    if LINK not in doc:
        marker = "## Distilling strategy rules"
        save(strategies, doc.replace(marker, f"- {LINK}: {headline}\n\n"+marker, 1))
    current_activity(f"Nothing running. {LINK} finished. {headline}",
                     "Use the paired boss and endpoint results to choose the next training experiment.")
