"""File the finished study, including rejected gates and unresolved worlds."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from study import CHAMPION, ROOT, RUN, SHA256
from sts_rl.paired_endpoints import digest

PAGE = "2026-11-18-act3-boss-refits.md"
RESEARCH = ROOT / "research"


def save(path, text):
    temporary = path.with_name(path.name + ".finishing")
    temporary.write_text(text)
    temporary.replace(path)


def describe(r):
    gain = r.get("difference_pp", r.get("complete_pair_difference_pp"))
    ci = r.get("ci95_pp", r.get("complete_pair_ci95_pp"))
    effect = f"{gain:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "Interval unavailable"
    if not r["complete"]:
        lo, hi = r["full_cohort_gain_bounds_pp"]
        effect += f" on complete pairs, full-cohort bounds [{lo:+.2f}, {hi:+.2f}]"
    return (f"{r['second']['wins']:,} vs {r['first']['wins']:,} wins on {r['n']:,} assigned worlds, "
            f"{r['paired_completed']:,} complete pairs. {effect}")


def statistical_verdict(lo, hi):
    if lo > 0:
        return "Better"
    if hi < 0:
        return "Worse"
    return "No clear difference"


def record_finish(choice, primary, confirmation):
    date = datetime.now(ZoneInfo("Europe/London")).date().isoformat()
    selected = Path(choice["checkpoint"])
    qualified = choice["confirmation_passed"]
    if digest(selected) != choice["sha256"] or digest(CHAMPION) != SHA256:
        raise ValueError("Confirmation checkpoint changed before filing")
    champion = CHAMPION
    if qualified:
        champion = CHAMPION.with_name("champion_" + choice["sha256"][:12] + ".pt")
        if not champion.exists():
            temporary = champion.with_suffix(".copying")
            shutil.copyfile(selected, temporary)
            if digest(temporary) != choice["sha256"]:
                raise ValueError("Champion export changed during copy")
            temporary.replace(champion)
        if digest(champion) != choice["sha256"]:
            raise ValueError("Champion export has conflicting contents")
        receipt = dict(promoted=date, by="Act 3 boss study confirmation rule", sha256=choice["sha256"],
                       parent_sha256=SHA256, source=str(selected.relative_to(RUN.parent)),
                       environment="env-v0.1.5.2", evidence=describe(confirmation),
                       baseline_games=dict(zip((confirmation["a"], confirmation["b"]), confirmation["folders"])))
        save(champion.with_suffix(".json"), json.dumps(receipt, indent=2) + "\n")

    verdict = (f"The {choice['arm']} endpoint passed independent confirmation and became champion."
               if qualified else f"The {choice['arm']} endpoint missed the confirmation threshold. The starting champion remains in place.")
    findings = []
    for name, r in (("Selected endpoint vs starting champion", confirmation), ("Joint endpoint vs frozen control", primary)):
        interval = r.get("ci95_pp", r.get("complete_pair_ci95_pp"))
        rating = statistical_verdict(*interval) if interval else "Inconclusive"
        findings.append((name, describe(r), "Confirmed" if r is confirmation else "Planned", rating))
    for i in (8, 16):
        folder = RUN / f"refit{i}"
        decision = json.loads((folder / "decision.json").read_text())
        fights = json.loads((Path(decision["fight_panel"]) / "summary.json").read_text())
        for mode in ("unsalted", "resampled"):
            metric = fights["groups"][f"BOSS3:{mode}"]["win"]
            diff, se = 100*metric["paired_difference"], 100*metric["se"]
            lo, hi = diff-1.96*se, diff+1.96*se
            n = metric["worlds"]
            counts = f"{round(n*metric['replacement_mean'])} vs {round(n*metric['champion_mean'])} of {n} identical starts"
            findings.append((f"Update {i} boss fights, {mode}", f"{counts}, {diff:+.2f} points [{lo:+.2f}, {hi:+.2f}]",
                             "Planned", statistical_verdict(lo, hi)))
        if decision["full_game_bounds_pp"] is not None:
            r = json.loads((RUN / f"gate{i}.json").read_text())
            ci = r.get("ci95_pp", r.get("complete_pair_ci95_pp"))
            findings.append((f"Update {i} whole-game gate", describe(r), "Planned",
                             statistical_verdict(*ci) if ci else "Inconclusive"))
    page = RESEARCH / "experiments" / PAGE
    doc = page.read_text()
    design = doc.split("## Results")[0].splitlines()
    design[2] = f"**Status:** finished {date}. **Verdict:** {verdict}"
    table = ["| Comparison | Result |", "|---|---|"] + [f"| {a} | {b} |" for a,b,_,_ in findings]
    decisions = []
    for i in (8, 16):
        d = json.loads((RUN / f"refit{i}/decision.json").read_text())
        decisions.append(f"Update {i} " + ("adopted its specialist." if d["adopted"] else "retained the active head.")
                         + (" The saved-fight result did not trigger the full-game gate." if d["full_game_bounds_pp"] is None else ""))
    if len(set(primary["checkpoint_sha256"].values())) == 1:
        decisions.append("Both refits were rejected, so the two endpoints are the same trained policy.")
    log = doc[doc.index("## Log"):]
    save(page, "\n".join(design).rstrip()+"\n\n## Results\n\n"+"\n".join(table)+"\n\n"+" ".join(decisions)
         +"\n\n## Conclusion\n\n"+verdict+" Both arms used one training seed. Confirmation measures the selected policy's total gain over its parent."
         +"\n\n## Next\n\nReview the saved boss-fight outcomes and the paired endpoint result before choosing the next training change.\n\n"+log)
    ledger = RESEARCH / "results.md"
    old = ledger.read_text()
    if PAGE not in old:
        marker = "|---|---|---|---|---|---|\n"
        new_rows = [f"| {date} | [Act 3 boss refits](experiments/{PAGE}) | {a}, env 0.1.5.2 | {b} | {e} | {v} |"
                    for a,b,e,v in findings]
        save(ledger, old.replace(marker, marker+"\n".join(new_rows)+"\n", 1))
    readme = RESEARCH / "README.md"
    doc = readme.read_text()
    if "## Now" in doc and PAGE in doc.split("## Now")[1].split("## Next")[0]:
        doc = doc[:doc.index("## Now")] + f"## Now\n\nNothing running. [Act 3 boss refits](experiments/{PAGE}) finished. {verdict}\n\n## Next\n\nReview the matched boss and full-game results before starting another run.\n\n" + doc[doc.index("## Open questions"):]
        if qualified and champion.name not in doc:
            marker = "|---|---|---|\n"
            doc = doc.replace("Confirmed; current default", "Confirmed; previous champion")
            doc = doc.replace(marker, marker+f"| **Champion from {date}:** {choice['arm']} endpoint (`champions/{champion.name}`) | {describe(confirmation)} | Confirmed; current default |\n", 1)
        save(readme, doc)
    strategies = RESEARCH / "strategies.md"
    doc = strategies.read_text()
    strategy_result = f"- [Act 3 boss specialists](experiments/{PAGE}) produced {describe(primary)} against frozen combat. {verdict}"
    old_line = next((line for line in doc.splitlines() if PAGE in line), None)
    if old_line:
        save(strategies, doc.replace(old_line, strategy_result, 1))
    else:
        save(strategies, doc.replace("## Distilling strategy rules", strategy_result+"\n\n## Distilling strategy rules", 1))
    status = RESEARCH / "status.md"
    if PAGE in status.read_text():
        save(status, f"# Status\n\nNothing running. {verdict} See the [finished experiment](experiments/{PAGE}) and [progress log](log.md).\n")
