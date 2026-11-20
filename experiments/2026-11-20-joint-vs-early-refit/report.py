"""Follow the two queued evaluations and publish the frozen comparison."""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import compare, digest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "data/rl_runs"
PLAN = json.loads((HERE / "panel.json").read_text())
OUT = RUNS / PLAN["run"]
N = PLAN["n"]
JOBS = {arm: f"champion-match-{arm}" for arm in PLAN["players"]}


def put(path, body):
    temporary = path.with_name(path.name + ".updating")
    temporary.write_text(body)
    temporary.replace(path)


def update(message, stage, *, log=False):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%d %b %H:%M GMT")
    event = dict(stamp=stamp, t=now.timestamp(), run=PLAN["run"], stage=stage, message=message)
    atomic_json(RUNS / "study_heartbeat.json", event)
    if log:
        path = ROOT / "research/log.md"
        title, entries = path.read_text().split("\n", 1)
        if not any(line.endswith(" — " + message) for line in entries.splitlines()):
            put(path, f"{title}\n\n- {stamp} — {message}\n" + entries.lstrip("\n"))
    print(stamp, message, flush=True)


def game_count(arm):
    progress = OUT / arm / "progress.jsonl"
    if progress.exists():
        for line in reversed(progress.read_text().splitlines()):
            try:
                return json.loads(line)["recorded"]
            except json.JSONDecodeError:
                continue  # A writer can be partway through the final line.
    return 0


def wait_for_games():
    last_counts, last_quarter = None, -1
    while True:
        queue = json.loads((RUNS / "queue.json").read_bytes())["runs"]
        jobs = {j["name"]: j for j in queue}
        counts = {arm: game_count(arm) for arm in JOBS}
        finished = []
        for arm, name in JOBS.items():
            job = jobs[name]
            status = job["status"]
            if status in ("done", "exited") and (OUT / arm / "summary.json").is_file():
                finished.append(arm)
                continue
            if status not in ("pending", "running", "paused"):
                raise RuntimeError(f"{name} ended with status {status} before producing a usable result")
            if status == "running":
                progress = OUT / arm / "progress.jsonl"
                last = max(job["started"], job.get("resumed_at", 0),
                           progress.stat().st_mtime if progress.exists() else 0)
                if time.time() - last > 600:
                    raise RuntimeError(f"{name} has made no recorded progress for ten active minutes")
        if len(finished) == len(JOBS):
            return
        if counts != last_counts:
            quarter = min(counts.values()) * 4 // N
            update(f"Joint player {counts['joint']:,} of {N:,} worlds. "
                   f"Early-refit champion {counts['early']:,} of {N:,}.",
                   "playing", log=quarter > last_quarter)
            last_counts, last_quarter = counts, quarter
        time.sleep(20)


def archive_winner(players, evidence, day):
    info = PLAN["players"]["joint"]
    destination = RUNS / "champions" / f"champion_{info['sha256'][:12]}.pt"
    if not destination.exists():
        temporary = destination.with_suffix(".copying")
        shutil.copyfile(players["joint"], temporary)
        if digest(temporary) != info["sha256"]:
            raise ValueError("Incomplete champion copy")
        temporary.replace(destination)
    if digest(destination) != info["sha256"]:
        raise ValueError("Archived champion differs from the evaluated player")
    atomic_json(destination.with_suffix(".json"), dict(
        promoted=day, by="Independent head-to-head against the early-refit champion",
        source=info["checkpoint"], sha256=info["sha256"], environment="env-v0.1.5.2",
        parent_sha256=json.loads((players["joint"].parents[1] / "production-plan.json").read_text())["parent_sha256"],
        replaced_sha256=PLAN["players"]["early"]["sha256"], evidence=evidence,
        baseline_games={arm: str(OUT / arm) for arm in JOBS},
    ))
    return str(destination.relative_to(RUNS))


def file_result(result, evidence, decision, promoted, champion, day):
    from bosses import boss_table

    research = ROOT / "research"
    page = research / "experiments" / (HERE.name + ".md")
    link = f"[Joint vs early refits](experiments/{page.name})"
    early, joint = result["first"], result["second"]
    for arm, label, counts in (("early", "Early refits", early), ("joint", "Joint training", joint)):
        completed = counts["completed"]
        label = f"{label} · {HERE.name[:10]} head-to-head"
        if completed < N:
            label += f" ({N - completed} unresolved)"
        atomic_json(RUNS / "fresh_eval" / f"{HERE.name}-{arm}.json", dict(
            label=label, assigned=N, summary=dict(n=completed, wins=counts["wins"], mode="greedy",
                                                 mean_floor=counts["mean_floor"])))
    interval = result.get("ci95_pp", result.get("complete_pair_ci95_pp"))
    title, remainder = page.read_text().split("## Next\n", 1)
    lines = title.split("## Results\n")[0].rstrip().splitlines()
    lines[2] = f"**Status** — finished {day}. **Verdict** — {decision}"
    results = (f"## Results\n\n{evidence}\n\n"
               "| Player | Wins on assigned worlds | Won only by this player | Simulator faults | Decision limits |\n"
               "|---|---:|---:|---:|---:|\n"
               f"| Early-refit champion | {early['wins']:,} ({100*early['wins']/N:.2f}%) | "
               f"{result['a_only']:,} | {early['faults']} | {early['cuts']} |\n"
               f"| Joint player | {joint['wins']:,} ({100*joint['wins']/N:.2f}%) | "
               f"{result['b_only']:,} | {joint['faults']} | {joint['cuts']} |\n\n"
               f"![Paired win-rate difference and 95% interval](../figures/{HERE.name}/win-difference.png)\n\n")
    results += boss_table(OUT, research / "figures" / HERE.name, result["paired_completed"])
    interpretation = ("The fresh games did not establish an advantage for either player. "
                      if interval[0] <= 0 <= interval[1] else "")
    results += (f"## Conclusion\n\n{decision} {interpretation}The lower 95% bound with unresolved outcomes assigned "
               f"against joint was {result['worst_case_ci95_pp'][0]:+.2f} percentage points. "
               "The interval measures variation across worlds for these two frozen players.\n")
    commits = sorted({json.loads((OUT / arm / "run.json").read_text())["git_commit"][:8] for arm in JOBS})
    remainder = remainder.split("\nEvaluator source", 1)[0].rstrip()
    put(page, "\n".join(lines) + "\n\n" + results + "\n## Next\n" + remainder
        + "\n\nEvaluator source recorded at " + ", ".join(f"`{c}`" for c in commits) + ".\n")

    ledger = research / "results.md"
    rows = [row for row in ledger.read_text().splitlines() if link not in row]
    verdict = "Better" if promoted else "Worse" if result["complete"] and interval[1] < 0 else "Inconclusive"
    row = f"| {day} | {link} | Full-game wins, joint vs early refits, env 0.1.5.2 | {evidence} | Confirmed | {verdict} |"
    rows.insert(rows.index("|---|---|---|---|---|---|") + 1, row)
    put(ledger, "\n".join(rows) + "\n")

    front = research / "README.md"
    body = front.read_text()
    old_now = body.split("## Now\n", 1)[1].split("\n## Next", 1)[0]
    if HERE.name in old_now:
        body = body.replace(old_now, f"\nNothing running. {decision} {evidence} {link}.\n")
    if promoted and champion not in body:
        rows = body.splitlines()
        previous = next(i for i, row in enumerate(rows) if PLAN["players"]["early"]["checkpoint"] in row)
        rows[previous] = rows[previous].replace("current default", "previous champion")
        rows.insert(previous, f"| **Joint combat and strategy champion** ({day}, `{champion}`) | "
                    f"{evidence} | Confirmed, current default |")
        body = "\n".join(rows) + "\n"
    put(front, body)
    strategies = research / "strategies.md"
    body = strategies.read_text()
    if link not in body:
        body = body.replace("\n## Joint sparse-win PPO", f"\n{decision} {evidence} {link}.\n\n## Joint sparse-win PPO")
        put(strategies, body)


def report():
    players = {arm: RUNS / spec["checkpoint"] for arm, spec in PLAN["players"].items()}
    for arm, checkpoint in players.items():
        if digest(checkpoint) != PLAN["players"][arm]["sha256"]:
            raise ValueError(f"Frozen {arm} checkpoint differs from the study plan")
    result = compare(OUT / "early", OUT / "joint", PLAN["world_start"], "early", "joint",
                     players["early"], players["joint"], N)
    delta = result.get("difference_pp", result.get("complete_pair_difference_pp"))
    interval = result.get("ci95_pp", result.get("complete_pair_ci95_pp"))
    if interval is None:
        raise RuntimeError("Too few complete pairs to estimate an interval")
    promoted = result["worst_case_ci95_pp"][0] > 0
    evidence = (f"Joint {result['second']['wins']:,} wins, early refits {result['first']['wins']:,} "
                f"on {N:,} worlds. Joint minus early {delta:+.2f} percentage points "
                f"(95% interval {interval[0]:+.2f} to {interval[1]:+.2f}, "
                f"{result['paired_completed']:,} complete pairs).")
    decision = "Joint is the new champion." if promoted else "The early-refit player retains the title."
    finished_at = max((OUT / arm / "summary.json").stat().st_mtime for arm in JOBS)
    day = datetime.fromtimestamp(finished_at, timezone.utc).date().isoformat()
    champion = archive_winner(players, evidence, day) if promoted else PLAN["players"]["early"]["checkpoint"]
    atomic_json(OUT / "verdict.json", dict(comparison=result, promoted=promoted, champion=champion,
                                           players=PLAN["players"], evidence=evidence, decision=decision))
    file_result(result, evidence, decision, promoted, champion, day)
    update(evidence + " " + decision, "finished", log=True)
    status = ROOT / "research/status.md"
    if HERE.name in status.read_text():
        put(status, f"# Status\n\nNothing running. {decision} {evidence} "
            f"[Head-to-head result](experiments/{HERE.name}.md).\n")


if __name__ == "__main__":
    try:
        wait_for_games()
        report()
    except Exception as error:
        update(f"The joint versus early-refit comparison needs attention. {error}", "attention", log=True)
        raise
    subprocess.run(["uv", "run", "--offline", "--no-project", "--with", "matplotlib==3.11.2",
                    "--python", sys.executable, "python", str(HERE / "plot.py"), str(OUT / "verdict.json"),
                    str(ROOT / "research/figures" / HERE.name)], check=True)
