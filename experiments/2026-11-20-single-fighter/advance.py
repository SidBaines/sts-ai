"""Launch ready evaluations through the Mac queue, then publish both paired tests."""
import json
import os
import subprocess
import sys
import time

from assess import freeze, report
from study import CFG, CHAMP, OUT, ROOT, RUNS, note

PREFIX = "one-fighter-"


def finished(jobs, name, marker):
    job = jobs.get(PREFIX+name)
    if not job or job["status"] in ("running", "pending", "paused"):
        return False
    if job["status"] not in ("done","exited") or not marker.is_file():
        raise RuntimeError(f"{name} ended in state {job['status']}. Inspect its queue log.")
    return True


def queue(jobs, name, lane, tool, args):
    if PREFIX+name in jobs:
        return
    cmd = [sys.executable, ROOT/"scripts/rl/queue_runner.py", "add", PREFIX+name, "--lane", PREFIX+lane,
           "--", "/usr/bin/env", "PYTHONUNBUFFERED=1", f"PYTHONPATH={ROOT/'src'}",
           "STS_LIGHTSPEED_BUILD="+os.environ["STS_LIGHTSPEED_BUILD"],
           sys.executable, "-u", ROOT/"scripts/rl"/tool, *args]
    subprocess.run([str(a) for a in cmd], check=True)


def evaluate(jobs, stage, arm, checkpoint):
    start,n = CFG[stage]
    out = OUT/stage/arm
    name = f"{stage}-{arm}"
    args = ["--checkpoint", checkpoint, "--world-start", start, "--n", n, "--processes", 7,
            "--games-per-process", 8, "--progress-every", 256, "--out", out]
    if stage == "development":
        args += ["--max-active-processes", 3]
    queue(jobs,name,name,"fast_eval.py",args)
    return finished(jobs,name,out/"summary.json")


def latest(folder):
    path = folder/"progress.jsonl"
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(max(0,path.stat().st_size-4096))
            for line in reversed(stream.read().splitlines()):
                try:
                    row = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if "recorded" in row or "completed" in row:
                    return row.get("recorded",row.get("completed"))
    return 0


def main():
    previous, last_quarter = None,-1
    while not (OUT/"STOP").exists():
        if (OUT/"verdict.json").exists():
            return
        jobs = {j["name"]:j for j in json.loads((RUNS/"queue.json").read_bytes())["runs"]}
        finished(jobs,"demonstrations",OUT/"demonstrations/summary.json")
        finished(jobs,"boss-bank",OUT/"bank/completion.json")
        grid_ready = finished(jobs,"freeze-boss-entries",OUT/"grid.json")
        ready = []
        for arm in ("mix25","mix50"):
            if finished(jobs,"fit-"+arm,OUT/arm/"summary.json"):
                ready.append(evaluate(jobs,"development",arm,OUT/arm/"player.pt"))
        if ready == [True,True]:
            if not (OUT/"selection.json").is_file():
                freeze()
            full = [evaluate(jobs,"confirmation",arm,ck) for arm,ck in
                    (("champion", CHAMP), ("student", OUT/"finalist/player.pt"))]
            if grid_ready:
                start,n = CFG["bank"]
                args = ["--checkpoint",CHAMP,"--combat",OUT/"finalist/head.pt","--bank",OUT/"bank",
                        "--bank-world-start",start,"--bank-n",n,"--starts-per-group",
                        json.dumps({"BOSS3":CFG["boss_starts"]}),"--resamples",CFG["resamples"],
                        "--grid",OUT/"grid.json","--workers",6,"--out",OUT/"boss-fights"]
                queue(jobs,"boss-fights","boss-test","fresh_panel.py",args)
            fights = finished(jobs,"boss-fights",OUT/"boss-fights/completion.json")
            if all(full) and fights:
                report()
                return
            counts = (latest(OUT/"confirmation/champion"), latest(OUT/"confirmation/student"),latest(OUT/"boss-fights"))
            if counts != previous:
                n = CFG["confirmation"][1]
                quarter = 4*min(counts[:2])//n
                note("paired tests", f"Single-fighter comparison. Champion {counts[0]:,} and student {counts[1]:,} "
                     f"of {n:,} full games each. Boss replays {counts[2]:,} of {4*CFG['boss_starts']:,}.",
                     log=quarter>last_quarter)
                previous,last_quarter = counts,quarter
        time.sleep(15)
    note("stopped","Single-fighter evaluation scheduling stopped at the study control file.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        note("attention",f"Single-fighter study needs attention. {exc}")
        raise
