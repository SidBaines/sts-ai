"""Record one local, process-checked heartbeat every fifteen minutes for both cadences."""
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from study import ROOT, RUN

EVERY_TWO = RUN.parent / "joint_a3_every2_from_0f52"
RUNS = RUN.parent
INTERVAL = 900


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_bytes()) if path.is_file() else {}
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.2)


def last_row(path):
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        start = max(0, path.stat().st_size - 32768)
        stream.seek(start)
        lines = stream.read().split(b"\n")[:-1]
        if start:
            lines = lines[1:]
        return json.loads(lines[-1]) if lines else {}


def atomic(path, value):
    temp = path.with_name(path.name + ".heartbeat-tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temp.replace(path)


def processes(pids):
    if not pids:
        return {}
    ps = subprocess.run(["ps", "-p", ",".join(map(str, sorted(pids))), "-o", "pid=,stat=,etime=,command="],
                        capture_output=True, text=True, timeout=10)
    if ps.returncode not in (0, 1) or ps.stderr.strip():
        raise RuntimeError(f"Cannot inspect live study processes: {ps.stderr.strip()}")
    states = {}
    for line in ps.stdout.splitlines():
        pid, status, elapsed, command = line.strip().split(None, 3)
        states[int(pid)] = dict(status=status, elapsed=elapsed, command=command)
    return states


def snapshot():
    queue = read(RUNS / "queue.json")["runs"]
    runner = read(RUNS / "runner_status.json")
    original = read(RUN / "monitor-status.json")
    dense = read(EVERY_TWO / "monitor-status.json")
    dense_active = (EVERY_TWO / "started.json").exists()
    study, folder = ("every two", EVERY_TWO) if dense_active else ("every eight / control", RUN)
    state = dense if dense_active else original
    running = [j for j in queue if j["name"].startswith(("joint-a3-", "a3-e2-"))
               and "smoke" not in j["name"] and j["status"] in ("running", "paused", "pending")]
    handles = {runner["pid"]: "queue_runner.py"}
    for status, script in [(original, "monitor.py"), (dense, "cadence.py")]:
        if status.get("pid") and status.get("stage") not in ("complete", "stopped"):
            handles[status["pid"]] = script
    for j in running:
        if j.get("pid"):
            handles[j["pid"]] = next((Path(s).name for s in j["cmd"] if str(s).endswith(".py")), "")
    live = processes(set(handles))
    issues = [f"missing process {pid} ({script})" for pid, script in handles.items()
              if pid not in live or script not in live[pid]["command"] or "Z" in live[pid]["status"]]
    if time.time() - runner["heartbeat"] > 150:
        issues.append("stale queue heartbeat")
    if state.get("stage") == "attention":
        issues.append(str(state.get("error", "monitor needs attention")))
    updates = {}
    for training in [RUN / "control", *RUN.glob("joint_after_*"), EVERY_TWO / "train"]:
        log = training / "log.jsonl"
        if log.exists():
            rows = log.read_bytes().split(b"\n")[:-1]
            updates[training.name] = max((r["iteration"] for r in map(json.loads, rows) if r["event"] == "train"), default=0)
    jobs = {}
    for job in running:
        args = job["cmd"]
        name = job["name"]
        info = dict(status=job["status"], pid=job.get("pid"))
        if "--out" in args:
            out = Path(args[args.index("--out")+1])
            ro = last_row(out / "progress.jsonl")
            info.update({k: ro[k] for k in ("recorded","completed","decisions","faults","censored") if k in ro})
            if "fit-" in name and out.is_dir():
                fit = read(out / "status.json")
                info["fit_step"] = fit.get("updates", fit.get("step"))
        jobs[name] = info
    t = time.time()
    now = datetime.fromtimestamp(t, ZoneInfo("Europe/London")).strftime("%d %b %H:%M %Z")
    stage = state.get("stage", "waiting")
    msg = (f"{study}, {stage}. Strategy updates: " + ", ".join(f"{k} {v}/24" for k,v in updates.items()) +
           ". " + ("On battery." if runner["on_battery"] else "On AC.") +
           ((" Needs attention: " + ", ".join(issues)) if issues else f" {len(live)} study and queue processes verified."))
    return dict(t=t, stamp=now, interval_seconds=INTERVAL, study=study, stage=stage, updates=updates, jobs=jobs,
                on_battery=runner["on_battery"], free_gib=round(shutil.disk_usage(folder).free / 2**30, 1),
                processes=live, issues=issues, message=msg)


def checked_snapshot():
    """Allow the queue one poll to reap a child that just exited."""
    row = snapshot()
    if any(issue.startswith("missing process ") for issue in row["issues"]):
        time.sleep(18)
        row = snapshot()
    return row


def record(log, event, *, next_due=None):
    try:
        row = checked_snapshot()
    except Exception as error:
        row = dict(t=time.time(), stamp=datetime.now(ZoneInfo("Europe/London")).strftime("%d %b %H:%M %Z"),
                   message=f"Heartbeat needs inspection: {error}", issues=[repr(error)], interval_seconds=INTERVAL)
    row.update(watcher_pid=os.getpid(), event=event)
    if next_due is not None:
        row["next_due"] = next_due
    with log.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic(RUNS / "study_heartbeat.json", row)
    print(row["stamp"], row["message"], flush=True)
    return row


def main():
    EVERY_TWO.mkdir(exist_ok=True)
    with (EVERY_TWO / "heartbeat.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = EVERY_TWO / "heartbeats.jsonl"
        previous = last_row(log)
        due = previous.get("next_due", previous.get("t", 0) + INTERVAL)
        if previous.get("issues") and time.time() < due:
            record(log, "recheck", next_due=due)
        while not (EVERY_TWO / "STOP_WATCH").exists():
            finished = read(EVERY_TWO / "monitor-status.json").get("stage") == "complete"
            if time.time() >= due or finished:
                next_due = due + INTERVAL if time.time() - due < INTERVAL else time.time() + INTERVAL
                record(log, "final" if finished else "heartbeat", next_due=next_due)
                if finished:
                    return
                # Keep the cadence on its original clock after a recheck or sleep.
                due = next_due
            time.sleep(max(1, min(30, due - time.time())))


if __name__ == "__main__":
    main()
