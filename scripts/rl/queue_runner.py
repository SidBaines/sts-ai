#!/usr/bin/env python3
"""Detached experiment queue with a power-aware babysitter.

    queue_runner.py add NAME [--lane L] -- CMD...   append a pending run (one active job per lane)
    queue_runner.py cancel NAME            mark pending/running run cancelled (kills it)
    queue_runner.py start                  launch the runner detached (idempotent)
    queue_runner.py status                 print queue + runner heartbeat

Runs execute one at a time, each in its own process group. On battery the
group is SIGSTOPped and caffeinate released; on AC it is SIGCONTed and the Mac
kept awake. A stalled run is stopped and marked failed for diagnosis. Scientific
jobs are never automatically restarted or resumed into an existing output.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "rl_runs"
QUEUE = RUNS / "queue.json"
LOCK = RUNS / "queue.lock"
STATUS = RUNS / "runner_status.json"
STALL_MINUTES = 40
LAUNCHED: dict[int, subprocess.Popen] = {}
ENV = {"PYTHONPATH": str(ROOT / "src"),
       "STS_LIGHTSPEED_BUILD": str(ROOT / "external" / "sts_lightspeed_rl" / "build013_clean")}


class Locked:
    def __enter__(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        self.fh = LOCK.open("w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def read_queue() -> dict:
    return json.loads(QUEUE.read_text()) if QUEUE.exists() else {"runs": []}


def write_queue(queue: dict) -> None:
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text(json.dumps(queue, indent=1))
    os.replace(tmp, QUEUE)


def on_battery() -> bool:
    try:
        return "Battery Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_log(run: dict) -> Path:
    """The trainer's log: under its --out directory, which need not match the queue name
    (resumed and branched runs). Using RUNS/<name> for those made the stall watchdog
    kill healthy runs STALL_MINUTES after launch (2026-10-13, ooc_main_s1_ext4)."""
    cmd = run.get("cmd", [])
    if "--out" in cmd and cmd.index("--out") + 1 < len(cmd):
        return ROOT / cmd[cmd.index("--out") + 1] / "log.jsonl"
    return RUNS / run["name"] / "log.jsonl"


def last_activity(run: dict) -> tuple[float, bool]:
    """Latest sign of progress, and whether the run writes a progress log at all."""
    log, stdout = run_log(run), RUNS / run["name"] / "stdout.log"
    times = [run.get("started", 0), run.get("resumed_at", 0)]
    times += [f.stat().st_mtime for f in (log, stdout) if f.exists()]
    return max(times), log.exists()


def launch_inputs(cmd: list[str]) -> list[str]:
    """Collect file-valued options; --bank may also be a boolean collector switch."""
    inputs = []
    for flag in ('--bank', '--records', '--baseline', '--teacher'):
        if flag not in cmd:
            continue
        for value in cmd[cmd.index(flag) + 1:]:
            if value.startswith('--'):
                break
            inputs.append(value)
    return inputs


def launch(run: dict) -> int:
    sys.path.insert(0, str(ROOT/'src'))
    from sts_rl.recording import begin_run
    out = RUNS / run["name"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = run['cmd']
    checkpoint = cmd[cmd.index('--checkpoint')+1] if '--checkpoint' in cmd else None
    inputs = launch_inputs(cmd)
    build = next((word.split('=',1)[1] for word in cmd if word.startswith('STS_LIGHTSPEED_BUILD=')), ENV['STS_LIGHTSPEED_BUILD'])
    run['launch_record'] = begin_run(out, {'queue_name':run['name'], 'cmd':cmd, 'lane':run.get('lane','main')},
                                     checkpoint, inputs, build_dir=build, preparation=run.get('preparation',False))
    stdout = (out / "stdout.log").open("a")
    log = run_log(run)
    run['log_start_bytes'] = log.stat().st_size if log.exists() else 0
    proc = subprocess.Popen(run["cmd"], cwd=ROOT, env={**os.environ, **ENV}, start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT)
    LAUNCHED[proc.pid] = proc
    stdout.close()
    return proc.pid


def runner() -> None:
    caffeinate = None
    while True:
        battery = on_battery()
        with Locked():
            queue = read_queue()
            running = [r for r in queue["runs"] if r["status"] in ("running", "paused")]
            for run in running:
                pid = run["pid"]
                if run.get("cancel"):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        pass
                    run.update(status="cancelled", ended=time.time())
                    continue
                # reap if exited
                finished = False
                try:
                    if pid in LAUNCHED:
                        code = LAUNCHED[pid].poll()
                        wpid = pid if code is not None else 0
                    else:
                        wpid, status = os.waitpid(pid, os.WNOHANG)
                        code = os.waitstatus_to_exitcode(status) if wpid else None
                    if wpid == pid:
                        finished = True
                        run["returncode"] = code
                        LAUNCHED.pop(pid, None)
                except ChildProcessError:
                    finished = not alive(pid)
                    run.setdefault("returncode", None)
                if finished:
                    log = run_log(run)
                    done = False
                    if log.exists():
                        with log.open('rb') as source:
                            source.seek(max(run.get('log_start_bytes',0), log.stat().st_size-2000))
                            done = b'"event": "done"' in source.read()
                    code = run.get("returncode")
                    # Non-trainer jobs (evals) write no log.jsonl; a runner restarted since
                    # launch cannot reap its predecessor's children, so code may be unknown.
                    status = "done" if code == 0 or (code is None and done) else ("exited" if code is None and not log.exists() else "failed")
                    run.update(status=status, ended=time.time())
                    continue
                if battery and run["status"] == "running":
                    os.killpg(pid, signal.SIGSTOP)
                    run["status"] = "paused"
                elif not battery and run["status"] == "paused":
                    os.killpg(pid, signal.SIGCONT)
                    run["status"] = "running"
                    run["resumed_at"] = time.time()
                last, has_log = last_activity(run)
                # Jobs without a progress log (evaluations) only print at the end.
                limit = STALL_MINUTES * 60 * (1 if has_log else 3)
                if run["status"] == "running" and time.time() - last > limit:
                    os.killpg(pid, signal.SIGKILL)
                    run.update(status="failed", ended=time.time(), reason="stalled; diagnosis required, no automatic retry")
            # One active job per lane (default lane "main").
            busy = {r.get("lane", "main") for r in queue["runs"] if r["status"] in ("running", "paused")}
            if not battery:
                for run in [r for r in queue["runs"] if r["status"] == "pending"]:
                    lane = run.get("lane", "main")
                    if lane in busy:
                        continue
                    busy.add(lane)
                    try:
                        run.update(pid=launch(run), status="running", started=time.time())
                    except (OSError, RuntimeError, ValueError) as exc:
                        run.update(status="failed", ended=time.time(), reason=f"launch: {exc}")
            write_queue(queue)
            active = [r for r in queue["runs"] if r["status"] == "running"]
        # keep awake only on AC while something runs
        want = bool(active) and not battery
        if want and (caffeinate is None or caffeinate.poll() is not None):
            caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        elif not want and caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
            caffeinate = None
        STATUS.write_text(json.dumps({"heartbeat": time.time(), "pid": os.getpid(), "on_battery": battery,
                                      "active": [r["name"] for r in active], "caffeinated": want}))
        time.sleep(15)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "add":
        name, cmd = sys.argv[2], sys.argv[sys.argv.index("--") + 1:]
        options = sys.argv[3:sys.argv.index("--")]
        lane = options[options.index("--lane") + 1] if "--lane" in options else "main"
        with Locked():
            queue = read_queue()
            if any(r["name"] == name for r in queue["runs"]):
                sys.exit(f"run {name} already queued")
            queue["runs"].append({"name": name, "cmd": cmd, "status": "pending", "added": time.time(), "lane": lane,
                                  'preparation':'--preparation' in options})
            write_queue(queue)
        print("queued", name)
    elif command == "cancel":
        with Locked():
            queue = read_queue()
            for run in queue["runs"]:
                if run["name"] == sys.argv[2]:
                    if run["status"] == "pending":
                        run["status"] = "cancelled"
                    else:
                        run["cancel"] = True
            write_queue(queue)
    elif command == "start":
        if STATUS.exists():
            status = json.loads(STATUS.read_text())
            if alive(status["pid"]) and time.time() - status["heartbeat"] < 120:
                print("runner already alive", status["pid"])
                return
        RUNS.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([sys.executable, __file__, "_run"], cwd=ROOT, start_new_session=True,
                                stdin=subprocess.DEVNULL, stdout=(RUNS / "runner.log").open("a"),
                                stderr=subprocess.STDOUT)
        print("runner started", proc.pid)
    elif command == "_run":
        runner()
    else:
        print(json.dumps({"status": json.loads(STATUS.read_text()) if STATUS.exists() else None,
                          "runs": [{k: r.get(k) for k in ("name", "status", "returncode")} for r in read_queue()["runs"]]},
                         indent=1))


if __name__ == "__main__":
    main()
