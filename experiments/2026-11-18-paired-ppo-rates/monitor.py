"""Keep the live log and advance the two planned evaluation panels through the shared queue."""

import fcntl
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from launch import ARMS, PREFIX, ROOT, RUN, native_build, queue
from panels import CONFIRM, PRIMARY, N, compare, digest

QUEUE = ROOT / "data/rl_runs/queue.json"
WORKER = QUEUE.with_name("runner_status.json")
LOG = ROOT / "research/log.md"
INCUMBENT = ROOT / "data/rl_runs/champions/champion_4058cbf0e271.pt"
TZ = ZoneInfo("Europe/London")


def atomic(path, value):
    temp = path.with_name(path.name + ".writing")
    temp.write_text(value)
    temp.replace(path)


def rows(path):
    if not path.is_file():
        return []
    # A producer may still be appending the last line.
    data = path.read_bytes()
    return [json.loads(line) for line in data.split(b"\n")[:-1] if line]


def read_status(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_bytes())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def stamp(t):
    return datetime.fromtimestamp(t, TZ).strftime("%Y-%m-%d %H:%M %Z")


class Monitor:
    def __init__(self):
        self.events_path = RUN / "progress.jsonl"
        self.events = {row["key"]: row for row in rows(self.events_path)}
        self.expected = {
            r["iteration"]: r
            for r in json.loads((RUN / "expected_worlds.json").read_text())
        }
        self.statuses = {}
        self.note(
            "opened",
            "Study opened. Both iteration-24 endpoints and both 16,384-world panels are fixed.",
            when=(RUN / "expected_worlds.json").stat().st_mtime,
        )

    def note(self, key, msg, *, when=None, **details):
        if key in self.events:
            return
        event = {
            "key": key,
            "t": time.time() if when is None else when,
            "message": msg,
            **details,
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        self.events[key] = event
        lines = [
            f"- {stamp(r['t'])} — {r['message']}"
            for r in sorted(self.events.values(), key=lambda v: v["t"], reverse=True)
        ]
        atomic(LOG, "# Learning-rate study progress\n\n" + "\n".join(lines) + "\n")
        print(f"{stamp(event['t'])} {msg}", flush=True)

    def train(self, arm):
        folder = RUN / arm
        for r in rows(folder / "log.jsonl"):
            kind = r["event"]
            i = r.get("iteration", 0)
            t = r["t"]
            if kind == "start":
                self.note(
                    f"{arm}-start",
                    f"{arm.capitalize()} started from the incumbent at learning rate {ARMS[arm]:g}.",
                    when=t,
                )
            elif kind == "train":
                key = f"{arm}-train-{i}"
                if key in self.events:
                    continue
                with gzip.open(folder / f"collection_{i:04d}.json.gz", "rt") as f:
                    receipt = json.load(f)
                seeds = [s["seed"] for s in receipt["tasks"]]
                expected = self.expected[i]
                h = hashlib.sha256(
                    json.dumps(seeds, separators=(",", ":")).encode()
                ).hexdigest()
                if h != expected["world_hash"]:
                    raise ValueError(f"{arm} iteration {i}: wrong training worlds")
                ep = sorted(
                    (v["world_seed"], v["action_trace_sha256"])
                    for v in receipt["episodes"]
                )
                actions = hashlib.sha256(
                    json.dumps(ep, separators=(",", ":")).encode()
                ).hexdigest()
                self.note(
                    key,
                    f"{arm.capitalize()} finished iteration {i}/24 on {len(seeds):,} scheduled worlds. "
                    f"Sampled wins {r['collect_wins']:,}/{r['games']:,}, faults {r['faults']}, "
                    f"iteration {r['iteration_seconds'] / 60:.1f} min.",
                    when=t,
                    world_hash=h,
                    actions_hash=actions,
                    seconds=r["iteration_seconds"],
                    faults=r["faults"],
                )
            elif kind == "eval":
                self.note(
                    f"{arm}-eval-{i}",
                    f"{arm.capitalize()} development evaluation at iteration {i}: "
                    f"{r['wins']:,}/{r['n']:,} wins, {r.get('faults', 0)} faults.",
                    when=t,
                )
            elif kind == "done":
                self.note(
                    f"{arm}-done", f"{arm.capitalize()} training completed.", when=t
                )
        for i in [1]:
            keys = [f"{a}-train-{i}" for a in ARMS]
            if all(k in self.events for k in keys):
                if len({self.events[k]["actions_hash"] for k in keys}) != 1:
                    raise ValueError(
                        "First training trajectories disagree before the first policy update"
                    )
                self.note(
                    "identical-first",
                    "Verified identical first-iteration trajectories for both arms on all 3,072 assigned worlds.",
                )

    def eval_job(self, jobs, phase, arm, checkpoint, start, lane):
        name = f"{PREFIX}{phase}-{arm}"
        folder = RUN / phase / arm
        if name not in jobs:
            queue(
                "add",
                name,
                "--lane",
                lane,
                "--",
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                "STS_LIGHTSPEED_BUILD=" + native_build(),
                sys.executable,
                "-u",
                "scripts/rl/fast_eval.py",
                "--checkpoint",
                checkpoint,
                "--world-start",
                start,
                "--n",
                N,
                "--processes",
                7,
                "--games-per-process",
                8,
                "--progress-every",
                256,
                "--out",
                folder,
            )
            self.note(
                name + "-submit", f"Queued {arm} on the {phase} panel ({N:,} worlds)."
            )
            return False
        pr = rows(folder / "progress.jsonl")
        if pr:
            for v in pr:
                if v["recorded"] % 4096 == 0:
                    self.note(
                        f"{name}-{v['recorded']}",
                        f"{phase.capitalize()} panel, {arm}: "
                        f"{v['recorded']:,}/{N:,} world records returned.",
                        when=v["t"],
                    )
        return jobs[name]["status"] == "done" and (folder / "summary.json").is_file()

    def evaluate(self, jobs, phase, names, checkpoints, start):
        finished = [
            self.eval_job(jobs, phase, a, c, start, f"ppo-rates-eval-{j}")
            for j, (a, c) in enumerate(zip(names, checkpoints))
        ]
        if not all(finished):
            return None
        p = RUN / (phase + ".json")
        if not p.exists():
            result = compare(
                RUN / phase / names[0],
                RUN / phase / names[1],
                start,
                *names,
                *checkpoints,
            )
            result["checkpoint_sha256"] = {
                a: digest(c) for a, c in zip(names, checkpoints)
            }
            if phase == "primary":
                # Select by observed wins; unknown-world bounds are reported separately.
                result["selected"] = (
                    names[1]
                    if result["second"]["wins"] > result["first"]["wins"]
                    else names[0]
                )
            atomic(p, json.dumps(result, indent=2) + "\n")
        r = json.loads(p.read_text())
        self.note(
            phase + "-result",
            f"{phase.capitalize()} panel finished. {names[0]} won {r['first']['wins']:,}, "
            f"{names[1]} won {r['second']['wins']:,}; {r['paired_completed']:,}/{N:,} complete pairs.",
        )
        return r

    def tick(self):
        for event in rows(RUN / "notes.jsonl"):
            self.note(event["key"], event["message"], when=event["t"])
        if (RUN / "STOP").exists():
            return "stopped"
        jobs = {
            j["name"]: j
            for j in json.loads(QUEUE.read_text())["runs"]
            if j["name"].startswith(PREFIX)
        }
        state = read_status(WORKER)
        if time.time() - state["heartbeat"] > 150 and any(
            job["status"] in ("running", "pending", "paused") for job in jobs.values()
        ):
            # The logger can wake before the power-aware queue after laptop sleep.
            self.note(
                f"stale-heartbeat-{state['heartbeat']}",
                "Waiting for the queue heartbeat to return before advancing the study.",
            )
            return "waiting for queue heartbeat"
        for name, j in jobs.items():
            status = j["status"]
            if status != self.statuses.get(name):
                if status in ("paused", "running"):
                    transition = (
                        j.get("resumed_at", j.get("started", time.time()))
                        if status == "running"
                        else state["heartbeat"]
                    )
                    self.note(
                        f"{name}-{status}-{int(transition)}",
                        f"{name} is {status}"
                        + (" on battery." if status == "paused" else "."),
                        when=transition,
                    )
                self.statuses[name] = status
            if status in ("failed", "cancelled", "exited"):
                # Successful explicit retries supersede only their own failed output/queue entry.
                later = [
                    x
                    for x in jobs.values()
                    if x.get("added", 0) > j.get("added", 0)
                    and x["cmd"][-1:] == ["--resume"]
                    and "--out" in x["cmd"]
                    and "--out" in j["cmd"]
                    and x["cmd"][x["cmd"].index("--out") + 1]
                    == j["cmd"][j["cmd"].index("--out") + 1]
                ]
                if not later:
                    raise RuntimeError(
                        f"{name} {status}: {j.get('reason', 'inspect stdout and child records')}"
                    )
        for arm in ARMS:
            self.train(arm)
        if not all(
            f"{a}-done" in self.events and (RUN / a / "iter_0024.pt").is_file()
            for a in ARMS
        ):
            return "training"
        if not all(f"{a}-train-{i}" in self.events for a in ARMS for i in range(1, 25)):
            raise ValueError("Incomplete training iteration set")
        endpoints = {a: RUN / a / "iter_0024.pt" for a in ARMS}
        result = self.evaluate(
            jobs, "primary", list(ARMS), list(endpoints.values()), PRIMARY
        )
        if result is None:
            return "paired evaluation"
        name = result["selected"]
        if digest(endpoints[name]) != result["checkpoint_sha256"][name]:
            raise ValueError("Selected endpoint changed")
        self.note(
            "freeze-selection",
            f"Selected and froze the {name} endpoint before confirmation.",
        )
        confirmed = self.evaluate(
            jobs,
            "confirmation",
            ["incumbent", name],
            [INCUMBENT, endpoints[name]],
            CONFIRM,
        )
        return "complete" if confirmed else "confirmation"


def main():
    with (RUN / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = Monitor()
        while True:
            try:
                stage = monitor.tick()
            except Exception as err:
                atomic(
                    RUN / "monitor-status.json",
                    json.dumps(
                        {"t": time.time(), "stage": "attention", "error": repr(err)}
                    )
                    + "\n",
                )
                raise
            atomic(
                RUN / "monitor-status.json",
                json.dumps({"t": time.time(), "stage": stage, "pid": os.getpid()})
                + "\n",
            )
            if stage in ("complete", "stopped"):
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
