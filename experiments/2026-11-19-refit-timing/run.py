"""Advance the two dependent schedules; each expensive stage runs in the shared queue."""

import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from study import (
    BANKS,
    CHAMPION,
    CHAMPION_SHA,
    CONFIRMATION,
    GATE_GAMES,
    LABELS,
    N,
    PPO,
    PREFIX,
    ROOT,
    RUN,
    SCHEDULES,
    SELECTION,
    submit,
    train,
)

sys.path.insert(0, str(ROOT / "src"))
from sts_rl.boss_refits import BossRefitJobs, read
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import compare, digest

RUNS = RUN.parent
ZONE = ZoneInfo("Europe/London")


def stamp(t):
    return datetime.fromtimestamp(t, ZONE).strftime("%d %b %H:%M %Z")


def rows(path):
    return (
        [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1] if line]
        if path.is_file()
        else []
    )


def save_text(path, text):
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(text)
    tmp.replace(path)


class Journal:
    def __init__(self):
        self.log = RUN / "progress.jsonl"
        self.seen = {r["key"] for r in rows(self.log)}

    def note(self, key, msg, *, public=True, when=None):
        if key in self.seen:
            return
        t = time.time() if when is None else when
        with self.log.open("a") as f:
            f.write(json.dumps(dict(key=key, t=t, message=msg)) + "\n")
        self.seen.add(key)
        if public:
            path = ROOT / "research/log.md"
            header, _, previous = path.read_text().partition("\n")
            save_text(path, f"{header}\n\n- {stamp(t)} — {msg}\n" + previous.lstrip("\n"))
        print(f"{stamp(t)} {msg}", flush=True)


class Arm(BossRefitJobs):
    def __init__(self, name, run, schedule, journal):
        super().__init__(
            run,
            PREFIX + name + "-",
            label_worlds=dict(zip(schedule, LABELS)),
            bank_worlds=dict(zip(schedule, BANKS)),
            gate_worlds=dict(zip(schedule, GATE_GAMES)),
            probe=RUN / "probe",
            submit=submit,
            lane_prefix="refit-timing",
        )
        self.name, self.journal = name, journal
        self.run.mkdir(exist_ok=True)
        self.events = {k[len(name) + 1 :]: True for k in journal.seen if k.startswith(name + "/")}

    def note(self, key, message):
        public = key.startswith(("decision-", "fight-")) or key.endswith(
            ("-submit", "-result", "-incomplete")
        )
        self.journal.note(self.name + "/" + key, f"{self.name.capitalize()} — {message}", public=public)
        self.events[key] = True


class Study:
    def __init__(self):
        self.journal = Journal()
        self.arms = {n: Arm(n, RUN / n, s, self.journal) for n, s in SCHEDULES.items()}
        self.evaluator = Arm("evaluation", RUN / "evaluation", (), self.journal)
        self.last_heartbeat = rows(RUN / "heartbeats.jsonl")
        self.last_heartbeat = self.last_heartbeat[-1]["t"] if self.last_heartbeat else 0
        self.known_plans = set()

    def refresh(self):
        self.queue = {
            r["name"]: r for r in read(RUNS / "queue.json")["runs"] if r["name"].startswith(PREFIX)
        }
        self.runner = read(RUNS / "runner_status.json")
        for arm in [*self.arms.values(), self.evaluator]:
            arm.jobs = self.queue
            arm.on_battery = self.runner["on_battery"]

    def annotate(self, folder, parent=None, update=0):
        if folder in self.known_plans:
            return
        plan = dict(
            iterations=24,
            label="Shared · starting champion"
            if parent is None
            else f"{folder.name.split('-')[0].capitalize()} · specialist adopted at update {update}",
        )
        if parent is not None:
            plan["parent_run"] = parent.relative_to(RUNS).as_posix()
            plan["iteration"] = update
        folder.mkdir(exist_ok=True)
        f = folder / "training-plan.json"
        if f.exists() and any(read(f).get(k) != v for k, v in plan.items() if k != "iterations"):
            raise ValueError("Dashboard branch ancestry changed")
        if not f.exists():
            atomic_json(f, plan)
        self.known_plans.add(folder)

    def milestone(self, folder, template, base_update, target):
        ck = folder / f"iter_{target:04d}.pt"
        if ck.is_file():
            return ck
        jobs = [
            j
            for j in self.queue.values()
            if "scripts/rl/train_ooc.py" in j["cmd"]
            and Path(j["cmd"][j["cmd"].index("--out") + 1]) == folder
        ]
        if any(j["status"] in ("running", "pending", "paused") for j in jobs):
            return None
        if self.runner["on_battery"]:
            return None
        if jobs and jobs[-1]["status"] == "cancelled":
            raise RuntimeError(f"Training cancelled for {folder.name}")
        failed = [j for j in jobs if j["status"] not in ("done",)]
        if len(failed) > 2:
            raise RuntimeError(f"Repeated PPO failure for {folder.name}")
        latest = folder / "latest.pt"
        if not jobs and base_update:
            # First creation of a diverged trajectory. PPO will restore identical strategy, optimizer and RNG.
            tmp = folder / "latest.copying"
            shutil.copyfile(template, tmp)
            if digest(tmp) != digest(template):
                raise ValueError("Incomplete continuation checkpoint")
            tmp.replace(latest)
        if jobs and not latest.exists():
            raise RuntimeError(f"Interrupted training lacks a resume state in {folder.name}")
        if latest.exists():
            import torch

            state = torch.load(latest, map_location="cpu", weights_only=False)
            if not (
                base_update <= state["iteration"] < target
                and state["optimizer_state"]
                and state["rng_state"]
                and "trainer_state" in state
            ):
                raise ValueError(f"Invalid continuation to update {target} in {folder.name}")
        name = f"{folder.name}-{target:02d}" + (f"-recovery{len(failed)}" if failed else "")
        train(name, folder, template, target, latest.exists())
        self.journal.note(
            name + "-train", f"Queued {folder.name} strategy training through update {target}."
        )
        self.refresh()
        return None

    def trajectory(self, name):
        folder, template, base_update = RUN / "shared", CHAMPION, 0
        self.annotate(folder)
        for i in (*SCHEDULES[name], 24):
            ck = self.milestone(folder, template, base_update, i)
            if ck is None:
                return None, f"{name} training to {i}"
            if i == 24:
                return ck, f"{name} endpoint frozen"
            result = self.arms[name].gate(i, ck)
            if result is None:
                return None, f"{name} refit at {i}"
            if result["adopted"]:
                parent = folder
                folder = RUN / f"{name}-{i:02d}"
                template = Path(result["checkpoint"])
                if digest(template) != result["checkpoint_sha256"]:
                    raise ValueError(f"Accepted head for {name} at {i} changed")
                self.annotate(folder, parent, i)
                base_update = i

    def progress(self):
        from sts_rl.common import train_tasks

        self.updates = {}
        for path in sorted(RUN.glob("*/log.jsonl")):
            for row in rows(path):
                i = row.get("iteration", 0)
                if row["event"] != "train" and row["event"] != "eval":
                    continue
                name = path.parent.name
                key = f"{name}-{row['event']}-{i}"
                if row["event"] == "train":
                    self.updates[name] = max(self.updates.get(name, 0), i)
                if key in self.journal.seen:
                    continue
                if row["event"] == "eval":
                    msg = f"{name} development at update {i}: {row['wins']}/{row['n']} wins, {row['faults']} faults."
                else:
                    with gzip.open(path.parent / f"collection_{i:04d}.json.gz", "rt") as f:
                        seeds = [x["seed"] for x in json.load(f)["tasks"]]
                    start = PPO["train_world_start"] + PPO["seed"] * 500_000_000 + i * 10_000_000
                    if seeds != [t.seed for t in train_tasks(start, PPO["games"])]:
                        raise ValueError(f"Training world mismatch at {name} update {i}")
                    msg = (
                        f"{name} finished update {i}/24 on {len(seeds):,} matched worlds, "
                        f"{row['collect_wins']:,} sampled wins and {row['faults']} faults."
                    )
                self.journal.note(key, msg, when=row["t"])

    def panel(self, phase, start, players, comparisons):
        output = RUN / (phase + ".json")
        if output.is_file():
            return read(output)
        seen, folders = {}, {}
        for name, ck in players.items():
            sha = digest(ck)
            if sha not in seen:
                seen[sha] = self.evaluator.evaluate_one(phase, name, ck, start, N, len(seen) % 2)
            folders[name] = seen[sha]
        if not all(folders.values()):
            return None
        reports = {}
        for a, b in comparisons:
            item = compare(folders[a], folders[b], start, a, b, players[a], players[b], N)
            item["checkpoint_sha256"] = {n: digest(players[n]) for n in (a, b)}
            item["folders"] = [str(folders[n]) for n in (a, b)]
            item["checkpoints"] = {n: str(players[n]) for n in (a, b)}
            if item["checkpoint_sha256"][a] == item["checkpoint_sha256"][b] and (
                item["a_only"] or item["b_only"]
            ):
                raise ValueError("Identical checkpoint has different completed outcomes")
            reports[b + "_vs_" + a] = item
            self.journal.note(
                f"{phase}/{b}/{a}",
                f"{phase.capitalize()}: {b} won {item['second']['wins']:,}, "
                f"{a} won {item['first']['wins']:,} on {item['n']:,} worlds, "
                f"{item['paired_completed']:,} complete pairs.",
            )
        atomic_json(output, reports)
        return reports

    def tick(self):
        self.refresh()
        self.progress()
        if (RUN / "STOP").exists():
            return "stopped"
        if time.time() - self.runner["heartbeat"] > 150:
            return "waiting for queue heartbeat"
        if self.runner["on_battery"]:
            return "paused on battery"
        endpoints, stages = {}, []
        for n in SCHEDULES:
            ck, stage = self.trajectory(n)
            stages.append(stage)
            if ck:
                endpoints[n] = ck
        if len(endpoints) < 2:
            return " / ".join(stages)
        primary = self.panel("selection", SELECTION, endpoints, [("early", "spaced")])
        if primary is None:
            return "paired endpoint selection"
        from sts_rl.boss_refits import primary_nominee

        nominee = primary_nominee(primary["spaced_vs_early"], "early", "spaced")
        other = "spaced" if nominee == "early" else "early"
        choice = dict(
            arm=nominee,
            checkpoint=str(endpoints[nominee]),
            sha256=digest(endpoints[nominee]),
            source_sha256=CHAMPION_SHA,
            endpoint_sha256={k: digest(v) for k, v in endpoints.items()},
        )
        if (RUN / "frozen.json").exists() and read(RUN / "frozen.json") != choice:
            raise ValueError("Frozen endpoints changed")
        if not (RUN / "frozen.json").exists():
            atomic_json(RUN / "frozen.json", choice)
            self.journal.note(
                "selected", f"Froze both endpoints. The selection rule nominated {nominee}."
            )
        if digest(CHAMPION) != CHAMPION_SHA:
            raise ValueError("Starting champion changed")
        confirmed = self.panel(
            "confirmation",
            CONFIRMATION,
            {**endpoints, "champion": CHAMPION},
            [(other, nominee), ("champion", "early"), ("champion", "spaced")],
        )
        if confirmed is None:
            return "independent confirmation"
        from report import finish

        finish(primary["spaced_vs_early"], confirmed, choice)
        self.journal.note(
            "finished", "Both timing schedules, all six gates and independent confirmation are complete."
        )
        return "complete"

    def heartbeat(self, stage):
        now = time.time()
        if now - self.last_heartbeat < 900 and stage not in ("attention", "complete", "stopped"):
            return
        jobs = {
            n: {"status": j["status"], "pid": j.get("pid")}
            for n, j in self.queue.items()
            if j["status"] in ("running", "pending", "paused")
        }
        running = [v["pid"] for v in jobs.values() if v.get("pid")]
        # One queue poll can elapse between a successful child exit and its terminal queue record.
        live = 0
        for pid in running:
            try:
                os.kill(pid, 0)
                live += 1
            except ProcessLookupError:
                pass
        status = f"{stage}. {live} live worker jobs, {len(jobs)} queued or active. "
        status += "On battery." if self.runner["on_battery"] else "On AC."
        event = dict(
            t=now,
            stamp=stamp(now),
            stage=stage,
            message=status,
            pid=os.getpid(),
            jobs=jobs,
            updates=self.updates,
            on_battery=self.runner["on_battery"],
            interval_seconds=900,
            free_gib=round(shutil.disk_usage(RUN).free / 2**30, 2),
        )
        with (RUN / "heartbeats.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        atomic_json(RUNS / "study_heartbeat.json", event)
        self.last_heartbeat = now


def main():
    with (RUN / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Study()
        while True:
            try:
                stage = study.tick()
            except Exception as exc:
                atomic_json(
                    RUN / "status.json",
                    dict(stage="attention", error=repr(exc), t=time.time(), pid=os.getpid()),
                )
                study.journal.note("attention-" + str(exc), f"Study needs attention: {exc}")
                raise
            study.heartbeat(stage)
            atomic_json(RUN / "status.json", dict(stage=stage, t=time.time(), pid=os.getpid()))
            if stage in ("complete", "stopped"):
                return
            time.sleep(20)


if __name__ == "__main__":
    main()
