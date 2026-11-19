"""Advance the fixed study using the power-aware queue. STOP prevents new submissions."""
import fcntl
import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from study import CHAMPION, CONFIG, PREFIX, ROOT, RUN, SHA256, submit, train

sys.path.insert(0, str(ROOT / "src"))
from sts_rl.paired_endpoints import compare, digest
from sts_rl.boss_refits import BossRefitJobs, package

GATE_WORLDS = {8: 149_000_000_000, 16: 150_000_000_000}
LABEL_WORLDS = {8: 90_850_008_000, 16: 90_850_016_000}
BANK_WORLDS = {8: 92_850_008_000, 16: 92_850_016_000}
FROZEN = RUN / "control"
QUEUE = ROOT / "data/rl_runs/queue.json"


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_bytes())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.1)


def lines(path):
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_bytes().split(b"\n")[:-1] if row]


def stamp(t):
    return datetime.fromtimestamp(t, ZoneInfo("Europe/London")).strftime("%d %b %H:%M %Z")


class Monitor(BossRefitJobs):
    def __init__(self, run=RUN, prefix=PREFIX, *, label_worlds=LABEL_WORLDS, bank_worlds=BANK_WORLDS,
                 gate_worlds=GATE_WORLDS, probe=RUN / "probe"):
        super().__init__(run, prefix, label_worlds=label_worlds, bank_worlds=bank_worlds,
                         gate_worlds=gate_worlds, probe=probe, submit=submit, lane_prefix="joint-a3")
        self.events = {r["key"]: r for r in lines(run / "progress.jsonl")}

    def note(self, key, message, t=None, **details):
        if key in self.events:
            return
        event = dict(key=key, t=t if t is not None else time.time(), message=message, **details)
        with (self.run / "progress.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        self.events[key] = event
        self.render_progress()
        print(message, flush=True)

    def render_progress(self):
        entries = sorted(self.events.values(), key=lambda e: e["t"], reverse=True)
        write(ROOT / "research/log.md", "# Act 3 boss refit progress\n\n" +
              "\n".join(f"- {stamp(e['t'])} — {e['message']}" for e in entries if not e["key"].startswith(PREFIX + "smoke")) + "\n")

    def training_progress(self, folder):
        from sts_rl.common import train_tasks
        for r in lines(folder / "log.jsonl"):
            i = r.get("iteration", 0)
            event = r["event"]
            key = f"{folder.name}-{event}-{i}"
            if key in self.events:
                continue
            if event == "train":
                start = CONFIG["train_world_start"] + CONFIG["seed"] * 500_000_000 + i * 10_000_000
                with gzip.open(folder / f"collection_{i:04d}.json.gz", "rt") as f:
                    seeds = [w["seed"] for w in json.load(f)["tasks"]]
                if seeds != [w.seed for w in train_tasks(start, CONFIG["games"])]:
                    raise ValueError(f"World roster mismatch at {folder.name} iteration {i}")
                self.note(key, f"{folder.name} finished update {i}/24 on {len(seeds):,} matched worlds. "
                          f"Sampled wins {r['collect_wins']:,}, faults {r['faults']}, "
                          f"elapsed {r['iteration_seconds']/60:.1f} min.", r["t"])
            elif event == "eval":
                self.note(key, f"{folder.name} development evaluation at update {i}: "
                          f"{r['wins']:,}/{r['n']:,} wins, {r['faults']} faults.", r["t"])
            elif event in ("start", "resume"):
                self.note(key, f"{folder.name} started at update {i} with learning rate 0.0001.", r["t"])

    def resume(self, name, folder, source, target, template=None):
        job = self.jobs.get(PREFIX + name)
        if job is None:
            folder.mkdir(exist_ok=True)
            if not (folder / "latest.pt").exists():
                tmp = folder / "latest.copy"
                shutil.copyfile(source, tmp)
                tmp.replace(folder / "latest.pt")
            train(name, folder, template or source, target, True)
            self.note(name + "-submit", f"Queued matched joint strategy training through update {target}.")
        else:
            self.check_trainer(name, folder, template or source, target)
        return folder

    def check_trainer(self, name, folder, template, target):
        for attempt in range(3):
            key = name if attempt == 0 else f"{name}-resume{attempt}"
            job = self.jobs.get(self.prefix + key)
            if job is None:
                if self.on_battery:
                    return
                import torch
                saved = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
                if (not 0 < saved["iteration"] < target or not saved["optimizer_state"] or
                        not saved["rng_state"] or "trainer_state" not in saved):
                    raise RuntimeError(f"Cannot safely resume {name} from its latest checkpoint")
                train(key, folder, template, target, True, prefix=self.prefix)
                self.note(key + "-submit", f"Resuming {name} from saved update {saved['iteration']} after interruption.")
                return
            state = job["status"]
            if state in ("pending", "running", "paused"):
                return
            if state == "done" and (folder / f"iter_{target:04d}.pt").is_file():
                return
            if state == "cancelled":
                raise RuntimeError(f"{key} was cancelled")
        raise RuntimeError(f"{name} needs inspection after two recovery attempts")

    def tick(self):
        if (RUN / "STOP").exists():
            return "stopped"
        self.jobs = {r["name"]: r for r in read(QUEUE)["runs"]
                     if r["name"].startswith(PREFIX) and not r["name"].startswith(PREFIX + "smoke")}
        probe = RUN / "probe/completion.json"
        if probe.exists() and "probe-finished" not in self.events:
            info = read(probe)
            self.note("probe-finished", f"Collected {info['completed']}/32 movement-probe worlds "
                      f"with {info['faults']} faults.", probe.stat().st_mtime)
        runner = read(QUEUE.with_name("runner_status.json"))
        self.on_battery = runner["on_battery"]
        if time.time() - runner["heartbeat"] > 150:
            return "waiting for queue heartbeat"
        for name, j in self.jobs.items():
            state = j["status"]
            transition = j.get("resumed_at", j.get("started", j["added"]))
            if state in ("paused", "running"):
                self.note(f"{name}-{state}-{int(transition)}", f"{name} is {state}" +
                          (" on battery." if state == "paused" else "."), transition if state == "running" else None)
        for folder in [FROZEN, *RUN.glob("joint_after_*")]:
            self.training_progress(folder)
        self.check_trainer("control", FROZEN, CHAMPION, 24)
        if self.on_battery:
            return "paused on battery"
        fixed_endpoint = FROZEN / "iter_0024.pt"
        if fixed_endpoint.exists() and not (RUN / "primary.json").exists():
            self.evaluate_one("primary", "control", fixed_endpoint, 152_000_000_000, 16_384, 0)
            # paired() may visit the same submission in this tick.
            self.jobs = {r["name"]: r for r in read(QUEUE)["runs"] if r["name"].startswith(PREFIX)}
        current = FROZEN
        for i, target in ((8, 16), (16, 24)):
            active = current / f"iter_{i:04d}.pt"
            if not active.exists():
                return f"training to update {i}"
            outcome = self.gate(i, active)
            if outcome is None:
                return f"update {i} specialist and gates"
            source = Path(outcome["checkpoint"])
            if digest(source) != outcome["checkpoint_sha256"]:
                raise ValueError("Frozen gate outcome changed")
            if outcome["adopted"]:
                name = f"joint_after_{i}" + ("_joint" if current != FROZEN else "_control")
                current = self.resume(name, RUN / name, source, target)
            elif i == 16 and current != FROZEN:
                initial = Path(read(RUN / "refit8/decision.json")["checkpoint"])
                self.resume("joint-continue", current, active, target, initial)
        endpoints = (FROZEN / "iter_0024.pt", current / "iter_0024.pt")
        if not all(p.exists() for p in endpoints):
            return "training to update 24"
        names = ("control", "joint")
        primary = self.paired("primary", names, endpoints, 152_000_000_000, 16_384)
        if primary is None:
            return "paired endpoint evaluation"
        j = int(primary["full_cohort_gain_bounds_pp"][0] > 0)
        selected = endpoints[j]
        choice = dict(arm=names[j], checkpoint=str(selected), sha256=digest(selected))
        if choice["sha256"] != primary["checkpoint_sha256"][names[j]]:
            raise ValueError("Selected endpoint changed")
        selection = RUN / "selection.json"
        if not selection.exists():
            write(selection, choice)
        if read(selection) != choice or digest(CHAMPION) != SHA256:
            raise ValueError("Frozen confirmation inputs changed")
        self.note("selected", f"Froze the {names[j]} endpoint for independent confirmation.")
        result = self.paired("confirmation", ("incumbent", names[j]), (CHAMPION, selected), 153_000_000_000, 16_384)
        if result is None:
            return "independent confirmation"
        qualifies = result["worst_case_ci95_pp"][0] > 0
        verdict = dict(**choice, confirmation_passed=qualifies, starting_champion=SHA256)
        write(RUN / "verdict.json", verdict)
        from finish import record_finish
        record_finish(verdict, primary, result)
        self.note("verdict", f"The {names[j]} endpoint " +
                  ("passed confirmation." if qualifies else "did not pass confirmation; the incumbent stays champion."))
        return "complete"


def main():
    with (RUN / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = Monitor()
        while True:
            try:
                stage = monitor.tick()
            except Exception as error:
                monitor.note("attention-" + str(error), f"Study needs attention: {error}")
                write(RUN / "monitor-status.json", dict(stage="attention", error=repr(error), time=time.time()))
                raise
            write(RUN / "monitor-status.json", dict(stage=stage, time=time.time(), pid=os.getpid()))
            if stage in ("stopped", "complete"):
                break
            time.sleep(25)


if __name__ == "__main__":
    main()
