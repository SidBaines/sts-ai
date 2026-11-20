"""Submit ready stages to the existing power-aware queue and file the overnight verdict."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from decisions import (
    choose_endpoint,
    endpoints,
    finish,
    first_selection,
    freeze_file,
    retention,
    seal_student,
)
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import digest
from study import CFG, CHAMP, HERE, OUT, ROOT, RUNS, check_inputs, models, note, read

PREFIX = "overnight-fighter-"
RL = ROOT / "scripts/rl"
QUEUE = RL / "queue_runner.py"


class Jobs:
    def __init__(self):
        self.rows = {j["name"]: j for j in read(RUNS / "queue.json")["runs"]}

    def ensure(self, name, lane, script, args, destination, marker, *, resume=False):
        """Keep completed work, retry a failed task at most twice and respect cancellation."""
        for attempt in range(1, 4):
            key = PREFIX + name + ("" if attempt == 1 else f"-r{attempt}")
            job = self.rows.get(key)
            if not job:
                if (OUT / "STOP").exists():
                    return False
                if (
                    attempt > 1
                    and destination.exists()
                    and not (resume and (destination / "latest.pt").is_file())
                ):
                    archive = OUT / "attempts" / (name + f"-attempt-{attempt - 1}")
                    archive.parent.mkdir(exist_ok=True)
                    if archive.exists():
                        raise RuntimeError(f"Retry archive already exists: {archive}")
                    destination.rename(archive)
                cmd = [
                    sys.executable,
                    QUEUE,
                    "add",
                    key,
                    "--lane",
                    PREFIX + lane,
                    "--",
                    "/usr/bin/env",
                    "PYTHONUNBUFFERED=1",
                    f"PYTHONPATH={ROOT / 'src'}",
                    "STS_LIGHTSPEED_BUILD=" + os.environ["STS_LIGHTSPEED_BUILD"],
                    sys.executable,
                    "-u",
                    script,
                    *args,
                    "--out",
                    destination,
                ]
                subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)
                self.rows[key] = {"name": key, "status": "pending", "cmd": [str(x) for x in cmd]}
                note(
                    "queued",
                    f"Queued {name}" + (f", recovery attempt {attempt}." if attempt > 1 else "."),
                )
                return False
            state = job["status"]
            if state in ("pending", "running", "paused"):
                return False
            if state in ("done", "exited") and marker.is_file():
                return True
            if state == "cancelled":
                raise RuntimeError(f"{key} was cancelled. Its outputs are preserved.")
        raise RuntimeError(f"{name} has exhausted three attempts. Inspect {RUNS / key / 'stdout.log'}.")

    def panel(self, stage, arm, checkpoint, lane, *, cap=None):
        start, count = CFG[stage]
        dest = OUT / stage / arm
        args = [
            "--checkpoint",
            checkpoint,
            "--world-start",
            start,
            "--n",
            count,
            "--processes",
            7,
            "--games-per-process",
            8,
            "--progress-every",
            256,
        ]
        if cap:
            args += ["--max-active-processes", cap]
        return self.ensure(
            f"{stage}-{arm}", lane, RL / "fast_eval.py", args, dest, dest / "summary.json"
        )


def bank(j):
    binding = OUT / "bank-selection.json"
    if binding.is_file():
        return read(binding)
    first, n = CFG["bank"]
    for number, start in enumerate([first, *CFG["backup_bank_starts"]]):
        name = "bank" if number == 0 else f"bank-backup-{number}"
        dest = OUT / name
        args = [
            "--mode",
            "eval",
            "--checkpoint",
            CHAMP,
            "--world-start",
            start,
            "--n",
            n,
            "--workers",
            6,
            "--bank",
            "--no-teacher-labels",
        ]
        if not j.ensure(name, "bank", RL / "fresh_collect.py", args, dest, dest / "completion.json"):
            return None
        if not read(dest / "completion.json").get("cohort_complete"):
            if not (dest / "unused.json").exists():
                atomic_json(dest / "unused.json", {"reason": "incomplete source cohort"})
                note(
                    "boss bank",
                    f"{name} had an incomplete source cohort. Using the next reserved cohort intact.",
                )
            continue
        pa = [
            "--bank",
            dest,
            "--bank-world-start",
            start,
            "--bank-n",
            n,
            "--starts-per-group",
            json.dumps({"BOSS3": CFG["boss_starts"]}),
            "--resamples",
            1,
        ]
        grid = OUT / (name + "-grid.json")
        if not j.ensure(
            name + "-grid", "bank", RL / "fresh_panel.py", [*pa, "--freeze-grid-only"], grid, grid
        ):
            return None
        if not read(grid)["coverage_sufficient"]:
            continue
        freeze_file(grid, OUT / "grid.json", digest(grid))
        result = {"bank": str(dest), "start": start, "n": n, "grid_hash": read(grid)["grid_hash"]}
        atomic_json(binding, result)
        note(
            "boss bank",
            f"Froze {CFG['boss_starts']:,} distinct Act 3 boss entries from {n:,} champion games.",
        )
        return result
    raise RuntimeError("All reserved boss source cohorts were incomplete or short of entries.")


def stage(j):
    bank_info = bank(j)
    long = OUT / "long-fit"
    done = j.ensure("long-fit", "gpu", HERE / "distil.py", ["long"], long, long / "summary.json")
    verify = OUT / "verification"
    checked = j.ensure(
        "verify-two-trainers",
        "verify",
        HERE / "training.py",
        ["smoke"],
        verify,
        verify / "finished.json",
    )
    candidates = models()
    screens = []
    for i, arm in enumerate(("original", "step-6400", "step-12800")):
        if arm == "original" or (candidates[arm] / "summary.json").exists():
            screens.append(
                j.panel(
                    "screen",
                    arm,
                    candidates[arm] / "player.pt",
                    "eval-a" if i != 1 else "eval-b",
                    cap=3,
                )
            )
    if not (done and len(screens) == 3 and all(screens)):
        return "Longer imitation and 4,096-world selection"
    choice = first_selection()
    if choice["feedback"]:
        data = OUT / "feedback-data"
        if not j.ensure(
            "feedback-data", "collect", HERE / "distil.py", ["collect"], data, data / "summary.json"
        ):
            return "Labelling the student's own games"
        prev = j.panel(
            "feedback_screen", "previous", candidates[choice["chosen"]] / "player.pt", "eval-a", cap=3
        )
        correct = candidates["feedback"]
        if not j.ensure(
            "feedback-fit", "gpu", HERE / "distil.py", ["feedback"], correct, correct / "summary.json"
        ):
            return "Fitting the student with teacher feedback"
        challenger = j.panel("feedback_screen", "feedback", correct / "player.pt", "eval-b", cap=3)
        if not (prev and challenger):
            return "Independent selection of the corrective fit"
    seal_student()
    ck = {"champion": CHAMP, "student": OUT / "finalist/player.pt"}
    complete = [
        j.panel("retention", a, path, lane) for (a, path), lane in zip(ck.items(), ("eval-a", "eval-b"))
    ]
    fights = OUT / "boss-fights"
    if bank_info:
        pa = [
            "--checkpoint",
            CHAMP,
            "--combat",
            OUT / "finalist/head.pt",
            "--bank",
            bank_info["bank"],
            "--bank-world-start",
            bank_info["start"],
            "--bank-n",
            bank_info["n"],
            "--starts-per-group",
            json.dumps({"BOSS3": CFG["boss_starts"]}),
            "--resamples",
            1,
            "--grid",
            OUT / "grid.json",
            "--workers",
            6,
        ]
        fight_done = j.ensure(
            "boss-fights", "bank", RL / "fresh_panel.py", pa, fights, fights / "completion.json"
        )
    else:
        fight_done = False
    if not (all(complete) and fight_done):
        return "49,152-world comparison and paired boss replays"
    ret = retention()
    from publish import record_retention

    record_retention()
    if not checked:
        return "Checking both joint-training paths"
    ck = endpoints()
    trained = {}
    for arm in ("two", "one"):
        dest = OUT / ("rl-" + arm)
        trained[arm] = j.ensure(
            "rl-" + arm, "gpu", HERE / "training.py", [arm], dest, dest / "finished.json", resume=True
        )
    # Both trainers use the large MPS setup in one lane. Once one endpoint is frozen,
    # evaluate it and its parent alongside the other trainer with limited CPU concurrency.
    evaluated = []
    for arm in ("two", "one"):
        if trained[arm]:
            for label, lane in ((arm, "eval-a"), (arm + "-parent", "eval-b")):
                evaluated.append(
                    j.panel(
                        "rl_panel", label, ck[label], lane, cap=2 if not all(trained.values()) else None
                    )
                )
    if not all(trained.values()):
        return "24 joint updates per player on matched worlds"
    if not all(evaluated):
        return "Comparing the two trained players and their starting parents"
    selected = choose_endpoint()
    last = {"incumbent": Path(ret["incumbent"]), "nominee": ck[selected["chosen"]]}
    if not all(
        [
            j.panel("rl_confirmation", a, path, lane)
            for (a, path), lane in zip(last.items(), ("eval-a", "eval-b"))
        ]
    ):
        return "Independent champion confirmation"
    finish()
    return "finished"


def latest(path):
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 16384))
        for line in reversed(f.read().splitlines()):
            try:
                return json.loads(line)
            except (UnicodeError, ValueError):
                continue
    return {}


def heartbeat(title, jobs):
    parts = []
    for name, j in jobs.rows.items():
        if not name.startswith(PREFIX) or j["status"] not in ("running", "paused"):
            continue
        cmd = j.get("cmd", [])
        if "--out" not in cmd:
            continue
        folder = Path(cmd[cmd.index("--out") + 1])
        key = name[len(PREFIX) :]
        if folder in (OUT / "rl-one", OUT / "rl-two"):
            rows = []
            log = folder / "log.jsonl"
            if log.is_file():
                for line in log.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            n = max((int(r["iteration"]) for r in rows if r.get("event") == "train"), default=0)
            parts.append(f"{key} {n}/24 updates")
        else:
            row = latest(folder / "progress.jsonl")
            count = row.get("recorded", row.get("step", row.get("completed")))
            total = row.get("total", cmd[cmd.index("--n") + 1] if "--n" in cmd else None)
            if count is not None:
                parts.append(f"{key} {int(count):,}" + (f"/{int(total):,}" if total else ""))
    message = title + (". " + ", ".join(parts) + "." if parts else ".")
    note("overnight", message, log=False)
    atomic_json(
        OUT / "status.json",
        dict(
            stage=title,
            time=time.time(),
            jobs=parts,
            free_gib=round(shutil.disk_usage(OUT).free / 2**30, 2),
        ),
    )


def main():
    check_inputs()
    if not (OUT / "world-plan.json").is_file():
        from reserve import reserve

        reserve()
    with (OUT / "advance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last, previous = 0, None
        while not (OUT / "STOP").exists():
            if (OUT / "verdict.json").exists():
                from publish import publish

                publish()
                return
            if time.time() - read(RUNS / "runner_status.json")["heartbeat"] > 180:
                subprocess.run([sys.executable, QUEUE, "start"], check=True, cwd=ROOT)
            j = Jobs()
            title = stage(j)
            if title == "finished":
                from publish import publish

                publish()
                return
            if title != previous:
                note("stage", title + ".")
                from publish import status

                status(title + ". The local power-aware queue is running the frozen overnight design.")
            if title != previous or time.monotonic() - last > 60:
                heartbeat(title, j)
                last = time.monotonic()
            previous = title
            time.sleep(15)
    note(
        "stopped",
        "Overnight scheduling stopped at the study control file. Saved outputs are available.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        note("attention", f"Overnight study needs attention: {type(exc).__name__}: {exc}")
        raise
