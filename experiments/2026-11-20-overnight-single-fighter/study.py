"""Fixed inputs and progress publication for the overnight single-fighter study."""

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "data/rl_runs"
CFG = json.loads((HERE / "config.json").read_bytes())
OUT = RUNS / CFG["run"]
OLD = RUNS / CFG["previous_run"]
DEMO = OLD / "demonstrations"
CHAMP = RUNS / CFG["champion"]
BASE = OLD / "finalist"


def read(path):
    return json.loads(path.read_bytes())


def check_inputs():
    if (
        digest(CHAMP) != CFG["champion_sha256"]
        or digest(BASE / "player.pt") != CFG["original_student_sha256"]
    ):
        raise ValueError("The starting champion or original student has changed")


def note(stage, message, *, log=True):
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%d %b %H:%M GMT")
    with (OUT / "progress.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_json(
            RUNS / "study_heartbeat.json",
            dict(run=CFG["run"], stage=stage, t=now.timestamp(), stamp=stamp, message=message),
        )
        if log:
            path = ROOT / "research/log.md"
            title, old = path.read_text().split("\n", 1)
            if not any(line.endswith(" — " + message) for line in old.splitlines()):
                tmp = path.with_suffix(".tmp")
                tmp.write_text(f"{title}\n\n- {stamp} — {message}\n" + old.lstrip("\n"))
                tmp.replace(path)
    print(stamp, message, flush=True)


def models():
    return {
        "original": BASE,
        **{f"step-{i}": OUT / "long-fit" / f"step-{i}" for i in CFG["snapshots"]},
        "feedback": OUT / "feedback-fit",
    }
