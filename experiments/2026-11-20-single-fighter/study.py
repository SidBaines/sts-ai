"""Settings, paths and live notes for the single-fighter experiment."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / "config.json").read_text())
RUNS = ROOT / "data/rl_runs"
OUT = RUNS / CFG["run"]
CHAMP = RUNS / CFG["champion"]
ROLES = ("ordinary", "boss")


def check_parent():
    if digest(CHAMP) != CFG["champion_sha256"]:
        raise ValueError("The teacher differs from the early-refit champion")


def note(stage, message, *, log=True):
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%d %b %H:%M GMT")
    with (OUT / "progress.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_json(RUNS / "study_heartbeat.json", dict(run=CFG["run"], stage=stage,
                    t=now.timestamp(), stamp=stamp, message=message))
        if log:
            path = ROOT / "research/log.md"
            title, old = path.read_text().split("\n", 1)
            if not any(row.endswith(" — " + message) for row in old.splitlines()):
                temp = path.with_suffix(".tmp")
                temp.write_text(f"{title}\n\n- {stamp} — {message}\n" + old.lstrip("\n"))
                temp.replace(path)
    print(stamp, message, flush=True)
