"""Launch the shared prefix and the unlabelled movement probe through the local queue."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PPO = json.loads((HERE / "ppo.json").read_bytes())
RUN = (ROOT / PPO["out"]).parent
CHAMPION = ROOT / PPO["checkpoint"]
CHAMPION_SHA = "9700741dca15cbf451e5ad574a43283f076cccbfdfd1a91f6a3f43375c122c21"
NATIVE = (ROOT / "external/sts_lightspeed_rl/build013_clean").resolve(strict=True)
PREFIX = "a3-timing-"
SCHEDULES = {"early": (2, 4, 6), "spaced": (8, 16, 22)}
LABELS = (90_875_002_000, 90_875_004_000, 90_875_006_000)
BANKS = (92_875_002_000, 92_875_004_000, 92_875_006_000)
GATE_GAMES = (176_001_000_000, 176_002_000_000, 176_003_000_000)
SELECTION = 180_000_000_000
CONFIRMATION = 181_000_000_000
N = 16_384


def submit(name, lane, tool, args, *, prefix=PREFIX):
    cmd = [
        sys.executable,
        "scripts/rl/queue_runner.py",
        "add",
        prefix + name,
        "--lane",
        lane,
        "--",
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "STS_LIGHTSPEED_BUILD=" + str(NATIVE),
        sys.executable,
        "-u",
        tool,
        *args,
    ]
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def train(name, folder, template, end, resume):
    args = []
    cfg = dict(PPO, checkpoint=str(template), out=str(folder), iterations=end)
    for key, value in cfg.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            args.append("--" + ("" if value else "no-") + flag)
        elif value is not None:
            args.extend(("--" + flag, value))
    if resume:
        args.append("--resume")
    submit(name, "refit-timing-ppo", "scripts/rl/train_ooc.py", args)


def launch():
    if hashlib.sha256(CHAMPION.read_bytes()).hexdigest() != CHAMPION_SHA:
        raise ValueError("Starting champion changed")
    RUN.mkdir(exist_ok=False)
    train("shared-08", RUN / "shared", CHAMPION, 8, False)
    submit(
        "probe",
        "refit-timing-probe",
        "scripts/rl/fresh_collect.py",
        [
            "--mode",
            "collect",
            "--no-teacher-labels",
            "--checkpoint",
            CHAMPION,
            "--world-start",
            93_875_000_000,
            "--n",
            32,
            "--workers",
            3,
            "--out",
            RUN / "probe",
        ],
    )


if __name__ == "__main__":
    launch()
