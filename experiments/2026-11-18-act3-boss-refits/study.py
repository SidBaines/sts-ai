"""Shared paths and command construction for the Act 3 boss study."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "ppo.json").read_text())
RUN = ROOT / "data/rl_runs/joint_a3_from_0f52"
CHAMPION = ROOT / CONFIG["checkpoint"]
SHA256 = "0f527a0fbf93b5cd856229d9a38ce2aa9c86b31f87f1c2c19eb2dd54823819ff"
PREFIX = "joint-a3-"
NATIVE = (ROOT / "external/sts_lightspeed_rl/build013_clean").resolve(strict=True)


def submit(name, lane, tool, args, *, prefix=PREFIX):
    cmd = [sys.executable, "scripts/rl/queue_runner.py", "add", prefix + name,
           "--lane", lane, "--", "env", "PYTHONDONTWRITEBYTECODE=1",
           "STS_LIGHTSPEED_BUILD=" + str(NATIVE), sys.executable, "-u", tool, *args]
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def train(name, folder, parent, end, resume, *, prefix=PREFIX):
    cfg = dict(CONFIG, checkpoint=str(parent), out=str(folder), iterations=end)
    args = []
    for key, value in cfg.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            args.append("--" + ("" if value else "no-") + flag)
        elif value is not None:
            args.extend(("--" + flag, str(value)))
    if resume:
        args.append("--resume")
    submit(name, "joint-a3-gpu", "scripts/rl/train_ooc.py", args, prefix=prefix)


if __name__ == "__main__":
    if hashlib.sha256(CHAMPION.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Champion checkpoint identity mismatch")
    RUN.mkdir(exist_ok=False)
    train("control", RUN / "control", CHAMPION, 24, False)
    submit("probe", "joint-a3-probe", "scripts/rl/fresh_collect.py",
           ["--mode", "collect", "--no-teacher-labels", "--checkpoint", CHAMPION,
            "--world-start", 93_850_000_000, "--n", 32, "--workers", 3, "--out", RUN / "probe"])
