"""Submit the two matched trainer jobs to one GPU lane of the power-aware queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUEUER = ROOT / "scripts/rl/queue_runner.py"
RUN = ROOT / "data/rl_runs/ppo_fresh_worlds_from_4058"
ARMS = {"control": 0.0003, "challenger": 0.0001}
PREFIX = "ppo-rates24-"


def queue(*args):
    subprocess.run([sys.executable, str(QUEUER), *map(str, args)], cwd=ROOT, check=True)


def native_build():
    return str((ROOT / "external/sts_lightspeed_rl/build013_clean").resolve(strict=True))


def flags(config):
    for key, value in config.items():
        if value is None:
            continue
        name = key.replace("_", "-")
        if isinstance(value, bool):
            yield "--" + ("" if value else "no-") + name
        else:
            yield "--" + name
            yield str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume-arm", choices=ARMS)
    p.add_argument("--retry-name", help="Unique queue name for a verified interrupted run")
    args = p.parse_args()
    config = json.loads((HERE / "control.json").read_text())
    ck = ROOT / config["checkpoint"]
    assert hashlib.sha256(ck.read_bytes()).hexdigest() == "4058cbf0e271e8cd0ada21576cf0583368c112c53fc99c0523cc26d9af80943e"
    if args.resume_arm:
        if not args.retry_name or not (RUN / args.resume_arm / "latest.pt").is_file():
            p.error("Resuming requires the saved training state and a unique queue name")
    elif RUN.exists():
        p.error(f"New study output already exists: {RUN}")
    else:
        RUN.mkdir(parents=True)
    for arm, lr in ARMS.items():
        if args.resume_arm and arm != args.resume_arm:
            continue
        settings = dict(config, lr=lr, out=str(RUN / arm), checkpoint=str(ck))
        cmd = ["env", "PYTHONDONTWRITEBYTECODE=1", "STS_LIGHTSPEED_BUILD=" + native_build(),
               sys.executable, "-u", "scripts/rl/train_ooc.py", *flags(settings)]
        if args.resume_arm:
            cmd.append("--resume")
        queue("add", args.retry_name or PREFIX + arm, "--lane", "ppo-rates-gpu", "--", *cmd)
    queue("start")


if __name__ == "__main__":
    main()
