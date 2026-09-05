"""Run a recorded finite list of MLX diagnostic arms sequentially."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def arm_log_path(output: Path) -> Path:
    """Append a log extension without removing a decimal from an arm name."""
    return output.with_name(output.name + ".log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    attempts = []
    for arm in plan["arms"]:
        command = [sys.executable, "scripts/experiments/astra_policy_probe.py", arm["mode"],
                   "--out", arm["out"], "--model", plan["model"],
                   "--contract", arm["contract"], "--batch-size", str(plan["batch_size"])]
        for name in ("source", "adapter", "seeds", "max_tokens", "max_decisions", "max_retries", "temperature"):
            if name in arm:
                command.extend(["--" + name.replace("_", "-"), str(arm[name])])
        if arm.get("constrained"):
            command.append("--constrained")
        output = Path(arm["out"])
        if output.exists():
            raise FileExistsError(f"fresh suite requires absent output: {output}")
        log_path = arm_log_path(output)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print("START", arm["out"], flush=True)
        with log_path.open("x") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=arm.get("timeout_seconds", 900))
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
        attempt = {"arm": arm, "command": command, "status": status,
                   "started": started, "wall_seconds": time.time()-started}
        attempts.append(attempt)
        args.plan.with_suffix(".attempts.json").write_text(json.dumps(attempts, indent=2))
        print("DONE", json.dumps(attempt), flush=True)
        if status != 0:
            print(log_path.read_text()[-2000:], flush=True)


if __name__ == "__main__":
    main()
