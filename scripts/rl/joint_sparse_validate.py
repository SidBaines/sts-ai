#!/usr/bin/env python3
"""Required regression suite and pinned native rollout smoke, via queue."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from sts_rl.recording import begin_run

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    record = begin_run(out, vars(a))
    env = {**os.environ, "STS_REQUIRE_SIMULATOR": "1", "PYTHONPATH": "src"}
    results = []
    for command in [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        [
            sys.executable,
            "scripts/run_rollout.py",
            "--agent",
            "random",
            "--seed",
            "3",
            "--max-act",
            "1",
            "--battle-simulations",
            "50",
        ],
    ]:
        start = time.monotonic()
        r = subprocess.run(command, env=env, timeout=600, check=False)
        results.append(
            {
                "command": command,
                "returncode": r.returncode,
                "seconds": time.monotonic() - start,
            }
        )
        if r.returncode:
            raise RuntimeError(f"Validation failed: {command}")
    (out / "acceptance.json").write_text(
        json.dumps({"record": record, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
