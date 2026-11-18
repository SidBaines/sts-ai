"""Time identical workloads in fresh processes with one selectable environment source tree."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "candidate",
        "workstream",
        "python",
        "native",
        "run-root",
        "out",
    ):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument(
        "--suite", choices=["eval", "collect", "train", "env"], required=True
    )
    p.add_argument("--order", default="abba")
    p.add_argument("--games", default=32, type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compare-diagnostics", action="store_true")
    p.add_argument("--collectors", type=int, default=2)
    p.add_argument("--production-shape", action="store_true")
    p.add_argument(
        "--combined",
        action="store_true",
        help="Candidate environment plus GPU diagnostics on arm b.",
    )
    p.add_argument("--original-workstream", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for i, arm in enumerate(args.order):
        source = (
            args.candidate
            if args.compare_diagnostics
            else {"a": args.baseline, "b": args.candidate}[arm]
        ).resolve()
        item = args.out / f"{i}-{arm}"
        workstream = (
            args.original_workstream
            if arm == "a" and args.original_workstream
            else args.workstream
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            STS_LIGHTSPEED_BUILD=str(args.native.resolve()),
            PYTHONPATH=os.pathsep.join(
                [str(source / "src"), str(workstream.resolve() / "src")]
            ),
        )
        if args.suite == "train":
            cmd = [
                HERE / "train_case.py",
                "--out",
                item,
                "--run-root",
                args.run_root,
                "--games",
                str(args.games),
                "--device",
                args.device,
                "--collectors",
                str(args.collectors),
            ]
            if args.production_shape:
                cmd.append("--production-shape")
            if (args.compare_diagnostics or args.combined) and arm == "b":
                cmd.append("--diagnostics-on-update-device")
        elif args.suite == "env":
            cmd = [source / "scripts/benchmark_env_step.py", "--repeats", "5"]
        else:
            ck = args.run_root / "champions/champion_4058cbf0e271.pt"
            cmd = [
                HERE / "profile_policy.py",
                "--output",
                item.with_suffix(".json"),
                "--checkpoint",
                ck,
                "--games",
                str(args.games),
                "--slots",
                "16",
            ]
            if args.suite == "collect":
                cmd.append("--collect")
        start = time.monotonic()
        with item.with_suffix(".log").open("x") as log:
            result = subprocess.run(
                [args.python, "-u", *cmd],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200 if args.production_shape else 600,
            )
        record = {
            "arm": arm,
            "suite": args.suite,
            "seconds_including_startup": time.monotonic() - start,
            "code": str(source),
            "device": args.device,
            "exit": result.returncode,
        }
        with (args.out / "timings.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)
        result.check_returncode()


if __name__ == "__main__":
    main()
