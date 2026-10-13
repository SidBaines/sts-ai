"""Queue an otherwise matched thinking-on arm after a frozen fight run finishes.

This separate launcher leaves the active sampler's fingerprinted files untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from scripts import sample_fight_bank as sample
from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


SCRIPT = Path(__file__).resolve()


def prepare(after: Path, output: Path) -> None:
    manifest, selected, settings = sample.validate(after)
    if settings["thinking"] is not False:
        raise ValueError("predecessor must have thinking disabled")
    output.mkdir(parents=True, exist_ok=False)
    lab.write_new(output / "selection.json", selected)
    lab.write_new(output / "settings.json", {**settings, "thinking": True})
    dependency = {
        "path": str(after.resolve()),
        "files": {name: file_sha256(after / name)
                  for name in ("manifest.json", "settings.json", "selection.json")},
        "trigger": "all scheduled attempts finished and predecessor run lock released",
    }
    copied = {**manifest,
              "settings_sha256": file_sha256(output / "settings.json"),
              "created_at": datetime.now(timezone.utc).isoformat(),
              "followup": dependency}
    lab.write_new(output / "manifest.json", copied)
    lab.write_new(output / "queue.json", {
        "launcher_sha256": file_sha256(SCRIPT),
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "after": str(after.resolve()), "output": str(output.resolve()),
    })
    for name, expected in copied["implementation"].items():
        source = SCRIPT.parent.parent / name
        if file_sha256(source) != expected:
            raise ValueError(f"implementation changed while preparing: {name}")
        target = output / "code_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(SCRIPT, output / "code_snapshot" / SCRIPT.name)
    sample.progress(output, copied, "queued")
    write_status(output, "queued")


def write_status(output: Path, status: str, **extra: Any) -> None:
    sample.atomic(output / "queue_status.json", {
        "status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
    })


def validate_dependency(output: Path) -> tuple[Path, dict[str, Any]]:
    queue = lab.read_json(output / "queue.json")
    if file_sha256(SCRIPT) != queue["launcher_sha256"]:
        raise ValueError("queue launcher changed")
    if file_sha256(output / "manifest.json") != queue["manifest_sha256"]:
        raise ValueError("queued manifest changed")
    manifest = lab.read_json(output / "manifest.json")
    after = Path(manifest["followup"]["path"])
    if queue["after"] != str(after) or queue["output"] != str(output.resolve()):
        raise ValueError("queue paths changed")
    for name, expected in manifest["followup"]["files"].items():
        if file_sha256(after / name) != expected:
            raise ValueError(f"predecessor configuration changed: {name}")
    return after, manifest


def lock_available(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def ready(output: Path) -> bool:
    after, manifest = validate_dependency(output)
    progress = lab.read_json(after / "progress.json")
    expected = len(manifest["order"])
    if (progress.get("status") != "finished"
            or progress.get("finished") != expected
            or progress.get("expected") != expected):
        return False
    if not all((after / f"{state_id}.result.json").exists() for state_id in manifest["order"]):
        return False
    return (lock_available(after / "run.lock")
            and lock_available(Path(manifest["root"]) / ".gui_jobs/worker.lock"))


def watch(output: Path, *, resume: bool = False) -> None:
    with (output / "queue.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            (output / "STOP").unlink(missing_ok=True)
        (output / "queue.pid").write_text(str(os.getpid()) + "\n")
        stopping = False

        def stop(_signum: int, _frame: Any) -> None:
            nonlocal stopping
            stopping = True
            (output / "STOP").touch()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not stopping and not (output / "STOP").exists():
                if ready(output):
                    after, _ = validate_dependency(output)
                    sample.validate(after)
                    sample.validate(output)
                    if (output / "STOP").exists():
                        break
                    write_status(output, "running")
                    _, manifest = validate_dependency(output)
                    budget = None
                    runner = sample.SCRIPT
                    if manifest.get("budget_policy"):
                        from scripts import thinking_fight_budget as budget
                        budget.validated(output)
                        runner = budget.SCRIPT
                    increase = None
                    with (output / "supervisor.log").open("ab") as log:
                        child = subprocess.Popen(
                            [sys.executable, str(runner), "resume", "--output", str(output)],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                (str(SCRIPT.parent.parent / "src"), str(SCRIPT.parent.parent)))},
                        )
                        while child.poll() is None:
                            if stopping or (output / "STOP").exists():
                                child.send_signal(signal.SIGTERM)
                            elif budget is not None:
                                previous = lab.read_json(output / "budget_status.json")
                                if previous["action"] in ("observe", "raise"):
                                    check = budget.assessment(budget.observations(output),
                                                              previous["max_tokens"])
                                    sample.atomic(output / "budget_status.json", check)
                                    if check["action"] == "raise":
                                        increase = check
                                        child.send_signal(signal.SIGTERM)
                                        child.wait(timeout=15)
                                        break
                            time.sleep(2)
                    if increase is not None and not stopping and not (output / "STOP").exists():
                        model_lock = Path(manifest["root"]) / ".gui_jobs/worker.lock"
                        if not lock_available(output / "run.lock") or not lock_available(model_lock):
                            raise RuntimeError("worker locks remain held after budget stop")
                        budget.raise_budget(output, increase)
                        write_status(output, "budget_increased")
                        continue
                    state = lab.read_json(output / "progress.json")
                    if budget is not None and state["status"] == "finished":
                        previous = lab.read_json(output / "budget_status.json")
                        if previous["action"] == "observe":
                            check = budget.assessment(budget.observations(output),
                                                      previous["max_tokens"], finished=True)
                            sample.atomic(output / "budget_status.json", check)
                            if check["action"] == "raise":
                                budget.raise_budget(output, check)
                                continue
                    write_status(output, state["status"] if child.returncode == 0 else "error",
                                 supervisor_returncode=child.returncode)
                    return
                after, _ = validate_dependency(output)
                state = lab.read_json(after / "progress.json")
                write_status(output, "waiting", after_status=state["status"],
                             after_finished=state["finished"], after_expected=state["expected"])
                time.sleep(10)
            write_status(output, "cancelled")
        except Exception as exc:
            write_status(output, "error", error=f"{type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "watch", "status", "stop"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        if args.after is None:
            parser.error("prepare requires --after")
        prepare(args.after.resolve(), output)
    elif args.command == "watch":
        watch(output, resume=args.resume)
    elif args.command == "stop":
        (output / "STOP").touch()
    else:
        import json
        report = {"queue": lab.read_json(output / "queue_status.json"),
                  "run": lab.read_json(output / "progress.json")}
        if (output / "budget_status.json").exists():
            report["budget"] = lab.read_json(output / "budget_status.json")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
