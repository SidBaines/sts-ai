"""Read-only overnight job monitoring and automatic three-arm report refreshes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.compare_fight_formats import DEFAULTS, ROOT
from scripts.sample_fight_bank import atomic
from sts_ai import fight_lab as lab


def observe(runs: dict[str, Path]) -> dict:
    process_table = subprocess.run(["ps", "-axo", "pid,ppid,etime,command"],
                                   text=True, capture_output=True, check=True, timeout=10).stdout.splitlines()
    state = {"updated_at": datetime.now(timezone.utc).isoformat(), "arms": {}, "alerts": []}
    for arm, path in runs.items():
        progress = lab.read_json(path / "progress.json")
        live = [line.strip() for line in process_table
                if str(path) in line and any(name in line for name in
                    ("sample_fight_bank.py", "thinking_fight_budget.py", "scripts.queue_thinking_fights"))
                and int(line.split()[0]) != os.getpid()]
        item = {"progress": progress, "live_processes": live}
        for name in ("queue_status.json", "budget_status.json", "budget_history.json"):
            if (path / name).exists():
                item[name] = lab.read_json(path / name)
        active = progress.get("active", {})
        if active.get("trace"):
            trace = path / active["trace"]
            if trace.exists():
                text = trace.read_bytes()
                item["active_saved_decisions"] = text.count(b"\n")
                item["active_trace_age_seconds"] = time.time() - trace.stat().st_mtime
        if arm != "old_format" and progress["status"] != "finished" and not live:
            state["alerts"].append(f"{arm}: unfinished with no verified live sampler/queue handle")
        if item.get("queue_status.json", {}).get("status") == "error":
            state["alerts"].append(f"{arm}: queue reports error")
        state["arms"][arm] = item
    return state


def terminal_results(runs: dict[str, Path], state: dict) -> bool:
    for arm in ("new_no_thinking", "new_thinking"):
        progress = state["arms"][arm]["progress"]
        manifest = lab.read_json(runs[arm] / "manifest.json")
        expected = len(manifest["order"])
        if progress["status"] != "finished" or progress["finished"] != expected:
            return False
        if not all((runs[arm] / f"{sid}.result.json").exists() for sid in manifest["order"]):
            return False
        if state["arms"][arm].get("queue_status.json", {}).get("status", "finished") != "finished":
            return False
    return True


def refresh_checkpoint(state: dict, done: bool) -> tuple:
    off = state["arms"]["new_no_thinking"]["progress"]
    thinking = state["arms"]["new_thinking"]
    n = thinking["progress"]["finished"]
    return (off["finished"] // 50, off["status"] == "finished",
            min(n, 5), n // 10, thinking.get("budget_status.json", {}).get("max_tokens"), done)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gemma_overnight_monitor_2026-09-06")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs/gemma_three_arm_comparison_2026-09-06")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    monitor_lock = (output / "monitor.lock").open("a")
    fcntl.flock(monitor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / "monitor.pid").write_text(str(os.getpid()) + "\n")
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_refresh = None
    while not stopping:
        try:
            state = observe(DEFAULTS)
            atomic(output / "status.json", state)
            with (output / "observations.jsonl").open("a") as handle:
                handle.write(json.dumps(state) + "\n")
            done = terminal_results(DEFAULTS, state)
            milestone = refresh_checkpoint(state, done)
            if milestone != last_refresh:
                command = [sys.executable, str(ROOT / "scripts/compare_fight_formats.py"),
                           "--output", str(args.report_output.resolve())]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        timeout=300, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
                with (output / "report_refresh.log").open("a") as log:
                    log.write(state["updated_at"] + "\n" + result.stdout + result.stderr)
                if result.returncode:
                    state["alerts"].append("report refresh failed; retry next poll")
                    atomic(output / "status.json", state)
                else:
                    last_refresh = milestone
                    if done:
                        atomic(output / "finished.json", {"updated_at": state["updated_at"],
                               "latest_report": lab.read_json(args.report_output / "latest.json")})
                        return
        except Exception as exc:
            with (output / "observation_errors.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": time.time(), "error": repr(exc)}) + "\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
