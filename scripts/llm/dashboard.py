#!/usr/bin/env python3
"""Read-only, dependency-free dashboard for local MLX teacher-feedback runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse

RATES = ("1e-5", "3e-5", "1e-4")
PHASES = ("all", "combat", "out_of_combat")


class Records:
    """Cache completed JSONL lines; tolerate a writer's unfinished final line."""
    def __init__(self):
        self.cache = {}

    def read(self, path):
        try:
            st = path.stat()
        except FileNotFoundError:
            return []
        key = (st.st_mtime_ns, st.st_size)
        if path in self.cache and self.cache[path][0] == key:
            return self.cache[path][1]
        rows = []
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                break
            if line.strip():
                row = json.loads(line)
                row.pop("output_token_ids", None)
                rows.append(row)
        self.cache[path] = (key, rows)
        return rows


def tail(path, count=7000):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - count))
        return stream.read().decode(errors="replace")


def metrics(rows, states):
    if not rows:
        return {}
    valid = [r for r in rows if r["valid"]]
    def mean(key):
        return statistics.fmean(r[key] for r in rows)
    tokens = sorted(r["reasoning_tokens"] for r in rows)
    at_a = [r for r in rows if r["order"][0] == states[r["id"]]["teacher_top"]]
    elsewhere = [r for r in rows if r["order"][0] != states[r["id"]]["teacher_top"]]
    return {
        "decisions": len(rows), "valid_count": len(valid),
        "valid_fraction": len(valid) / len(rows),
        "mean_teacher_probability": mean("teacher_probability"),
        "mean_reward": mean("reward"),
        "teacher_top_agreement": mean("teacher_top_agreement"),
        "median_reasoning_tokens": statistics.median(tokens),
        "p95_reasoning_tokens": tokens[min(len(tokens) - 1, (95 * len(tokens) + 99) // 100 - 1)],
        "empty_reasoning_fraction": sum(t == 0 for t in tokens) / len(tokens),
        "capped_fraction": mean("capped"),
        "first_displayed_option_rate": (statistics.fmean(r["first_displayed_option"] for r in valid)
                                         if valid else None),
        "content_only_a_expectation": (statistics.fmean(1 / len(r["order"]) for r in valid)
                                        if valid else None),
        "top_at_a": {"n": len(at_a), "agree": sum(r["teacher_top_agreement"] for r in at_a)},
        "top_elsewhere": {"n": len(elsewhere), "agree": sum(r["teacher_top_agreement"] for r in elsewhere)},
    }


class Dashboard:
    def __init__(self, root, screen, reference):
        self.root, self.screen, self.reference = root, screen, reference
        self.records = Records()
        self.lock = threading.Lock()
        self.cached_at, self.cached = 0, None
        rows = []
        for name in (screen, reference):
            for config_path in (root / "data/llm_runs" / name).glob("lr_*/config.json"):
                settings = json.loads(config_path.read_text())["settings"]
                for key in ("train", "probe", "train_eval"):
                    if settings.get(key):
                        path = Path(settings[key])
                        rows += self.records.read(path if path.is_absolute() else root / path)
        self.states = {row["id"]: row for row in rows}

    def processes(self):
        try:
            result = subprocess.run(["ps", "-axo", "pid,ppid,pgid,stat,etime,args"],
                                    capture_output=True, text=True, timeout=3, check=True)
            return [line.split(None, 5) for line in result.stdout.splitlines()[1:]]
        except (OSError, subprocess.SubprocessError):
            return None

    def run(self, screen, reference, processes):
        folder = self.root / "data/llm_runs" / screen
        screen_log = tail(folder / "screen.log")
        match = re.search(r"screen_pgid\s+(\d+)", tail(folder / "pids.txt"))
        pgid = match.group(1) if match else None
        group = [p for p in (processes or []) if len(p) == 6 and p[2] == pgid]
        trainer = next((p for p in group if "-m sts_llm.mlx_training " in p[5]), None)
        screen_alive = any(p[0] == pgid for p in group)
        arms = []
        rates = sorted(p.name[3:] for p in folder.glob("lr_*") if (p / "config.json").exists())
        for rate in rates:
            arm_folder = folder / f"lr_{rate}"
            config_file = arm_folder / "config.json"
            config = json.loads(config_file.read_text()) if config_file.exists() else {}
            settings = config.get("settings", {})
            iterations = self.records.read(arm_folder / "iterations.jsonl")
            evaluations = self.records.read(arm_folder / "evaluations.jsonl")
            progress_file = arm_folder / "evaluation_progress.json"
            evaluation_progress = json.loads(progress_file.read_text()) if progress_file.exists() else None
            completed = max((r["iteration"] for r in iterations), default=0)
            is_live = trainer is not None and f"lr_{rate}" in trainer[5]
            log = tail(folder / f"lr_{rate}.log")
            exits = re.findall(rf"end learning rate {re.escape(rate)} exit (\d+)", screen_log)
            exit_code = int(exits[-1]) if exits else None
            outcome_file = folder / "outcomes.json"
            if outcome_file.exists():
                for outcome in json.loads(outcome_file.read_text()):
                    if outcome["arm"]["key"] == rate:
                        exit_code = outcome["exit_code"]
            failed = (exit_code is not None and exit_code != 0) or "Traceback (most recent call last)" in log
            summary_path = arm_folder / "run_summary.json"
            run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            stopped_training = '\"event\": \"stopped\"' in log
            final_step = run_summary.get("last_iteration", completed) if run_summary else completed
            final_eval = any(r["iteration"] == final_step and (final_step > 0 or settings.get("eval_only")) for r in evaluations)
            if failed:
                status = "failed"
            elif run_summary and final_eval:
                status = "complete" if run_summary["stop_reason"] in ("iterations_complete", "evaluation_only") else "stopped"
            elif completed >= settings.get("iterations", 48) and final_eval and exit_code == 0:
                status = "complete"
            elif is_live:
                if stopped_training:
                    status = "evaluation paused" if "T" in trainer[3] else "evaluating"
                else:
                    status = "paused" if "T" in trainer[3] else "running"
            elif exit_code == 0 or '"event": "stopped"' in log:
                status = "stopped"
            elif config and processes is not None and not screen_alive:
                status = "interrupted"
            elif not config:
                status = "queued" if screen_alive and not reference else "not started"
            else:
                status = "waiting" if processes is not None else "process unknown"
            paths = [p for p in [arm_folder / "iterations.jsonl", arm_folder / "evaluations.jsonl",
                                 folder / f"lr_{rate}.log", progress_file] if p.exists()]
            age = time.time() - max((p.stat().st_mtime for p in paths), default=time.time())
            power_log = folder / "babysitter.log"
            since_power_event = time.time() - power_log.stat().st_mtime if power_log.exists() else age
            active_age = min(age, since_power_event)
            times = [r["iteration_seconds"] for r in iterations[-8:]]
            recent_seconds = statistics.median(times) if times else None
            stall_after = max(600, (recent_seconds or 100) * 5)
            all_generations = []
            passes = []
            pass_size = max(1, config.get("train_states", 32) // settings.get("states_per_iteration", 4))
            for start in range(1, completed + 1, pass_size):
                end = min(completed, start + pass_size - 1)
                rows = []
                for i in range(start, end + 1):
                    rows.extend(self.records.read(arm_folder / "generations" / f"iter_{i:04d}.jsonl"))
                all_generations.extend(rows)
                blocks = {phase: metrics([r for r in rows if phase == "all" or r["phase"] == phase], self.states)
                          for phase in PHASES}
                passes.append({"pass": (start - 1) // pass_size + 1, "iteration": end,
                               "complete": end == start + pass_size - 1, **blocks})
            for row in iterations:
                # Add the menu-size-matched reference to the recorded iteration summaries.
                raw = self.records.read(arm_folder / "generations" / f"iter_{row['iteration']:04d}.jsonl")
                for phase in PHASES:
                    selected = [r for r in raw if r["valid"] and (phase == "all" or r["phase"] == phase)]
                    if row.get(phase):
                        row[phase]["content_only_a_expectation"] = (statistics.fmean(1 / len(r["order"]) for r in selected)
                                                                   if selected else None)
            arms.append({"rate": rate, "label": settings.get("label") or rate, "status": status, "completed": completed,
                         "target": settings.get("iterations", 48), "config": config,
                         "iterations": iterations, "evaluations": evaluations, "passes": passes,
                         "totals": {phase: metrics([r for r in all_generations
                                                   if phase == "all" or r["phase"] == phase], self.states)
                                    for phase in PHASES},
                         "last_write_age_seconds": age, "recent_iteration_seconds": recent_seconds,
                         "possible_stall": is_live and status == "running" and active_age > stall_after,
                         "long_evaluation": status == "evaluating" and active_age > 1200,
                         "active_record_age_seconds": active_age,
                         "run_summary": run_summary,
                         "evaluation_progress": evaluation_progress,
                         "process": {"pid": trainer[0], "state": trainer[3], "elapsed": trainer[4]} if is_live else None,
                         "log_tail": log[-3000:], "exit_code": exit_code,
                         "stop_requested": (arm_folder / "STOP").exists()})
        return {"name": screen, "reference": reference, "arms": arms,
                "screen_alive": screen_alive if processes is not None else None,
                "finished": "SCREEN_DONE" in screen_log, "log_tail": screen_log,
                "babysitter_log": tail(folder / "babysitter.log", 2000),
                "stop_requested": (folder / "STOP").exists(),
                "alerts": self.records.read(folder / "alerts.jsonl"),
                "stop_decision": json.loads((folder / "stop_decision.json").read_text())
                if (folder / "stop_decision.json").exists() else None}

    def snapshot(self):
        with self.lock:
            if time.monotonic() - self.cached_at < 3 and self.cached:
                return self.cached
            processes = self.processes()
            try:
                power = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                       text=True, timeout=3).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                power = "Power status unavailable"
            self.cached = {"updated": datetime.now(timezone.utc).isoformat(), "power": power,
                           "screens": [self.run(self.screen, False, processes),
                                       self.run(self.reference, True, processes)]}
            self.cached_at = time.monotonic()
            return self.cached

    def traces(self, screen, rate, iteration, kind):
        if screen not in (self.screen, self.reference) or kind not in ("generations", "evaluations"):
            raise ValueError("Unknown run")
        allowed = {p.name[3:] for p in (self.root / "data/llm_runs" / screen).glob("lr_*") if (p / "config.json").exists()}
        if rate not in allowed:
            raise ValueError("Unknown arm")
        if not 0 <= iteration <= 100000:
            raise ValueError("Unknown iteration")
        with self.lock:
            rows = self.records.read(self.root / "data/llm_runs" / screen / f"lr_{rate}" / kind / f"iter_{iteration:04d}.jsonl")
            return [{**row, "observation": self.states[row["id"]]["observation"],
                     "teacher_top": self.states[row["id"]]["teacher_top"]} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--screen", default="broader_training_20261026")
    parser.add_argument("--reference", default="stage1_screen_ps_20261026")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    dashboard = Dashboard(args.root, args.screen, args.reference)
    page = Path(__file__).with_name("dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            try:
                if route.path == "/":
                    body, mime = page, "text/html; charset=utf-8"
                elif route.path == "/api/status":
                    body, mime = json.dumps(dashboard.snapshot(), allow_nan=False).encode(), "application/json"
                elif route.path == "/api/traces":
                    q = parse_qs(route.query)
                    result = dashboard.traces(q["screen"][0], q["rate"][0], int(q["iteration"][0]), q["kind"][0])
                    body, mime = json.dumps(result, allow_nan=False).encode(), "application/json"
                elif route.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError) as exc:
                self.send_error(400, str(exc))
            except Exception as exc:
                print(f"dashboard error: {exc!r}", flush=True)
                self.send_error(500, "Cannot read current run records; see dashboard log")

        def log_message(self, fmt, *values):
            if len(values) > 1 and str(values[1]) != "200":
                super().log_message(fmt, *values)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LLM dashboard: http://{args.host}:{args.port} (read-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
