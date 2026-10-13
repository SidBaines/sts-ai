"""Background single-fight jobs for the GUI, using the existing isolated CLI."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Any, Callable
import uuid

from sts_ai import fight_lab as lab


class RunBusyError(RuntimeError):
    pass


class FightJobs:
    """One active GUI job at a time; browsing stays independent of native/MLX work."""

    def __init__(self, root: Path, collection: dict[str, Any], *,
                 command_builder: Callable[[Path, Path], list[str]] | None = None,
                 timeout: float = 1830):
        self.root = root.resolve()
        self.collection = collection
        self.directory = self.root / ".gui_jobs"
        self.directory.mkdir(exist_ok=True)
        self.command_builder = command_builder or self._command
        self.timeout = timeout
        self.lock = threading.RLock()
        self.active: str | None = None
        self.process: subprocess.Popen | None = None
        self.run_lock = None
        self.closed = False
        self.jobs = {p.stem: lab.read_json(p) for p in self.directory.glob("*.json")}
        # Graceful shutdown cancels workers; a hard restart keeps finished artifacts
        # and marks jobs without a result interrupted instead of presenting a win.
        for job in self.jobs.values():
            if job["status"] == "running":
                self._finish(job, "interrupted")

    def _command(self, selected: Path, output: Path) -> list[str]:
        script = Path(__file__).resolve().parents[2] / "scripts/fight_lab.py"
        return [sys.executable, str(script), "eval", "--root", str(self.root),
                "--selection", str(selected), "--output", str(output)]

    def _save(self, job: dict[str, Any]) -> None:
        path = self.directory / f"{job['job_id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def start(self, state_id: str) -> dict[str, Any]:
        selected = lab.selection(self.collection, lab.StateFilter(), [state_id])
        with self.lock:
            if self.closed:
                raise RunBusyError("the viewer is shutting down")
            if self.active is not None:
                raise RunBusyError("a fight is already running; wait or cancel it first")
            run_lock = (self.directory / "worker.lock").open("a")
            try:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                run_lock.close()
                raise RunBusyError("another viewer worker is still running a fight")
            self.run_lock = run_lock
            job_id = "gui_" + uuid.uuid4().hex[:12]
            input_dir = self.directory / job_id
            input_dir.mkdir()
            selected_path = input_dir / "selection.json"
            lab.write_new(selected_path, selected)
            output = self.root / "runs" / job_id
            job = {"job_id": job_id, "state_id": state_id, "status": "running",
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "output": str(output.relative_to(self.root)),
                   "selection_sha256": selected["selection_sha256"]}
            self.jobs[job_id] = job
            self._save(job)
            try:
                with (input_dir / "launcher.log").open("x") as log:
                    self.process = subprocess.Popen(self.command_builder(selected_path, output),
                        cwd=Path(__file__).resolve().parents[2],
                        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                             "PYTHONUNBUFFERED": "1"},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(run_lock.fileno(),))
            except OSError as exc:
                self.run_lock.close()
                self.run_lock = None
                job["error"] = str(exc)
                self._finish(job, "error")
                return self.get(job_id)
            self.active = job_id
            threading.Thread(target=self._watch, args=(job_id, self.process), daemon=True).start()
            return self.get(job_id)

    def _finish(self, job: dict[str, Any], fallback: str) -> None:
        report_path = self.root / job["output"] / "report.json"
        if report_path.exists():
            try:
                report = lab.read_json(report_path)
                episode = report["episodes"][0]
                job["result"] = episode
                job["status"] = "completed" if report["completed"] == 1 else "incomplete"
                if episode.get("status") != "ok":
                    job["status"] = episode.get("status", "error")
                meta = (self.root / job["output"] / episode["trace"]).with_suffix(".meta.json")
                if meta.exists():
                    job["trajectory"] = job["output"] + "/" + episode["trace"]
            except (ValueError, KeyError, IndexError):
                job["status"] = fallback
        else:
            job["status"] = fallback
        if job["status"] != "completed":
            logs = sorted((self.root / job["output"]).glob("*.log"))
            logs.append(self.directory / job["job_id"] / "launcher.log")
            for path in logs:
                if path.exists() and path.stat().st_size:
                    with path.open("rb") as handle:
                        handle.seek(max(0, path.stat().st_size - 4000))
                        job["error"] = handle.read().decode(errors="replace")
                    break
        self._save(job)

    def _watch(self, job_id: str, process: subprocess.Popen) -> None:
        fallback = "error"
        try:
            process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._kill(process)
            process.wait()
            fallback = "timeout"
        with self.lock:
            if self.jobs[job_id]["status"] == "running":
                self._finish(self.jobs[job_id], fallback)
            if self.active == job_id:
                self.active = None
                self.process = None
                if self.run_lock is not None:
                    self.run_lock.close()
                    self.run_lock = None

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                # The CLI launches its own native/MLX worker: kill the whole group.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs[job_id]
            if self.active == job_id and self.process is not None:
                self._kill(self.process)
                self.process.wait()
                self._finish(job, "cancelled")
                self.active = None
                self.process = None
                if self.run_lock is not None:
                    self.run_lock.close()
                    self.run_lock = None
            return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = dict(self.jobs[job_id])
            job["decisions"] = 0
            for path in (self.root / job["output"]).glob("*.jsonl"):
                # A writer may be midway through the final line; count complete rows.
                with path.open("rb") as handle:
                    job["decisions"] += sum(line.endswith(b"\n") for line in handle)
            return job

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self.get(k) for k in sorted(self.jobs, key=lambda k: self.jobs[k]["created_at"], reverse=True)]

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if self.active is not None:
                self.cancel(self.active)
