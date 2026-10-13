"""Interruptible, resumable sampling of a frozen fight bank with one MLX model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


SCRIPT = Path(__file__).resolve()


def atomic(path: Path, value: Any) -> None:
    import json
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sampling_seed(state_id: str) -> int:
    return int(lab.digest({"state_id": state_id, "replicate": 0})[:8], 16)


def implementation() -> dict[str, str]:
    root = SCRIPT.parent.parent
    names = ["scripts/sample_fight_bank.py", "src/sts_ai/fight_lab.py",
             "src/sts_ai/agents.py", "src/sts_ai/agent_factory.py", "src/sts_ai/local_tasks/runner.py",
             "src/sts_ai/rollout.py", "src/sts_ai/provenance.py", "src/sts_ai/lightspeed.py",
             "src/sts_ai/glossary.py", "src/sts_ai/turn_math.py", "src/sts_ai/prompting.py",
             "src/sts_ai/combat_public_v4.py", "src/sts_ai/glossary_v4.py", "src/sts_ai/turn_math_v4.py",
             "src/sts_ai/combat_public_v5.py", "src/sts_ai/glossary_v5.py",
             "src/sts_ai/train/sft_format.py", "src/sts_ai/fight_analysis.py"]
    return {name: file_sha256(root / name) for name in names}


def runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")}


def baseline_cohort(root: Path, baseline: Path) -> tuple[list[str], dict[str, Any]]:
    """Freeze attempted starts, including errors and partials, without cherry-picking."""
    manifest = lab.read_json(baseline / "manifest.json")
    if Path(manifest["root"]).resolve() != root.resolve():
        raise ValueError("baseline belongs to a different bank")
    if (manifest["model_id"], manifest["model_revision"]) != (lab.MODEL_ID, lab.MODEL_REVISION):
        raise ValueError("baseline model identity differs")
    if file_sha256(baseline / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("baseline sampling settings changed")
    selected = lab.read_json(baseline / "selection.json")
    lab.validate_selection(lab.read_json(root / "collection.json"), selected)
    if selected["selection_sha256"] != manifest["selection_sha256"]:
        raise ValueError("baseline selection changed")
    allowed = {row["state_id"] for row in selected["states"]}
    if set(manifest["order"]) != allowed or len(manifest["order"]) != len(allowed):
        raise ValueError("baseline schedule differs from selection")
    ids: list[str] = []
    hashes = {name: file_sha256(baseline / name)
              for name in ("manifest.json", "settings.json", "selection.json")}
    for state_id in manifest["order"]:
        trace = baseline / f"{state_id}_r0.jsonl"
        result_path = baseline / f"{state_id}.result.json"
        if not result_path.exists() and not (trace.exists() and trace.stat().st_size):
            continue
        if manifest["sampling_seeds"].get(state_id) != sampling_seed(state_id):
            raise ValueError("baseline uses different sampling seeds")
        ids.append(state_id)
        if result_path.exists():
            saved = lab.read_json(result_path)
            if saved["state_id"] != state_id:
                raise ValueError("baseline result identity changed")
            if saved["status"] == "ok":
                for path, key in ((trace, "trace_sha256"),
                                  (trace.with_suffix(".meta.json"), "meta_sha256")):
                    if file_sha256(path) != saved[key]:
                        raise ValueError("baseline finished trajectory changed")
        for path in (trace, trace.with_suffix(".meta.json"), result_path):
            if path.exists():
                hashes[path.name] = file_sha256(path)
    if not ids:
        raise ValueError("baseline has no attempted fights")
    return ids, {"path": str(baseline.resolve()), "files": hashes,
                 "cohort_rule": "all result files plus nonempty active traces, in baseline order"}


def prepare(root: Path, output: Path, *, baseline: Path | None = None,
            combat_observation: str | None = None) -> None:
    collection = lab.read_json(root / "collection.json")
    baseline_ids, baseline_info = baseline_cohort(root, baseline) if baseline else (None, None)
    selected = lab.selection(collection, lab.StateFilter(battle_start=True), baseline_ids)
    if not selected["states"]:
        raise ValueError("no combat starts")
    ids = [row["state_id"] for row in selected["states"]]
    seeds = {state_id: sampling_seed(state_id) for state_id in ids}
    if len(set(seeds.values())) != len(ids):
        raise ValueError("sampling seed collision")
    if baseline_ids is not None:
        ids = baseline_ids
    else:
        random.Random(20260906).shuffle(ids)
    settings = {
        "agent": "mlx", "temperature": 1.0, "top_p": 0.95, "top_k": 64,
        "thinking": False, "max_tokens": 4096, "max_decisions": 1000,
        "rollout_index": 0, "adapter_path": None}
    if baseline is not None:
        previous = lab.read_json(baseline / "settings.json")
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError("baseline sampling settings differ from this arm")
    if combat_observation is not None:
        if combat_observation not in ("combat_public_v3", "combat_public_v4", "combat_public_v5"):
            raise ValueError("unsupported policy observation")
        settings["combat_observation"] = combat_observation
    output.mkdir(parents=True, exist_ok=False)
    lab.write_new(output / "selection.json", selected)
    lab.write_new(output / "settings.json", settings)
    lab.write_new(output / "manifest.json", {
        "root": str(root), "model_id": lab.MODEL_ID, "model_revision": lab.MODEL_REVISION,
        "implementation": implementation(), "selection_sha256": selected["selection_sha256"],
        "runtime_versions": runtime_versions(),
        "settings_sha256": file_sha256(output / "settings.json"),
        "order": ids, "sampling_seeds": seeds, "order_seed": 20260906,
        "baseline": baseline_info,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stall_seconds": 300, "fight_seconds": 3600})
    for name in implementation():
        source = SCRIPT.parent.parent / name
        if source.exists():
            target = output / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def validate(output: Path) -> tuple[dict, dict, dict]:
    manifest = lab.read_json(output / "manifest.json")
    selected = lab.read_json(output / "selection.json")
    settings = lab.read_json(output / "settings.json")
    lab.validate_selection(lab.read_json(Path(manifest["root"]) / "collection.json"), selected)
    if selected["selection_sha256"] != manifest["selection_sha256"]:
        raise ValueError("selection changed")
    if file_sha256(output / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("settings changed")
    if implementation() != manifest["implementation"]:
        raise ValueError("sampling implementation changed; use a new arm")
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("sampling runtime versions changed")
    if manifest.get("baseline"):
        baseline = manifest["baseline"]
        for name, expected in baseline["files"].items():
            if file_sha256(Path(baseline["path"]) / name) != expected:
                raise ValueError(f"baseline artifact changed: {name}")
    if manifest["model_id"] != lab.MODEL_ID or manifest["model_revision"] != lab.MODEL_REVISION:
        raise ValueError("model identity changed")
    ids = {row["state_id"] for row in selected["states"]}
    if len(manifest["order"]) != len(ids) or set(manifest["order"]) != ids:
        raise ValueError("schedule differs from selection")
    if manifest["sampling_seeds"] != {state_id: sampling_seed(state_id) for state_id in ids}:
        raise ValueError("sampling seeds changed")
    for state_id in manifest["order"]:
        path = output / f"{state_id}.result.json"
        if path.exists():
            saved = lab.read_json(path)
            if saved["state_id"] != state_id:
                raise ValueError("result identity changed")
            if saved["status"] == "ok":
                trace = output / f"{state_id}_r0.jsonl"
                if file_sha256(trace) != saved["trace_sha256"] or file_sha256(trace.with_suffix(".meta.json")) != saved["meta_sha256"]:
                    raise ValueError("finished trajectory changed")
    return manifest, selected, settings


def archive_partial(output: Path, state_id: str) -> None:
    files = list(output.glob(f"{state_id}_r0.*"))
    if files:
        archive = output / "partials" / f"{state_id}_{time.time_ns()}"
        archive.mkdir(parents=True)
        for path in files:
            path.rename(archive / path.name)


def result(output: Path, state_id: str, seconds: float) -> dict:
    trace = output / f"{state_id}_r0.jsonl"
    meta = lab.read_json(trace.with_suffix(".meta.json"))
    records = lab.read_records(trace)
    metrics = meta["extra"]["local_task"]["metrics"]
    return {"state_id": state_id, "status": "ok", "trace": trace.name,
            "seconds": seconds, "stopped_reason": meta["stopped_reason"], "metrics": metrics,
            "n_invalid": meta["n_invalid"], "undefined_behavior_evoked": meta["undefined_behavior_evoked"],
            "decisions": len(records), "policy_seed": sampling_seed(state_id),
            "trace_sha256": file_sha256(trace), "meta_sha256": file_sha256(trace.with_suffix(".meta.json")),
            "trace_digest": lab.trace_digest(records),
            "generation_cap_hits": sum(r["agent"]["completion_tokens"] >= 4096 for r in records)}


def worker(output: Path) -> None:
    from huggingface_hub import snapshot_download
    from sts_ai.agent_factory import build_agent

    manifest, selected, settings = validate(output)
    model = snapshot_download(lab.MODEL_ID, revision=lab.MODEL_REVISION, local_files_only=True)
    agent = build_agent("mlx", model=model, temperature=settings["temperature"],
                        top_p=settings["top_p"], top_k=settings["top_k"],
                        thinking=settings["thinking"], max_tokens=settings["max_tokens"],
                        max_retries=0, output_contract="action_text")
    for state_id in manifest["order"]:
        if (output / "STOP").exists():
            return
        destination = output / f"{state_id}.result.json"
        if destination.exists():
            continue
        archive_partial(output, state_id)
        started = time.time()
        atomic(output / "active.json", {"state_id": state_id, "started_at": started,
                                        "trace": f"{state_id}_r0.jsonl"})
        try:
            lab.evaluate_worker(Path(manifest["root"]), selected, state_id,
                {**settings, "policy_seed": manifest["sampling_seeds"][state_id]},
                output / f"{state_id}_r0.jsonl", agent=agent)
            report = result(output, state_id, time.time() - started)
        except Exception:
            report = {"state_id": state_id, "status": "error", "error": traceback.format_exc(),
                      "trace": f"{state_id}_r0.jsonl", "seconds": time.time() - started}
        atomic(destination, report)
        print(f"{state_id}: {report.get('stopped_reason', report['status'])}", flush=True)
    atomic(output / "active.json", {})


def progress(output: Path, manifest: dict, status: str) -> dict:
    episodes = [lab.read_json(output / f"{state_id}.result.json")
                for state_id in manifest["order"] if (output / f"{state_id}.result.json").exists()]
    complete = [r for r in episodes if r.get("metrics", {}).get("completed")
                and not r.get("undefined_behavior_evoked")]
    active = lab.read_json(output / "active.json") if (output / "active.json").exists() else {}
    report = {"status": status, "expected": len(manifest["order"]), "finished": len(episodes),
              "completed": len(complete), "wins": sum(r["metrics"]["won"] for r in complete),
              "incomplete": len(episodes) - len(complete),
              "invalid_decisions": sum(r.get("n_invalid", 0) for r in episodes),
              "generation_cap_hits": sum(r.get("generation_cap_hits", 0) for r in episodes),
              "stopped_reasons": dict(Counter(r.get("stopped_reason", r["status"]) for r in episodes)),
              "updated_at": datetime.now(timezone.utc).isoformat(), "active": active}
    atomic(output / "report.json", {**report, "episodes": episodes})
    atomic(output / "progress.json", report)
    return report


def supervise(output: Path) -> None:
    manifest, _, _ = validate(output)
    lock_paths = [output / "run.lock", Path(manifest["root"]) / ".gui_jobs/worker.lock"]
    locks = []
    try:
        for path in lock_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a")
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this run or a GUI model worker is already active") from None
    (output / "STOP").unlink(missing_ok=True)
    (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    failures = 0
    child = None
    try:
        while not stopping and not (output / "STOP").exists():
            p = progress(output, manifest, "running")
            if p["finished"] == p["expected"]:
                progress(output, manifest, "finished")
                return
            atomic(output / "active.json", {})
            with (output / "worker.log").open("a") as log:
                child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--output", str(output)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=tuple(h.fileno() for h in locks))
            launched = time.time()
            timeout = False
            while child.poll() is None:
                if stopping or (output / "STOP").exists():
                    break
                p = progress(output, manifest, "running")
                active = p["active"]
                began = active.get("started_at", launched)
                trace = output / active.get("trace", "nonexistent")
                last_write = max(began, trace.stat().st_mtime if trace.exists() else began)
                if time.time() - last_write > manifest["stall_seconds"] or time.time() - began > manifest["fight_seconds"]:
                    timeout = True
                    break
                time.sleep(2)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if stopping or (output / "STOP").exists():
                break
            if timeout:
                active = lab.read_json(output / "active.json")
                if active.get("state_id"):
                    state_id = active["state_id"]
                    target = output / f"{state_id}.result.json"
                    if not target.exists():
                        atomic(target, {"state_id": state_id, "status": "timeout", "trace": active["trace"]})
            # A watchdog result has been recorded and the next worker will skip
            # that attempt. Only unexplained crashes count toward the stop limit.
            failures = failures + 1 if child.returncode != 0 and not timeout else 0
            if failures >= 3:
                progress(output, manifest, "error")
                return
        progress(output, manifest, "paused")
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        for handle in locks:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "resume", "worker", "status", "stop"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--baseline-run", type=Path,
                        help="rerun all attempted starts from this arm, including partials")
    parser.add_argument("--combat-observation", choices=("combat_public_v3", "combat_public_v4", "combat_public_v5"), default="combat_public_v5")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "start":
        if args.root is None:
            parser.error("start requires --root")
        prepare(args.root.resolve(), output,
                baseline=args.baseline_run.resolve() if args.baseline_run else None,
                combat_observation=args.combat_observation)
        supervise(output)
    elif args.command == "resume":
        supervise(output)
    elif args.command == "worker":
        worker(output)
    elif args.command == "stop":
        (output / "STOP").touch()
    else:
        import json
        print(json.dumps(lab.read_json(output / "progress.json"), indent=2))


if __name__ == "__main__":
    main()
