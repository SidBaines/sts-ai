from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from sts_ai import provenance
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.local_tasks import base
from sts_ai.local_tasks.runner import replay_task_start
from sts_ai.local_tasks.start_state import (
    START_STATE_SIGNATURE_SCHEMA_VERSION,
    build_start_state_signature,
    start_state_payload_sha256,
)
from sts_ai.rollout import current_git_sha


VALIDATION_SCHEMA_VERSION = 2
RESULT_STATUSES = ("success", "failure", "timeout")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest_identity(
    manifest: dict[str, Any],
    source_manifest_path: Path,
) -> dict[str, Any]:
    source_manifest_path = Path(source_manifest_path)
    return {
        "path": str(source_manifest_path),
        "sha256": sha256_file(source_manifest_path),
        "task_id": str(manifest["task_id"]),
        "manifest_version": manifest.get("version"),
        "n_windows": len(manifest["windows"]),
    }


def source_rollouts_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Content-address every unique JSONL referenced by manifest windows."""
    stems = sorted(
        {
            str(window["source_stem"])
            for window in manifest["windows"]
        }
    )
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for stem in stems:
        path = base.source_rollout_dir(manifest) / f"{stem}.jsonl"
        stat = path.stat()
        size_bytes = int(stat.st_size)
        total_bytes += size_bytes
        files.append(
            {
                "source_stem": stem,
                "path": str(path),
                "size_bytes": size_bytes,
                "sha256": sha256_file(path),
            }
        )
    content_entries = [
        {
            "source_stem": item["source_stem"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in files
    ]
    canonical = json.dumps(
        content_entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "n_files": len(files),
        "total_bytes": total_bytes,
        "content_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def simulator_binary_identity(env: Any) -> dict[str, Any]:
    module_path = getattr(getattr(env, "sts", None), "__file__", None)
    if not module_path:
        raise RuntimeError("loaded simulator module has no __file__")
    path = Path(str(module_path)).resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _window_fields(window: dict[str, Any], window_index: int) -> dict[str, Any]:
    return {
        "window_index": window_index,
        "window_id": str(window["window_id"]),
        "world_seed": int(window["world_seed"]),
        "split": window.get("split"),
    }


def validate_one_window(
    manifest: dict[str, Any],
    *,
    window_index: int,
    task: Any,
    combat_observation: str,
    battle_simulations: int,
    max_act: int,
) -> dict[str, Any]:
    """Replay one window in the current process.

    The public validator never calls this directly: it invokes the CLI's hidden
    single-window mode in a child process so a native crash or hang is contained.
    """
    window = manifest["windows"][window_index]
    started = time.monotonic()
    result = _window_fields(window, window_index)
    loaded_simulator: dict[str, Any] | None = None
    try:
        env = LightspeedHybridEnv(
            world_seed=int(window["world_seed"]),
            combat_control="llm",
            combat_observation=combat_observation,
            battle_simulations=battle_simulations,
            max_act=max_act,
        )
        loaded_simulator = simulator_binary_identity(env)
        applied = replay_task_start(env, window, task)
        summary = env.summary()
        if not isinstance(summary.get("combat"), dict):
            raise RuntimeError("replay did not stop at a live combat state")
        completion = task.completion_reason(summary)
        if completion is not None:
            raise RuntimeError(
                "replayed local task is already complete: " f"{completion}"
            )
        signature = build_start_state_signature(env)
        final_simulator = simulator_binary_identity(env)
        if final_simulator != loaded_simulator:
            raise RuntimeError(
                "loaded simulator binary changed while validating this window"
            )
    except Exception as exc:  # noqa: BLE001 - failure is persisted for audit
        result.update(
            {
                "status": "failure",
                "returncode": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if loaded_simulator is not None:
            result["simulator_binary"] = loaded_simulator
        return result
    result.update(
        {
            "status": "success",
            "returncode": 0,
            "error_type": None,
            "error": None,
            "elapsed_seconds": time.monotonic() - started,
            "n_pre_actions_applied": applied,
            "fixed_start_sha256": window.get("fixed_start_sha256"),
            "start_state_signature": signature,
            "simulator_binary": loaded_simulator,
        }
    )
    return result


def _excerpt(value: str | bytes | None, *, limit: int = 8_000) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = value.strip()
    if not value:
        return None
    if len(value) <= limit:
        return value
    return value[-limit:]


def classify_child_result(
    *,
    window: dict[str, Any],
    window_index: int,
    returncode: int,
    result_payload: dict[str, Any] | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    expected = _window_fields(window, window_index)
    if result_payload is not None:
        identity_matches = all(
            result_payload.get(key) == value
            for key, value in expected.items()
        )
        status = result_payload.get("status")
        if identity_matches and status in ("success", "failure"):
            result = dict(result_payload)
            result["returncode"] = returncode
            if status == "success" and returncode != 0:
                result.update(
                    {
                        "status": "failure",
                        "error_type": "ChildProcessError",
                        "error": "single-window validator reported success but "
                        f"exited with return code {returncode}",
                    }
                )
            child_stdout = _excerpt(stdout)
            child_stderr = _excerpt(stderr)
            if child_stdout is not None:
                result["child_stdout"] = child_stdout
            if child_stderr is not None:
                result["child_stderr"] = child_stderr
            return result

    error = f"single-window validator exited with return code {returncode}"
    if result_payload is not None:
        error += " and wrote an invalid or mismatched result payload"
    result = {
        **expected,
        "status": "failure",
        "returncode": returncode,
        "error_type": "ChildProcessError",
        "error": error,
        "elapsed_seconds": elapsed_seconds,
    }
    child_stdout = _excerpt(stdout)
    child_stderr = _excerpt(stderr)
    if child_stdout is not None:
        result["child_stdout"] = child_stdout
    if child_stderr is not None:
        result["child_stderr"] = child_stderr
    return result


def run_window_subprocess(
    *,
    task_id: str,
    manifest_path: Path,
    window: dict[str, Any],
    window_index: int,
    timeout_seconds: float,
    combat_observation: str,
    battle_simulations: int,
    max_act: int,
    script_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sts-local-task-validation-") as tmp:
        result_path = Path(tmp) / "result.json"
        command = [
            sys.executable,
            str(script_path),
            "--_validate-window",
            "--task",
            task_id,
            "--manifest",
            str(Path(manifest_path).resolve()),
            "--window-index",
            str(window_index),
            "--result",
            str(result_path),
            "--combat-observation",
            combat_observation,
            "--battle-simulations",
            str(battle_simulations),
            "--max-act",
            str(max_act),
        ]
        child_env = dict(os.environ)
        src_path = str(repo_root / "src")
        old_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = (
            src_path
            if not old_pythonpath
            else os.pathsep.join((src_path, old_pythonpath))
        )
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                **_window_fields(window, window_index),
                "status": "timeout",
                "returncode": None,
                "error_type": "TimeoutExpired",
                "error": f"replay exceeded {timeout_seconds:g} seconds",
                "elapsed_seconds": time.monotonic() - started,
                **(
                    {"child_stdout": output}
                    if (output := _excerpt(exc.stdout)) is not None
                    else {}
                ),
                **(
                    {"child_stderr": output}
                    if (output := _excerpt(exc.stderr)) is not None
                    else {}
                ),
            }

        payload: dict[str, Any] | None = None
        if result_path.exists():
            try:
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = None
        return classify_child_result(
            window=window,
            window_index=window_index,
            returncode=completed.returncode,
            result_payload=payload,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=time.monotonic() - started,
        )


def _assert_complete_results(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    source_window_ids = [
        str(window["window_id"]) for window in manifest["windows"]
    ]
    if len(source_window_ids) != len(set(source_window_ids)):
        raise ValueError("source manifest contains duplicate window_id values")
    expected = [
        (index, str(window["window_id"]))
        for index, window in enumerate(manifest["windows"])
    ]
    observed = [
        (result.get("window_index"), str(result.get("window_id")))
        for result in results
    ]
    if observed != expected:
        raise ValueError(
            "validation results must cover every source window exactly once and "
            f"in source order: expected={expected!r}, observed={observed!r}"
        )
    invalid_statuses = sorted(
        {
            str(result.get("status"))
            for result in results
            if result.get("status") not in RESULT_STATUSES
        }
    )
    if invalid_statuses:
        raise ValueError(f"invalid validation statuses: {invalid_statuses!r}")
    missing_audit_fields = [
        str(result.get("window_id"))
        for result in results
        if "returncode" not in result or "error" not in result
    ]
    if missing_audit_fields:
        raise ValueError(
            "validation results must record returncode and error for every window: "
            f"{missing_audit_fields!r}"
        )
    incomplete_successes: list[str] = []
    for result in results:
        if result.get("status") != "success":
            continue
        signature = result.get("start_state_signature")
        payload = signature.get("payload") if isinstance(signature, dict) else None
        valid_signature = (
            isinstance(signature, dict)
            and signature.get("schema_version")
            == START_STATE_SIGNATURE_SCHEMA_VERSION
            and isinstance(signature.get("sha256"), str)
            and isinstance(payload, dict)
            and start_state_payload_sha256(payload) == signature.get("sha256")
        )
        binary = result.get("simulator_binary")
        if not valid_signature or not isinstance(binary, dict) or not binary.get("sha256"):
            incomplete_successes.append(str(result.get("window_id")))
    if incomplete_successes:
        raise ValueError(
            "successful validation results must record the public start-state "
            "signature and loaded simulator binary: "
            f"{incomplete_successes!r}"
        )


def _validated_simulator_binary(
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    identities = {
        json.dumps(
            result["simulator_binary"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ): result["simulator_binary"]
        for result in results
        if isinstance(result.get("simulator_binary"), dict)
    }
    if len(identities) > 1:
        raise ValueError(
            "validation windows loaded different simulator binaries; discard the "
            "mixed-build report"
        )
    return next(iter(identities.values()), None)


def build_validation_report(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    source_manifest_path: Path,
    timeout_seconds: float,
    combat_observation: str,
    battle_simulations: int,
    max_act: int,
    source_rollout_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _assert_complete_results(manifest, results)
    counts = Counter(str(result["status"]) for result in results)
    loaded_simulator = _validated_simulator_binary(results)
    current_rollout_content = source_rollouts_identity(manifest)
    if (
        source_rollout_content is not None
        and source_rollout_content != current_rollout_content
    ):
        raise RuntimeError(
            "referenced source rollout JSONL content changed before the validation "
            "report was assembled"
        )
    return {
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "task_id": str(manifest["task_id"]),
        "source_manifest": source_manifest_identity(manifest, source_manifest_path),
        "source_rollouts": current_rollout_content,
        "validator": {
            "git_sha": current_git_sha(),
            "source_sha256": provenance.file_sha256(Path(__file__)),
            "replay_runner_sha256": provenance.file_sha256(
                Path(__file__).with_name("runner.py")
            ),
            "start_state_signature_sha256": provenance.file_sha256(
                Path(__file__).with_name("start_state.py")
            ),
            "python_wrapper_sha256": provenance.file_sha256(
                Path(__file__).parents[1] / "lightspeed.py"
            ),
            "simulator_patch_sha256": provenance.file_sha256(
                Path(__file__).parents[3] / "patches/sts_lightspeed_python_api.patch"
            ),
            "simulator_binary": loaded_simulator,
        },
        "settings": {
            "isolation": "one_subprocess_per_window",
            "timeout_seconds": timeout_seconds,
            "combat_control": "llm",
            "combat_observation": combat_observation,
            "battle_simulations": battle_simulations,
            "max_act": max_act,
        },
        "n_windows": len(results),
        "status_counts": {
            status: int(counts.get(status, 0)) for status in RESULT_STATUSES
        },
        "all_windows_accounted_for": True,
        "results": results,
    }


def validate_manifest_isolated(
    manifest: dict[str, Any],
    *,
    source_manifest_path: Path,
    task_id: str,
    timeout_seconds: float,
    combat_observation: str,
    battle_simulations: int,
    max_act: int,
    script_path: Path,
    repo_root: Path,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    initial_source_identity = source_manifest_identity(
        manifest, source_manifest_path
    )
    initial_rollout_identity = source_rollouts_identity(manifest)
    results: list[dict[str, Any]] = []
    windows = manifest["windows"]
    for window_index, window in enumerate(windows):
        result = run_window_subprocess(
            task_id=task_id,
            manifest_path=source_manifest_path,
            window=window,
            window_index=window_index,
            timeout_seconds=timeout_seconds,
            combat_observation=combat_observation,
            battle_simulations=battle_simulations,
            max_act=max_act,
            script_path=script_path,
            repo_root=repo_root,
        )
        results.append(result)
        if progress is not None:
            progress(window_index + 1, len(windows), result)
    final_source_identity = source_manifest_identity(manifest, source_manifest_path)
    if final_source_identity != initial_source_identity:
        raise RuntimeError(
            "source manifest changed while replay validation was running; "
            "discarding the mixed-version result"
        )
    final_rollout_identity = source_rollouts_identity(manifest)
    if final_rollout_identity != initial_rollout_identity:
        raise RuntimeError(
            "referenced source rollout JSONL content changed while replay validation "
            "was running; discarding the mixed-version result"
        )
    report = build_validation_report(
        manifest,
        results,
        source_manifest_path=source_manifest_path,
        timeout_seconds=timeout_seconds,
        combat_observation=combat_observation,
        battle_simulations=battle_simulations,
        max_act=max_act,
        source_rollout_content=final_rollout_identity,
    )
    if report["source_manifest"] != final_source_identity:
        raise RuntimeError(
            "source manifest changed before the validation report was assembled"
        )
    return report


def build_validated_manifest(
    manifest: dict[str, Any],
    validation_report: dict[str, Any],
    *,
    source_manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    results = validation_report["results"]
    _assert_complete_results(manifest, results)
    if validation_report.get("validation_schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ValueError(
            "validation report schema version differs from the current validator"
        )
    expected_identity = source_manifest_identity(manifest, source_manifest_path)
    if validation_report.get("source_manifest") != expected_identity:
        raise ValueError(
            "validation report source-manifest identity does not match the source"
        )
    expected_rollouts = source_rollouts_identity(manifest)
    if validation_report.get("source_rollouts") != expected_rollouts:
        raise ValueError(
            "validation report source-rollout content identity does not match the "
            "currently referenced JSONL files"
        )
    if validation_report.get("task_id") != manifest.get("task_id"):
        raise ValueError("validation report task_id differs from source manifest")

    passing_results = {
        str(result["window_id"]): result
        for result in results
        if result["status"] == "success"
    }
    windows = [
        {
            **window,
            "start_state_signature": passing_results[str(window["window_id"])][
                "start_state_signature"
            ],
        }
        for window in manifest["windows"]
        if str(window["window_id"]) in passing_results
    ]
    excluded = [
        {
            key: result.get(key)
            for key in (
                "window_index",
                "window_id",
                "world_seed",
                "split",
                "status",
                "returncode",
                "error_type",
                "error",
            )
        }
        for result in results
        if result["status"] != "success"
    ]
    validated = dict(manifest)
    validated.update(
        {
            "n_windows": len(windows),
            "split_counts": base.split_counts(windows),
            "label_counts": base.label_counts(windows),
            "windows": windows,
            "validation": {
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "source_manifest": expected_identity,
                "source_rollouts": expected_rollouts,
                "report_path": str(report_path),
                "settings": validation_report["settings"],
                "validator": validation_report.get("validator"),
                "n_source_windows": len(manifest["windows"]),
                "n_validated_windows": len(windows),
                "n_excluded_windows": len(excluded),
                "excluded_windows": excluded,
                "all_source_windows_accounted_for": True,
            },
        }
    )
    return validated
