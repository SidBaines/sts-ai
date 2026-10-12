"""Dependency-free identities for a final evaluation embargo and immutable inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

FREEZE_VERSION = "nonllm_final_freeze_v1"


class FreezeChangedError(ValueError):
    """A selected checkpoint, panel definition, or backend changed after freeze."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_identity(value: str | Path) -> dict[str, Any]:
    reference = Path(value).absolute()
    path = reference.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a frozen input file: {path}")
    return {"reference_path": str(reference), "path": str(path),
            "sha256": _sha_file(path), "bytes": path.stat().st_size}


def capture_final_freeze(checkpoints: Mapping[str, str | Path],
                         panel_files: Mapping[str, str | Path],
                         backend_build_dir: str | Path, *,
                         generated_definitions: Mapping[str, Any] | None = None,
                         expected_backend_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Hash inputs only; keep the campaign's legacy selected-path fields separate.

    Generated seed cohorts need their exact kind/split/seed/horizon/act/search
    definition in generated_definitions, since they do not use task JSONL files.
    This reads checkpoint bytes, never loads a model or reads evaluation results.
    """
    build_reference = Path(backend_build_dir).absolute()
    build = build_reference.resolve(strict=True)
    manifest = _file_identity(build / "build_manifest.json")
    provenance = json.loads(Path(manifest["path"]).read_text())
    name = provenance.get("module_file")
    if not isinstance(name, str) or Path(name).name != name or not name.startswith("slaythespire."):
        raise ValueError("Invalid frozen native module filename")
    module = _file_identity(build / name)
    if module["sha256"] != provenance.get("module_sha256"):
        raise ValueError("Native manifest and binary checksum disagree")
    if expected_backend_provenance is not None and dict(expected_backend_provenance) != provenance:
        raise ValueError("Native backend differs from the expected provenance")
    # JSON round-trip creates a detached immutable-by-identity specification.
    definitions = json.loads(json.dumps(generated_definitions or {}, sort_keys=True))
    snapshot = {"freeze_version": FREEZE_VERSION, "frozen_at": time.time(),
                "checkpoints": {name: _file_identity(path) for name, path in sorted(checkpoints.items())},
                "panel_files": {name: _file_identity(path) for name, path in sorted(panel_files.items())},
                "generated_definitions": definitions,
                "backend": {"reference_dir": str(build_reference), "build_dir": str(build), "manifest": manifest,
                            "module": module, "provenance": provenance}}
    snapshot["digest"] = _digest(snapshot)
    return snapshot


def verify_final_freeze(snapshot: Mapping[str, Any], *,
                        backend_build_dir: str | Path | None = None) -> None:
    """Call before every final command; reject mutation rather than selecting anew."""
    payload = {key: value for key, value in snapshot.items() if key != "digest"}
    if snapshot.get("freeze_version") != FREEZE_VERSION or _digest(payload) != snapshot.get("digest"):
        raise FreezeChangedError("Frozen identity record was altered or has an unsupported version")
    backend = snapshot["backend"]
    reference_dir = backend_build_dir if backend_build_dir is not None else backend["reference_dir"]
    if str(Path(reference_dir).resolve()) != backend["build_dir"]:
        raise FreezeChangedError("Native backend path changed after final selection")
    files = [(f"checkpoint {name}", row) for name, row in snapshot["checkpoints"].items()]
    files += [(f"panel {name}", row) for name, row in snapshot["panel_files"].items()]
    files += [("native manifest", backend["manifest"]), ("native module", backend["module"])]
    for label, row in files:
        try:
            actual = _file_identity(row["reference_path"])
        except (OSError, ValueError) as exc:
            raise FreezeChangedError(f"Frozen {label} is missing: {row['path']}") from exc
        if actual != row:
            raise FreezeChangedError(f"Frozen {label} changed: {row['path']}")
