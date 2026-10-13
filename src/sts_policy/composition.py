"""Atomic composition of independently trained, validated candidate checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sts_policy.encoding import ENCODER_VERSION
from sts_policy.model import CandidateActorCritic
from sts_policy.model_agent import CompositeActorCritic, policy_from_payload
from sts_policy.training import file_identity, load_checkpoint, save_checkpoint

PROVENANCE_KEYS = ("environment_version", "observation_version", "python_interface_sha256",
                   "runtime_python_version", "native_provenance", "encoder_code_sha256", "model_code_sha256")


def training_environment_record() -> dict[str, Any]:
    """File-based identity only: never initialize/import the native simulator."""
    from sts_env.provenance import interface_identity
    from sts_env.env import ENVIRONMENT_VERSION, OBSERVATION_VERSION
    directory = Path(__file__).parent
    record = {**interface_identity(), "environment_version":ENVIRONMENT_VERSION,"observation_version":OBSERVATION_VERSION,
              "encoder_code_sha256": file_identity(directory / "encoding.py"),
              "model_code_sha256": file_identity(directory / "model.py")}
    build = os.environ.get("STS_LIGHTSPEED_BUILD")
    if build:
        manifest_path = Path(build) / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        module = manifest.get("module_file", "")
        if not module or Path(module).name != module:
            raise ValueError("native manifest module_file must be a local basename")
        actual = file_identity(Path(build) / module)
        if actual != manifest.get("module_sha256"):
            raise ValueError("native module does not match its recorded build manifest")
        record.update(sourced_env_manifest_sha256=file_identity(manifest_path),
                      sourced_env_manifest=manifest, native_module_sha256=actual)
    return record


def compose_checkpoint(combat_path: str | Path, ooc_path: str | Path,
                       output: str | Path) -> dict[str, Any]:
    """Copy source parameters into a new composite; sources are never overwritten.

    Legacy candidate files lack environment provenance. Preserve that limitation
    explicitly while enforcing source digests, encoder/config/state compatibility
    and equality of every environment/code provenance field actually recorded.
    """
    inputs = {"combat": Path(combat_path).resolve(), "ooc": Path(ooc_path).resolve()}
    destination = Path(output).resolve()
    if destination in inputs.values():
        raise ValueError("composed destination must not overwrite a source checkpoint")
    sources, models = {}, {}
    for role, path in inputs.items():
        before = file_identity(path)
        payload = load_checkpoint(path)
        if file_identity(path) != before:
            raise ValueError("source checkpoint changed during composition")
        model = policy_from_payload(payload)
        if not isinstance(model, CandidateActorCritic):
            raise ValueError("composition sources must be candidate checkpoints, not nested composites")
        models[role] = model
        sources[role] = {"path": str(path), "sha256": before, "step": payload.get("step", 0),
                         "model_config": payload["model_config"], "metadata": payload["metadata"]}
    common = {}
    for key in PROVENANCE_KEYS:
        values = [sources[role]["metadata"].get(key) for role in ("combat", "ooc")]
        if values[0] != values[1]:
            raise ValueError(f"composition source provenance mismatch: {key}")
        if values[0] is not None:
            common[key] = values[0]
    environments = [sources[role]["metadata"].get("training_environment") for role in ("combat", "ooc")]
    if all(isinstance(value, dict) for value in environments):
        for key in ("python_interface_sha256", "runtime_python_version", "encoder_code_sha256",
                    "model_code_sha256", "native_module_sha256"):
            if environments[0].get(key) != environments[1].get(key):
                raise ValueError(f"composition source training-environment mismatch: {key}")
        common["training_environment"] = environments[0]
    model = CompositeActorCritic(models["combat"], models["ooc"])
    identity = hashlib.sha256(json.dumps({r:s["sha256"] for r,s in sources.items()},
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = Path(__file__).parent
    provenance = {"sources": sources, "common_recorded_provenance": common,
                  "legacy_environment_provenance_missing": not (
                      common.get("native_provenance") or
                      all(isinstance(value, dict) and value.get("native_module_sha256") for value in environments)),
                  "composition_code_sha256": file_identity(__file__),
                  "encoder_code_sha256": file_identity(directory / "encoding.py"),
                  "model_code_sha256": file_identity(directory / "model.py"),
                  "wrapper_code_sha256": file_identity(directory / "model_agent.py")}
    save_checkpoint(destination, model, model_config=model.metadata(), step=0,
                    metadata={"encoder_version": ENCODER_VERSION, "model_kind": "composite",
                              "model_config_version": 2, "stage": "composition",
                              "data_identity": "composed:" + identity,
                              "composition_provenance": provenance})
    return {"output": str(destination), "sha256": file_identity(destination),
            "model_config": model.metadata(), "composition_provenance": provenance}
