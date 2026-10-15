from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_build_dir() -> Path:
    return repo_root() / "external" / "sts_lightspeed" / "build"


def expected_metadata() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("native_manifest.json").read_text())


def native_metadata(build_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Verify and return the identity of the exact binary about to be loaded."""
    path = Path(build_dir or os.environ.get("STS_LIGHTSPEED_BUILD", default_build_dir())).resolve()
    try:
        manifest = json.loads((path / "build_manifest.json").read_text())
        for key, value in expected_metadata().items():
            if manifest.get(key) != value:
                raise ValueError(f"{key} differs from the pinned release")
        if manifest.get("python_cache_tag") != sys.implementation.cache_tag:
            raise ValueError("Python interpreter ABI differs from the simulator build")
        name = manifest["module_file"]
        if Path(name).name != name or not name.startswith("slaythespire."):
            raise ValueError("invalid native module filename")
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if digest != manifest.get("module_sha256"):
            raise ValueError("simulator binary checksum differs from its build manifest")
        return dict(manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Simulator build at {path} is missing, stale, or incompatible: {exc}. "
            "Run scripts/build_lightspeed.sh with the Python interpreter used to run your agent."
        ) from exc


build_metadata = native_metadata


def import_lightspeed(build_dir: str | os.PathLike[str] | None = None) -> ModuleType:
    path = Path(build_dir or os.environ.get("STS_LIGHTSPEED_BUILD", default_build_dir())).resolve()
    metadata = native_metadata(path)
    expected_file = path / metadata["module_file"]
    existing = sys.modules.get("slaythespire")
    if existing is not None:
        if Path(existing.__file__).resolve() != expected_file:
            raise RuntimeError("Another simulator build is already loaded in this process")
        if getattr(existing, "_verified_build_metadata", None) != metadata:
            raise RuntimeError("Loaded simulator has another identity; restart the process after rebuilding")
        return existing
    sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("slaythespire")
    except ImportError as exc:
        raise RuntimeError(f"Could not load the verified simulator at {path}: {exc}") from exc
    finally:
        sys.path.remove(str(path))
    if Path(module.__file__).resolve() != expected_file:
        raise RuntimeError("Imported simulator does not match the verified build")
    module._verified_build_metadata = dict(metadata)
    return module
