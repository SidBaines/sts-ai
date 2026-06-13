from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_build_dir() -> Path:
    return repo_root() / "external" / "sts_lightspeed" / "build"


def import_lightspeed(build_dir: str | os.PathLike[str] | None = None) -> ModuleType:
    """Import the locally built sts_lightspeed pybind module."""
    path = Path(build_dir or os.environ.get("STS_LIGHTSPEED_BUILD", default_build_dir()))
    if not path.exists():
        raise RuntimeError(
            f"sts_lightspeed build directory does not exist: {path}. "
            "Run scripts/build_lightspeed.sh first."
        )

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    try:
        return importlib.import_module("slaythespire")
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import slaythespire from {path}. "
            "Run scripts/build_lightspeed.sh first."
        ) from exc
