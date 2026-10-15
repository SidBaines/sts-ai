"""Build-time source verification and binary provenance (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


def source_identity(source: Path, root: Path) -> dict:
    expected = json.loads((root / "src/sts_env/_backend/native_manifest.json").read_text())
    patch = (root / "patches/sts_lightspeed_python_api.patch").read_bytes()
    if hashlib.sha256(patch).hexdigest() != expected["patch_sha256"]:
        raise RuntimeError("Versioned patch changed without updating native_manifest.json")
    actual = subprocess.check_output(["git", "-C", str(source), "diff", "--", "bindings", "include", "src"], text=True)
    actual = "\n".join("" if line == " " else line for line in actual.split("\n"))
    if actual.encode() != patch:
        raise RuntimeError("Simulator source contains changes not matching the release patch")
    other = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only"], text=True).splitlines()
    if any(not (name.startswith(("bindings/", "include/", "src/")) or name in {"pybind11", "json"}) for name in other):
        raise RuntimeError("Simulator has unversioned changes outside patched sources")
    untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "--", "bindings", "include", "src"], text=True)
    if untracked.strip():
        raise RuntimeError("Simulator contains unversioned source files")
    for name in ("json", "pybind11"):
        dependency = source / name
        if (dependency / ".git").exists():
            dirty = subprocess.check_output(["git", "-C", str(dependency), "status", "--porcelain", "--untracked-files=normal"], text=True)
            if dirty.strip():
                raise RuntimeError(f"Simulator dependency {name} has unversioned changes")
    return expected


def main() -> None:
    command, source_arg, root_arg, *extra = sys.argv[1:]
    source, root = Path(source_arg), Path(root_arg)
    metadata = source_identity(source, root)
    if command == "check-source":
        return
    if command != "write":
        raise ValueError(command)
    build = Path(extra[0])
    json_pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD:json"], text=True).strip()
    json_actual = subprocess.check_output(["git", "-C", str(source / "json"), "rev-parse", "HEAD"], text=True).strip()
    if json_pin != json_actual:
        raise RuntimeError("JSON submodule is not at the upstream pinned commit")
    name = "slaythespire" + sysconfig.get_config_var("EXT_SUFFIX")
    binary = build / name
    metadata.update(module_file=name, module_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    python_cache_tag=sys.implementation.cache_tag,
                    python_version=sys.version.split()[0], platform=sysconfig.get_platform(),
                    json_commit=json_actual,
                    cmake_version=subprocess.check_output([sys.executable, "-m", "cmake", "--version"], text=True).splitlines()[0])
    cache = (build / "CMakeCache.txt").read_text()
    compiler = next(line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="))
    metadata["compiler_version"] = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    (build / "build_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
