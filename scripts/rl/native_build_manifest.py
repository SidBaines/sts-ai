"""Verify research sources without changing them; publish binary provenance.

The source digest is SHA256 of canonical JSON (sorted keys, compact separators)
mapping every verified non-submodule path to the SHA256 of its bytes. Both
patches are applied only to a private archive of the declared upstream commit.
No native module is imported and no third-party Python packages are needed.
Publication is valid only immediately after a successful build in a fresh output
directory, with matching pre-build and post-build source checks. The build script
enforces the new-or-empty output prerequisite; this helper alone cannot establish
that an existing binary was built from the checked source.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile


# https://github.com/pybind/pybind11/releases/tag/v2.13.6 (peeled commit).
# Do not trust a locally moved tag as an immutable dependency identity.
PYBIND11_TAG = "v2.13.6"
PYBIND11_COMMIT = "a2e59f0e7065404b44dfe92a28aca47ba1378dc4"


def _run(*args: str) -> bytes:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if result.returncode:
        raise RuntimeError(f"{' '.join(args[:3])} failed: "
                           f"{result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def _git(source: Path, *args: str) -> bytes:
    return _run("git", "--no-replace-objects", "-C", str(source), *args)


def _repository_head(source: Path) -> str:
    if not (source / ".git").exists():
        raise RuntimeError(f"Missing Git checkout: {source}")
    top = Path(os.fsdecode(_git(source, "rev-parse", "--show-toplevel")).strip())
    if top.resolve() != source.resolve():
        raise RuntimeError(f"Not an independent Git checkout: {source}")
    return _git(source, "rev-parse", "HEAD").decode().strip()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(directory: Path) -> dict[str, tuple[int, bytes]]:
    """Include executable bits as well as bytes; reject links and special files."""
    result = {}
    for parent, dirs, files in os.walk(directory, followlinks=False):
        if Path(parent) == directory:
            dirs.remove(".git")
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise RuntimeError(f"Unsupported source link: {Path(parent) / name}")
        for name in files:
            path = Path(parent) / name
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                data, git_mode = path.read_bytes(), 0o100755 if mode & 0o111 else 0o100644
            else:
                raise RuntimeError(f"Unsupported source file type: {path}")
            result[path.relative_to(directory).as_posix()] = (git_mode, data)
    return result


def _reconstruct(source: Path, pin: str, baseline: dict, patches: tuple[bytes, bytes], dest: Path) -> dict:
    archive = _git(source, "archive", "--format=tar", pin)
    # Reject archive links/traversal before extraction, rather than following an
    # arbitrary path when applying patches. The pinned simulator uses regular files.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                raise RuntimeError(f"Unsupported upstream archive entry: {member.name}")
            target = dest / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream:
                    target.write_bytes(stream.read())
                target.chmod(member.mode)
    # Archive attributes (including local .git/info/attributes) can omit files or
    # substitute text. Restore any such entries from the immutable Git blobs.
    for name, (mode, blob) in baseline.items():
        target = dest / name
        data = target.read_bytes() if target.is_file() else b""
        blob_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if not target.is_file() or blob_sha != blob:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_git(source, "show", f"{pin}:{name}"))
        target.chmod(mode & 0o777)
    # Isolate git apply from any enclosing repository, even if TMPDIR is inside
    # the source checkout. Only this disposable repository is ever initialized.
    _git(dest, "init", "--quiet", "--template=")
    for label, patch in zip(("core", "research"), patches):
        # The patch file lives outside the reconstructed tree/digest.
        patch_file = dest.parent / f"{label}.patch"
        patch_file.write_bytes(patch)
        _git(dest, "apply", "--check", str(patch_file))
        _git(dest, "apply", str(patch_file))
    return _files(dest)


def _cache(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            result[key.split(":", 1)[0]] = value
    return result


def _check_extras(source: Path, expected: dict, gitlinks: dict, build: Path | None) -> None:
    tracked = set()
    for entry in _git(source, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        info, path = entry.split(b"\t", 1)
        mode, _, stage = info.split()
        name = os.fsdecode(path)
        if stage != b"0" or name not in expected.keys() | gitlinks.keys():
            raise RuntimeError(f"Unexpected tracked source/index entry: {name}")
        expected_mode = 0o160000 if name in gitlinks else expected[name][0]
        # A staged core patch is fine, but staged deletion/type changes are not.
        if int(mode, 8) != expected_mode:
            raise RuntimeError(f"Unexpected index file mode: {name}")
        tracked.add(name)
    # Newly added patch files may be untracked. Baseline files must remain tracked.
    if not gitlinks.keys() <= tracked:
        raise RuntimeError("Missing dependency gitlink in source index")

    untracked = {os.fsdecode(p) for p in _git(source, "ls-files", "--others", "-z").split(b"\0") if p}
    build_dirs = set()
    if build is not None:
        build_dirs.add(build.resolve())
    # Permit prior CMake outputs beside the requested build. A build/cache must
    # not hide new files inside any versioned input or dependency directory.
    for name in untracked:
        if Path(name).name == "CMakeCache.txt":
            cache_path = source / name
            home = _cache(cache_path).get("CMAKE_HOME_DIRECTORY", "")
            if home and Path(home).resolve() == source:
                build_dirs.add(cache_path.parent.resolve())
    input_dirs = {source / PurePosixPath(name).parts[0] for name in expected if "/" in name}
    input_dirs.update(source / name for name in gitlinks)
    for directory in build_dirs:
        if (any((source / name).is_relative_to(directory) for name in expected)
                or any(directory.is_relative_to(parent) for parent in input_dirs)):
            raise RuntimeError(f"Build directory contains versioned source: {directory}")
    for name in sorted(untracked - expected.keys()):
        path = source / name
        if any(path.is_relative_to(directory) for directory in build_dirs):
            continue
        # Include ignored files: an ignore rule must not hide a new build input.
        raise RuntimeError(f"Unexpected unversioned source file: {name}")


def source_identity(source: Path, root: Path, build: Path | None = None) -> dict:
    source, root = source.resolve(), root.resolve()
    expected = json.loads((root / "src/sts_env/_backend/native_manifest.json").read_text())
    core = (root / "patches/sts_lightspeed_python_api.patch").read_bytes()
    overlay = (root / "patches/sts_lightspeed_rl_panel.patch").read_bytes()
    if _sha(core) != expected["patch_sha256"]:
        raise RuntimeError("Core patch SHA does not match native_manifest.json")
    pin = expected["upstream_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise RuntimeError("Upstream pin must be an immutable full commit SHA")
    head = _repository_head(source)
    _git(source, "merge-base", "--is-ancestor", pin, head)
    gitlinks = {}
    baseline = {}
    for entry in _git(source, "ls-tree", "-r", "-z", pin).split(b"\0"):
        if entry:
            info, path = entry.split(b"\t", 1)
            mode, _, sha = info.split()
            name = os.fsdecode(path)
            if mode == b"160000":
                gitlinks[name] = sha.decode()
            else:
                if mode not in {b"100644", b"100755"}:
                    raise RuntimeError(f"Unsupported upstream file mode: {name}")
                baseline[name] = (int(mode, 8), sha.decode())
    if gitlinks.keys() != {"json", "pybind11"} or expected["pybind11_tag"] != PYBIND11_TAG:
        raise RuntimeError("Unexpected upstream dependency layout or pybind11 pin")
    identities = {}
    for name, dep_pin in (("json", gitlinks["json"]), ("pybind11", PYBIND11_COMMIT)):
        dependency = source / name
        actual = _repository_head(dependency)
        if actual != dep_pin:
            raise RuntimeError(f"Dependency {name} pin mismatch: {actual} != {dep_pin}")
        if _git(dependency, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none").strip():
            raise RuntimeError(f"Dependency {name} has unversioned changes")
        identities[f"{name}_commit"] = actual
    with tempfile.TemporaryDirectory(prefix="sts-rl-source-") as directory:
        reconstructed = Path(directory) / "tree"
        reconstructed.mkdir()
        files = _reconstruct(source, pin, baseline, (core, overlay), reconstructed)
    _check_extras(source, files, gitlinks, build)
    tracked = {os.fsdecode(p) for p in _git(source, "ls-files", "-z").split(b"\0") if p}
    if not (baseline.keys() & files.keys()) <= tracked:
        raise RuntimeError("Versioned upstream files missing from source index")
    hashes = {}
    for name, (mode, data) in sorted(files.items()):
        path = source / name
        if any(parent.is_symlink() for parent in path.parents if parent != source):
            raise RuntimeError(f"Source path traverses a symlink: {name}")
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Missing or non-regular source file: {name}")
        actual_mode = 0o100755 if path.stat().st_mode & 0o111 else 0o100644
        if actual_mode != mode or path.read_bytes() != data:
            raise RuntimeError(f"Source does not match pinned core + research patches: {name}")
        hashes[name] = _sha(data)
    if _repository_head(source) != head:
        raise RuntimeError("Source HEAD changed during verification")
    return {**expected, **identities, "rl_panel_patch_sha256": _sha(overlay),
            "source_upstream_commit": pin, "source_head": head,
            "source_tree_sha256": _sha(_canonical(hashes))}


def write_manifest(source: Path, root: Path, build: Path, checked: dict) -> dict:
    if not isinstance(checked, dict) or not checked:
        raise RuntimeError("A checked source identity from the pre-build check is required")
    cache = _cache(build / "CMakeCache.txt")
    if Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() != source.resolve():
        raise RuntimeError("CMake cache refers to a different source checkout")
    cmake, compiler = cache["CMAKE_COMMAND"], cache["CMAKE_CXX_COMPILER"]
    name = "slaythespire" + sysconfig.get_config_var("EXT_SUFFIX")
    binary_metadata = dict(module_file=name, module_sha256=_sha((build / name).read_bytes()),
                    python_cache_tag=sys.implementation.cache_tag, python_version=sys.version.split()[0],
                    python_executable=sys.executable, python_soabi=sysconfig.get_config_var("SOABI"),
                    platform=sysconfig.get_platform(), variant="sts_rl_panel",
                    cmake_executable=cmake, compiler_executable=compiler,
                    cmake_version=_run(cmake, "--version").decode().splitlines()[0],
                    compiler_version=_run(compiler, "--version").decode().splitlines()[0])
    # Reconstruct and compare again AFTER the build and tool queries, immediately
    # before publication. Never trust the precheck or relabel changed inputs.
    metadata = source_identity(source, root, build)
    if checked != metadata:
        raise RuntimeError("Source identity changed since the pre-build check")
    metadata.update(binary_metadata)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=build,
                                         prefix=".build_manifest-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(metadata, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, build / "build_manifest.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-source")
    write = commands.add_parser("write")
    for command in (check, write):
        command.add_argument("source", type=Path)
        command.add_argument("root", type=Path)
    check.add_argument("build", type=Path, nargs="?")
    write.add_argument("build", type=Path)
    write.add_argument("--checked-source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-source":
        result = source_identity(args.source, args.root, args.build)
    else:
        checked = json.loads(args.checked_source.read_text())
        result = write_manifest(args.source, args.root, args.build, checked)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
