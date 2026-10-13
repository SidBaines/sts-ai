"""Preserve experiment source bytes at launch, alongside content hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path


def snapshot_experiment_sources(destination: Path, *, root: Path | None = None) -> dict[str, str]:
    root = (root or Path.cwd()).resolve()
    if destination.exists():
        raise ValueError("source snapshot destination must be new")
    files = [path for area in ("src/sts_env","src/sts_policy","scripts")
             for path in (root/area).rglob("*") if path.is_file() and path.suffix in {".py",".json"}]
    files += [root/"pyproject.toml",root/"patches/sts_lightspeed_python_api.patch"]
    hashes = {}
    for path in sorted(set(files)):
        relative = path.relative_to(root)
        data = path.read_bytes()
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("source snapshot verification failed")
        hashes[str(relative)] = digest
    return hashes
