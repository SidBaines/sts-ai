"""Identity of the installed Python interface, independent of checkout paths."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


def interface_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"python_interface_sha256": digest.hexdigest(),
            "runtime_python_version": sys.version.split()[0]}
