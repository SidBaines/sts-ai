"""Private, replayable RPC fault evidence. Never part of policy observations."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


class FaultRecordingError(RuntimeError):
    """Evidence could not be persisted; stop rather than silently lose faults."""


def serializable(value):
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_fault(directory, record):
    path = Path(directory) / f"fault-{time.time_ns()}-{uuid.uuid4().hex}.json"
    temporary = path.with_suffix('.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open('x') as output:
            json.dump({'schema': 'sts_rpc_fault_v1', **record}, output, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        raise FaultRecordingError(f'Cannot persist simulator fault to {path}: {exc}') from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Preserve the evidence-write error, including invalid directory paths.
    return str(path)
