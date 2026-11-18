"""Compare peak memory and write time for a real, indented fast-evaluation summary."""

import argparse
import hashlib
import json
import os
import resource
import time
from itertools import islice
from pathlib import Path

from sts_rl.fresh_study import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=["original", "streamed"], required=True)
    args = p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    data = json.loads(args.source.read_bytes())
    with args.source.open("rb") as source:
        expected = hashlib.file_digest(source, "sha256").hexdigest()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    if args.arm == "original":
        atomic_json(args.out, data)
    else:
        temp = args.out.with_suffix(".tmp")
        chunks = json.JSONEncoder(sort_keys=True, allow_nan=False, indent=2).iterencode(
            data
        )
        with temp.open("xb") as stream:
            while batch := list(islice(chunks, 8192)):
                stream.write("".join(batch).encode())
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(args.out)
    elapsed = time.perf_counter() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.out.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    print(
        json.dumps(
            dict(
                arm=args.arm,
                seconds=elapsed,
                peak_mib=peak / 2**20,
                before_mib=before / 2**20,
                bytes=args.out.stat().st_size,
                sha256=expected,
            )
        )
    )
    args.out.unlink()


if __name__ == "__main__":
    main()
