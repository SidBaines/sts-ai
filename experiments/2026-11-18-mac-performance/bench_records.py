"""Time the production-format gzip training receipt and verify exact decompressed bytes."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from sts_rl.recording import write_collection


def buffered(path, record):
    # Use Python's C JSON encoder in bounded top-level list elements. Preserve the order.
    def encode(value):
        return json.dumps(value, separators=(",", ":"))

    with gzip.open(path, "wt", compresslevel=1) as stream:
        stream.write("{")
        for index, (key, value) in enumerate(record.items()):
            if index:
                stream.write(",")
            stream.write(encode(key) + ":")
            if isinstance(value, list):
                stream.write("[")
                for i, item in enumerate(value):
                    if i:
                        stream.write(",")
                    stream.write(encode(item))
                stream.write("]")
            else:
                stream.write(encode(value))
        stream.write("}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    raw = gzip.decompress(args.source.read_bytes())
    doc = json.loads(raw)
    rows = []
    with TemporaryDirectory() as tmp:
        for arm in ("original", "bounded", "whole", "whole", "bounded", "original"):
            path = Path(tmp) / (arm + ".gz")
            t = time.perf_counter()
            if arm == "original":
                write_collection(path, doc)
            elif arm == "bounded":
                buffered(path, doc)
            else:
                with gzip.open(path, "wt", compresslevel=1) as f:
                    f.write(json.dumps(doc, separators=(",", ":")) + "\n")
            sec = time.perf_counter() - t
            reread = gzip.decompress(path.read_bytes())
            assert reread == raw
            rows.append(
                dict(
                    arm=arm,
                    seconds=sec,
                    raw_bytes=len(raw),
                    episodes=len(doc["episodes"]),
                    sha256=hashlib.sha256(reread).hexdigest(),
                )
            )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
