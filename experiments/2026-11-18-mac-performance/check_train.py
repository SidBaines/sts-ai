"""Compare learned tensors, optimizer moments, RNG and game outcomes in two benchmark runs."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch


def check_equal(a, b, path=""):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise ValueError(f"Changed tensor at {path}")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            check_equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            check_equal(x, y, f"{path}/{i}")
    else:
        assert a == b, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    rs = [args.reference, args.candidate]
    ck = [
        torch.load(r / "latest.pt", map_location="cpu", weights_only=False) for r in rs
    ]
    for k in (
        "model_state",
        "optimizer_state",
        "trainer_state",
        "rng",
        "local_rng",
        "steps_total",
        "iteration",
    ):
        if k in ck[0]:
            check_equal(ck[0][k], ck[1][k], k)
    rng_results = {}
    for k, state in ck[0]["rng_state"].items():
        try:
            check_equal(state, ck[1]["rng_state"][k], k)
            rng_results[k] = "equal"
        except (ValueError, AssertionError):
            if k not in ("python", "numpy"):
                raise
            rng_results[k] = "different (global RNG)"
    games = [
        json.loads(gzip.decompress((r / "collection_0001.json.gz").read_bytes()))
        for r in rs
    ]
    assert not any(x["faults"] for x in games)
    check_equal(games[0]["episodes"], games[1]["episodes"], "games")
    for name in ("eval_0000.json", "eval_0001.json"):
        ev = [json.loads((r / name).read_bytes()) for r in rs]
        check_equal(ev[0]["records"], ev[1]["records"], name)
    print(
        "Equal game, evaluation, optimizer and model records:",
        len(games[0]["episodes"]),
        "games.",
        rng_results,
    )


if __name__ == "__main__":
    main()
