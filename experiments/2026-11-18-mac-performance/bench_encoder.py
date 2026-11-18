"""Test bounded scalar/parser caching on actual champion public decisions, outside the immutable encoder."""

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sts_policy import encoding
from sts_rl.batched_runner import BatchedPolicy, configure_cpu, run_in_process
from sts_rl.controlled_policy import policy_factory


class PublicSample(BatchedPolicy):
    def __init__(self, study, size):
        super().__init__(study)
        self.size = size
        self.sample = []
        self.rng = random.Random(29)
        self.count = 0

    def _menu(self, observation):
        obs = super()._menu(observation)
        self.count += 1
        if len(self.sample) < self.size:
            self.sample.append(obs)
        else:
            i = self.rng.randrange(self.count)
            if i < self.size:
                self.sample[i] = obs
        return obs


class ShortTextRegex:
    def __init__(self):
        self.cached = lru_cache(maxsize=8192)(re.search)

    def search(self, pattern, text, flags=0):
        if len(text) <= 256:
            return self.cached(pattern, text, flags)
        return re.search(pattern, text, flags)

    def __getattr__(self, key):
        return getattr(re, key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("out", type=Path)
    args = p.parse_args()
    configure_cpu()
    sampler = PublicSample(policy_factory(args.checkpoint), 2048)
    games = run_in_process(range(91_001_000_000, 91_001_000_024), sampler, 16)
    assert all(x["status"] == "completed" for x in games)
    old_bag = encoding._bag

    def skip_zero(target, prefix, text, weight=1.0):
        if weight != 0.0:
            old_bag(target, prefix, text, weight)

    rows = []
    for arm in (
        "original",
        "skip_zero",
        "regex",
        "both",
        "both",
        "regex",
        "skip_zero",
        "original",
    ):
        encoding._bag = skip_zero if arm in ("both", "skip_zero") else old_bag
        encoding.re = ShortTextRegex() if arm in ("both", "regex") else re
        t = time.perf_counter()
        features = [encoding.encode_observation(obs) for obs in sampler.sample]
        elapsed = time.perf_counter() - t
        h = hashlib.sha256()
        for f in features:
            for x in (f.state_features, f.action_features, f.action_indices):
                h.update(x.tobytes())
            h.update(json.dumps(f.coverage, sort_keys=True).encode())
        rows.append(
            dict(arm=arm, seconds=elapsed, rows=len(features), hash=h.hexdigest())
        )
    encoding.re = re
    encoding._bag = old_bag
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        len(set(x["hash"] for x in rows)),
        "feature hashes, phases",
        Counter(o["phase"] for o in sampler.sample),
    )


if __name__ == "__main__":
    main()
