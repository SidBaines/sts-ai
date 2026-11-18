"""Attribute the reference teacher's wall time and test caching menu-index translation."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from sts_policy.curriculum import TaskSpec
from sts_rl.common import WORKER_CONFIG
from sts_rl.panel import PanelEpisode


def time_case(ep, method, arm):
    env = ep.episode.env
    cls = type(env._env.bc)
    original_map = env.display_index_for_bits
    original_search = cls.search_best_action
    cache = {}
    native_seconds = 0.0

    def cached(bits):
        if bits not in cache:
            cache[bits] = original_map(bits)
        return cache[bits]

    def measured(battle, *args, **kwargs):
        nonlocal native_seconds
        start = time.perf_counter()
        try:
            return original_search(battle, *args, **kwargs)
        finally:
            native_seconds += time.perf_counter() - start

    cls.search_best_action = measured
    env.display_index_for_bits = cached if arm == "cached" else original_map
    futures = list(range(101, 125))
    try:
        start = time.perf_counter()
        if method == "votes":
            result = ep.search_votes_public(1000, futures, 51)
            assert sum(result.values()) == 24
        else:
            result = ep.search_details_public(
                1000, futures, [51 + i for i in futures], 51
            )
            assert len(result) == 24
        elapsed = time.perf_counter() - start
    finally:
        cls.search_best_action = original_search
        env.display_index_for_bits = original_map
    return result, elapsed, native_seconds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--blocks", type=int, default=1, help="Four distinct source worlds per block"
    )
    args = p.parse_args()
    if args.blocks < 1:
        p.error("--blocks must be positive")
    games = json.loads(args.games.read_bytes())["records"]
    wins = [r for r in games if r["win"]]
    report = []
    cases = [(1, False), (1, True), (2, True), (3, True)] * args.blocks
    if len(wins) < len(cases):
        raise ValueError("Too few completed source worlds")
    for game, (act, boss) in zip(wins, cases):
        ep = PanelEpisode(dict(WORKER_CONFIG))
        obs, _ = ep.reset(
            TaskSpec(
                seed=game["world_seed"],
                kind="run",
                seek_combat=False,
                max_act=3,
                horizon=4096,
            )
        )
        for i, action in enumerate(game["action_prefix"]):
            room = obs["state"]["room"].rsplit(".", 1)[-1]
            if (
                obs["phase"] == "combat"
                and obs["state"]["act"] == act
                and (room == "BOSS") == boss
            ):
                break
            obs, _, done, cut, _ = ep.step(action)
            assert not (done or cut)
        else:
            raise RuntimeError("No matching fight entry in supplied game")
        before = ep.episode.env.summary()
        for method in ("votes", "details"):
            reference = None
            for arm in ("original", "cached", "cached", "original"):
                result, elapsed, native = time_case(ep, method, arm)
                assert before == ep.episode.env.summary()
                if reference is None:
                    reference = result
                assert result == reference
                report.append(
                    dict(
                        seed=game["world_seed"],
                        act=act,
                        room=room,
                        step=i,
                        method=method,
                        arm=arm,
                        seconds=elapsed,
                        native_seconds=native,
                        result_sha256=hashlib.sha256(
                            json.dumps(result, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                )
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(act, room, "verified", flush=True)


if __name__ == "__main__":
    main()
