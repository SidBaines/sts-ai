"""Benchmark in-process reset/step; run with PYTHONPATH=src (no IPC or search).

The deterministic policy and worlds also drive the original-output regression.
Use --profile for cProfile's cumulative breakdown (timings then include its cost).
"""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import pstats
from time import perf_counter

from sts_env.env import _Episode


SEEDS = (3, 8, 9, 12, 18, 19, 22, 50)
CONFIG = {"mode": "full", "ascension": 0, "max_act": 3, "max_steps": 5000,
          "battle_simulations": 50, "build_dir": None, "ooc_seed": None}


def choose_action(observation: dict, step_count: int) -> int:
    actions = observation["legal_actions"]
    if observation["phase"] == "combat":
        return next((a["index"] for a in actions
                     if a["description"].startswith("play")), 0)
    return step_count % len(actions)


class TimedEpisode(_Episode):
    def __init__(self, config: dict, seed: int) -> None:
        self.timings: Counter[str] = Counter()
        super().__init__(config, seed)

    def _observe(self):
        start = perf_counter()
        result = super()._observe()
        self.timings["observation (state, text, menu)"] += perf_counter() - start
        return result

    def _info(self, observation, reason):
        start = perf_counter()
        result = super()._info(observation, reason)
        self.timings["info"] += perf_counter() - start
        return result

    def _advance(self):
        start = perf_counter()
        super()._advance()
        self.timings["advance"] += perf_counter() - start


def benchmark(seeds: list[int], repeats: int) -> None:
    timings: Counter[str] = Counter()
    elapsed = 0.0
    decisions = 0
    for _ in range(repeats):
        for seed in seeds:
            episode = TimedEpisode(dict(CONFIG), seed)
            start = perf_counter()
            observation, _ = episode.reset()
            elapsed += perf_counter() - start
            while not episode.finished:
                action = choose_action(observation, episode.steps)
                start = perf_counter()
                observation, _, _, _, _ = episode.step(action)
                elapsed += perf_counter() - start
                decisions += 1
            if episode.steps >= CONFIG["max_steps"]:
                raise RuntimeError(f"Seed {seed} hit the decision bound")
            timings.update(episode.timings)
    timings["execution and other step work"] = elapsed - sum(timings.values())
    print(f"{len(seeds) * repeats} games, {decisions} decisions; "
          f"{decisions / elapsed:.1f} decisions/s; {elapsed * 1e6 / decisions:.1f} us/decision")
    print("Split includes reset; excludes construction, policy choice and IPC:")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds * 1e6 / decisions:.1f} us/decision "
              f"({100 * seconds / elapsed:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or args.repeats < 1:
        parser.error("Provide seeds and a positive repeat count")
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(benchmark, seeds, args.repeats)
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(35)
    else:
        benchmark(seeds, args.repeats)


if __name__ == "__main__":
    main()
