"""Benchmark in-process reset/step; run with PYTHONPATH=src (no IPC or search).

The deterministic policy and worlds also drive the step-output regression
fixture. Use --profile for cProfile's cumulative breakdown (timings then include
its cost). Use --record-fixture PATH only for a deliberate, versioned change to
observations, info or rewards, and commit the new fixture with that release.
"""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
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


def digest(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def hashes(observation, info, reward, identities) -> list[str]:
    """Hash one transition, pinning source/runtime/build identities to the fixture's."""
    normalized = {key: identities.get(key, value) for key, value in info.items()}
    return [digest(observation), digest(normalized), digest(reward)]


def record_fixture(path: Path, seeds: list[int]) -> None:
    games, identities = [], None
    for seed in seeds:
        episode = _Episode(dict(CONFIG), seed)
        if identities is None:
            identities = {**episode.interface_identity, "native_provenance": episode.env.native_provenance}
        observation, info = episode.reset()
        game = {"seed": seed, "reset": hashes(observation, info, None, identities), "steps": []}
        screens: Counter[str] = Counter()
        while not episode.finished:
            screens[str(observation["state"].get("screen_state"))] += 1
            action = choose_action(observation, episode.steps)
            observation, reward, terminated, truncated, info = episode.step(action)
            game["steps"].append(hashes(observation, info, reward, identities) + [terminated, truncated])
        if truncated:
            raise RuntimeError(f"Seed {seed} did not finish; fixture games must end without truncation")
        coverage = {"screens": dict(screens), "final_act": observation["state"].get("act"),
                    "final_floor": observation["state"].get("floor"), "outcome": info.get("outcome")}
        games.append({"seed": seed, "reset": game["reset"], "coverage": coverage, "steps": game["steps"]})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "patches"],
                                capture_output=True, text=True).stdout.strip())
    header = {"source_commit": commit + ("+uncommitted" if dirty else ""),
              "policy": "first play in combat, else index 0; outside combat step_count modulo menu length",
              "config": CONFIG, "identities": identities}
    # One transition per line keeps the fixture reviewable in diffs.
    blocks = []
    for game in games:
        steps = ",\n".join("      " + json.dumps(step) for step in game["steps"])
        meta = json.dumps({key: value for key, value in game.items() if key != "steps"})
        blocks.append(f'    {meta[:-1]}, "steps": [\n{steps}\n    ]}}')
    path.write_text(json.dumps(header, indent=2)[:-2] + ',\n  "games": [\n' + ",\n".join(blocks) + "\n  ]\n}\n")
    print(f"Recorded {len(games)} games, {sum(len(g['steps']) for g in games)} decisions to {path}")


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
    parser.add_argument("--record-fixture", type=Path, metavar="PATH",
                        help="write per-step hashes for the regression fixture instead of timing")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or args.repeats < 1:
        parser.error("Provide seeds and a positive repeat count")
    if args.record_fixture:
        record_fixture(args.record_fixture, seeds)
    elif args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(benchmark, seeds, args.repeats)
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(35)
    else:
        benchmark(seeds, args.repeats)


if __name__ == "__main__":
    main()
