"""Compare isolated full-game CPU policies with immutable native baselines."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

from astra_isolated_env import IsolatedLightspeedEnv
from sts_ai.agents import RandomLegalAgent, SimpleHeuristicAgent
from sts_ai.rollout import run_rollout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_difference(left: Path, right: Path) -> dict | None:
    a = [json.loads(line) for line in left.read_text().splitlines()]
    b = [json.loads(line) for line in right.read_text().splitlines()]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"decision_index": i, "differing_fields": sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k)),
                    "baseline_state": x["state"], "isolated_state": y["state"]}
    return {"baseline_decisions": len(a), "isolated_decisions": len(b)} if len(a) != len(b) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/experiments/astra_5sep/isolation_fullgame"))
    parser.add_argument("--baseline", type=Path, default=Path("data/experiments/astra_5sep/baselines"))
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--rpc-timeout-seconds", type=float, default=8)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Use fresh output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    comparisons = []
    started = time.monotonic()
    try:
        for policy in ("random", "heuristic"):
            arm = f"{policy}_search50"
            for seed in range(args.seed_start, args.seed_start + args.seed_count):
                output = args.out / arm / f"seed_{seed}_r0.jsonl"
                baseline = args.baseline / arm / output.name
                baseline_meta = json.loads(baseline.with_suffix(".meta.json").read_text())
                agent = RandomLegalAgent() if policy == "random" else SimpleHeuristicAgent()
                with IsolatedLightspeedEnv(seed, ascension=0, battle_simulations=50,
                    boss_simulation_multiplier=2.0, max_act=3, combat_control="search",
                    combat_observation="combat_public_v2", rpc_timeout_seconds=args.rpc_timeout_seconds) as env:
                    result = run_rollout(env, agent, max_decisions=250, output_path=output,
                        rollout_index=0, run_meta={"battle_simulations": 50,
                            "extra": {"isolated_env": True, "rpc_timeout_seconds": args.rpc_timeout_seconds,
                                      "max_act": 3, "max_decisions": 250, "boss_simulation_multiplier": 2.0}})
                    isolation = env.isolation_metadata
                output.touch(exist_ok=True)
                equal = sha256(baseline) == sha256(output)
                row = {"arm": arm, "world_seed": seed, "baseline_trace": str(baseline),
                       "isolated_trace": str(output), "exact_trace_equal": equal,
                       "baseline_sha256": sha256(baseline), "isolated_sha256": sha256(output),
                       "baseline_floor": baseline_meta["final_floor"], "isolated_floor": result.terminal_state.get("floor"),
                       "baseline_stop": baseline_meta["stopped_reason"], "isolated_stop": result.stopped_reason,
                       "isolated_error": result.error, "isolation": isolation,
                       "first_difference": None if equal else first_difference(baseline, output)}
                comparisons.append(row)
                with (args.out / "comparisons.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"{arm} seed={seed} equal={equal} floor={row['isolated_floor']} stop={result.stopped_reason}", flush=True)
    finally:
        IsolatedLightspeedEnv.close_all()
        summary = {"planned": 2 * args.seed_count, "observed": len(comparisons),
                   "exact_matches": sum(row["exact_trace_equal"] for row in comparisons),
                   "elapsed_seconds": time.monotonic() - started,
                   "isolated_stop_counts": dict(Counter(row["isolated_stop"] for row in comparisons)),
                   "arms": {}, "comparisons": comparisons}
        for arm in sorted({row["arm"] for row in comparisons}):
            rows = [row for row in comparisons if row["arm"] == arm]
            summary["arms"][arm] = {"n": len(rows), "exact_matches": sum(row["exact_trace_equal"] for row in rows),
                "baseline_mean_floor": statistics.mean(row["baseline_floor"] for row in rows),
                "isolated_mean_floor": statistics.mean(row["isolated_floor"] for row in rows if row["isolated_floor"] is not None)}
        (args.out / "verification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
