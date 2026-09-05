"""Bounded, fresh-world OOC policy/search-budget calibration (2026-09-05)."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def worker(args: argparse.Namespace) -> None:
    from sts_ai.agents import RandomLegalAgent, SimpleHeuristicAgent
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.rollout import current_git_sha, run_rollout

    env = LightspeedHybridEnv(
        world_seed=args.seed, ascension=0, battle_simulations=args.simulations,
        boss_simulation_multiplier=2.0, max_act=args.max_act,
        combat_control="search", combat_observation="combat_public_v2",
    )
    agent = RandomLegalAgent() if args.agent == "random" else SimpleHeuristicAgent()
    if args.audit:
        from astra_information_audit import AuditedAgent
        agent = AuditedAgent(env, agent, args.output.with_suffix(".audit.jsonl"))
    result = run_rollout(
        env, agent, max_decisions=args.max_decisions, output_path=args.output,
        rollout_index=0,
        run_meta={
            "battle_simulations": args.simulations, "git_sha": current_git_sha(),
            "extra": {"experiment": "astra_5sep_baselines", "max_act": args.max_act,
                      "max_decisions": args.max_decisions,
                      "boss_simulation_multiplier": 2.0},
        },
    )
    if result.error:
        raise SystemExit(2)


def run_case(spec: tuple[str, int, int], args: argparse.Namespace) -> dict:
    agent, simulations, seed = spec
    arm = f"{agent}_search{simulations}"
    output = args.out / arm / f"seed_{seed}_r0.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--agent", agent, "--simulations", str(simulations),
               "--seed", str(seed), "--output", str(output),
               "--max-act", str(args.max_act), "--max-decisions", str(args.max_decisions)]
    if args.audit:
        command.append("--audit")
    started = time.monotonic()
    row = {"arm": arm, "agent": agent, "simulations": simulations,
           "world_seed": seed, "rollout_index": 0, "command": command,
           "trace": str(output), "timeout_seconds": args.timeout}
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=args.timeout)
        row.update(returncode=completed.returncode,
                   status="completed" if completed.returncode == 0 else "error")
        log = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as exc:
        row.update(returncode=None, status="timeout")
        log = "Timed out; partial trace is preserved and has no final score.\n"
        for stream in (exc.stdout, exc.stderr):
            if stream:
                log += stream.decode(errors="replace") if isinstance(stream, bytes) else stream
    row["elapsed_seconds"] = time.monotonic() - started
    output.with_suffix(".log").write_text(log)
    meta_path = output.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        row["meta"] = str(meta_path)
        for key in ("final_floor", "final_act", "final_hp", "outcome", "n_decisions",
                    "n_invalid", "stopped_reason", "undefined_behavior_evoked", "policy_seed"):
            row[key] = meta[key]
        if meta["error"]:
            row["status"] = "error"
            row["error"] = meta["error"]
    elif row["status"] == "completed":
        row["status"] = "missing_metadata"
    write_json(output.with_suffix(".attempt.json"), row)
    return row


def summarize(rows: list[dict], expected: list[tuple[str, int, int]]) -> dict:
    from collections import Counter
    from sts_ai.eval_stats import bootstrap_ci, sign_test

    expected_ids = {(a, b, s) for a, b, s in expected}
    actual_ids = {(r["agent"], r["simulations"], r["world_seed"]) for r in rows}
    if actual_ids != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("Attempt accounting is not the complete planned cohort")
    arms = {}
    for arm in sorted({r["arm"] for r in rows}):
        cohort = [r for r in rows if r["arm"] == arm]
        scored = [r for r in cohort if r["status"] == "completed"]
        floor = [r["final_floor"] for r in scored]
        arms[arm] = {
            "n_planned": len(cohort), "n_scored": len(scored),
            "attempt_statuses": dict(Counter(r["status"] for r in cohort)),
            "stop_reasons": dict(Counter(r.get("stopped_reason", r["status"]) for r in cohort)),
            "mean_floor_completed": statistics.mean(floor) if floor else None,
            "median_floor_completed": statistics.median(floor) if floor else None,
            "min_floor_completed": min(floor) if floor else None,
            "max_floor_completed": max(floor) if floor else None,
            "wins": sum(r.get("outcome") == "GameOutcome.PLAYER_VICTORY" for r in scored),
            "n_invalid": sum(r.get("n_invalid", 0) for r in scored),
            "ub_flagged": sum(r.get("undefined_behavior_evoked", False) for r in cohort),
            "elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in cohort),
        }
    contrasts = {}
    pairs = [("random_search50", "heuristic_search50"),
             ("random_search500", "heuristic_search500"),
             ("random_search50", "random_search500"),
             ("heuristic_search50", "heuristic_search500")]
    for base, trained in pairs:
        groups = [{r["world_seed"]: r for r in rows if r["arm"] == arm and r["status"] == "completed"}
                  for arm in (base, trained)]
        seeds = sorted(set(groups[0]) & set(groups[1]))
        deltas = [groups[1][s]["final_floor"] - groups[0][s]["final_floor"] for s in seeds]
        contrasts[f"{trained}_minus_{base}"] = {
            "n_paired_complete_worlds": len(seeds), "world_seeds": seeds,
            "mean_floor_delta": statistics.mean(deltas) if deltas else None,
            "bootstrap_ci_95": list(bootstrap_ci(deltas)) if deltas else None,
            "sign_test": sign_test(deltas),
            "excluded_worlds": sorted({s for _, _, s in expected} - set(seeds)),
        }
    return {
        "cohort_accounting_complete": True, "n_attempts": len(rows), "arms": arms,
        "paired_contrasts": contrasts,
        "limitations": [
            "One policy sample per world; heuristic is a minimal fallback baseline, not expert play.",
            "Timeouts/errors are unscored; complete-pair contrasts may have policy-dependent selection.",
            "Search budget changes both solver strength and runtime under a fixed wall-clock cap.",
            "Fresh diagnostic seeds are not the historical LLM evaluation cohort.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--agent", choices=("random", "heuristic"))
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/astra_5sep/baselines")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be new or empty; immutable attempt artifacts")
    args.out.mkdir(parents=True, exist_ok=True)
    specs = [(agent, budget, seed)
             for agent in ("random", "heuristic") for budget in (50, 500)
             for seed in range(args.seed_start, args.seed_start + args.seed_count)]
    write_json(args.out / "plan.json", {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "expected_specs": specs, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, spec, args) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(specs)} {row['arm']} seed={row['world_seed']} "
                  f"{row['status']} floor={row.get('final_floor')} elapsed={row['elapsed_seconds']:.1f}", flush=True)
    rows.sort(key=lambda r: (r["arm"], r["world_seed"]))
    write_json(args.out / "attempts.json", rows)
    report = summarize(rows, specs)
    write_json(args.out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
