"""Counterfactual OOC action values under explicit continuation policies."""
from __future__ import annotations

import argparse
from collections import Counter
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


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def worker(args: argparse.Namespace) -> None:
    from sts_ai.agents import RandomLegalAgent, SimpleHeuristicAgent
    from sts_ai.interactive.replay import replay_actions
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.rollout import prepare_decision, run_rollout

    row = rows(args.cohort)[args.row_index]
    source = rows(ROOT / row["source_trace"])
    index = row["decision_index"]
    if [r["decision_index"] for r in source] != list(range(len(source))):
        raise ValueError("source decision indices are not contiguous")
    if source[index]["state_text"] != row["state_text"]:
        raise ValueError("cohort differs from source trace")
    env = LightspeedHybridEnv(
        world_seed=row["world_seed"], battle_simulations=50,
        boss_simulation_multiplier=2.0, max_act=3, combat_control="search",
        combat_observation="combat_public_v2",
    )
    replay_actions(env, [r["selected_action"] for r in source[:index]])
    status, view = prepare_decision(env)
    if status != "ok" or view is None:
        raise ValueError(f"source replay ended before target: {status}")
    if view["state_text"] != row["state_text"]:
        raise ValueError("target state_text replay mismatch")
    if view["legal_action_dicts"] != row["legal_actions"]:
        raise ValueError("target legal_actions replay mismatch")
    env.step(args.action_index)
    agent = SimpleHeuristicAgent() if args.continuation == "heuristic" else RandomLegalAgent()
    policy_seed = 905 if args.continuation == "heuristic" else int(args.continuation.removeprefix("random"))
    result = run_rollout(env, agent, max_decisions=250-index-1, policy_seed=policy_seed)
    terminal = result.terminal_state
    write_json(args.output, {
        "probe_id": row["probe_id"], "row_index": args.row_index,
        "world_seed": row["world_seed"], "decision_index": index,
        "action_index": args.action_index,
        "action_description": row["legal_actions"][args.action_index]["description"],
        "continuation": args.continuation, "policy_seed": policy_seed,
        "search_simulations": 50, "replay_exact": True,
        "status": "simulator_error" if result.error else "completed",
        "stopped_reason": result.stopped_reason, "error": result.error,
        "final_floor": terminal.get("floor"), "final_act": terminal.get("act"),
        "final_hp": terminal.get("cur_hp"), "outcome": str(terminal.get("outcome")),
        "undefined_behavior_evoked": terminal.get("undefined_behavior_evoked", False),
        "n_total_decisions": index + 1 + len(result.decisions),
    })


def run_case(spec: tuple[int, int, str], args: argparse.Namespace) -> dict:
    row_index, action_index, continuation = spec
    output = args.out / continuation / f"row_{row_index:03d}_action_{action_index:02d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--cohort", str(args.cohort), "--row-index", str(row_index),
               "--action-index", str(action_index), "--continuation", continuation,
               "--output", str(output)]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
        if completed.returncode == 0 and output.exists():
            result = json.loads(output.read_text())
        else:
            result = {"status": "worker_error", "returncode": completed.returncode,
                      "error": (completed.stdout + completed.stderr)[-8000:]}
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "error": "per-action subprocess exceeded timeout"}
    result.update(row_index=row_index, action_index=action_index, continuation=continuation,
                  elapsed_seconds=time.monotonic()-started, timeout_seconds=args.timeout)
    write_json(output, result)
    return result


def summarize(cohort: list[dict], results: list[dict], continuations: list[str]) -> tuple[list[dict], dict]:
    by_case = {(r["row_index"], r["action_index"], r["continuation"]): r for r in results}
    expected = {(i, a["index"], c) for i, row in enumerate(cohort)
                for a in row["legal_actions"] for c in continuations}
    if set(by_case) != expected or len(results) != len(expected):
        raise ValueError("attempt cohort incomplete or duplicated")
    labels = []
    for i, row in enumerate(cohort):
        label = {k: row[k] for k in ("probe_id", "world_seed", "decision_index", "probe_screen", "source_trace")}
        label["legal_actions"] = row["legal_actions"]
        label["source_action_index"] = row["selected_action"]["index"]
        label["values"] = {}
        for c in continuations:
            cases = [by_case[i, a["index"], c] for a in row["legal_actions"]]
            complete = all(r["status"] == "completed" and r["stopped_reason"] == "terminal" for r in cases)
            values = {str(r["action_index"]): r["final_floor"] for r in cases if r["status"] == "completed"}
            entry = {"complete_terminal_action_set": complete, "action_final_floors": values,
                     "statuses": dict(Counter(r["status"] for r in cases))}
            if complete:
                best, worst = max(values.values()), min(values.values())
                entry.update(value_range=best-worst, max_final_floor=best,
                             top_action_indices=[int(a) for a,v in values.items() if v == best],
                             first_action_regret=best-values["0"],
                             uniform_random_expected_regret=best-statistics.mean(values.values()),
                             source_random_action_regret=best-values[str(label["source_action_index"])])
            label["values"][c] = entry
        random_entries = [label["values"][c] for c in continuations if c.startswith("random")]
        if random_entries and all(v["complete_terminal_action_set"] for v in random_entries):
            mean_values = {str(a["index"]): statistics.mean(v["action_final_floors"][str(a["index"])] for v in random_entries)
                           for a in row["legal_actions"]}
            best = max(mean_values.values())
            label["random3_mean"] = {"action_final_floors": mean_values,
                                     "value_range": best-min(mean_values.values()),
                                     "top_action_indices": [int(a) for a,v in mean_values.items() if v == best]}
        labels.append(label)
    per_policy = {}
    for c in continuations:
        good = [r["values"][c] for r in labels if r["values"][c]["complete_terminal_action_set"]]
        per_policy[c] = {"n_complete_states": len(good),
                         "n_value_informative": sum(v["value_range"] > 0 for v in good)}
        for field in ("value_range", "first_action_regret", "uniform_random_expected_regret", "source_random_action_regret"):
            per_policy[c][f"mean_{field}"] = statistics.mean(v[field] for v in good) if good else None
    sensitivity = []
    for label in labels:
        h = label["values"]["heuristic"]
        r = label.get("random3_mean")
        if h["complete_terminal_action_set"] and r and (h["value_range"] > 0 or r["value_range"] > 0):
            overlap = set(h["top_action_indices"]) & set(r["top_action_indices"])
            sensitivity.append({"probe_id": label["probe_id"], "top_set_overlap": bool(overlap),
                                "heuristic_range": h["value_range"], "random3_range": r["value_range"]})
    report = {"n_states": len(cohort), "n_actions": sum(len(r["legal_actions"]) for r in cohort),
              "n_attempts": len(results), "attempt_statuses": dict(Counter(r["status"] for r in results)),
              "stop_reasons": dict(Counter(r.get("stopped_reason", r["status"]) for r in results)),
              "per_continuation": per_policy,
              "heuristic_vs_random3_informative_states": len(sensitivity),
              "heuristic_vs_random3_top_set_overlap": statistics.mean(s["top_set_overlap"] for s in sensitivity) if sensitivity else None,
              "sensitivity_by_state": sensitivity,
              "limitations": ["Privileged exact-world playout labels, not human-public optimal-action truth.",
                              "Values depend on continuation policy, fixed hidden world and search50 combat.",
                              "Random continuations use three shared per-action policy RNG seeds905/906/907.",
                              "Floor-only values create many ties and can miss HP/deck-value differences.",
                              "States share source worlds; uncertainty calculations must cluster by world."]}
    return labels, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--cohort", type=Path, default=ROOT / "data/experiments/astra_5sep/policy/cohort.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "data/experiments/astra_5sep/teacher")
    p.add_argument("--row-index", type=int)
    p.add_argument("--action-index", type=int)
    p.add_argument("--continuation", default="heuristic")
    p.add_argument("--output", type=Path)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--continuations", nargs="+", default=["heuristic", "random905", "random906", "random907"])
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.out.exists() and any(args.out.iterdir()):
        p.error("output directory must be fresh")
    if not 1 <= args.workers <= 4:
        p.error("use1–4 workers")
    cohort = rows(args.cohort)
    specs = [(i, a["index"], c) for c in args.continuations
             for i,row in enumerate(cohort) for a in row["legal_actions"]]
    write_json(args.out / "plan.json", {"cohort": str(args.cohort),
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_states": len(cohort), "n_attempts": len(specs), "timeout": args.timeout,
        "workers": args.workers, "continuations": args.continuations,
        "search_simulations": 50, "max_act": 3, "max_total_decisions": 250,
        "replay_contract": "exact state_text and legal_actions at saved target"})
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case,spec,args) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f"{len(results)}/{len(specs)} {dict(Counter(r['status'] for r in results))}",flush=True)
    labels, report = summarize(cohort,results,args.continuations)
    (args.out / "labels.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in labels))
    write_json(args.out / "summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
