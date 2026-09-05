"""Privileged MAP/REST action lookahead with fixed heuristic continuations."""
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

from sts_ai.agents import SimpleHeuristicAgent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import prepare_decision, run_rollout
from sts_ai.schemas import AgentDecision, LegalAction

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


class LookaheadAgent:
    name = "privileged_ooc_lookahead"

    def __init__(self, env: LightspeedHybridEnv, mode: str, choices_path: Path) -> None:
        self.env = env
        self.mode = mode
        self.choices_path = choices_path
        self.prefix: list[dict] = []
        self.fallback = SimpleHeuristicAgent()
        self.config = {"lookahead_screens": ["MAP_SCREEN"] + (["REST_ROOM"] if mode == "map_rest" else []),
                       "teacher_privilege": "exact_world_simulator_state",
                       "continuation": "SimpleHeuristicAgent", "search_simulations": 50}

    def reseed(self, policy_seed: int) -> None:
        return None

    def _value(self, action_index: int, state_text: str, actions: list[dict]) -> dict:
        branch = LightspeedHybridEnv(world_seed=self.env.world_seed, battle_simulations=50,
                                     max_act=3, combat_control="search", combat_observation="combat_public_v2")
        for i, record in enumerate(self.prefix):
            status, view = prepare_decision(branch)
            if status != "ok" or view is None:
                raise ValueError(f"prefix terminal at{i}")
            if view["state_text"] != record["state_text"] or view["legal_action_dicts"] != record["legal_actions"]:
                raise ValueError(f"prefix public view mismatch at{i}")
            branch.step(record["action_index"])
        status, view = prepare_decision(branch)
        if status != "ok" or view is None or view["state_text"] != state_text or view["legal_action_dicts"] != actions:
            raise ValueError("target public view mismatch")
        branch.step(action_index)
        result = run_rollout(branch, SimpleHeuristicAgent(), max_decisions=249-len(self.prefix), policy_seed=905)
        if result.error or result.stopped_reason != "terminal":
            raise ValueError(f"continuation incomplete: {result.stopped_reason} {result.error}")
        state = result.terminal_state
        return {"action_index": action_index, "final_floor": state["floor"],
                "outcome": str(state["outcome"]), "final_hp": state["cur_hp"],
                "stopped_reason": result.stopped_reason, "replay_exact": True,
                "undefined_behavior_evoked": state.get("undefined_behavior_evoked", False)}

    def choose_action(self, state_text: str, legal_actions: list[LegalAction], phase: str | None = None) -> AgentDecision:
        screen = str(self.env.gc.screen_state).split(".")[-1]
        actions = [self.env.action_dict(a) for a in legal_actions]
        heuristic = self.fallback.choose_action(state_text, legal_actions, phase)
        selected = heuristic.action_index
        if screen in self.config["lookahead_screens"] and len(actions) > 1:
            record = {"world_seed": self.env.world_seed, "decision_index": len(self.prefix),
                      "screen": screen, "floor": self.env.summary()["floor"],
                      "legal_actions": actions, "heuristic_action_index": selected,
                      "candidates": []}
            for action in legal_actions:
                try:
                    value = self._value(action.index, state_text, actions)
                    record["candidates"].append(value)
                except Exception as exc:
                    record["error"] = {"action_index": action.index, "type": type(exc).__name__, "message": str(exc)}
                    with self.choices_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    raise
            selected = max(record["candidates"], key=lambda v: v["final_floor"])["action_index"]
            record["selected_action_index"] = selected
            record["changed_from_heuristic"] = selected != heuristic.action_index
            record["value_range"] = max(v["final_floor"] for v in record["candidates"]) - min(v["final_floor"] for v in record["candidates"])
            with self.choices_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        self.prefix.append({"state_text": state_text, "legal_actions": actions, "action_index": selected})
        return AgentDecision(action_index=selected, raw_response="privileged map/rest lookahead" if selected != heuristic.action_index else "heuristic or tied lookahead")


def worker(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = LightspeedHybridEnv(world_seed=args.seed, battle_simulations=50, max_act=3,
                             combat_control="search", combat_observation="combat_public_v2")
    agent = (SimpleHeuristicAgent() if args.mode == "heuristic" else
             LookaheadAgent(env, args.mode, args.output.with_suffix(".choices.jsonl")))
    result = run_rollout(env, agent, max_decisions=250, output_path=args.output,
                         run_meta={"battle_simulations": 50, "extra": {"experiment": "astra_5sep_ooc_lookahead",
                                   "max_decisions": 250, "max_act": 3, "mode": args.mode}})
    if result.error:
        raise SystemExit(2)


def run_case(spec: tuple[str, int], args: argparse.Namespace) -> dict:
    mode, seed = spec
    out = args.out / mode / f"seed_{seed}_r0.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
               "--seed", str(seed), "--output", str(out)]
    start = time.monotonic()
    row = {"mode": mode, "world_seed": seed, "trace": str(out)}
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
        row.update(status="completed" if done.returncode == 0 else "error", returncode=done.returncode)
        out.with_suffix(".log").write_text(done.stdout + done.stderr)
    except subprocess.TimeoutExpired:
        row.update(status="timeout", returncode=None)
    row["elapsed_seconds"] = time.monotonic()-start
    meta_path = out.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for key in ("final_floor", "final_hp", "outcome", "stopped_reason", "n_decisions", "undefined_behavior_evoked"):
            row[key] = meta[key]
    choices_path = out.with_suffix(".choices.jsonl")
    choices = [json.loads(l) for l in choices_path.read_text().splitlines()] if choices_path.exists() else []
    row.update(n_lookahead_choices=len(choices), n_changed=sum(r.get("changed_from_heuristic", False) for r in choices),
               n_candidates=sum(len(r["candidates"]) for r in choices), n_choice_errors=sum("error" in r for r in choices))
    write_json(out.with_suffix(".attempt.json"), row)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", choices=("heuristic", "map_only", "map_rest"))
    p.add_argument("--output", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "data/experiments/astra_5sep/lookahead")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-count", type=int, default=8)
    p.add_argument("--modes", nargs="+", choices=("heuristic", "map_only", "map_rest"), default=["map_only", "map_rest"])
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.out.exists() and any(args.out.iterdir()):
        p.error("output dir must be fresh")
    modes = args.modes
    specs = [(m,s) for m in modes for s in range(args.seed_start,args.seed_start+args.seed_count)]
    write_json(args.out / "plan.json", {"specs": specs, "timeout_per_game_seconds": 90,
                                       "workers": 4, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                                       "primary_endpoint": "paired world-seed final-floor delta vs heuristic, bootstrap95%CI",
                                       "fixed_policy": "all nonlookahead choices SimpleHeuristic; every candidate exact-prefix replay plus heuristic continuation search50; maximize final floor,tie-first",
                                       "exclusions": "none planned; count every error and timeout, report complete-pair floor estimates separately",
                                       "max_act": 3, "max_total_decisions": 250})
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_case,spec,args) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result(); rows.append(row)
            print(json.dumps(row),flush=True)
    if {(r["mode"],r["world_seed"]) for r in rows} != set(specs):
        raise ValueError("missing planned game")
    from sts_ai.eval_stats import bootstrap_ci
    summary = {"attempt_statuses": dict(Counter(r["status"] for r in rows)), "arms": {}}
    for mode in modes:
        arm = [r for r in rows if r["mode"] == mode]
        good = [r for r in arm if r["status"] == "completed" and r.get("stopped_reason") == "terminal"]
        if "heuristic" in modes:
            base = {r["world_seed"]:r["final_floor"] for r in rows if r["mode"] == "heuristic" and r["status"] == "completed" and r.get("stopped_reason") == "terminal"}
        else:
            base = {r["world_seed"]: json.loads((ROOT / f"data/experiments/astra_5sep/baselines/heuristic_search50/seed_{r['world_seed']}_r0.meta.json").read_text())["final_floor"] for r in good}
        paired = [r for r in good if r["world_seed"] in base]
        deltas = [r["final_floor"] - base[r["world_seed"]] for r in paired]
        summary["arms"][mode] = {"n_planned": len(arm), "n_terminal": len(good),
                                "mean_floor_terminal": statistics.mean(r["final_floor"] for r in good) if good else None,
                                "mean_floor_delta_vs_heuristic_complete_pairs": statistics.mean(deltas) if deltas else None,
                                "bootstrap_ci_95": bootstrap_ci(deltas) if deltas else None,
                                "n_lookahead_choices": sum(r["n_lookahead_choices"] for r in arm),
                                "n_changed": sum(r["n_changed"] for r in arm),
                                "n_candidates": sum(r["n_candidates"] for r in arm),
                                "n_choice_errors": sum(r["n_choice_errors"] for r in arm),
                                "n_paired_worlds": len(paired),
                                "excluded_seeds": [r["world_seed"] for r in arm if r not in paired]}
    summary["limitations"] = ["Privileged deterministic future-world lookahead, not a public-information policy.",
                              "One heuristic continuation per action; no optimality claim.",
                              f"{args.seed_count} diagnostic seeds; fixed continuation policy and search budget.",
                              "Fail-closed replay; failed games excluded from paired floor estimates and counted separately."]
    write_json(args.out / "attempts.json",rows)
    write_json(args.out / "summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
