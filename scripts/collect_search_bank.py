"""Timed native-search collection with full traces and independent replay checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from sts_ai import fight_lab as lab
from sts_ai.agents import SimpleHeuristicAgent
from sts_ai.provenance import file_sha256
from sts_ai.schemas import AgentDecision, LegalAction
from sts_ai.teacher import displayed_action_index


SCRIPT = Path(__file__).resolve()


def search_json(value: Any) -> Any:
    """Keep all search diagnostics except opaque native Action objects."""
    if isinstance(value, dict):
        return {key: search_json(item) for key, item in value.items() if key != "action"}
    if isinstance(value, (list, tuple)):
        return [search_json(item) for item in value]
    return value


class SearchAgent(SimpleHeuristicAgent):
    """Replan on every displayed decision; OOC retains the pilot heuristic."""

    name = "native_search_per_decision"

    def __init__(self, env: Any, simulations: int) -> None:
        self.env = env
        self.simulations = simulations

    def choose_action(self, state_text: str, legal_actions: list[LegalAction],
                      phase: str | None = None) -> AgentDecision:
        if phase != "combat":
            return super().choose_action(state_text, legal_actions, phase)
        started = time.perf_counter()
        result = self.env.search_best_combat_action(self.simulations, search_seed=1)
        # Native search describes v2 actions; render v3 before deduplication so
        # identical hand copies remain interchangeable even when their bits differ.
        description = result["action"].describe(
            self.env.bc, include_card_type=True, include_computed_damage=True)
        index = displayed_action_index(legal_actions, bits=result["bits"], description=description)
        return AgentDecision(action_index=index, raw_response="native search",
                             latency_s=time.perf_counter() - started,
                             metadata={"search": search_json(result)})


def run_worker(root: Path, seed: int, output: Path) -> None:
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.rollout import run_rollout

    config = lab.read_json(root / "config.json")
    env = LightspeedHybridEnv(world_seed=seed, **config["env"])
    run_rollout(env, SearchAgent(env, config["simulations"]),
                max_decisions=config["max_decisions"], output_path=output,
                run_meta={"extra": {"fight_lab": config}})


def process(command: list[str], log: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    with log.open("x") as handle:
        try:
            result = subprocess.run([sys.executable, *command], stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            status = "ok" if result.returncode == 0 else "error"
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
    return {"status": status, "returncode": code, "seconds": time.monotonic() - started,
            "log": str(log)}


def collect(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("sources", "checks"):
        (root / name).mkdir()
    config = {"env": {"ascension": 0, "max_act": 1, "combat_control": "llm",
                      "combat_observation": "combat_public_v3", "battle_simulations": args.simulations},
              "agent": SearchAgent.name, "simulations": args.simulations, "search_seed": 1,
              "max_decisions": args.max_decisions, "seconds": args.seconds,
              "seed_start": args.seed_start, "world_timeout": args.world_timeout,
              "split_rule": "world_seed % 4 == 0: eval; otherwise train",
              "cli_sha256": file_sha256(SCRIPT),
              "implementation_sha256": file_sha256(Path(lab.__file__)),
              "validation": "independent duplicate plus exact public-prefix replay per included world"}
    lab.write_new(root / "config.json", config)
    sources, states, reports = [], [], []
    interface = None
    started = time.monotonic()
    deadline = started + args.seconds
    seed = args.seed_start
    while time.monotonic() < deadline:
        relative = f"sources/seed_{seed}_r0.jsonl"
        output = root / relative
        duplicate = root / "checks" / output.name
        report = {"world_seed": seed, "source": relative, "status": "excluded", "checks": []}
        commands = [([str(SCRIPT), "worker", "--root", str(root), "--seed", str(seed),
                      "--output", str(target)], target.with_suffix(".log"))
                    for target in (output, duplicate)]
        commands.append(([str(SCRIPT.with_name("fight_lab.py")), "_verify", "--root", str(root),
                          "--seed", str(seed), "--output", str(output)],
                         root / "checks" / f"replay_{seed}.log"))
        for command, log in commands:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                report["reason"] = "collection_deadline"
                break
            check = process(command, log, min(args.world_timeout, remaining))
            report["checks"].append(check)
            if check["status"] != "ok":
                report["reason"] = ("collection_deadline" if time.monotonic() >= deadline
                                    else check["status"])
                break
        if len(report["checks"]) == 3 and all(c["status"] == "ok" for c in report["checks"]):
            records = lab.read_records(output)
            meta = lab.read_json(output.with_suffix(".meta.json"))
            other_meta = lab.read_json(duplicate.with_suffix(".meta.json"))
            if not records or meta["stopped_reason"] not in ("terminal", "max_decisions") or meta["undefined_behavior_evoked"]:
                report["reason"] = "empty_or_failed_rollout"
            elif lab.trace_digest(records) != lab.trace_digest(lab.read_records(duplicate)) or any(
                meta[k] != other_meta[k] for k in ("stopped_reason", "outcome", "undefined_behavior_evoked")
            ):
                report["reason"] = "cross_process_divergence"
            else:
                current = {k: v for k, v in meta["extra"]["interface_provenance"].items() if k in (
                    "python_serializer_sha256", "glossary_sha256", "simulator_patch_sha256",
                    "simulator_binary_sha256", "combat_observation")}
                if interface is not None and current != interface:
                    raise ValueError("interface changed during collection")
                interface = current
                report["status"] = "included"
                sources.append({"path": relative, "sha256": file_sha256(output),
                                "trace_sha256": lab.trace_digest(records),
                                "stopped_reason": meta["stopped_reason"], "n_records": len(records)})
                states.extend(lab.index_records(records, relative))
        reports.append(report)
        lab.write_new(root / "checks" / f"seed_{seed}.json", report)
        progress = {"elapsed_seconds": time.monotonic() - started, "attempted_worlds": len(reports),
                    "included_worlds": len(sources), "states": len(states),
                    "battle_starts": sum(row["battle_start"] for row in states), "latest": report}
        temporary = root / "progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(root / "progress.json")
        print(json.dumps({k: v for k, v in progress.items() if k != "latest"}), flush=True)
        seed += 1
    elapsed = time.monotonic() - started
    lab.write_new(root / "collection.json", {"kind": "fight_lab_collection", "version": 1,
        "config": config, "replay_interface": interface or {}, "sources": sources,
        "states": states, "reports": reports})
    lab.write_new(root / "summary.json", {"collection_seconds": elapsed,
        "included_worlds": len(sources), "attempted_worlds": len(reports), "states": len(states),
        "combat_states": sum(row["phase"] == "combat" for row in states),
        "battle_starts": sum(row["battle_start"] for row in states),
        "actual_seeds": [report["world_seed"] for report in reports],
        "bytes_before_summary": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "note": "Duration includes duplicate generation and replay. All excluded/partial files retained."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--seed-start", type=int, default=3001)
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--world-timeout", type=float, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.seconds, args.simulations, args.max_decisions, args.world_timeout) <= 0:
        parser.error("duration, budgets, and timeout must be positive")
    if args.mode == "worker":
        if args.seed is None or args.output is None:
            parser.error("worker requires --seed and --output")
        run_worker(args.root, args.seed, args.output)
    else:
        collect(args)


if __name__ == "__main__":
    main()
