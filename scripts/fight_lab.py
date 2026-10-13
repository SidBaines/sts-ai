"""Collect, filter, replay, and view deterministic single-fight experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from sts_ai import fight_lab as lab
from sts_ai.provenance import file_sha256


SCRIPT = Path(__file__).resolve()


def worker(args: list[str], log: Path, timeout: float) -> dict:
    with log.open("x") as handle:
        try:
            result = subprocess.run([sys.executable, str(SCRIPT), *args], stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            return {"status": "ok" if result.returncode == 0 else "error",
                    "returncode": result.returncode, "log": log.name}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "log": log.name}


def collect(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "sources").mkdir()
    (root / "checks").mkdir()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide unique world seeds")
    config = {"env": {"ascension": args.ascension, "max_act": args.max_act,
                      "combat_control": "llm", "combat_observation": "combat_public_v3",
                      "battle_simulations": 50}, "agent": "heuristic",
              "max_decisions": args.max_decisions, "seeds": seeds,
              "timeout_seconds": args.timeout,
              "split_rule": "world_seed % 4 == 0: eval; otherwise train",
              "implementation_sha256": file_sha256(Path(lab.__file__)),
              "cli_sha256": file_sha256(SCRIPT)}
    lab.write_new(root / "config.json", config)
    states, sources, reports = [], [], []
    replay_interface = None
    for seed in seeds:
        print(f"seed {seed}: collecting twice and checking replay", flush=True)
        relative = f"sources/seed_{seed}_r0.jsonl"
        output = root / relative
        duplicate = root / "checks" / output.name
        report = {"world_seed": seed, "source": relative, "status": "excluded", "checks": []}
        for target in (output, duplicate):
            check = worker(["_collect", "--root", str(root), "--seed", str(seed),
                            "--output", str(target)], target.with_suffix(".log"), args.timeout)
            report["checks"].append(check)
            if check["status"] != "ok":
                break
        if len(report["checks"]) == 2 and all(c["status"] == "ok" for c in report["checks"]):
            records = lab.read_records(output)
            meta = lab.read_json(output.with_suffix(".meta.json"))
            other_meta = lab.read_json(duplicate.with_suffix(".meta.json"))
            if not records:
                report["reason"] = "empty_trace"
            elif meta["stopped_reason"] not in ("terminal", "max_decisions") or meta["undefined_behavior_evoked"]:
                report["reason"] = "simulator_or_policy_error"
            elif lab.trace_digest(records) != lab.trace_digest(lab.read_records(duplicate)) or any(
                meta[k] != other_meta[k] for k in ("stopped_reason", "outcome", "undefined_behavior_evoked")
            ):
                report["reason"] = "cross_process_divergence"
            else:
                check = worker(["_verify", "--root", str(root), "--seed", str(seed),
                                "--output", str(output)], root / "checks" / f"replay_{seed}.log", args.timeout)
                report["checks"].append(check)
                if check["status"] == "ok":
                    report["status"] = "included"
                    interface = meta["extra"]["interface_provenance"]
                    current = {k: v for k, v in interface.items() if k in (
                        "python_serializer_sha256", "glossary_sha256", "simulator_patch_sha256",
                        "simulator_binary_sha256", "combat_observation")}
                    if replay_interface is not None and current != replay_interface:
                        raise ValueError("simulator/interface changed during collection")
                    replay_interface = current
                    sources.append({"path": relative, "sha256": file_sha256(output),
                                    "trace_sha256": lab.trace_digest(records),
                                    "stopped_reason": meta["stopped_reason"], "n_records": len(records)})
                    states.extend(lab.index_records(records, relative))
                else:
                    report["reason"] = "replay_failed"
        reports.append(report)
        lab.write_new(root / "checks" / f"seed_{seed}.json", report)
        print(f"seed {seed}: {report['status']} {report.get('reason', '')}", flush=True)
    lab.write_new(root / "collection.json", {"kind": "fight_lab_collection", "version": 1,
        "config": config, "replay_interface": replay_interface or {},
        "sources": sources, "states": states, "reports": reports})
    print(f"{len(sources)}/{len(seeds)} sources included; {len(states)} states; "
          f"{sum(s['battle_start'] for s in states)} battle starts. {root}")


def evaluate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    selected = lab.read_json(args.selection)
    lab.validate_selection(lab.read_json(root / "collection.json"), selected)
    if not selected["states"] or any(r["phase"] != "combat" for r in selected["states"]):
        raise ValueError("select one or more combat states before evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {"agent": args.agent, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k,
                "max_tokens": args.max_tokens, "thinking": args.thinking,
                "max_decisions": args.max_decisions, "rollout_index": args.rollout_index,
                "adapter_path": str(args.adapter.resolve()) if args.adapter else None}
    if getattr(args, "combat_observation", None):
        settings["combat_observation"] = args.combat_observation
    lab.write_new(args.output / "settings.json", settings)
    lab.write_new(args.output / "selection.json", selected)
    reports = []
    for row in selected["states"]:
        state_id = row["state_id"]
        output = args.output / f"{state_id}_r{args.rollout_index}.jsonl"
        print(f"evaluating {state_id}", flush=True)
        report = worker(["_eval", "--root", str(root), "--selection", str(args.output / "selection.json"),
            "--settings", str(args.output / "settings.json"), "--state-id", state_id,
            "--output", str(output)], output.with_suffix(".log"), args.timeout)
        report["state_id"] = state_id
        report["trace"] = output.name
        if report["status"] == "ok":
            meta = lab.read_json(output.with_suffix(".meta.json"))
            report["stopped_reason"] = meta["stopped_reason"]
            report["metrics"] = meta["extra"]["local_task"]["metrics"]
            report["n_invalid"] = meta["n_invalid"]
            report["undefined_behavior_evoked"] = meta["undefined_behavior_evoked"]
            records = lab.read_records(output)
            report["generation_cap_hits"] = sum(
                r["agent"].get("completion_tokens", 0) >= settings["max_tokens"] for r in records)
        reports.append(report)
        lab.write_new(args.output / f"{state_id}.result.json", report)
        print(json.dumps(report), flush=True)
    complete = [r for r in reports if r.get("metrics", {}).get("completed")
                and not r.get("undefined_behavior_evoked")]
    lab.write_new(args.output / "report.json", {"expected": len(selected["states"]),
        "completed": len(complete), "wins": sum(r["metrics"]["won"] for r in complete),
        "invalid_decisions": sum(r.get("n_invalid", 0) for r in reports),
        "generation_cap_hits": sum(r.get("generation_cap_hits", 0) for r in reports), "episodes": reports})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--root", type=Path, required=True, help="Fresh collection directory")
    c.add_argument("--seeds", default=",".join(str(i) for i in range(1001, 1013)))
    c.add_argument("--max-decisions", type=int, default=240)
    c.add_argument("--max-act", type=int, default=1)
    c.add_argument("--ascension", type=int, default=0)
    c.add_argument("--timeout", type=float, default=60)
    f = sub.add_parser("filter")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--phase", choices=["", "combat", "out_of_combat"], default="combat")
    f.add_argument("--enemy", default="")
    f.add_argument("--battle-start", action="store_true")
    f.add_argument("--hp-below", type=float)
    f.add_argument("--act", type=int)
    f.add_argument("--floor-min", type=int)
    f.add_argument("--floor-max", type=int)
    f.add_argument("--turn", type=int)
    f.add_argument("--split", choices=["", "train", "eval"], default="")
    e = sub.add_parser("eval")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--selection", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--agent", choices=["heuristic", "first", "random", "mlx"], default="mlx")
    e.add_argument("--adapter", type=Path)
    e.add_argument("--thinking", action="store_true")
    e.add_argument("--combat-observation", choices=("combat_public_v3", "combat_public_v4", "combat_public_v5"), default="combat_public_v5")
    e.add_argument("--temperature", type=float, default=0.0)
    e.add_argument("--top-p", type=float, default=1.0)
    e.add_argument("--top-k", type=int, default=-1)
    e.add_argument("--max-tokens", type=int, default=4096)
    e.add_argument("--max-decisions", type=int, default=200)
    e.add_argument("--rollout-index", type=int, default=0)
    e.add_argument("--timeout", type=float, default=1800)
    s = sub.add_parser("serve")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--port", type=int, default=8010)
    for name in ("_collect", "_verify", "_eval"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int)
        p.add_argument("--selection", type=Path)
        p.add_argument("--settings", type=Path)
        p.add_argument("--state-id")
    args = parser.parse_args()
    for key in ("timeout", "max_decisions", "max_tokens"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.command == "collect":
        collect(args)
    elif args.command == "filter":
        filters = lab.StateFilter(**{k: getattr(args, k) for k in asdict(lab.StateFilter())})
        selected = lab.selection(lab.read_json(args.root / "collection.json"), filters)
        lab.write_new(args.output, selected)
        print(f"Selected {len(selected['states'])} states → {args.output}")
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "serve":
        import uvicorn
        from sts_ai.fight_lab_server import create_app
        uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)
    elif args.command == "_collect":
        lab.collect_worker(lab.read_json(args.root / "config.json"), args.seed, args.output)
    elif args.command == "_verify":
        from sts_ai.lightspeed import LightspeedHybridEnv
        from sts_ai.state_sanity import phantom_power_findings
        records = lab.read_records(args.output)
        for record in records:
            enemies = (record["state"].get("combat") or {}).get("enemies", [])
            if any(e["name"] == "GREMLIN_NOB" for e in enemies):
                if phantom_power_findings(record["state_text"], task_id="gremlin_nob"):
                    raise ValueError("known phantom powers detected")
        env = LightspeedHybridEnv(world_seed=args.seed, **lab.read_json(args.root / "config.json")["env"])
        lab.replay_to(env, records, len(records) - 1)
        print(f"Verified {len(records)} public states")
    elif args.command == "_eval":
        lab.evaluate_worker(args.root, lab.read_json(args.selection), args.state_id,
                            lab.read_json(args.settings), args.output)


if __name__ == "__main__":
    main()
