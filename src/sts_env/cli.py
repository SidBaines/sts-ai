"""Dependency-free command line tools for agents, evaluation and trace replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from sts_env.agents import NativeAgent, choose_action, load_agent

TRACE_SCHEMA = "sts_env_trace_v1"


class AgentError(Exception):
    """Carries an exception raised by agent code, so it is never recorded as a simulator fault."""


def _agent_call(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except Exception as exc:
        raise AgentError(exc) from exc


def make_env(config: dict[str, Any]) -> Any:
    # Lazy: --help, inspect and doctor do not import the simulator.
    from sts_env.env import SlayTheSpireEnv

    return SlayTheSpireEnv(**config)


def observation_hash(observation: dict[str, Any]) -> str:
    payload = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_record(stream: TextIO | None, record: dict[str, Any]) -> None:
    if stream is not None:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()


def env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "max_act": args.max_act,
        "max_steps": args.max_steps,
        "step_timeout": args.step_timeout,
        "battle_simulations": args.battle_simulations,
        "ascension": args.ascension,
        "ooc_seed": args.ooc_seed,
    }


def run_episode(
    *, seed: int, policy_seed: int, agent_spec: str, agent_config: dict[str, Any],
    config: dict[str, Any], output: Path | None = None, human: bool = False,
    show_steps: bool = False, agent_cache: dict[str, Any] | None = None,
    raise_agent_errors: bool = False,
) -> dict[str, Any]:
    """Play one episode and return its end record.

    Errors are recorded rather than raised. ``error["source"]`` is ``"agent"``
    for exceptions from loading, resetting or querying the agent (including an
    illegal returned action) and ``"environment"`` for everything else. With
    ``raise_agent_errors`` an agent exception is re-raised after the trace's end
    record is written, so callers can stop on, for example, out-of-memory errors.
    """
    # Claim output before loading an expensive model or launching the simulator.
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("x", encoding="utf-8") if output is not None else None
    env = None
    started = time.monotonic()
    steps = 0
    total_reward = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}
    status = "error"
    error = None
    agent_exception = None
    try:
        write_record(stream, {
            "type": "header", "schema": TRACE_SCHEMA, "seed": seed,
            "policy_seed": policy_seed, "agent": "human" if human else agent_spec,
            "agent_config": agent_config, "env_config": config,
            **environment_identity(),
        })
        agent = agent_cache.get("agent") if agent_cache is not None else None
        if not human and agent is None:
            agent = _agent_call(load_agent, agent_spec, agent_config)
        if agent is not None and callable(getattr(agent, "reset", None)):
            _agent_call(agent.reset, policy_seed)
            if agent_cache is not None:
                agent_cache["agent"] = agent
        env = make_env(config)
        observation, info = env.reset(seed=seed)
        write_record(stream, {"type": "reset", "observation": observation, "info": info})
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        while not (terminated or truncated):
            if not observation["legal_actions"]:
                # No public decisions remain after reset (e.g. automated mode limit).
                terminated = bool(info.get("terminated", False))
                truncated = not terminated
                break
            if human:
                action = human_action(observation)
                if action is None:
                    status = "quit"
                    truncated = True
                    break
            elif isinstance(agent, NativeAgent):
                # The native baseline queries the simulator, not agent code.
                action = choose_action(agent, observation, env)
            else:
                action = _agent_call(choose_action, agent, observation, env)
            before_hash = observation_hash(observation)
            if show_steps:
                description = next(a["description"] for a in observation["legal_actions"] if a["index"] == action)
                print(f"seed={seed} step={steps} phase={observation['phase']} action={action}: {description}", file=sys.stderr)
            # Persist attempted actions even if the native call fails or stalls.
            write_record(stream, {"type": "action", "step": steps, "action": action, "observation_hash": before_hash})
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            write_record(stream, {
                "type": "step", "step": steps, "action": action,
                "observation": observation, "reward": reward,
                "terminated": terminated, "truncated": truncated, "info": info,
            })
            steps += 1
        if status != "quit":
            status = "truncated" if truncated else "terminated"
    except (KeyboardInterrupt, EOFError):
        status = "interrupted"
        truncated = True
    except AgentError as exc:
        agent_exception = exc.__cause__
        error = {"kind": type(agent_exception).__name__, "message": str(agent_exception), "source": "agent"}
        truncated = True
    except Exception as exc:
        error = {"kind": type(exc).__name__, "message": str(exc), "source": "environment"}
        truncated = True
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                if error is None:
                    error = {"kind": type(exc).__name__, "message": f"close failed: {exc}",
                             "source": "environment"}
                    status = "error"
        summary = {
            "type": "end", "seed": seed, "policy_seed": policy_seed,
            "status": status, "steps": steps, "reward": total_reward,
            "terminated": terminated, "truncated": truncated,
            "info": info, "error": error, "elapsed_seconds": time.monotonic() - started,
        }
        write_record(stream, summary)
        if stream is not None:
            stream.close()
    if raise_agent_errors and agent_exception is not None:
        raise agent_exception
    return summary


def human_action(observation: dict[str, Any]) -> int | None:
    print(observation["text"])
    for action in observation["legal_actions"]:
        print(f"  {action['index']}: {action['description']}")
    legal = {action["index"] for action in observation["legal_actions"]}
    while True:
        answer = input("Action index (q to quit): ").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            return None
        try:
            value = int(answer)
        except ValueError:
            value = None
        if value in legal:
            return value
        print("Choose one of the displayed legal action indices.")


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        if ":" in item:
            pieces = item.split(":")
            if len(pieces) != 2:
                raise ValueError("seed ranges use START:STOP (stop exclusive)")
            start, stop = map(int, pieces)
            if stop <= start:
                raise ValueError("seed ranges must be increasing")
            seeds.extend(range(start, stop))
        else:
            seeds.append(int(item))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least one seed, without duplicates")
    return seeds


def load_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return parse_seeds(args.seeds)
    content = args.seed_file.read_text(encoding="utf-8")
    if content.lstrip().startswith("["):
        values = json.loads(content)
        if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError("seed file JSON must be a list of integers")
        return parse_seeds(",".join(map(str, values)))
    return parse_seeds(",".join(content.split()))


def read_trace(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"trace record at line {line_number} must be an object")
            records.append(record)
    if not records or records[0].get("schema") != TRACE_SCHEMA or records[0].get("type") != "header":
        raise ValueError(f"expected a {TRACE_SCHEMA} header")
    expected_step = 0
    reset_seen = False
    pending = None
    ended = False
    for record in records[1:]:
        kind = record.get("type")
        if ended:
            raise ValueError("trace has records after its end")
        if kind == "reset" and not reset_seen and pending is None and expected_step == 0:
            reset_seen = True
        elif kind == "action" and reset_seen and pending is None:
            if record.get("step") != expected_step:
                raise ValueError("trace action has a nonsequential step index")
            pending = record
        elif kind == "step" and pending is not None:
            if record.get("step") != expected_step or record.get("action") != pending.get("action"):
                raise ValueError(f"trace action/step records disagree at step {expected_step}")
            expected_step += 1
            pending = None
        elif kind == "end":
            ended = True
        else:
            raise ValueError(f"unexpected {kind!r} trace record")
    return records


def inspect_trace(path: Path, show_steps: bool = False) -> dict[str, Any]:
    records = read_trace(path)
    header = records[0]
    ends = [r for r in records if r.get("type") == "end"]
    if show_steps:
        for record in records:
            if record.get("type") == "step":
                print(json.dumps(record, sort_keys=True))
    return {
        "path": str(path), "schema": header["schema"], "seed": header["seed"],
        "agent": header["agent"], "env_config": header["env_config"],
        "completed_steps": sum(r.get("type") == "step" for r in records),
        "attempted_actions": sum(r.get("type") == "action" for r in records),
        "summary": ends[-1] if ends else None, "complete": bool(ends),
    }


def replay_trace(path: Path) -> dict[str, Any]:
    records = read_trace(path)
    header = records[0]
    for key, value in environment_identity().items():
        if header.get(key) != value:
            raise ValueError(f"replay {key} mismatch: saved {header.get(key)!r}, installed {value!r}")
    expected_resets = [r for r in records if r.get("type") == "reset"]
    if len(expected_resets) != 1:
        raise ValueError("replay needs exactly one successful reset record")
    expected_provenance = expected_resets[0]["info"].get("native_provenance")
    if not isinstance(expected_provenance, dict) or not expected_provenance:
        raise ValueError("replay requires saved native provenance")
    steps = [r for r in records if r.get("type") == "step"]
    actions = [r for r in records if r.get("type") == "action"]
    if len(actions) not in {len(steps), len(steps) + 1}:
        raise ValueError("trace action/step records are inconsistent")
    for index, expected in enumerate(steps):
        action = actions[index]
        if expected.get("step") != index or action.get("step") != index or action.get("action") != expected.get("action"):
            raise ValueError(f"trace action/step records disagree at step {index}")
    env = make_env(header["env_config"])
    try:
        observation, info = env.reset(seed=header["seed"])
        # Artifact paths can change when a checkout moves; binary/patch hashes,
        # upstream revision, interpreter and platform remain strict identities.
        actual_provenance = info.get("native_provenance")
        if not isinstance(actual_provenance, dict) or {
            key: value for key, value in actual_provenance.items() if key != "module_file"
        } != {key: value for key, value in expected_provenance.items() if key != "module_file"}:
            raise ValueError("replay native provenance mismatch")
        if observation_hash(observation) != observation_hash(expected_resets[0]["observation"]):
            raise ValueError("replay mismatch at reset: public observation differs")
        for expected in steps:
            pending = actions[expected["step"]]
            if pending.get("observation_hash") != observation_hash(observation):
                raise ValueError(f"replay mismatch before step {expected['step']}: action observation hash differs")
            observation, reward, terminated, truncated, _ = env.step(expected["action"])
            actual = (observation_hash(observation), reward, terminated, truncated)
            wanted = (observation_hash(expected["observation"]), expected["reward"], expected["terminated"], expected["truncated"])
            if actual != wanted:
                raise ValueError(f"replay mismatch at step {expected['step']}: observation/reward/termination differs")
    finally:
        env.close()
    attempted = len(actions)
    return {"path": str(path), "verified_steps": len(steps), "verified_prefix": True,
            "uncompleted_actions": attempted - len(steps),
            "complete": any(r.get("type") == "end" for r in records)}


def doctor() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("STS_LIGHTSPEED_BUILD", root / "external/sts_lightspeed/build"))
    extensions = sorted(str(path) for pattern in ("slaythespire*.so", "slaythespire*.pyd") for path in build.glob(pattern))
    metadata = None
    metadata_error = None
    if extensions:
        try:
            from sts_env._backend.lightspeed_import import native_metadata

            metadata = native_metadata(build)
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            metadata_error = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "repo_root": str(root), "build_dir": str(build),
        "native_extensions": extensions, "ready": bool(metadata),
        "native_provenance": metadata, "metadata_error": metadata_error,
        "check": "build manifest and binary identity; run a bounded rollout to verify execution",
        "build_command": "bash scripts/build_lightspeed.sh",
    }


def environment_identity() -> dict[str, str]:
    from sts_env.env import ENVIRONMENT_VERSION, OBSERVATION_VERSION
    from sts_env.provenance import interface_identity

    return {"environment_version": ENVIRONMENT_VERSION,
            "observation_version": OBSERVATION_VERSION, **interface_identity()}


def json_config(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("agent config must be a JSON object")
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not (0 < number < float("inf")):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def add_episode_flags(parser: argparse.ArgumentParser, *, human: bool = False) -> None:
    if not human:
        parser.add_argument("--agent", default="random", help="first, random, native, or importable module:factory")
        parser.add_argument("--agent-config", type=json_config, default={}, metavar="JSON", help="factory keyword arguments as a JSON object")
    parser.add_argument("--policy-seed", type=int, help="agent RNG seed (default: world seed)")
    parser.add_argument("--mode", choices=("full", "ooc", "combat"), default="ooc", help="full: user controls everything; ooc: native combat; combat: automated OOC")
    parser.add_argument("--ascension", type=int, choices=range(21), default=0, metavar="0..20")
    parser.add_argument("--ooc-seed", type=int, help="separate seed for automatic native out-of-combat choices")
    parser.add_argument("--max-act", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--max-steps", type=positive_int, default=1500)
    parser.add_argument("--step-timeout", type=positive_float, default=30.0, help="native reset/step deadline in seconds")
    parser.add_argument("--battle-simulations", type=positive_int, default=50)
    parser.add_argument("--show-steps", action="store_true", help="print each decision to stderr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sts-env", description="Slay the Spire environment and agent tools")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one seed with a builtin or your own agent")
    add_episode_flags(run)
    run.add_argument("--seed", type=int, default=3)
    run.add_argument("--output", type=Path, help="new JSONL trace path; existing files are protected")
    evaluate = sub.add_parser("evaluate", help="evaluate paired seed sets serially with separate traces")
    add_episode_flags(evaluate)
    seeds = evaluate.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seeds", help="comma-separated integers or START:STOP ranges (stop exclusive)")
    seeds.add_argument("--seed-file", type=Path, help="JSON integer list or one integer per line")
    evaluate.add_argument("--output", type=Path, required=True, help="new directory for per-seed traces and summary.json")
    evaluate.add_argument("--stop-on-agent-error", action="store_true",
                          help="stop the batch after an episode whose error came from the agent (e.g. out of memory)")
    play = sub.add_parser("play", help="choose legal action indices interactively")
    add_episode_flags(play, human=True)
    play.add_argument("--seed", type=int, default=3)
    play.add_argument("--output", type=Path)
    inspect = sub.add_parser("inspect", help="inspect a trace without loading the simulator")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--show-steps", action="store_true")
    replay = sub.add_parser("replay", help="re-execute and verify a saved trace or its completed prefix")
    replay.add_argument("path", type=Path)
    sub.add_parser("doctor", help="read-only Python and simulator build diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"run", "play"}:
            summary = run_episode(
                seed=args.seed, policy_seed=args.seed if args.policy_seed is None else args.policy_seed,
                agent_spec=getattr(args, "agent", "human"), agent_config=getattr(args, "agent_config", {}),
                config=env_config(args), output=args.output, human=args.command == "play", show_steps=args.show_steps,
            )
            print(json.dumps(summary, sort_keys=True))
            return 1 if summary["status"] in {"error", "interrupted"} else 0
        if args.command == "evaluate":
            seeds = load_seeds(args)
            args.output.mkdir(parents=True, exist_ok=False)
            summaries = []
            agent_cache: dict[str, Any] = {}
            for seed in seeds:
                policy_seed = seed if args.policy_seed is None else args.policy_seed
                try:
                    result = run_episode(
                        seed=seed, policy_seed=policy_seed,
                        agent_spec=args.agent, agent_config=args.agent_config, config=env_config(args),
                        output=args.output / f"seed-{seed}.jsonl", show_steps=args.show_steps,
                        agent_cache=agent_cache,
                    )
                except OSError as exc:
                    # A path failure must not erase the batch's completed outcomes.
                    result = {"type": "end", "seed": seed, "policy_seed": policy_seed,
                              "status": "error", "steps": 0, "reward": 0.0,
                              "terminated": False, "truncated": True, "info": {},
                              "error": {"kind": type(exc).__name__, "message": str(exc),
                                        "source": "environment"},
                              "elapsed_seconds": 0.0}
                summaries.append(result)
                print(json.dumps(result, sort_keys=True), file=sys.stderr)
                if result["status"] == "interrupted":
                    break
                if args.stop_on_agent_error and (result.get("error") or {}).get("source") == "agent":
                    break
            report = {
                "seeds_requested": seeds, "episodes": summaries,
                "completed": len(summaries), "errors": sum(r["status"] in {"error", "interrupted"} for r in summaries),
                "error_sources": {name: sum((r.get("error") or {}).get("source") == name for r in summaries)
                                  for name in ("agent", "environment")},
                "mean_reward": sum(r["reward"] for r in summaries) / len(summaries),
                "statuses": {name: sum(r["status"] == name for r in summaries) for name in ("terminated", "truncated", "error", "interrupted")},
                "outcomes": {str(name): sum(r["info"].get("outcome") == name for r in summaries)
                             for name in sorted({r["info"].get("outcome") for r in summaries}, key=str)},
                "reward_population": "all attempted episodes, including errors",
            }
            with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print(json.dumps(report, sort_keys=True))
            return 1 if report["errors"] else 0
        if args.command == "inspect":
            result = inspect_trace(args.path, args.show_steps)
        elif args.command == "replay":
            result = replay_trace(args.path)
        else:
            result = doctor()
        print(json.dumps(result, sort_keys=True))
        return 1 if args.command == "doctor" and not result["ready"] else 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        print(f"sts-env: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
