from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sts_ai.agent_factory import agent_label, build_agent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import run_rollout


def truncate_output(text: str | None, limit: int = 20_000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def parse_seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of hybrid sts_lightspeed rollouts.")
    parser.add_argument("--agent", choices=["first", "random", "heuristic", "mlx"], default="heuristic")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. 1,2,3")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument("--battle-simulations", type=int, default=2_000)
    parser.add_argument("--boss-simulation-multiplier", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "rollouts")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--seed-timeout-seconds",
        type=float,
        default=None,
        help="If set, run each seed in an isolated subprocess and kill it after this many seconds.",
    )

    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    seeds = parse_seed_list(args)
    label = agent_label(
        args.agent,
        model=args.model,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )

    shared_agent = None
    if args.agent == "mlx" and args.seed_timeout_seconds is None:
        shared_agent = build_agent(
            args.agent,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
            thinking=args.thinking,
        )

    print(f"agent={label} seeds={seeds}")
    for seed in seeds:
        output = args.output_dir / label / f"seed_{seed}.jsonl"
        error_output = output.with_suffix(".error.json")
        if output.exists() and not args.overwrite:
            print(f"skip existing: {output}")
            continue
        if output.exists():
            output.unlink()
        if error_output.exists():
            error_output.unlink()

        if args.seed_timeout_seconds is not None:
            run_seed_subprocess(args, seed, output, error_output)
            continue

        agent = shared_agent
        if agent is None:
            agent = build_agent(args.agent, seed=seed)

        env = LightspeedHybridEnv(
            seed=seed,
            ascension=args.ascension,
            battle_simulations=args.battle_simulations,
            boss_simulation_multiplier=args.boss_simulation_multiplier,
        )
        try:
            result = run_rollout(env, agent, max_decisions=args.max_decisions, output_path=output)
        except Exception as exc:  # noqa: BLE001 - keep later seeds running after agent or harness failures.
            payload = {
                "seed": seed,
                "stopped_reason": "batch_error",
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "phase": "run_rollout",
                    "decision_index": None,
                },
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch(exist_ok=True)
            error_output.parent.mkdir(parents=True, exist_ok=True)
            error_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"seed={seed} failed error={payload['error']} output={error_output}")
            continue

        if result.error is not None:
            payload = {
                "seed": seed,
                "stopped_reason": result.stopped_reason,
                "terminal_state": result.terminal_state,
                "decisions": len(result.decisions),
                "error": result.error,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch(exist_ok=True)
            error_output.parent.mkdir(parents=True, exist_ok=True)
            error_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(
            f"seed={seed} decisions={len(result.decisions)} "
            f"stopped={result.stopped_reason} final={result.terminal_state} output={output}"
        )


def run_seed_subprocess(args: argparse.Namespace, seed: int, output: Path, error_output: Path) -> None:
    command = [
        sys.executable,
        "scripts/run_rollout.py",
        "--agent",
        args.agent,
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--max-retries",
        str(args.max_retries),
        "--seed",
        str(seed),
        "--ascension",
        str(args.ascension),
        "--max-decisions",
        str(args.max_decisions),
        "--battle-simulations",
        str(args.battle_simulations),
        "--boss-simulation-multiplier",
        str(args.boss_simulation_multiplier),
        "--output",
        str(output),
        "--error-output",
        str(error_output),
    ]
    if args.thinking:
        command.append("--thinking")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"

    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=args.seed_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "seed": seed,
            "stopped_reason": "timeout",
            "timeout_seconds": args.seed_timeout_seconds,
            "output": str(output),
            "error": {
                "type": "TimeoutExpired",
                "message": f"seed exceeded {args.seed_timeout_seconds} seconds",
                "phase": "subprocess",
                "decision_index": None,
            },
            "stdout": truncate_output(exc.stdout),
            "stderr": truncate_output(exc.stderr),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        error_output.parent.mkdir(parents=True, exist_ok=True)
        error_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"seed={seed} timed_out after={args.seed_timeout_seconds}s output={output} error={error_output}")
        return

    if completed.returncode != 0 and not error_output.exists():
        payload = {
            "seed": seed,
            "stopped_reason": "subprocess_error",
            "returncode": completed.returncode,
            "output": str(output),
            "error": {
                "type": "SubprocessError",
                "message": f"run_rollout exited with code {completed.returncode}",
                "phase": "subprocess",
                "decision_index": None,
            },
            "stdout": truncate_output(completed.stdout),
            "stderr": truncate_output(completed.stderr),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        error_output.parent.mkdir(parents=True, exist_ok=True)
        error_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = "failed" if error_output.exists() else "completed"
    print(f"seed={seed} {status} returncode={completed.returncode} output={output}")


if __name__ == "__main__":
    main()
