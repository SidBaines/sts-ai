from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts_ai.agent_factory import build_agent as make_agent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import run_rollout


def build_agent(args: argparse.Namespace):
    return make_agent(
        args.agent,
        seed=args.seed,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        thinking=args.thinking,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hybrid sts_lightspeed rollout.")
    parser.add_argument("--agent", choices=["first", "random", "heuristic", "mlx"], default="first")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument(
        "--combat-control",
        choices=["search", "llm"],
        default="search",
        help="'search' (default): battles auto-resolved by the built-in C++ search agent (hybrid). "
        "'llm': each in-combat decision is surfaced to the agent (full control).",
    )
    parser.add_argument("--battle-simulations", type=int, default=2_000)
    parser.add_argument("--boss-simulation-multiplier", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--error-output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path("data") / "rollouts" / f"rollout_{args.agent}_{args.seed}.jsonl"
    if output.exists():
        output.unlink()

    env = LightspeedHybridEnv(
        seed=args.seed,
        ascension=args.ascension,
        battle_simulations=args.battle_simulations,
        boss_simulation_multiplier=args.boss_simulation_multiplier,
        combat_control=args.combat_control,
    )
    agent = build_agent(args)
    result = run_rollout(env, agent, max_decisions=args.max_decisions, output_path=output)

    print(f"wrote: {output}")
    print(f"stopped_reason: {result.stopped_reason}")
    print(f"decisions: {len(result.decisions)}")
    print(f"terminal_state: {result.terminal_state}")
    if result.error is not None:
        print(f"error: {result.error}")
        if args.error_output is not None:
            payload = {
                "seed": args.seed,
                "stopped_reason": result.stopped_reason,
                "terminal_state": result.terminal_state,
                "decisions": len(result.decisions),
                "error": result.error,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch(exist_ok=True)
            args.error_output.parent.mkdir(parents=True, exist_ok=True)
            args.error_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        sys.exit(2)

    if args.error_output is not None and args.error_output.exists():
        args.error_output.unlink()


if __name__ == "__main__":
    main()
