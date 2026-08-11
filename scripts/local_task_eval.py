#!/usr/bin/env python
"""Evaluate a model or adapter on a detachable local-curriculum task."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from sts_ai.agent_factory import build_agent
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.local_tasks import base, get_task
from sts_ai.local_tasks.runner import (
    LocalTaskEnv,
    inject_local_task_meta,
    replay_task_start,
    run_local_task_episode,
    window_rollout_indices,
    windows_for_split,
    one_window_per_seed,
)
from sts_ai.prompting import NEUTRAL_FRAME, OUTPUT_CONTRACTS, REASONING_ACTION_OUTPUT
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.rollout import current_git_sha
from sts_ai.seeding import derive_policy_seed, rollout_stem
from sts_ai.streaming_rollout import run_streaming_rollouts


LOCAL_TASK_EVAL_CONFIG_VERSION = 1
REQUIRED_INTERFACE_DIGESTS = (
    "prompt_probe_sha256",
    "chat_template_probe_hash",
    "python_serializer_sha256",
    "glossary_sha256",
    "prompting_sha256",
    "simulator_patch_sha256",
    "simulator_binary_sha256",
)


def build_eval_config(
    args: argparse.Namespace,
    *,
    task_id: str,
    manifest_sha256: str,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Canonical command/cohort contract stored in every eval sidecar."""
    adapter_path = (
        str(Path(args.adapter_path).expanduser().resolve())
        if args.adapter_path is not None
        else None
    )
    window_ids = sorted(str(window["window_id"]) for window in windows)
    rollout_identities = sorted(
        (int(window["world_seed"]), rollout_index)
        for window in windows
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window)
    )
    return {
        "version": LOCAL_TASK_EVAL_CONFIG_VERSION,
        "task_id": task_id,
        "split": args.split,
        "source_manifest_sha256": manifest_sha256,
        "model_id": args.model,
        "framing": NEUTRAL_FRAME,
        "backend": args.backend,
        "adapter_path": adapter_path,
        "adapter_provenance": adapter_provenance(args.adapter_path),
        "combat_observation": args.combat_observation,
        "max_decisions": args.max_decisions,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_retries": args.max_retries,
        "thinking": args.thinking,
        "output_contract": args.output_contract,
        "preserve_special_tokens": args.preserve_special_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "concurrency": args.concurrency if args.backend == "vllm" else 1,
        "rollouts_per_window": args.rollouts_per_window,
        "battle_simulations": args.battle_simulations,
        "max_act": args.max_act,
        "expected_window_count": len(window_ids),
        "expected_window_ids_sha256": hashlib.sha256(
            json.dumps(window_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "expected_rollout_count": len(rollout_identities),
        "expected_rollout_identities_sha256": hashlib.sha256(
            json.dumps(rollout_identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def validate_resume_meta(
    meta_path: Path,
    *,
    expected_eval_config: dict[str, Any],
    window: dict[str, Any],
    world_seed: int,
    rollout_index: int,
) -> None:
    """Fail closed unless a completed sidecar belongs to this exact command."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from invalid meta {meta_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"cannot resume from non-object meta {meta_path}")

    problems: list[str] = []
    expected_identity = {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "policy_seed": derive_policy_seed(world_seed, rollout_index),
    }
    for name, expected in expected_identity.items():
        if meta.get(name) != expected:
            problems.append(f"{name}: stored={meta.get(name)!r}, expected={expected!r}")

    jsonl_path = Path(str(meta_path).removesuffix(".meta.json") + ".jsonl")
    try:
        trace_lines = [
            line
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        problems.append(f"trace: missing/unreadable {jsonl_path}: {exc}")
        trace_lines = []
    if not trace_lines:
        problems.append("trace: contains no decision records")
    for line_number, line in enumerate(trace_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"trace line {line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            problems.append(f"trace line {line_number}: not a JSON object")
            continue
        if record.get("decision_index") != line_number - 1:
            problems.append(
                f"trace line {line_number} decision_index: "
                f"stored={record.get('decision_index')!r}, expected={line_number - 1}"
            )
        for name, expected in expected_identity.items():
            if record.get(name) != expected:
                problems.append(
                    f"trace line {line_number} {name}: "
                    f"stored={record.get(name)!r}, expected={expected!r}"
                )

    extra = meta.get("extra")
    if not isinstance(extra, dict):
        problems.append("extra: missing/non-object")
        extra = {}
    if extra.get("local_task_eval") is not True:
        problems.append("extra.local_task_eval: not true")
    stored_eval_config = extra.get("local_task_eval_config")
    if stored_eval_config != expected_eval_config:
        problems.append(
            "extra.local_task_eval_config differs from the current command/cohort: "
            f"stored={stored_eval_config!r}, expected={expected_eval_config!r}"
        )

    # Cross-check generic rollout provenance so a hand-edited/stale contract
    # cannot authorize sidecars produced by a different model configuration.
    top_expected = {
        "model_id": expected_eval_config["model_id"],
        "framing": expected_eval_config["framing"],
        "temperature": expected_eval_config["temperature"],
        "max_tokens": expected_eval_config["max_tokens"],
        "thinking": expected_eval_config["thinking"],
        "max_retries": expected_eval_config["max_retries"],
        "battle_simulations": expected_eval_config["battle_simulations"],
        "combat_control": "llm",
    }
    for name, expected in top_expected.items():
        if meta.get(name) != expected:
            problems.append(f"{name}: stored={meta.get(name)!r}, expected={expected!r}")
    meta_decisions = meta.get("n_decisions")
    if (
        isinstance(meta_decisions, bool)
        or not isinstance(meta_decisions, int)
        or meta_decisions < 0
    ):
        problems.append(f"n_decisions: invalid {meta_decisions!r}")
    elif len(trace_lines) != meta_decisions:
        problems.append(
            f"trace length {len(trace_lines)} disagrees with "
            f"meta.n_decisions {meta_decisions}"
        )
    expected_adapter_path = expected_eval_config["adapter_path"]
    flat_extra_expected = {
        "task_id": expected_eval_config["task_id"],
        "split": expected_eval_config["split"],
        "source_manifest_sha256": expected_eval_config["source_manifest_sha256"],
        "adapter_path": expected_adapter_path,
        "adapter_provenance": expected_eval_config["adapter_provenance"],
        "combat_observation": expected_eval_config["combat_observation"],
        "competence_interface_version": expected_eval_config["combat_observation"],
        "output_contract": expected_eval_config["output_contract"],
        "orchestrator": (
            "streaming" if expected_eval_config["backend"] == "vllm" else "serial"
        ),
        "concurrency": expected_eval_config["concurrency"],
    }
    for name, expected in flat_extra_expected.items():
        stored = extra.get(name)
        if name == "adapter_path" and stored is not None:
            stored = str(Path(stored).expanduser().resolve())
        if stored != expected:
            problems.append(f"extra.{name}: stored={stored!r}, expected={expected!r}")

    agent_config = extra.get("agent_config")
    if not isinstance(agent_config, dict):
        problems.append("extra.agent_config: missing/non-object")
    else:
        agent_expected = {
            "model_id": expected_eval_config["model_id"],
            "framing": expected_eval_config["framing"],
            "temperature": expected_eval_config["temperature"],
            "max_tokens": expected_eval_config["max_tokens"],
            "thinking": expected_eval_config["thinking"],
            "max_retries": expected_eval_config["max_retries"],
            "output_contract": expected_eval_config["output_contract"],
            "adapter_path": expected_adapter_path,
        }
        for name, expected in agent_expected.items():
            stored = agent_config.get(name)
            if name == "adapter_path" and stored is not None:
                stored = str(Path(stored).expanduser().resolve())
            if stored != expected:
                problems.append(
                    f"extra.agent_config.{name}: stored={stored!r}, expected={expected!r}"
                )
        if expected_eval_config["backend"] == "vllm":
            for name in ("top_p", "top_k", "enable_prefix_caching"):
                expected = expected_eval_config[name]
                if agent_config.get(name) != expected:
                    problems.append(
                        f"extra.agent_config.{name}: "
                        f"stored={agent_config.get(name)!r}, expected={expected!r}"
                    )
            if agent_config.get("backend") != "vllm":
                problems.append("extra.agent_config.backend: expected 'vllm'")
            preserve = agent_config.get("preserve_special_tokens")
            requested_preserve = expected_eval_config["preserve_special_tokens"]
            expected_preserve = {"on": True, "off": False}.get(requested_preserve)
            if not isinstance(preserve, bool) or (
                expected_preserve is not None and preserve != expected_preserve
            ):
                problems.append(
                    "extra.agent_config.preserve_special_tokens disagrees with "
                    f"request {requested_preserve!r}: stored={preserve!r}"
                )
        reasoning_mode = agent_config.get("reasoning_mode")
        if expected_eval_config["thinking"]:
            if reasoning_mode not in {"native", "prompted"}:
                problems.append(
                    "extra.agent_config.reasoning_mode must be native/prompted "
                    "when --thinking is enabled"
                )
        elif reasoning_mode != "none":
            problems.append(
                "extra.agent_config.reasoning_mode must be 'none' without --thinking"
            )

    interface = extra.get("interface_provenance")
    if not isinstance(interface, dict):
        problems.append("extra.interface_provenance: missing/non-object")
    else:
        for name in REQUIRED_INTERFACE_DIGESTS:
            digest = interface.get(name)
            # chat_template_probe_hash is sft_format's short 16-hex probe
            # digest by design; the rest are full SHA-256 file digests.
            pattern = (
                r"[0-9a-f]{16}"
                if name == "chat_template_probe_hash"
                else r"[0-9a-f]{64}"
            )
            if not isinstance(digest, str) or re.fullmatch(pattern, digest) is None:
                problems.append(
                    f"extra.interface_provenance.{name}: missing/invalid digest"
                )
        interface_expected = {
            "combat_observation": expected_eval_config["combat_observation"],
            "competence_interface_version": expected_eval_config["combat_observation"],
            "output_contract": expected_eval_config["output_contract"],
        }
        for name, expected in interface_expected.items():
            if interface.get(name) != expected:
                problems.append(
                    f"extra.interface_provenance.{name}: "
                    f"stored={interface.get(name)!r}, expected={expected!r}"
                )

    local_task = extra.get("local_task")
    if not isinstance(local_task, dict):
        problems.append("extra.local_task: missing/non-object")
    else:
        local_expected = {
            "task_id": expected_eval_config["task_id"],
            "window_id": window["window_id"],
            "source_stem": window["source_stem"],
            "split": expected_eval_config["split"],
        }
        for name, expected in local_expected.items():
            if local_task.get(name) != expected:
                problems.append(
                    f"extra.local_task.{name}: "
                    f"stored={local_task.get(name)!r}, expected={expected!r}"
                )
        reward = local_task.get("reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            problems.append("extra.local_task.reward: missing/non-finite numeric")
        if not isinstance(local_task.get("metrics"), dict):
            problems.append("extra.local_task.metrics: missing/non-object")

    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(
            f"refusing to resume incompatible completed episode {meta_path}; "
            f"pass --overwrite to replace it:\n  - {detail}"
        )


def preflight_resume_outputs(
    args: argparse.Namespace,
    *,
    windows: list[dict[str, Any]],
    expected_eval_config: dict[str, Any],
    task: Any | None = None,
) -> None:
    expected: dict[Path, tuple[dict[str, Any], int, int]] = {}
    for window in windows:
        world_seed = int(window["world_seed"])
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            path = args.output_dir / f"{rollout_stem(world_seed, rollout_index)}.meta.json"
            if path in expected:
                raise ValueError(
                    f"manifest/cohort maps multiple windows to output identity {path.name}"
                )
            expected[path] = (window, world_seed, rollout_index)

    unexpected = sorted(set(args.output_dir.rglob("seed_*_r*.meta.json")) - set(expected))
    if unexpected:
        raise ValueError(
            "output directory contains completed episodes outside the current "
            f"command/cohort; use a clean directory: {[str(path) for path in unexpected]!r}"
        )
    if args.overwrite:
        return
    for path, (window, world_seed, rollout_index) in expected.items():
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            local_task = (
                (value.get("extra") or {}).get("local_task")
                if isinstance(value, dict)
                else None
            )
            if not isinstance(local_task, dict):
                jsonl_path = Path(str(path).removesuffix(".meta.json") + ".jsonl")
                if task is None or not jsonl_path.exists() or not jsonl_path.read_text(
                    encoding="utf-8"
                ).strip():
                    raise ValueError(
                        f"cannot safely repair completed episode {path}; local-task "
                        "metrics or a non-empty trace are missing"
                    )
                if not inject_local_task_meta(path, jsonl_path, task, window):
                    raise ValueError(f"failed to repair local-task metrics in {path}")
            validate_resume_meta(
                path,
                expected_eval_config=expected_eval_config,
                window=window,
                world_seed=world_seed,
                rollout_index=rollout_index,
            )


def postflight_eval_outputs(
    args: argparse.Namespace,
    *,
    windows: list[dict[str, Any]],
    expected_eval_config: dict[str, Any],
) -> None:
    """Require every requested episode to finish with a valid trace and meta."""
    missing: list[str] = []
    for window in windows:
        world_seed = int(window["world_seed"])
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            meta_path = args.output_dir / (
                f"{rollout_stem(world_seed, rollout_index)}.meta.json"
            )
            if not meta_path.exists():
                missing.append(meta_path.name)
                continue
            validate_resume_meta(
                meta_path,
                expected_eval_config=expected_eval_config,
                window=window,
                world_seed=world_seed,
                rollout_index=rollout_index,
            )
    if missing:
        raise ValueError(f"local-task eval incomplete; missing completed metas: {missing!r}")


def _run_vllm_eval(
    *,
    args: argparse.Namespace,
    task,
    windows: list[dict],
    run_meta: dict,
    agent,
) -> int:
    by_seed = one_window_per_seed(windows)
    specs = []
    for window in windows:
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            out = args.output_dir / (
                f"{rollout_stem(int(window['world_seed']), rollout_index)}.jsonl"
            )
            if out.with_suffix(".meta.json").exists():
                continue  # completed episode from a prior (crashed/resumed) run
            if out.exists():
                out.unlink()  # partial episode: meta is written last, so no meta = incomplete
            specs.append((int(window["world_seed"]), rollout_index))

    def make_env(seed: int) -> LocalTaskEnv:
        env = LightspeedHybridEnv(
            world_seed=seed,
            combat_control="llm",
            combat_observation=args.combat_observation,
            battle_simulations=args.battle_simulations,
            max_act=args.max_act,
        )
        replay_task_start(env, by_seed[int(seed)], task)
        return LocalTaskEnv(env, task)

    results = run_streaming_rollouts(
        specs,
        make_env,
        agent,
        output_for=lambda ws, ri: args.output_dir / f"{rollout_stem(ws, ri)}.jsonl",
        concurrency=args.concurrency,
        max_decisions=args.max_decisions,
        max_retries=args.max_retries,
        run_meta=run_meta,
    )
    # Post-process every expected episode (not just this run's results) so
    # episodes completed by an earlier crashed process also get task metrics.
    for window in windows:
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            out = args.output_dir / (
                f"{rollout_stem(int(window['world_seed']), rollout_index)}.jsonl"
            )
            inject_local_task_meta(out.with_suffix(".meta.json"), out, task, window)
    return len(results)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-task eval.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("mlx", "vllm"), default="mlx")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument(
        "--combat-observation",
        choices=("legacy", "combat_public_v1", "combat_public_v2", "combat_public_v3"),
        default="legacy",
        help="Combat state serializer. Pin this within a matched eval comparison.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--output-contract",
        choices=OUTPUT_CONTRACTS,
        default=REASONING_ACTION_OUTPUT,
        help="Assistant JSON schema. Use action_only for the COMP-005 teacher "
        "positive control; the historical reasoning_action prompt remains default.",
    )
    parser.add_argument(
        "--preserve-special-tokens",
        choices=("auto", "on", "off"),
        default="auto",
        help="vLLM-only: preserve native special tokens in completions.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument(
        "--rollouts-per-window",
        type=int,
        default=1,
        help="Sample K episodes per task window (policy seed varies with the "
        "rollout index; use temperature > 0 or all K repeats are identical).",
    )
    parser.add_argument("--battle-simulations", type=int, default=50)
    parser.add_argument("--max-act", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = base.load_manifest(args.manifest)
    task = get_task(args.task)
    if manifest["task_id"] != task.task_id:
        raise ValueError(f"manifest task_id={manifest['task_id']!r} != --task {args.task!r}")
    windows = windows_for_split(manifest, args.split)
    if not windows:
        raise ValueError(
            f"manifest contains no windows for requested split {args.split!r}"
        )
    manifest_sha256 = file_sha256(args.manifest)
    if manifest_sha256 is None:
        raise ValueError(f"could not fingerprint source manifest: {args.manifest}")
    eval_config = build_eval_config(
        args,
        task_id=task.task_id,
        manifest_sha256=manifest_sha256,
        windows=windows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_resume_outputs(
        args,
        windows=windows,
        expected_eval_config=eval_config,
        task=task,
    )
    agent = build_agent(
        args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_retries=args.max_retries,
        thinking=args.thinking,
        preserve_special_tokens={"auto": None, "on": True, "off": False}[
            args.preserve_special_tokens
        ],
        enable_prefix_caching=args.enable_prefix_caching,
        adapter_path=args.adapter_path,
        output_contract=args.output_contract,
    )
    run_meta = {
        "git_sha": current_git_sha(),
        "battle_simulations": args.battle_simulations,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": {
            "local_task_eval": True,
            "task_id": task.task_id,
            "split": args.split,
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": manifest_sha256,
            "adapter_path": args.adapter_path,
            "adapter_provenance": eval_config["adapter_provenance"],
            "combat_observation": args.combat_observation,
            "output_contract": args.output_contract,
            "orchestrator": "streaming" if args.backend == "vllm" else "serial",
            "concurrency": args.concurrency if args.backend == "vllm" else 1,
            "local_task_eval_config": eval_config,
        },
    }
    if args.backend == "vllm":
        if args.overwrite:
            for window in windows:
                for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
                    out = args.output_dir / (
                        f"{rollout_stem(int(window['world_seed']), rollout_index)}.jsonl"
                    )
                    for path in (
                        out,
                        out.with_suffix(".meta.json"),
                        out.with_suffix(".error.json"),
                    ):
                        if path.exists():
                            path.unlink()
        completed = _run_vllm_eval(
            args=args,
            task=task,
            windows=windows,
            run_meta=run_meta,
            agent=agent,
        )
        postflight_eval_outputs(
            args,
            windows=windows,
            expected_eval_config=eval_config,
        )
        print(f"completed local-task eval episodes: {completed}")
        return

    completed = 0
    for window in windows:
        world_seed = int(window["world_seed"])
        # Local eval outputs identify task windows (and the sample index within
        # a window), not source rollout files: rollout_index = ordinal * K + k.
        for rollout_index in window_rollout_indices(window, args.rollouts_per_window):
            out = args.output_dir / f"{rollout_stem(world_seed, rollout_index)}.jsonl"
            if out.with_suffix(".meta.json").exists() and not args.overwrite:
                continue
            for path in (out, out.with_suffix(".meta.json"), out.with_suffix(".error.json")):
                if path.exists():
                    path.unlink()
            env = LightspeedHybridEnv(
                world_seed=world_seed,
                combat_control="llm",
                combat_observation=args.combat_observation,
                battle_simulations=args.battle_simulations,
                max_act=args.max_act,
            )
            replay_task_start(env, window, task)
            run_local_task_episode(
                task=task,
                env=env,
                agent=agent,
                window=window,
                max_decisions=args.max_decisions,
                output_path=out,
                run_meta=run_meta,
                rollout_index=rollout_index,
            )
            completed += 1
    postflight_eval_outputs(
        args,
        windows=windows,
        expected_eval_config=eval_config,
    )
    print(f"completed local-task eval episodes: {completed}")


if __name__ == "__main__":
    main()
