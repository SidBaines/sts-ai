"""In-process GRPO outer loop over streaming rollouts and TRL adapter updates.

Periodic eval is intentionally out of v1: run eval separately with
``run_until.py --split eval`` and compare adapters with ``compare_paired.py``.
"""
from __future__ import annotations

import datetime
import gc
import json
import logging
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from sts_ai.seeding import expand_specs, rollout_stem
from sts_ai.streaming_rollout import run_streaming_rollouts
from sts_ai.rollout import current_git_sha
from sts_ai.train import train_pg_trl
from sts_ai.train.pg_dataset import build_pg_dataset

train_pg_trl_train = train_pg_trl.train

log = logging.getLogger(__name__)


def floor_stats(rollouts_dir: Path) -> dict[str, Any]:
    """Aggregate the reward signal (final floor / outcome) from a directory of
    ``*.meta.json`` rollout sidecars. Pure: reads only the filesystem."""
    floors: list[int] = []
    n_rollouts = 0
    n_win = 0
    n_budget_truncated = 0
    for meta_path in sorted(Path(rollouts_dir).glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n_rollouts += 1
        try:
            floors.append(int(meta.get("final_floor", 0) or 0))
        except (TypeError, ValueError):
            floors.append(0)
        if "VICTORY" in str(meta.get("outcome", "")):
            n_win += 1
        extra = meta.get("extra") or {}
        if extra.get("budget_truncated"):
            n_budget_truncated += 1
    return {
        "n_rollouts": n_rollouts,
        "mean_floor": statistics.mean(floors) if floors else 0.0,
        "median_floor": statistics.median(floors) if floors else 0.0,
        "max_floor": max(floors) if floors else 0,
        "min_floor": min(floors) if floors else 0,
        "n_win": n_win,
        "win_rate": (n_win / n_rollouts) if n_rollouts else 0.0,
        "budget_truncated_rate": (n_budget_truncated / n_rollouts) if n_rollouts else 0.0,
    }


def _train_metrics(adapter_dir: Path) -> dict[str, float]:
    """Mean of each numeric metric across the persisted TRL log history."""
    log_path = Path(adapter_dir) / "trainer_log.json"
    if not log_path.exists():
        return {}
    try:
        history = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    collected: dict[str, list[float]] = {}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if key in ("epoch", "step", "total_flos"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            collected.setdefault(key, []).append(float(value))
    return {key: statistics.mean(values) for key, values in collected.items() if values}


def _free_accelerator_memory() -> None:
    """Best-effort release of cached GPU memory before vLLM re-acquires it.

    No-op (and silent) when torch is absent, so the dependency-free unit tier is
    unaffected. The authoritative free is inside the trainer; this is insurance
    against a co-resident wake() OOM at the start of the next GRPO iteration.
    """
    try:
        import gc

        import torch
    except Exception:
        return
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        log.warning("GPU memory free failed (continuing): %s", exc)


def _hf_upload(api: Any, repo_id: str, folder: Path, path_in_repo: str) -> bool:
    """Best-effort upload of a folder to an HF model repo; never raises."""
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            repo_type="model",
        )
        return True
    except Exception as exc:  # noqa: BLE001 - durable storage is best-effort
        log.warning("HF upload failed (%s -> %s): %s", folder, path_in_repo, exc)
        return False


def select_iteration_seeds(
    train_seeds: list[int],
    iteration: int,
    seeds_per_iter: int,
) -> list[int]:
    """Return a deterministic rotating seed window for one GRPO iteration."""
    if iteration < 0:
        raise ValueError("iteration must be >= 0")
    if seeds_per_iter < 1:
        raise ValueError("seeds_per_iter must be >= 1")
    if not train_seeds:
        return []
    if seeds_per_iter >= len(train_seeds):
        return list(train_seeds)

    start = (iteration * seeds_per_iter) % len(train_seeds)
    return [
        train_seeds[(start + offset) % len(train_seeds)]
        for offset in range(seeds_per_iter)
    ]


def _manifest_path(dataset_path: Path) -> Path:
    return Path(str(dataset_path) + ".manifest.json")


def _write_dataset(
    dataset_path: Path,
    examples: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    manifest_path = _manifest_path(dataset_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _run_meta(
    *,
    iteration: int,
    num_iterations: int,
    group_size: int,
    seeds: list[int],
    concurrency: int,
    max_decisions: int,
    output_contract: str | None = None,
    ooc_output_contract: str | None = None,
    policy_seed_salt: int = 0,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "orchestrator": "grpo_loop",
        "iteration": iteration,
        "num_iterations": num_iterations,
        "group_size": group_size,
        "seeds": list(seeds),
        "concurrency": concurrency,
        "max_decisions": max_decisions,
        "policy_seed_salt": policy_seed_salt,
    }
    if output_contract is not None:
        extra["output_contract"] = output_contract
    if ooc_output_contract is not None:
        extra["ooc_output_contract"] = ooc_output_contract
    return {
        "git_sha": current_git_sha(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": extra,
    }


def stratified_example_cap(
    examples: list[dict[str, Any]],
    cap: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically subsample to at most ``cap`` examples, round-robin
    across trajectories so every trajectory's advantage stays represented.

    Decisions within a trajectory share one broadcast advantage, so rows are
    highly redundant; capping per-trajectory both bounds the training pass and
    mildly normalizes the length bias of row-mean PG losses (longer episodes
    otherwise contribute proportionally more gradient mass)."""
    if cap <= 0 or len(examples) <= cap:
        return list(examples)
    rng = random.Random(seed)
    by_stem: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_stem[str(example.get("stem"))].append(example)
    queues = []
    for stem in sorted(by_stem):
        rows = by_stem[stem]
        rng.shuffle(rows)
        queues.append(rows)
    rng.shuffle(queues)
    capped: list[dict[str, Any]] = []
    cursor = 0
    while len(capped) < cap:
        progressed = False
        for rows in queues:
            if cursor < len(rows):
                capped.append(rows[cursor])
                progressed = True
                if len(capped) >= cap:
                    break
        if not progressed:
            break
        cursor += 1
    return capped


def _format_health(rollouts_dir: Path) -> dict[str, float]:
    """Decision-level format diagnostics for one iteration's rollouts.

    Complements the episode-level `agent_invalid_rate`: when RL stalls, these
    separate "the policy can't emit valid actions" from "the policy emits valid
    actions but chooses badly", and track which parser path (exact copy /
    stripped menu index / unique prefix / index fallback) carries the run."""
    n = invalid = retried = 0
    resolution: Counter[str] = Counter()
    for jsonl_path in sorted(rollouts_dir.glob("*.jsonl")):
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            agent = record.get("agent") or {}
            metadata = agent.get("metadata") or {}
            n += 1
            if agent.get("retries", 0) > 0:
                retried += 1
            if not agent.get("valid", True):
                invalid += 1
                continue
            resolution[
                metadata.get("semantic_match")
                or ("index_fallback" if metadata.get("semantic_fallback") else "exact")
            ] += 1
    denom = max(n, 1)
    health = {
        "n_decisions": float(n),
        "decision_invalid_rate": invalid / denom,
        "decision_retry_rate": retried / denom,
    }
    for key, count in resolution.items():
        health[f"resolution_{key}_rate"] = count / denom
    return health


def run_grpo(
    *,
    agent: Any,
    make_env: Callable[[int], Any],
    base_model: str,
    tokenizer: Any,
    tokenizer_id: str,
    framing: str,
    train_seeds: list[int],
    out_dir: Path,
    num_iterations: int,
    group_size: int = 8,
    seeds_per_iter: int = 8,
    start_iteration: int = 0,
    init_adapter_path: str | None = None,
    concurrency: int = 48,
    max_decisions: int = 1500,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
    learning_rate: float = 1e-5,
    std_norm: bool = True,
    eps: float = 1e-6,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    gradient_checkpointing: bool = False,
    wandb_project: str | None = None,
    run_name: str | None = None,
    wandb_config: dict[str, Any] | None = None,
    hf_repo: str | None = None,
    hf_private: bool = True,
    output_contract: str | None = None,
    ooc_output_contract: str | None = None,
    train_example_cap: int | None = None,
    policy_seed_salt: int = 0,
    build_dataset_fn: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = build_pg_dataset,
    train_fn: Callable[..., Path] = train_pg_trl_train,
    run_streaming_fn: Callable[..., Any] = run_streaming_rollouts,
) -> dict[str, Any]:
    """Run in-process GRPO with vLLM sleep/wake and LoRA hot-swap.

    Telemetry is opt-in and best-effort: pass ``wandb_project`` to stream a
    per-iteration reward/advantage/training dashboard to a single wandb run, and
    ``hf_repo`` to incrementally push each iteration's adapter + dataset to a
    HuggingFace model repo (so a mid-run pod failure does not lose progress).

    Resume: pass ``start_iteration=N`` and ``init_adapter_path=<latest adapter>``
    to continue an interrupted run from iteration N (the loop runs
    ``range(start_iteration, num_iterations)`` so seed rotation stays aligned, and
    seeds the first resumed iteration's training from the supplied adapter). No
    cross-iteration optimizer state is needed — each iteration trains a fresh
    optimizer from ``current_adapter`` — so the adapter weights are sufficient to
    resume. The latest adapter is on HF when ``hf_repo`` was set.

    Eval is intentionally not interleaved in v1; run it separately with
    ``run_until.py --split eval`` and ``compare_paired.py``.
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_decisions < 1:
        raise ValueError("max_decisions must be >= 1")
    if start_iteration < 0:
        raise ValueError("start_iteration must be >= 0")
    if start_iteration >= num_iterations:
        raise ValueError("start_iteration must be < num_iterations")
    if not train_seeds:
        raise ValueError("train_seeds must not be empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Telemetry setup (both opt-in, both best-effort) -------------------
    wandb_run = None
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config=wandb_config or {},
                reinit=False,
            )
            log.info("wandb run initialised: project=%s name=%s", wandb_project, run_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb init failed; continuing without wandb: %s", exc)
            wandb_run = None

    hf_api = None
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            hf_api = HfApi()
            hf_api.create_repo(repo_id=hf_repo, repo_type="model", private=hf_private, exist_ok=True)
            log.info("HF repo ready: %s (private=%s)", hf_repo, hf_private)
        except Exception as exc:  # noqa: BLE001
            log.warning("HF repo init failed; continuing without HF upload: %s", exc)
            hf_api = None

    def _wandb_log(payload: dict[str, Any], *, step: int) -> None:
        if wandb_run is None:
            return
        try:
            wandb_run.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb log failed at step %s: %s", step, exc)

    iterations: list[dict[str, Any]] = []
    current_adapter: str | None = init_adapter_path
    if start_iteration > 0 or init_adapter_path is not None:
        log.info(
            "GRPO resume: start_iteration=%s init_adapter_path=%s",
            start_iteration,
            init_adapter_path,
        )

    for iteration in range(start_iteration, num_iterations):
        _free_accelerator_memory()
        agent.wake()
        seeds = select_iteration_seeds(train_seeds, iteration, seeds_per_iter)
        specs = expand_specs(seeds, group_size)
        iter_dir = out_dir / f"iter_{iteration}"
        rollouts_dir = iter_dir / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Per-iteration salt: fresh sampling streams each iteration (standard
        # GRPO exploration), and a non-zero base (bumped by the restart
        # supervisor) re-rolls a wedged iteration instead of deterministically
        # replaying into the same state-dependent simulator hang. Iteration 0
        # at base 0 stays byte-identical to the historical unsalted stream.
        iteration_salt = policy_seed_salt + iteration
        # Supervised-restart resume: an episode whose meta sidecar exists was
        # completed by an earlier attempt at THIS iteration — same policy by
        # construction (every attempt resumes from the same prior adapter;
        # per-attempt salts only vary the sampling stream) — so only missing
        # episodes are (re)played. In-flight leftovers (jsonl without meta)
        # rerun from scratch; _Slot unlinks their partial files on open.
        pending = [
            spec
            for spec in specs
            if not (rollouts_dir / f"{rollout_stem(*spec)}.meta.json").exists()
        ]
        if len(pending) < len(specs):
            log.info(
                "GRPO iter=%s resume: %s/%s episodes already complete, running %s",
                iteration,
                len(specs) - len(pending),
                len(specs),
                len(pending),
            )
        try:
            run_streaming_fn(
                pending,
                make_env,
                agent,
                output_for=lambda ws, ri, d=rollouts_dir: (
                    d / f"{rollout_stem(ws, ri)}.jsonl"
                ),
                concurrency=concurrency,
                max_decisions=max_decisions,
                run_meta=_run_meta(
                    iteration=iteration,
                    num_iterations=num_iterations,
                    group_size=group_size,
                    seeds=seeds,
                    concurrency=concurrency,
                    max_decisions=max_decisions,
                    output_contract=output_contract,
                    ooc_output_contract=ooc_output_contract,
                    policy_seed_salt=iteration_salt,
                ),
                hint_cfg=None,
                policy_seed_salt=iteration_salt,
            )
        finally:
            # Backend sleep() owns the full release choreography (for MLX:
            # drop refs -> gc -> THEN clear the buffer cache; clearing before
            # gc strands a model image in the cache and the trainer's load
            # swap-storms — observed on 2026-08-17 in both dry-runs' iter-1).
            agent.sleep()
            gc.collect()

        dataset_contract_kwargs: dict[str, Any] = {}
        if output_contract is not None:
            dataset_contract_kwargs["output_contract"] = output_contract
        if ooc_output_contract is not None:
            dataset_contract_kwargs["ooc_output_contract"] = ooc_output_contract
        examples, manifest = build_dataset_fn(
            rollouts_dir,
            framing=framing,
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            mode="group",
            std_norm=std_norm,
            eps=eps,
            **dataset_contract_kwargs,
            loss_mask_mode="action",
        )
        manifest["n_examples_total"] = len(examples)
        if train_example_cap is not None:
            examples = stratified_example_cap(
                examples, train_example_cap, seed=iteration
            )
        manifest["n_examples_trained"] = len(examples)
        manifest["train_example_cap"] = train_example_cap

        dataset_path = iter_dir / "pg.jsonl"
        manifest_path = _write_dataset(dataset_path, examples, manifest)

        new_adapter = iter_dir / "adapter"
        train_fn(
            dataset_path=dataset_path,
            base_model=base_model,
            out_adapter_dir=new_adapter,
            init_adapter_path=current_adapter,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
            learning_rate=learning_rate,
            per_device_batch_size=per_device_batch_size,
            grad_accum=grad_accum,
            gradient_checkpointing=gradient_checkpointing,
            manifest_path=manifest_path,
            loss_mask_mode="action",
        )
        current_adapter = str(new_adapter)
        agent.set_adapter(current_adapter)

        rollout_stats = floor_stats(rollouts_dir)
        train_stats = _train_metrics(new_adapter)
        advantage_report = manifest.get("advantage_report", {})
        label_report = manifest.get("label_report", {})

        stats = {
            "iteration": iteration,
            "seeds": seeds,
            "n_specs": len(specs),
            "n_examples": len(examples),
            "advantage_report": advantage_report,
            "rollout_stats": rollout_stats,
            "train_metrics": train_stats,
            "current_adapter": current_adapter,
        }
        iterations.append(stats)
        log.info(
            "grpo iter=%s specs=%s examples=%s mean_floor=%.2f win_rate=%.2f "
            "agent_invalid_rate=%.3f adapter=%s advantage_report=%s",
            iteration,
            len(specs),
            len(examples),
            rollout_stats["mean_floor"],
            rollout_stats["win_rate"],
            label_report.get("agent_invalid_rate", 0.0),
            current_adapter,
            advantage_report,
        )

        format_health = _format_health(rollouts_dir)
        log.info("grpo iter=%s format_health=%s", iteration, format_health)

        # --- wandb: one clean iteration-indexed dashboard ------------------
        wandb_payload: dict[str, Any] = {
            "reward/mean_floor": rollout_stats["mean_floor"],
            "reward/median_floor": rollout_stats["median_floor"],
            "reward/max_floor": rollout_stats["max_floor"],
            "reward/win_rate": rollout_stats["win_rate"],
            "health/n_rollouts": rollout_stats["n_rollouts"],
            "health/budget_truncated_rate": rollout_stats["budget_truncated_rate"],
            "health/agent_invalid_rate": label_report.get("agent_invalid_rate", 0.0),
            "health/n_act_boss_clear": label_report.get("n_act_boss_clear", 0),
            "data/n_examples": len(examples),
            "data/n_examples_total": manifest.get("n_examples_total"),
            "data/n_trajectories_with_advantage": manifest.get("n_trajectories_with_advantage", 0),
            "advantage/mean": advantage_report.get("advantage_mean"),
            "advantage/min": advantage_report.get("advantage_min"),
            "advantage/max": advantage_report.get("advantage_max"),
            "advantage/n_groups": advantage_report.get("n_groups"),
            "advantage/n_zero_variance_groups": advantage_report.get("n_zero_variance_groups"),
        }
        for key, value in format_health.items():
            wandb_payload[f"format/{key}"] = value
        for key, value in train_stats.items():
            wandb_payload[f"train/{key}"] = value
        _wandb_log(
            {k: v for k, v in wandb_payload.items() if v is not None},
            step=iteration,
        )

        # --- HF: incremental durable storage of this iteration -------------
        if hf_api is not None:
            _hf_upload(hf_api, hf_repo, iter_dir, f"grpo/iter_{iteration}")

    summary = {"iterations": iterations, "final_adapter": current_adapter}

    # --- Final flush: summary to disk, HF, and wandb -----------------------
    try:
        (out_dir / "grpo_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("failed to write grpo_summary.json: %s", exc)

    if hf_api is not None and current_adapter is not None:
        _hf_upload(hf_api, hf_repo, Path(current_adapter), "grpo/final_adapter")
        summary_file = out_dir / "grpo_summary.json"
        if summary_file.exists():
            try:
                hf_api.upload_file(
                    path_or_fileobj=str(summary_file),
                    path_in_repo="grpo/grpo_summary.json",
                    repo_id=hf_repo,
                    repo_type="model",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("HF upload of summary failed: %s", exc)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb finish failed: %s", exc)

    return summary
