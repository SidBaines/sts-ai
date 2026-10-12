"""Fixed OOC PPO controls and matched live-development evaluation commands.

This module only prepares commands; the campaign owns GPU scheduling, deadline
accounting, policy selection and subsequent joint/composite experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def ooc_screen_recipes() -> list[dict[str, Any]]:
    """Five equal 90-second controls; every condition starts from one policy."""
    return [
        {"name": "ooc_lr0.0001_progress0.05", "learning_rate": 1e-4, "reward_mode": "progress", "progress_floor_weight": .05},
        {"name": "ooc_lr0.0003_progress0.05", "learning_rate": 3e-4, "reward_mode": "progress", "progress_floor_weight": .05},
        {"name": "ooc_lr0.001_progress0.05", "learning_rate": 1e-3, "reward_mode": "progress", "progress_floor_weight": .05},
        {"name": "ooc_lr0.0001_sparse", "learning_rate": 1e-4, "reward_mode": "sparse", "progress_floor_weight": .05},
        {"name": "ooc_lr0.0001_progress0.02", "learning_rate": 1e-4, "reward_mode": "progress", "progress_floor_weight": .02},
    ]


def ooc_training_command(source: str | Path, output: str | Path, recipe: dict[str, Any], *,
                         root: str | Path = DEFAULT_ROOT, seconds: float = 90.) -> list[str]:
    root, output = Path(root), Path(output)
    return [str(root / ".venv/bin/python"), str(root / "scripts/train_nonllm.py"), "ppo",
            "--checkpoint", str(source), "--output", str(output), "--metrics", str(output.with_suffix(".metrics.json")),
            "--task-kind", "ooc", "--split", "train", "--world-start", "22000", "--episodes", "64",
            "--seed", "11", "--horizon", "512", "--max-act", "1", "--simulations", "200",
            "--gamma", "1", "--clip-ratio", ".1", "--entropy-coef", ".01", "--goal-act-clear",
            "--learning-rate", str(recipe["learning_rate"]), "--reward-mode", str(recipe["reward_mode"]),
            "--progress-floor-weight", str(recipe["progress_floor_weight"]), "--terminal-hp-weight", "0",
            "--device", "mps", "--inference-device", "cpu", "--workers", "8", "--threads", "2",
            "--batch-size", "1024", "--updates", "100000", "--seconds", str(seconds),
            "--fast-observations", "--fast-replay", "--deferred-bootstrap", "--warm-optimizer"]


def ooc_evaluation_command(checkpoint: str | Path, metrics: str | Path, *,
                           root: str | Path = DEFAULT_ROOT, seconds: float = 120.) -> list[str]:
    root, metrics = Path(root), Path(metrics)
    return [str(root / ".venv/bin/python"), str(root / "scripts/train_nonllm.py"), "evaluate",
            "--checkpoint", str(checkpoint), "--output", str(metrics.with_suffix(".unused.pt")), "--metrics", str(metrics),
            "--task-kind", "ooc", "--split", "dev", "--world-start", "10000", "--episodes", "32",
            "--seed", "777", "--horizon", "1024", "--max-act", "1", "--simulations", "200",
            "--gamma", "1", "--goal-act-clear", "--terminal-hp-weight", "0",
            "--device", "cpu", "--inference-device", "cpu", "--workers", "8", "--threads", "2",
            "--seconds", str(seconds), "--fast-observations", "--fast-replay", "--deferred-bootstrap"]


def ooc_screen_plan(source: str | Path, directory: str | Path, *, root: str | Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    """Return recipe + train/eval arguments without executing any command."""
    directory = Path(directory)
    plans = []
    for recipe in ooc_screen_recipes():
        checkpoint = directory / (recipe["name"] + ".pt")
        metrics = directory / (recipe["name"] + "_dev.metrics.json")
        plans.append({**recipe, "checkpoint": str(checkpoint), "training_seconds": 90,
                      "training_command": ooc_training_command(source, checkpoint, recipe, root=root),
                      "evaluation_command": ooc_evaluation_command(checkpoint, metrics, root=root),
                      "selection_metrics": ["act_clear_count", "mean_floor", "fault_count", "cohort_completed"],
                      "future_composite_rule": "Reevaluate health combat alternatives with an improved OOC policy before joint freeze; keep current combat selection separate."})
    return plans
