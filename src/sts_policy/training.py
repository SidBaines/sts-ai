"""Supervised training and recoverable, versioned policy checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
import math
from typing import Any, Callable, Sequence

import torch

from .losses import behavioral_cloning_loss

CHECKPOINT_VERSION = 1


def seed_training(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def file_identity(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, model: torch.nn.Module, *, model_config: dict[str, Any],
                    optimizer: torch.optim.Optimizer | None = None, step: int = 0,
                    metadata: dict[str, Any] | None = None) -> None:
    """Atomic replace: an interruption cannot partially overwrite a good policy."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": CHECKPOINT_VERSION, "model_config": model_config,
               "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
               "step": step, "metadata": metadata or {}, "python_rng": random.getstate(),
               "torch_rng": torch.get_rng_state(), "mps_rng": None}
    if torch.backends.mps.is_available():
        payload["mps_rng"] = torch.mps.get_rng_state().cpu()
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path, model: torch.nn.Module | None = None, *,
                    optimizer: torch.optim.Optimizer | None = None, restore_rng: bool = False,
                    expected_data_identity: str | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported policy checkpoint version")
    if expected_data_identity is not None and payload["metadata"].get("data_identity") != expected_data_identity:
        raise ValueError("checkpoint dataset identity differs from the requested dataset")
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and payload["optimizer_state"] is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("mps_rng") is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(payload["mps_rng"])
    return payload


def warm_optimizer(optimizer: torch.optim.Optimizer, payload: dict[str, Any], *,
                   method: str, task_kind: str, reward_config: dict[str, Any],
                   learning_rate: float) -> dict[str, Any]:
    """Carry Adam moments across compatible PPO chunks, resetting across stages.

    A new task bank/horizon may be an explicit curriculum change. Preserve its
    identity separately; do not pretend it is an identical-data resume.
    """
    metadata = payload.get("metadata", {})
    config = metadata.get("config", {})
    if metadata.get("stage") != method:
        return {"restored": False, "reason": "training_method_changed"}
    if method != "ppo":
        return {"restored": False, "reason": "only_ppo_chunk_carry_supported"}
    if config.get("task_kind") != task_kind:
        return {"restored": False, "reason": "task_kind_changed"}
    if metadata.get("reward_config") != reward_config:
        return {"restored": False, "reason": "reward_config_changed"}
    if payload.get("optimizer_state") is None:
        return {"restored": False, "reason": "optimizer_state_missing"}
    optimizer.load_state_dict(payload["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return {"restored": True, "reason": "compatible_ppo_chunk", "source_step": payload.get("step", 0),
            "source_data_identity": metadata.get("data_identity"), "learning_rate": learning_rate}


def _label_position(record: dict[str, Any]) -> int:
    observation = record["observation"]
    label = record.get("teacher_index", record.get("action_index", record.get("label")))
    indices = [action["index"] for action in observation["legal_actions"]]
    if label not in indices:
        raise ValueError(f"teacher action {label!r} is absent from legal menu")
    return indices.index(label)


def train_bc(model: torch.nn.Module, records: Sequence[dict[str, Any]], *,
             collate: Callable[..., Any], forward: Callable[..., Any],
             device: str = "cpu", learning_rate: float = 3e-4, batch_size: int = 128,
             epochs: int = 3, seed: int = 0, max_seconds: float | None = None,
             max_steps: int | None = None,
             reseed: bool = True,
             metric_interval: int = 1, bucket_actions: bool = False,
             optimizer: torch.optim.Optimizer | None = None,
             phase_weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Train public observations; privileged teacher provenance stays outside features.

    collate(observations, device) -> batch; forward(model,batch) -> logits,values,mask.
    """
    if not records or batch_size < 1 or epochs < 1 or metric_interval < 1:
        raise ValueError("BC requires data and positive epochs/batch size")
    if reseed:
        seed_training(seed)
    model.to(device).train()
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random
    started = time.monotonic()
    losses, accuracies, examples, steps = [], [], 0, 0
    labels_by_record = [_label_position(record) for record in records]
    if any(not record["observation"].get("legal_actions") for record in records):
        raise ValueError("BC rows must have legal candidates")
    pending_metrics = []

    def flush_metrics() -> None:
        if pending_metrics:
            rows = torch.stack(pending_metrics).cpu().tolist()
            for loss_value, accuracy in rows:
                if not math.isfinite(loss_value):
                    raise FloatingPointError("non-finite behavioral cloning loss")
                losses.append(loss_value)
                accuracies.append(accuracy)
            pending_metrics.clear()
    order = list(range(len(records)))
    stop = False
    for _ in range(epochs):
        rng.shuffle(order)
        if bucket_actions:
            buckets = {}
            for index in order:
                count = len(records[index]["observation"]["legal_actions"])
                buckets.setdefault((count - 1).bit_length(), []).append(index)
            batches = [bucket[offset:offset + batch_size] for bucket in buckets.values()
                       for offset in range(0, len(bucket), batch_size)]
            rng.shuffle(batches)
        else:
            batches = [order[offset:offset + batch_size] for offset in range(0, len(order), batch_size)]
        for indices in batches:
            if ((max_seconds is not None and time.monotonic() - started >= max_seconds)
                    or (max_steps is not None and steps >= max_steps)):
                stop = True
                break
            chosen = [records[i] for i in indices]
            batch = collate([record["observation"] for record in chosen], device)
            logits, _, mask = forward(model, batch)
            labels = torch.tensor([labels_by_record[index] for index in indices], device=device)
            if phase_weights:
                per_row = torch.nn.functional.cross_entropy(logits.masked_fill(~mask, -1e9), labels, reduction="none")
                weights = torch.tensor([phase_weights.get(record["observation"].get("phase", ""), 1.)
                                        for record in chosen], device=device)
                loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-6)
            else:
                # Labels/menu support were verified on CPU before training.
                # Direct CE avoids Categorical's per-batch GPU validation sync.
                loss = torch.nn.functional.cross_entropy(logits.masked_fill(~mask, -1e9), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            steps += 1
            examples += len(chosen)
            pending_metrics.append(torch.stack((loss.detach(),
                (logits.masked_fill(~mask, -1e9).argmax(-1) == labels).float().mean().detach())))
            if len(pending_metrics) >= metric_interval:
                flush_metrics()
        if stop:
            break
    flush_metrics()
    return {"steps": steps, "examples": examples, "loss": sum(losses) / max(1, len(losses)),
            "accuracy": sum(accuracies) / max(1, len(accuracies)),
            "elapsed_seconds": time.monotonic() - started, "metric_interval": metric_interval,
            "bucket_actions": bucket_actions, "optimizer": optimizer}


@torch.no_grad()
def evaluate_bc(model: torch.nn.Module, records: Sequence[dict[str, Any]], *,
                collate: Callable[..., Any], forward: Callable[..., Any],
                device: str = "cpu", batch_size: int = 256) -> dict[str, Any]:
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    phase_counts: dict[str, list[int]] = {}
    menu_counts: dict[str, list[int]] = {}
    phase_menu_counts: dict[str, dict[str, list[int]]] = {}
    for offset in range(0, len(records), batch_size):
        chosen = records[offset:offset + batch_size]
        logits, _, mask = forward(model, collate([r["observation"] for r in chosen], device))
        labels = torch.tensor([_label_position(r) for r in chosen], device=device)
        from .losses import masked_distribution
        losses = -masked_distribution(logits, mask).log_prob(labels)
        matches = (logits.masked_fill(~mask, -1e9).argmax(-1) == labels).cpu().tolist()
        loss_sum += float(losses.sum().cpu())
        correct += sum(matches)
        count += len(chosen)
        for record, match in zip(chosen, matches):
            phase = str(record["observation"].get("phase", "unknown"))
            totals = phase_counts.setdefault(phase, [0, 0])
            totals[0] += int(match)
            totals[1] += 1
            menu = "forced" if len(record["observation"]["legal_actions"]) == 1 else "choice"
            menu_total = menu_counts.setdefault(menu, [0, 0])
            menu_total[0] += int(match)
            menu_total[1] += 1
            phase_menu_total = phase_menu_counts.setdefault(phase, {}).setdefault(menu, [0, 0])
            phase_menu_total[0] += int(match)
            phase_menu_total[1] += 1
    return {"examples": count, "loss": loss_sum / max(1, count),
            "accuracy": correct / max(1, count),
            "phase_accuracy": {k: v[0] / v[1] for k, v in phase_counts.items()},
            "phase_counts": {k: v[1] for k, v in phase_counts.items()},
            "menu_accuracy": {k: v[0] / v[1] for k, v in menu_counts.items()},
            "menu_counts": {k: v[1] for k, v in menu_counts.items()},
            "phase_menu_accuracy": {phase: {menu: values[0] / values[1] for menu, values in counts.items()}
                                    for phase, counts in phase_menu_counts.items()},
            "phase_menu_counts": {phase: {menu: values[1] for menu, values in counts.items()}
                                  for phase, counts in phase_menu_counts.items()}}
