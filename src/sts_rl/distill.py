"""Distil fight-panel teacher labels into the OOC head, with behaviour rehearsal.

Loss = CE(student, soft teacher target) on labelled OOC decisions
     + beta * KL(original || student) on rehearsal OOC observations.
Combat head is frozen (bit-identical). Splits by world seed for validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch

from sts_policy.model_agent import cached_public_collator, forward_policy

from .common import ORIGINAL_CHECKPOINT, JsonlLog, load_policy, save_policy
from .recording import begin_run


@dataclass(frozen=True)
class DistillConfig:
    records: str
    out: str
    checkpoint: str = ORIGINAL_CHECKPOINT
    tau: float = 0.05
    beta: float = 1.0
    lr: float = 1e-4
    batch: int = 256
    epochs: int = 4
    val_fraction: float = 0.1
    types: str = "card,rest,rest_select"
    head: str = "ooc"              # which head to train: ooc or combat (the other stays frozen)
    seed: int = 0
    threads: int = 4


def load_records(directory: str | Path) -> list[dict[str, Any]]:
    rows = []
    for shard in sorted(Path(directory).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def soft_target(record: dict[str, Any], tau: float) -> list[float] | None:
    """Distribution over legal positions; options aggregated by their first action (max score)."""
    legal = record["obs"]["legal_actions"]
    position = {int(a["index"]): i for i, a in enumerate(legal)}
    target = [0.0] * len(legal)
    if "votes" in record:  # search votes over hidden draw orders
        total = sum(record["votes"].values())
        for index, count in record["votes"].items():
            if int(index) not in position:
                return None
            target[position[int(index)]] = count / total
        return target
    if "scores" not in record:  # hard label (e.g. smith's card choice)
        if record["label_index"] not in position:
            return None
        target[position[record["label_index"]]] = 1.0
        return target
    best: dict[int, float] = {}
    for item in record["scores"]:
        if item["score"] is None:
            continue
        first = int(item["option"][0])
        best[first] = max(best.get(first, -1e9), item["score"])
    if len(best) < 2 or any(k not in position for k in best):
        return None
    top = max(best.values())
    weights = {k: math.exp((v - top) / max(tau, 1e-6)) for k, v in best.items()}
    total = sum(weights.values())
    for k, w in weights.items():
        target[position[k]] = w / total
    return target


def distill(cfg: DistillConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    if (out/'distilled.pt').exists() or (out/'log.jsonl').exists():
        raise ValueError('Distillation output contains history; choose a fresh output')
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "log.jsonl")
    run_record = begin_run(out, asdict(cfg), cfg.checkpoint, inputs=[cfg.records])
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    student, template = load_policy(cfg.checkpoint)
    reference, _ = load_policy(cfg.checkpoint)
    reference.eval()
    frozen_head = "combat" if cfg.head == "ooc" else "ooc"
    for name, p in student.named_parameters():
        p.requires_grad_(name.startswith(cfg.head + "."))
    combat_before = {k: v.clone() for k, v in student.state_dict().items() if k.startswith(frozen_head + ".")}
    types = set(cfg.types.split(","))
    rows = load_records(cfg.records)
    labelled, rehearsal = [], []
    for r in rows:
        if r["type"] == "rehearsal":
            rehearsal.append(r)
        elif r["type"] in types:
            t = soft_target(r, cfg.tau)
            if t is not None:
                r["target"] = t
                labelled.append(r)
    seeds = sorted({r["seed"] for r in rows})
    rng.shuffle(seeds)
    val_seeds = set(seeds[: max(1, int(len(seeds) * cfg.val_fraction))])
    split = lambda xs: ([x for x in xs if x["seed"] not in val_seeds], [x for x in xs if x["seed"] in val_seeds])
    lab_train, lab_val = split(labelled)
    reh_train, reh_val = split(rehearsal)
    log.emit({"event": "start", "config": asdict(cfg), 'run_record': run_record, "labelled": len(labelled), "rehearsal": len(rehearsal),
              "train": [len(lab_train), len(reh_train)], "val": [len(lab_val), len(reh_val)],
              "by_type": {t: sum(r["type"] == t for r in labelled) for t in types}})
    all_obs = [r["obs"] for r in labelled + rehearsal]
    collate = cached_public_collator(all_obs)

    def batch_tensors(records: Sequence[dict[str, Any]]):
        state, actions, mask = collate([r["obs"] for r in records], "cpu")
        return state, actions, mask

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        student.eval()
        res: dict[str, float] = {}
        if lab_val:
            s, a, m = batch_tensors(lab_val)
            logits, _, _ = forward_policy(student, (s, a, m))
            target = torch.zeros_like(logits)
            for i, r in enumerate(lab_val):
                target[i, : len(r["target"])] = torch.tensor(r["target"])
            res["teacher_top1"] = float((logits.argmax(-1) == target.argmax(-1)).float().mean())
            ref_logits, _, _ = forward_policy(reference, (s, a, m))
            res["original_teacher_top1"] = float((ref_logits.argmax(-1) == target.argmax(-1)).float().mean())
        if reh_val:
            s, a, m = batch_tensors(reh_val)
            logits, _, _ = forward_policy(student, (s, a, m))
            ref_logits, _, _ = forward_policy(reference, (s, a, m))
            res["rehearsal_argmax_agree"] = float((logits.argmax(-1) == ref_logits.argmax(-1)).float().mean())
            lp, lq = torch.log_softmax(ref_logits, -1), torch.log_softmax(logits, -1)
            res["rehearsal_kl"] = float((lp.exp() * (lp - lq)).masked_fill(~m, 0).sum(-1).mean())
        student.train()
        return res

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr)
    log.emit({"event": "eval", "epoch": 0, **evaluate()})
    steps = 0
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        rng.shuffle(lab_train)
        rng.shuffle(reh_train)
        n_batches = max(1, math.ceil(len(lab_train) / cfg.batch))
        reh_per = max(1, len(reh_train) // n_batches) if reh_train else 0
        for b in range(n_batches):
            chunk = lab_train[b * cfg.batch:(b + 1) * cfg.batch]
            loss = torch.zeros(())
            if chunk:
                s, a, m = batch_tensors(chunk)
                logits, _, _ = forward_policy(student, (s, a, m))
                target = torch.zeros_like(logits)
                for i, r in enumerate(chunk):
                    target[i, : len(r["target"])] = torch.tensor(r["target"])
                logp = torch.log_softmax(logits, -1).masked_fill(~m, 0)
                loss = loss - (target * logp).sum(-1).mean()
            if reh_per:
                rchunk = reh_train[b * reh_per:(b + 1) * reh_per]
                if rchunk:
                    s, a, m = batch_tensors(rchunk)
                    logits, _, _ = forward_policy(student, (s, a, m))
                    with torch.no_grad():
                        ref_logits, _, _ = forward_policy(reference, (s, a, m))
                    lp, lq = torch.log_softmax(ref_logits, -1), torch.log_softmax(logits, -1)
                    loss = loss + cfg.beta * (lp.exp() * (lp - lq)).masked_fill(~m, 0).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            steps += 1
        log.emit({"event": "eval", "epoch": epoch, "steps": steps, "loss": float(loss.detach()),
                  "seconds": time.time() - started, **evaluate()})
    state = student.state_dict()
    if not all(torch.equal(v, state[k]) for k, v in combat_before.items()):
        raise RuntimeError(f"{frozen_head} head changed during {cfg.head} distillation")
    path = out / "distilled.pt"
    save_policy(path, student, template, metadata={"stage": "sts_rl_distill", "config": asdict(cfg),
                                                 'run_record': run_record}, optimizer=optimizer, step=steps, rng=rng)
    log.emit({"event": "done", "checkpoint": str(path)})
    return {"checkpoint": str(path)}
