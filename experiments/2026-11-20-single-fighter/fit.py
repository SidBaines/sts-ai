"""Distil role-specific preferences into one ordinary combat head and export a two-head player."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from sts_policy.model_agent import CompositeActorCritic
from sts_rl.common import JsonlLog, load_policy, save_policy
from sts_rl.fresh_study import atomic_json
from sts_rl.paired_endpoints import digest
from study import CFG, CHAMP, OUT, ROLES, check_parent, note


class Menus:
    def __init__(self, directory, split):
        arrays = {key: [] for key in ("states", "actions", "lengths", "scores", "seeds", "chosen")}
        roles = []
        for path in sorted(directory.glob(split + "-*.npz")):
            with np.load(path) as data:
                for i, role in enumerate(ROLES):
                    for key in arrays:
                        arrays[key].append(data[f"{role}_{key}"])
                    roles.append(np.full(len(arrays["lengths"][-1]), i, dtype=np.int8))
        for key in arrays:
            setattr(self, key, np.concatenate(arrays[key]))
            arrays[key].clear()
        self.role = np.concatenate(roles)
        self.groups = [np.flatnonzero(self.role == i) for i in range(2)]
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths, dtype=np.int64)))
        if (self.offsets[-1] != len(self.actions) or len(self.actions) != len(self.scores)
                or len(self.states) != len(self.role) or np.any(self.lengths < 2)
                or np.any(self.chosen < 0) or np.any(self.chosen >= self.lengths)):
            raise ValueError("Encoded combat menus and legal teacher scores disagree")

    def indices(self, ids):
        n = self.lengths[ids]
        positions = np.arange((int(max(n)) + 3) // 4 * 4)
        mask = positions < n[:, None]
        flat = self.offsets[ids, None] + np.minimum(positions, n[:, None] - 1)
        return flat, mask

    def batch(self, ids, device):
        flat, mask = self.indices(ids)
        arrays = (self.states[ids], self.actions[flat], mask, self.scores[flat], self.role[ids])
        return tuple(torch.as_tensor(a, device=device) for a in arrays)


def calibrate(data, seed):
    """Give each teacher 80% mean probability on its top-scoring legal choices."""
    temperatures = []
    rng = np.random.default_rng(seed)
    for group in data.groups:
        selected = rng.choice(group, min(8192, len(group)), replace=False)
        flat, mask = data.indices(selected)
        scores = np.where(mask, data.scores[flat], -np.inf).astype(np.float64)
        scores -= scores.max(axis=1, keepdims=True)
        best = scores >= -1e-7
        def top_mass(t):
            w = np.exp(scores/t)
            return np.mean((w * best).sum(axis=1) / w.sum(axis=1))
        lo, hi = 1e-6, 1000.
        if not top_mass(hi) <= .8 <= top_mass(lo):
            raise ValueError("Training choices do not bracket the calibration target")
        for _ in range(36):
            mid = (lo*hi)**.5
            if top_mass(mid) > .8:
                lo = mid
            else:
                hi = mid
        temperatures.append((lo*hi)**.5)
    return temperatures


def losses(head, batch, temperatures):
    state, actions, mask, target, role = batch
    prediction = head.actor_logits(state, actions, mask)
    def log_probs(scores, scale):
        centered = scores - scores.masked_fill(~mask, -1e9).amax(dim=1, keepdim=True)
        return torch.log_softmax((centered/scale).masked_fill(~mask, -1e6), dim=1)
    log_teacher = log_probs(target, temperatures[role.long(), None])
    log_student = log_probs(prediction, temperatures[0])
    kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=1)
    top = prediction.masked_fill(~mask, -1e9).argmax(dim=1, keepdim=True)
    regret = target.masked_fill(~mask, -1e9).amax(dim=1) - target.gather(1, top).flatten()
    return kl, regret


@torch.no_grad()
def validate(head, data, temps, device):
    stats = {}
    for name, ids in zip(ROLES, data.groups):
        ids = ids[np.argsort(data.lengths[ids], kind="stable")]
        sums = torch.zeros(2, device=device)
        for begin in range(0, len(ids), 256):
            batch = data.batch(ids[begin:begin+256], device)
            kl, regret = losses(head, batch, temps)
            sums += torch.stack([kl.sum(), (regret <= 1e-6).float().sum()])
        loss, agreement = (sums/len(ids)).cpu().tolist()
        stats[name] = dict(choices=len(ids), kl=loss, top_agreement=agreement)
    return stats


def export(head, original, template, out, best_step, fraction):
    head.cpu()
    player = CompositeActorCritic(head, original.ooc)
    meta = dict(stage="single_fighter_distillation", config=dict(checkpoint=str(CHAMP), boss_share=fraction))
    for filename, model, extra in (("head.pt", head, {"kind": "candidate", "phase": "combat"}),
                                   ("player.pt", player, {})):
        clean = {k:template[k] for k in ("format_version",)}
        clean.update(model_config=model.metadata(), metadata={"encoder_version": template["metadata"]["encoder_version"]})
        save_policy(out / filename, model, clean, metadata={**meta, **extra}, step=best_step,
                    trainer_state=template["trainer_state"] if filename == "player.pt" else {})
    saved, payload = load_policy(out / "player.pt")
    single, _ = load_policy(out / "head.pt")
    if type(saved) is not CompositeActorCritic or set(saved._modules) != {"combat", "ooc"}:
        raise ValueError("Exported player must use exactly one fighter")
    for name, wanted in (("ooc", original.ooc), ("combat", single)):
        actual = getattr(saved, name).state_dict()
        if any(not torch.equal(v, actual[k]) for k,v in wanted.state_dict().items()):
            raise ValueError(f"Exported {name} weights differ")
    return dict(player_sha256=digest(out/"player.pt"), head_sha256=digest(out/"head.pt"),
                architecture=payload["model_config"])


def fit(fraction, out):
    check_parent()
    if fraction not in CFG["boss_shares"]:
        raise ValueError("Unexpected mixture")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the matched fits on this Mac")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(CFG["seed"])
    directory = OUT / "demonstrations"
    summary = json.loads((directory / "summary.json").read_bytes())
    train, val = Menus(directory, "train"), Menus(directory, "validation")
    for split, data in (("train", train), ("validation", val)):
        if [len(g) for g in data.groups] != [summary[split][r] for r in ROLES]:
            raise ValueError("Shard sample counts differ from the collection summary")
    if np.intersect1d(train.seeds, val.seeds).size:
        raise ValueError("Training and imitation validation share a world")
    device = torch.device("mps")
    scales = calibrate(train, CFG["seed"])
    temps = torch.tensor(scales, dtype=torch.float32, device=device)
    original, template = load_policy(CHAMP)
    head = copy.deepcopy(original.combat).to(device)
    for name, p in head.named_parameters():
        p.requires_grad_(not name.startswith("critic."))
    parameters = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=CFG["lr"], weight_decay=0)
    rng = np.random.default_rng(CFG["seed"])
    log = JsonlLog(out/"progress.jsonl")
    initial = validate(head, val, temps, device)
    log.emit(dict(event="initial", temperatures=scales, validation=initial))
    note("fitting", f"Fitting the single fighter with {fraction:.0%} boss examples in each update.")
    best, best_step, best_state, best_metrics = float("inf"), 0, None, None
    boss_count = round(CFG["batch"] * fraction)
    started = time.monotonic()
    for step in range(1, CFG["steps"]+1):
        ids = np.concatenate([rng.choice(train.groups[0], CFG["batch"]-boss_count),
                              rng.choice(train.groups[1], boss_count)])
        loss = losses(head, train.batch(ids, device), temps)[0].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        opt.step()
        if step % CFG["check_every"]:
            continue
        metrics = validate(head, val, temps, device)
        score = sum(m["kl"] for m in metrics.values()) / 2
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation loss")
        if score < best:
            best, best_step, best_metrics = score, step, metrics
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        log.emit(dict(event="validation", step=step, total=CFG["steps"], validation=metrics,
                      best_step=best_step, seconds=time.monotonic()-started))
        note("fitting", f"Single fighter with {fraction:.0%} boss examples has finished {step:,} updates "
             f"of at most {CFG['steps']:,}.", log=False)
        if step >= CFG["min_steps"] and step-best_step >= CFG["patience"]*CFG["check_every"]:
            break
    head.load_state_dict(best_state)
    saved = export(head, original, template, out, best_step, fraction)
    atomic_json(out/"summary.json", dict(boss_share=fraction, temperatures=scales, initial=initial,
                validation=best_metrics, best_step=best_step, actual_steps=step, **saved))
    note("fit complete", f"Finished the {fraction:.0%} boss mix. Its validation-selected fighter was saved at update {best_step:,}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boss-share", required=True, type=float)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    fit(a.boss_share, a.out)
