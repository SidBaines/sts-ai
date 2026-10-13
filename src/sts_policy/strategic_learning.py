"""Choice-only PPO, separate value fitting and measured original-policy retention."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Any, Sequence

import numpy as np
import torch

from . import entities
from .entity_model import ResidualOOCPolicy
from .strategic_collection import StrategicTransition

TRAINER_VERSION = "protected_strategic_ppo_v3_v4"


@dataclass(frozen=True)
class StrategicConfig:
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    batch_size: int = 256
    epochs: int = 1
    clip_ratio: float = .1
    entropy_weight: float = .001
    rehearsal_weight: float = .1
    max_actor_norm: float = .5
    max_critic_norm: float = 1.
    post_update_kl_stop: float = .01
    actor_act_balanced: bool = False

    def __post_init__(self):
        if any(not math.isfinite(float(value)) for value in asdict(self).values()):
            raise ValueError("protected PPO settings must be finite")
        if min(self.actor_lr, self.critic_lr, self.batch_size, self.epochs, self.max_actor_norm, self.max_critic_norm) <= 0:
            raise ValueError("positive learning rates, batch counts and gradient bounds required")
        if not 0 < self.clip_ratio < 1 or min(self.entropy_weight, self.rehearsal_weight, self.post_update_kl_stop) < 0:
            raise ValueError("invalid protected PPO settings")


def optimizers(model: ResidualOOCPolicy, config: StrategicConfig):
    # Exclude immutable prior completely. Zero weight decay also avoids changing
    # unused parameters in either separate optimizer after zero-gradient steps.
    return (torch.optim.Adam(model.actor_parameters(), lr=config.actor_lr),
            torch.optim.Adam(model.critic_parameters(), lr=config.critic_lr))


def actor_loss_weights(rows, indices, balanced):
    weights=np.zeros(len(rows),np.float32)
    counts=Counter(int(rows[i].observation['state']['act']) for i in indices)
    for i in indices:
        act=int(rows[i].observation['state']['act'])
        weights[i]=len(indices)/(len(counts)*counts[act]) if balanced else 1.
    return weights


class OptimizerBundle:
    """Explicit checkpoint payload for the two disjoint optimizer histories."""
    def __init__(self, actor, critic):
        self.actor, self.critic = actor, critic

    def state_dict(self):
        return {"version": "disjoint_strategic_optimizers_v1", "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()}

    def load_state_dict(self, state):
        if state.get("version") != "disjoint_strategic_optimizers_v1":
            raise ValueError("incompatible strategic optimizer history")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


def _encoded(row: StrategicTransition) -> entities.EncodedEntities:
    if row.encoded is None:
        row.encoded = entities.encode_observation(row.observation)
    return row.encoded


def _batches(indices: Sequence[int], encoded: Sequence[entities.EncodedEntities], batch_size: int, rng: random.Random):
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in indices:
        observation = encoded[i]
        key = ((max(1, len(observation.action_indices))-1).bit_length(),
               (max(1, len(observation.entity_ids))-1).bit_length())
        buckets.setdefault(key, []).append(i)
    batches = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        batches.extend(bucket[start:start+batch_size] for start in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    numbers = np.asarray(values, dtype=np.float64)
    return {"mean": float(numbers.mean()) if len(numbers) else None,
            "p95": float(np.quantile(numbers, .95)) if len(numbers) else None,
            "max": float(numbers.max()) if len(numbers) else None}


@torch.no_grad()
def drift(model: ResidualOOCPolicy, encoded: Sequence[entities.EncodedEntities], *,
          device: str = "cpu", batch_size: int = 256,
          behavior: Sequence[np.ndarray | None] | None = None) -> dict[str, Any]:
    model.to(device).eval()
    original_kl, tv, changed, entropy, local_kl = [], [], [], [], []
    for start in range(0, len(encoded), batch_size):
        chosen = encoded[start:start+batch_size]
        batch = entities.collate_observations(chosen, device)
        logits, _ = model(*batch)
        prior_logits, _ = model.prior(batch[0], batch[1], batch[11])
        prior_logits = prior_logits.gather(1, batch[10]).masked_fill(~batch[2], -1e9)
        # MPS flushes float32 subnormal probabilities to zero. Computing
        # p * log(p) there can yield 0 * -inf => NaN even when public logits and
        # the actual distributions are finite. Diagnostics use CPU float64;
        # gradients remain on the chosen training device.
        current_logits = logits.detach().cpu().numpy().astype(np.float64)
        original_logits = prior_logits.detach().cpu().numpy().astype(np.float64)
        for i, observation in enumerate(chosen):
            size = len(observation.action_indices)
            if not size:
                raise ValueError("drift requires legal choice observations")
            def distribution_from_logits(numbers):
                values = numbers[:size]
                if not np.isfinite(values).all():
                    raise FloatingPointError("non-finite policy diagnostic logits")
                shifted = values-values.max()
                log = shifted-np.log(np.exp(shifted).sum())
                return np.exp(log), log
            current, current_log = distribution_from_logits(current_logits[i])
            original, original_log = distribution_from_logits(original_logits[i])
            original_kl.append(max(0., float(np.sum(original*(original_log-current_log)))))
            tv.append(float(.5*np.abs(original-current).sum()))
            changed.append(int(original_logits[i, :size].argmax() != current_logits[i, :size].argmax()))
            entropy.append(float(-np.sum(current*current_log)/max(1e-6, math.log(size))))
            reference = behavior[start+i] if behavior is not None else None
            if reference is not None:
                reference = np.asarray(reference, np.float64)
                if len(reference) != size or not np.isfinite(reference).all() or (reference < 0).any() or reference.sum() <= 0:
                    raise FloatingPointError("invalid behavior distribution in final KL diagnostic")
                reference = reference/reference.sum()
                positive = reference > 0
                value = float(np.sum(reference[positive]*(np.log(reference[positive])-current_log[positive])))
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite final behavior KL diagnostic")
                local_kl.append(max(0., value))
    return {"choices": len(encoded), "original_kl": _summary(original_kl), "original_tv": _summary(tv),
            "changed_greedy_fraction": sum(changed)/max(1, len(changed)), "normalized_entropy": _summary(entropy),
            "behavior_kl_post_update": _summary(local_kl)}


@torch.no_grad()
def calibration(model: ResidualOOCPolicy, rows: Sequence[StrategicTransition], *, device="cpu", batch_size=256):
    completed = [row for row in rows if row.completed_return is not None]
    predictions = []
    model.to(device).eval()
    for start in range(0, len(completed), batch_size):
        _, values = model(*entities.collate_observations([_encoded(row) for row in completed[start:start+batch_size]], device))
        predictions.extend(values.cpu().tolist())
    groups = {}
    for row, prediction in zip(completed, predictions):
        key = f"{row.observation['phase']}:act{row.observation['state'].get('act')}"
        groups.setdefault(key, []).append((prediction, row.completed_return))
    def scores(pairs):
        if not pairs:
            return {"rows": 0}
        array = np.asarray(pairs)
        errors = array[:, 0]-array[:, 1]
        variance = float(np.var(array[:, 1]))
        return {"rows": len(pairs), "bias": float(errors.mean()), "mse": float(np.mean(errors*errors)),
                "explained_variance": 1-float(np.var(errors))/variance if variance > 1e-12 else None,
                "return_mean": float(array[:, 1].mean()), "prediction_mean": float(array[:, 0].mean())}
    return {"overall": scores(list(zip(predictions, [row.completed_return for row in completed]))),
            "by_phase_act": {key: scores(pairs) for key, pairs in groups.items()},
            "note": "Completed realized returns; correlated within world, not exact expected-value calibration."}


def update(model: ResidualOOCPolicy, rows: Sequence[StrategicTransition], *,
           rehearsal: Sequence[entities.EncodedEntities], config: StrategicConfig,
           actor_optimizer: torch.optim.Optimizer, critic_optimizer: torch.optim.Optimizer,
           seed: int, device: str = "mps", critic_only: bool = False,
           completed_targets: bool = False) -> dict[str, Any]:
    if not rows:
        raise ValueError("protected update requires valid transitions")
    started = time.monotonic()
    encoded = [_encoded(row) for row in rows]
    actor_indices = [i for i, row in enumerate(rows) if row.actor_choice]
    critic_indices = [i for i, row in enumerate(rows) if not completed_targets or row.completed_return is not None]
    if not critic_indices:
        raise ValueError("no completed targets available for critic calibration")
    advantages = np.asarray([row.advantage for row in rows], np.float32)
    chosen_advantages = advantages[actor_indices]
    mean, std = (float(chosen_advantages.mean()), float(chosen_advantages.std())) if actor_indices else (0., 1.)
    advantages = (advantages-mean)/max(std, 1e-8)
    actor_weights=actor_loss_weights(rows,actor_indices,config.actor_act_balanced)
    actor_by_act,weighted_by_act=Counter(),Counter()
    rng = random.Random(seed)
    actor_steps, critic_steps, actor_exposures, critic_exposures, rehearsal_exposures = 0, 0, 0, 0, 0
    unique_actor, unique_critic = set(), set()
    actor_norms, critic_norms, losses, ratios, clip_fractions = [], [], [], [], []
    actor_after_norms, critic_after_norms = [], []
    stop_reason = None
    before_calibration = calibration(model, rows, device=device, batch_size=config.batch_size)
    model.to(device).train()
    for epoch in range(config.epochs):
        if not critic_only:
            for indices in _batches(actor_indices, encoded, config.batch_size, rng):
                batch = entities.collate_observations([encoded[i] for i in indices], device)
                logits, _ = model(*batch)
                log_probs = torch.log_softmax(logits, -1)
                positions = torch.tensor([rows[i].action_position for i in indices], device=device)
                current = log_probs.gather(1, positions[:, None]).squeeze(-1)
                old = torch.tensor([rows[i].behavior_log_prob for i in indices], device=device)
                ratio = (current-old).exp()
                adv = torch.as_tensor(advantages[indices], device=device)
                weights=torch.as_tensor(actor_weights[indices],device=device)
                policy_loss = -(torch.minimum(ratio*adv, ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio)*adv)*weights).mean()
                probabilities = log_probs.exp()
                entropy = (-((probabilities*log_probs).sum(-1))*weights).mean()
                loss = policy_loss-config.entropy_weight*entropy
                if rehearsal and config.rehearsal_weight:
                    picked = rng.sample(range(len(rehearsal)), min(config.batch_size, len(rehearsal)))
                    fixed = entities.collate_observations([rehearsal[i] for i in picked], device)
                    new_logits, _ = model(*fixed)
                    with torch.no_grad():
                        prior, _ = model.prior(fixed[0], fixed[1], fixed[11])
                        old_log = torch.log_softmax(prior.gather(1, fixed[10]).masked_fill(~fixed[2], -1e9), -1)
                    anchor = (old_log.exp() * (old_log-torch.log_softmax(new_logits, -1))).sum(-1).mean()
                    loss = loss+config.rehearsal_weight*anchor
                    rehearsal_exposures += len(picked)
                actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), config.max_actor_norm, error_if_nonfinite=True)
                after = torch.nn.utils.get_total_norm([p.grad for p in model.actor_parameters() if p.grad is not None], error_if_nonfinite=True)
                actor_optimizer.step()
                before_value, after_value = torch.stack((norm, after)).cpu().tolist()
                actor_norms.append(before_value)
                actor_after_norms.append(after_value)
                losses.append(float(policy_loss.detach().cpu()))
                ratios.append(float(ratio.detach().mean().cpu()))
                clip_fractions.append(float(((ratio.detach()-1).abs()>config.clip_ratio).float().mean().cpu()))
                actor_steps += 1
                actor_exposures += len(indices)
                unique_actor.update(indices)
                for i in indices:
                    act=int(rows[i].observation['state']['act'])
                    actor_by_act[act]+=1;weighted_by_act[act]+=float(actor_weights[i])
            model.eval()
            choice_encoded = [encoded[i] for i in actor_indices]
            current_drift = drift(model, choice_encoded, device=device, batch_size=config.batch_size,
                                  behavior=[rows[i].behavior_distribution for i in actor_indices])
            if (current_drift["behavior_kl_post_update"]["mean"] or 0) > config.post_update_kl_stop:
                stop_reason = "post_epoch_behavior_kl"
            model.train()
        for indices in _batches(critic_indices, encoded, config.batch_size, rng):
            batch = entities.collate_observations([encoded[i] for i in indices], device)
            _, values = model(*batch)
            targets = torch.tensor([rows[i].completed_return if completed_targets else rows[i].returns for i in indices], device=device)
            loss = .5*(values-targets).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.critic_parameters(), config.max_critic_norm, error_if_nonfinite=True)
            after = torch.nn.utils.get_total_norm([p.grad for p in model.critic_parameters() if p.grad is not None], error_if_nonfinite=True)
            critic_optimizer.step()
            before_value, after_value = torch.stack((norm, after)).cpu().tolist()
            critic_norms.append(before_value)
            critic_after_norms.append(after_value)
            critic_steps += 1
            critic_exposures += len(indices)
            unique_critic.update(indices)
        if stop_reason:
            break
    model.eval()
    choice_encoded = [encoded[i] for i in actor_indices]
    output = {"trainer_version": TRAINER_VERSION, "config": asdict(config), "critic_only": critic_only,
        "completed_targets": completed_targets, "actor_choice_rows": len(actor_indices), "critic_rows": len(critic_indices),
        "actor_steps": actor_steps, "critic_steps": critic_steps, "actor_exposures": actor_exposures,
        "critic_exposures": critic_exposures, "rehearsal_exposures": rehearsal_exposures,
        "unique_actor_rows_used": len(unique_actor), "unique_critic_rows_used": len(unique_critic),
        "actual_actor_exposures_by_act":dict(actor_by_act),"actual_actor_loss_mass_by_act":dict(weighted_by_act),
        "actor_grad_norm_before_clip": _summary(actor_norms), "critic_grad_norm_before_clip": _summary(critic_norms),
        "actor_grad_norm_after_clip": _summary(actor_after_norms),
        "critic_grad_norm_after_clip": _summary(critic_after_norms),
        "gradient_interference": "zero by disjoint actor/critic parameter sets; original prior excluded",
        "policy_loss": _summary(losses), "ratio_mean": _summary(ratios), "clip_fraction": _summary(clip_fractions),
        "advantage_before_normalization": {"mean": mean, "std": std}, "stop_reason": stop_reason,
        "buffer_drift": drift(model, choice_encoded, device=device, batch_size=config.batch_size,
                              behavior=[rows[i].behavior_distribution for i in actor_indices]),
        "rehearsal_drift": drift(model, rehearsal, device=device, batch_size=config.batch_size),
        "calibration_before": before_calibration, "calibration_after": calibration(model, rows, device=device, batch_size=config.batch_size),
        "rows_by_phase_act_choice": dict(Counter(f"{r.observation['phase']}:act{r.observation['state'].get('act')}:{'choice' if r.actor_choice else 'critic_only'}" for r in rows)),
        "elapsed_seconds": time.monotonic()-started}
    if not math.isfinite(output["elapsed_seconds"]):
        raise FloatingPointError("invalid update duration")
    return output
