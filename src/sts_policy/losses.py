"""Masked categorical BC and genuine clipped PPO objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


def masked_distribution(
    logits: Tensor, mask: Tensor
) -> torch.distributions.Categorical:
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("logits and Boolean mask must share shape")
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError("every decision needs at least one legal candidate")
    if not bool(torch.isfinite(logits[mask]).all().item()):
        raise ValueError("legal action logits must be finite")
    # A finite sentinel avoids 0 * -inf entropy gradients on MPS.
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def masked_kl_divergence(
    reference_logits: Tensor, logits: Tensor, mask: Tensor
) -> Tensor:
    """KL(reference || current) without rounded-probability support checks.

    A finite legal logit has positive mathematical probability even when its
    float32 softmax rounds to zero. Categorical KL can incorrectly return inf
    across that boundary. Work directly with normalized log probabilities and
    widen CPU/CUDA arithmetic; MPS uses float32 because it lacks float64.
    Padded entries are zeroed *before* subtraction, avoiding 0 * (inf - inf).
    """
    if (
        reference_logits.shape != logits.shape
        or logits.shape != mask.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("reference/current logits and Boolean mask must share shape")
    if reference_logits.device != logits.device or logits.device != mask.device:
        raise ValueError("KL inputs must share a device")
    if not reference_logits.is_floating_point() or not logits.is_floating_point():
        raise ValueError("KL logits must be floating point")
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError("every KL decision needs a legal candidate")
    if not bool(torch.isfinite(reference_logits[mask]).all().item()) or not bool(
        torch.isfinite(logits[mask]).all().item()
    ):
        raise ValueError("legal KL logits must be finite")
    dtype = torch.float32 if logits.device.type == "mps" else torch.float64
    reference = torch.log_softmax(
        reference_logits.to(dtype).masked_fill(~mask, -torch.inf), -1
    ).masked_fill(~mask, 0.0)
    current = torch.log_softmax(
        logits.to(dtype).masked_fill(~mask, -torch.inf), -1
    ).masked_fill(~mask, 0.0)
    divergence = (
        (reference.exp() * (reference - current)).masked_fill(~mask, 0.0).sum(-1)
    )
    if not bool(torch.isfinite(divergence).all().item()):
        raise FloatingPointError("nonfinite masked log-space KL")
    return divergence.clamp_min(0.0)  # Roundoff can produce tiny negative KL.


def behavioral_cloning_loss(logits: Tensor, mask: Tensor, labels: Tensor) -> Tensor:
    if not bool(mask.gather(1, labels[:, None]).all().item()):
        raise ValueError("teacher labels refer to a padded/illegal candidate")
    return -masked_distribution(logits, mask).log_prob(labels).mean()


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    target_kl: float = 0.03
    kl_guard_before_update: bool = False


def generalized_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    next_values: Sequence[float],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Bootstrap time limits, but never propagate GAE across episode resets.

    next_values is measured on the actual post-step observation, never the reset
    observation. A genuine terminal sets its bootstrap to zero. Truncation only
    ends the recurrence, retaining its value bootstrap.
    """
    lengths = {len(x) for x in (rewards, values, terminated, truncated, next_values)}
    if len(lengths) != 1:
        raise ValueError("trajectory arrays must share length")
    advantages = [0.0] * len(rewards)
    carry = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 0.0 if terminated[index] else next_values[index]
        delta = rewards[index] + gamma * bootstrap - values[index]
        continues = not (terminated[index] or truncated[index])
        carry = delta + gamma * gae_lambda * carry * continues
        advantages[index] = carry
    return advantages, [a + v for a, v in zip(advantages, values)]


def ppo_objective(
    logits: Tensor,
    mask: Tensor,
    values: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    config: PPOConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    distribution = masked_distribution(logits, mask)
    log_probs = distribution.log_prob(actions)
    log_ratio = log_probs - old_log_probs.detach()
    ratios = log_ratio.exp()
    unclipped = ratios * advantages
    clipped = ratios.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (values - returns).square().mean()
    entropy = distribution.entropy().mean()
    loss = (
        policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy
    )
    metrics = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "approx_kl": ((ratios - 1) - log_ratio).mean().detach(),
        "clip_fraction": ((ratios - 1).abs() > config.clip_ratio)
        .float()
        .mean()
        .detach(),
        "ratio_mean": ratios.mean().detach(),
    }
    return loss, metrics
