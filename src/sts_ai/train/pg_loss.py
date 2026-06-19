"""Policy-gradient loss helpers for clipped GRPO-style updates."""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["grpo_loss", "selective_logps"]


def selective_logps(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Return log-probs for each token under the preceding causal position."""
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:].unsqueeze(-1)
    return torch.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=shifted_targets,
    ).squeeze(-1)


def _float_metric(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def grpo_loss(
    logp_new: Tensor,
    logp_old: Tensor,
    logp_ref: Tensor | None,
    advantages: Tensor,
    completion_mask: Tensor,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
    loss_type: str = "grpo",
) -> tuple[Tensor, dict[str, float]]:
    """Compute the clipped policy-gradient loss with optional Schulman k3 KL."""
    if loss_type != "grpo":
        raise ValueError(f"unsupported loss_type: {loss_type!r}")

    mask = completion_mask.to(dtype=logp_new.dtype)
    denom = mask.sum().clamp(min=1.0)
    ratio = torch.exp(logp_new - logp_old)
    adv = advantages.unsqueeze(1)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg = torch.minimum(ratio * adv, clipped_ratio * adv)

    if kl_beta > 0 and logp_ref is not None:
        log_ratio_ref_new = logp_ref - logp_new
        kl = torch.exp(log_ratio_ref_new) - log_ratio_ref_new - 1.0
        per_token = pg - kl_beta * kl
    else:
        kl = torch.zeros_like(logp_new)
        per_token = pg

    loss = -((per_token * mask).sum() / denom)
    clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).to(
        dtype=logp_new.dtype,
    )
    metrics: dict[str, float] = {
        "mean_kl": _float_metric((kl * mask).sum() / denom),
        "mean_ratio": _float_metric((ratio * mask).sum() / denom),
        "clip_fraction": _float_metric((clipped * mask).sum() / denom),
        "mean_advantage": _float_metric(advantages.mean()),
    }
    return loss, metrics
