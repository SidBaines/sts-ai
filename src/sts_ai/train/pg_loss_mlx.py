"""MLX policy-gradient loss helpers for clipped GRPO-style updates."""
from __future__ import annotations

import mlx.core as mx

__all__ = ["grpo_loss", "selective_logps"]


def selective_logps(logits: mx.array, input_ids: mx.array) -> mx.array:
    """Return log-probs for each token under the preceding causal position."""
    logits_f32 = logits[:, :-1, :].astype(mx.float32)
    targets = input_ids[:, 1:]
    selected = mx.take_along_axis(logits_f32, targets[..., None], axis=-1).squeeze(-1)
    return selected - mx.logsumexp(logits_f32, axis=-1)


def _float_metric(value: mx.array) -> float:
    return float(value.item())


def grpo_loss(
    logp_new: mx.array,
    logp_old: mx.array,
    logp_ref: mx.array | None,
    advantages: mx.array,
    completion_mask: mx.array,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
    loss_type: str = "grpo",
) -> tuple[mx.array, dict[str, float]]:
    """Return the clipped GRPO loss and scalar training metrics."""
    if loss_type != "grpo":
        raise ValueError(f"unsupported loss_type: {loss_type!r}")
    mask = completion_mask.astype(logp_new.dtype)
    mask_f32 = mask.astype(mx.float32)
    denom = mx.maximum(mask_f32.sum(), 1.0)
    ratio = mx.exp(logp_new - logp_old)
    adv = advantages[:, None]
    clipped_ratio = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg = mx.minimum(ratio * adv, clipped_ratio * adv)
    if kl_beta > 0 and logp_ref is not None:
        log_ratio_ref_new = logp_ref - logp_new
        kl = mx.exp(log_ratio_ref_new) - log_ratio_ref_new - 1.0
        per_token = pg - kl_beta * kl
    else:
        kl = mx.zeros_like(logp_new)
        per_token = pg
    loss = -((per_token.astype(mx.float32) * mask_f32).sum() / denom)
    clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).astype(
        logp_new.dtype
    )
    metrics = {
        "mean_kl": _float_metric((kl.astype(mx.float32) * mask_f32).sum() / denom),
        "mean_ratio": _float_metric(
            (ratio.astype(mx.float32) * mask_f32).sum() / denom
        ),
        "clip_fraction": _float_metric(
            (clipped.astype(mx.float32) * mask_f32).sum() / denom
        ),
        "mean_advantage": _float_metric(advantages.astype(mx.float32).mean()),
    }
    return loss, metrics
