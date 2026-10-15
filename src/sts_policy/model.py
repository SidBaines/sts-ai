"""Shared candidate actor and critic; action indices are deliberately absent."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from sts_policy.encoding import STATE_DIM, ACTION_DIM


@dataclass(frozen=True)
class ModelConfig:
    size: str = "small"
    context: str = "mean"
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM


class CandidateActorCritic(nn.Module):
    """Small/medium/large MLP and optional permutation-invariant menu context.

    The same scorer evaluates every legal candidate. Mean candidate embeddings
    provide a cheap DeepSets context without quadratic attention overhead.
    """

    def __init__(self, size: str = "small", context: str = "mean", *,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        if size not in {"small", "medium", "large"}:
            raise ValueError("size must be small, medium, or large")
        if context not in {"none", "mean"}:
            raise ValueError("context must be none or mean")
        self.config = ModelConfig(size, context, state_dim, action_dim)
        width, depth = {"small": (128, 2), "medium": (256, 2), "large": (512, 3)}[size]
        def mlp(input_dim: int) -> nn.Sequential:
            layers = [nn.Linear(input_dim, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers.extend([nn.Linear(width, width), nn.SiLU()])
            return nn.Sequential(*layers)
        self.state_encoder = mlp(state_dim)
        self.action_encoder = mlp(action_dim)
        scorer_input = width * (4 if context == "mean" else 3)
        self.scorer = nn.Sequential(nn.Linear(scorer_input, width), nn.SiLU(), nn.Linear(width, 1))
        self.critic = nn.Sequential(nn.Linear(width * (2 if context == "mean" else 1), width),
                                    nn.SiLU(), nn.Linear(width, 1))

    def forward(self, state: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        return self._forward(state, actions, mask, actor_only=False)

    def actor_logits(self, state, actions, mask):
        """Identical actor arithmetic without the unused legacy critic."""
        return self._forward(state, actions, mask, actor_only=True)[0]

    def _forward(self, state, actions, mask, *, actor_only):
        if state.ndim != 2 or actions.ndim != 3 or mask.shape != actions.shape[:2]:
            raise ValueError("Expected state[B,D], actions[B,N,A], mask[B,N]")
        state_embedding = self.state_encoder(state)
        action_embedding = self.action_encoder(actions)
        state_candidates = state_embedding.unsqueeze(1).expand_as(action_embedding)
        score_features = [state_candidates, action_embedding, state_candidates * action_embedding]
        critic_features = [state_embedding]
        if self.config.context == "mean":
            weights = mask.unsqueeze(-1).to(action_embedding.dtype)
            context = (action_embedding * weights).sum(1) / weights.sum(1).clamp_min(1)
            score_features.append(context.unsqueeze(1).expand_as(action_embedding))
            critic_features.append(context)
        logits = self.scorer(torch.cat(score_features, -1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        value = (state.new_zeros(len(state)) if actor_only else
                 self.critic(torch.cat(critic_features, -1)).squeeze(-1))
        return logits, value

    def metadata(self) -> dict:
        return asdict(self.config)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
