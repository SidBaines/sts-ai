"""Behavior-preserving residual actor and separate typed critic for OOC learning."""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from . import entities
from .legacy_bridge import BRIDGE_VERSION, project_legacy_menu
from .model_agent import policy_from_config, policy_from_payload
from .training import load_checkpoint, save_checkpoint, file_identity
from pathlib import Path

MODEL_VERSION = "typed_residual_ooc_v2"


def code_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {"typed_encoder_sha256": file_identity(root / "entities.py"),
            "typed_model_sha256": file_identity(__file__),
            "legacy_bridge_sha256": file_identity(root / "legacy_bridge.py")}


def mlp(inputs: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())


class EntityTower(nn.Module):
    """Uncapped DeepSets entities and direct source/target candidate relations."""
    def __init__(self, width: int = 128, embedding: int = 16) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(len(entities.TOKENS), embedding, padding_idx=0)
                                         for _ in range(4)])
        self.entity_encoder = mlp(4 * embedding + entities.ENTITY_DIM, width)
        # Keep the old controlled public semantics available to the trainable
        # residual, including screen/selection purpose and complex event prose.
        # Typed entities add distinctions that the historical hashes discarded.
        self.action_encoder = mlp(4 * embedding + entities.ACTION_DIM + 128, width)
        self.state_encoder = mlp(2 * width + entities.GLOBAL_DIM + 256, width)

    def embedded(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([embedding(ids[..., i]) for i, embedding in enumerate(self.embeddings)], -1)

    def forward(self, global_values, entity_ids, entity_values, weights, action_ids, action_values,
                links, mask, legacy_state, legacy_actions):
        entity = self.entity_encoder(torch.cat((self.embedded(entity_ids), entity_values), -1))
        weighted = entity * weights.unsqueeze(-1)
        mean = weighted.sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        total = weighted.sum(1) / 20
        state = self.state_encoder(torch.cat((mean, total, global_values, legacy_state), -1))
        actions = self.action_encoder(torch.cat((self.embedded(action_ids), action_values, legacy_actions), -1))
        menu_weights = mask.unsqueeze(-1).to(actions.dtype)
        menu = (actions * menu_weights).sum(1) / menu_weights.sum(1).clamp_min(1)
        # Membership masks only perform joins. Visible source indices are absent.
        relations = torch.einsum("bnre,bew->bnrw", links, entity)
        relations = relations / links.sum(-1, keepdim=True).clamp_min(1)
        return state, actions, menu, relations[:, :, 0], relations[:, :, 1]


class ResidualOOCPolicy(nn.Module):
    def __init__(self, prior_config: dict[str, Any], width: int = 128, embedding: int = 16) -> None:
        super().__init__()
        if width < 16 or embedding < 1:
            raise ValueError("invalid typed residual dimensions")
        self.prior = policy_from_config(prior_config)
        self.prior.requires_grad_(False)
        self.actor = EntityTower(width, embedding)
        self.actor_score = nn.Sequential(nn.Linear(6 * width, width), nn.SiLU(), nn.Linear(width, 1))
        # Critic parameters are completely separate from actor/prior parameters.
        self.value = EntityTower(width, embedding)
        self.value_score = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.actor_score[-1].weight)
        nn.init.zeros_(self.actor_score[-1].bias)
        nn.init.zeros_(self.value_score[-1].weight)
        nn.init.zeros_(self.value_score[-1].bias)
        self.config = {"kind": MODEL_VERSION, "prior_config": prior_config, "width": width,
                       "embedding": embedding, "encoder_version": entities.ENCODER_VERSION,
                       "vocabulary_sha256": entities.VOCABULARY_SHA256, "bridge_version": BRIDGE_VERSION}

    def metadata(self) -> dict[str, Any]:
        return dict(self.config)

    def actor_parameters(self):
        return [*self.actor.parameters(), *self.actor_score.parameters()]

    def critic_parameters(self):
        return [*self.value.parameters(), *self.value_score.parameters()]

    def train(self, mode: bool = True):
        super().train(mode)
        self.prior.eval()
        return self

    def forward(self, legacy_state, legacy_actions, mask, global_values, entity_ids,
                entity_values, weights, action_ids, action_values, links, representatives, prior_mask):
        with torch.no_grad():
            prior_logits, prior_value = self.prior(legacy_state, legacy_actions, prior_mask)
            prior_logits = prior_logits.gather(1, representatives)
        typed = (global_values, entity_ids, entity_values, weights, action_ids, action_values, links, mask,
                 legacy_state, legacy_actions)
        state, actions, menu, sources, targets = self.actor(*typed)
        expanded = state.unsqueeze(1).expand_as(actions)
        residual = self.actor_score(torch.cat((expanded, actions, expanded * actions,
                                               menu.unsqueeze(1).expand_as(actions), sources, targets), -1)).squeeze(-1)
        value_state, _, value_menu, _, _ = self.value(*typed)
        value_residual = self.value_score(torch.cat((value_state, value_menu), -1)).squeeze(-1)
        # Combat actions remain the immutable prior. The separate typed critic
        # can learn microdecision values without moving any combat actor/prior.
        ooc = 1 - global_values[:, 0]
        logits = (prior_logits + residual * ooc.unsqueeze(-1)).masked_fill(~mask, -1e9)
        return logits, prior_value + value_residual


def forward_policy(model: nn.Module, batch: Any):
    logits, values = model(*batch)
    return logits, values, batch[2]


def initialize_from_legacy(checkpoint: str, *, width: int = 128, embedding: int = 16) -> tuple[ResidualOOCPolicy, dict[str, Any]]:
    before = file_identity(checkpoint)
    payload = load_checkpoint(checkpoint)
    prior = policy_from_payload(payload)
    if file_identity(checkpoint) != before:
        raise ValueError("legacy source checkpoint changed while loading")
    model = ResidualOOCPolicy(payload["model_config"], width, embedding)
    model.prior.load_state_dict(prior.state_dict(), strict=True)
    return model, {"prior_checkpoint": str(checkpoint), "prior_sha256": before,
                   "prior_environment_provenance": payload["metadata"].get("training_environment"),
                   "transfer": "frozen legacy projection plus initially zero typed OOC residual",
                   "stochastic_duplicate_note": "Combat uses deduplicated greedy prior; OOC preserves every offer."}


def load_policy(checkpoint: str) -> tuple[ResidualOOCPolicy, dict[str, Any]]:
    payload = load_checkpoint(checkpoint)
    config = dict(payload["model_config"])
    expected = {"kind": MODEL_VERSION, "encoder_version": entities.ENCODER_VERSION,
                "vocabulary_sha256": entities.VOCABULARY_SHA256, "bridge_version": BRIDGE_VERSION}
    for key, value in expected.items():
        if config.pop(key, None) != value:
            raise ValueError(f"typed checkpoint identity mismatch: {key}")
    if set(config) != {"prior_config", "width", "embedding"}:
        raise ValueError("unexpected typed model configuration")
    if payload["metadata"].get("encoder_version") != entities.ENCODER_VERSION:
        raise ValueError("typed checkpoint encoder metadata mismatch")
    if payload["metadata"].get("policy_code_identity") != code_identity():
        raise ValueError("typed checkpoint policy code identity mismatch; migrate explicitly")
    model = ResidualOOCPolicy(**config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def save_policy(path: str, model: ResidualOOCPolicy, *, metadata: dict[str, Any],
                optimizer=None, step: int = 0) -> None:
    save_checkpoint(path, model, model_config=model.metadata(), optimizer=optimizer, step=step,
                    metadata={**metadata, "encoder_version": entities.ENCODER_VERSION,
                              "model_kind": MODEL_VERSION, "policy_code_identity": code_identity()})


class EntityAgent:
    """Greedy public CLI agent; avoids typed encoding for immutable combat."""
    def __init__(self, checkpoint: str, device: str = "cpu", seed: int = 0) -> None:
        self.model, _ = load_policy(checkpoint)
        self.model.to(device).eval()
        self.device = device

    def reset(self, seed: int) -> None:
        # Greedy inference has no sampling/global RNG side effects.
        pass

    @torch.no_grad()
    def choose_actions(self, observations: Sequence[dict[str, Any]]) -> list[int]:
        from .encoding import encode_observation as legacy_encode, collate_observations as legacy_collate
        results = [0] * len(observations)
        for combat in (True, False):
            rows = [i for i, obs in enumerate(observations) if (obs.get("phase") == "combat") == combat]
            if not rows:
                continue
            chosen = [observations[i] for i in rows]
            if combat:
                projected = [project_legacy_menu(obs) for obs in chosen]
                logits, _ = self.model.prior(*legacy_collate([legacy_encode(p.observation) for p in projected], self.device))
                picks = [p.current_indices[position] for p, position in zip(projected, logits.argmax(-1).cpu().tolist())]
            else:
                logits, _ = self.model(*entities.collate_public(chosen, self.device))
                picks = [int(obs["legal_actions"][position]["index"]) for obs, position in zip(chosen, logits.argmax(-1).cpu().tolist())]
            for i, pick in zip(rows, picks):
                results[i] = pick
        return results

    def choose_action(self, observation: dict[str, Any]) -> int:
        return self.choose_actions([observation])[0]


def make_agent(checkpoint: str, device: str = "cpu", seed: int = 0) -> EntityAgent:
    return EntityAgent(checkpoint, device, seed)
