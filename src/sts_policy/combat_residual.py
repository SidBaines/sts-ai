"""A separately versioned combat residual over an immutable strategic policy.

Native combat supervision can improve tactical play without moving any OOC or
original-prior weights. Publicly indistinguishable candidates share BC targets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from . import entities
from .entity_model import EntityTower, ResidualOOCPolicy, MODEL_VERSION as BASE_VERSION, code_identity as base_code_identity, load_policy as load_base
from .legacy_bridge import BRIDGE_VERSION
from .training import load_checkpoint, save_checkpoint, file_identity

MODEL_VERSION = "typed_combat_residual_v2"


def code_identity() -> dict[str, str]:
    return {**base_code_identity(), "combat_residual_sha256": file_identity(__file__)}


def _base_from_config(config: dict[str, Any]) -> ResidualOOCPolicy:
    options = dict(config)
    expected = {"kind": BASE_VERSION, "encoder_version": entities.ENCODER_VERSION,
                "vocabulary_sha256": entities.VOCABULARY_SHA256, "bridge_version": BRIDGE_VERSION}
    for key, value in expected.items():
        if options.pop(key, None) != value:
            raise ValueError(f"combat base identity mismatch: {key}")
    if set(options) != {"prior_config", "width", "embedding"}:
        raise ValueError("unexpected combat base configuration")
    return ResidualOOCPolicy(**options)


class CombatResidualPolicy(nn.Module):
    def __init__(self, base: ResidualOOCPolicy, width: int = 128, embedding: int = 16) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.combat_actor = EntityTower(width, embedding)
        self.combat_score = nn.Sequential(nn.Linear(6*width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.combat_score[-1].weight)
        nn.init.zeros_(self.combat_score[-1].bias)
        self.config = {"kind": MODEL_VERSION, "base": base.metadata(), "width": width,
                       "embedding": embedding, "encoder_version": entities.ENCODER_VERSION,
                       "vocabulary_sha256": entities.VOCABULARY_SHA256}

    @property
    def prior(self):
        return self.base.prior

    def metadata(self):
        return dict(self.config)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def combat_parameters(self):
        return [*self.combat_actor.parameters(), *self.combat_score.parameters()]

    def forward(self, *batch):
        with torch.no_grad():
            logits, values = self.base(*batch)
        typed = (*batch[3:10], batch[2], batch[0], batch[1])
        state, actions, menu, sources, targets = self.combat_actor(*typed)
        expanded = state.unsqueeze(1).expand_as(actions)
        residual = self.combat_score(torch.cat((expanded, actions, expanded*actions,
            menu.unsqueeze(1).expand_as(actions), sources, targets), -1)).squeeze(-1)
        logits = (logits+residual*batch[3][:, 0, None]).masked_fill(~batch[2], -1e9)
        return logits, values


def initialize(checkpoint: str, *, seed: int = 0) -> tuple[CombatResidualPolicy, dict]:
    torch.manual_seed(seed)
    base, payload = load_base(checkpoint)
    model = CombatResidualPolicy(base)
    return model, {"base_checkpoint": checkpoint, "base_sha256": file_identity(checkpoint),
                   "base_metadata": payload["metadata"], "training_environment": payload["metadata"].get("training_environment"),
                   "initialization": "immutable strategic policy plus zero combat residual", "seed": seed}


def save_policy(path: str, model: CombatResidualPolicy, *, metadata: dict,
                optimizer=None, step=0):
    save_checkpoint(path, model, model_config=model.metadata(), optimizer=optimizer, step=step,
        metadata={**metadata, "encoder_version": entities.ENCODER_VERSION, "model_kind": MODEL_VERSION,
                  "combat_policy_code_identity": code_identity()})


def load_policy(checkpoint: str) -> tuple[CombatResidualPolicy, dict]:
    payload = load_checkpoint(checkpoint)
    config = dict(payload["model_config"])
    for key, expected in {"kind": MODEL_VERSION, "encoder_version": entities.ENCODER_VERSION,
                          "vocabulary_sha256": entities.VOCABULARY_SHA256}.items():
        if config.pop(key, None) != expected:
            raise ValueError(f"combat checkpoint identity mismatch: {key}")
    if payload["metadata"].get("combat_policy_code_identity") != code_identity():
        raise ValueError("combat checkpoint implementation identity mismatch")
    base = _base_from_config(config.pop("base"))
    if set(config) != {"width", "embedding"}:
        raise ValueError("unexpected combat residual configuration")
    model = CombatResidualPolicy(base, **config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def equivalent_positions(encoded: entities.EncodedEntities, teacher_position: int) -> np.ndarray:
    """Exactly equal inputs to this scorer, excluding routing-only source IDs.

    Source/target relation bags retain every public entity field and multiplicity.
    A free, retained, upgraded or otherwise distinct card is never collapsed.
    This is conservative model indistinguishability, not a native-future oracle.
    """
    def signature(position):
        relations = []
        for group in (0, 1):
            members = np.flatnonzero(encoded.action_links[position, group])
            relations.append(tuple(sorted((encoded.entity_ids[i].tobytes(), encoded.entity_values[i].tobytes())
                                           for i in members)))
        return (encoded.legacy_actions[position].tobytes(), encoded.action_ids[position].tobytes(),
                encoded.action_values[position].tobytes(), *relations)
    teacher = signature(teacher_position)
    return np.asarray([signature(i) == teacher for i in range(len(encoded.action_indices))], np.bool_)


def equivalence_loss(logits, mask, targets):
    if targets.shape != mask.shape or not bool((targets & mask).any(-1).all()):
        raise ValueError("every teacher target needs a legal equivalent candidate")
    log_prob = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
    return -torch.logsumexp(log_prob.masked_fill(~targets, -1e9), -1).mean()


class CombatAgent:
    def __init__(self, checkpoint: str, device="cpu", seed=0):
        self.model, _ = load_policy(checkpoint)
        self.model.to(device).eval()
        self.device = device

    def reset(self, seed: int):
        pass

    @torch.no_grad()
    def choose_actions(self, observations: Sequence[dict[str, Any]]):
        logits, _ = self.model(*entities.collate_public(observations, self.device))
        return [int(obs["legal_actions"][position]["index"])
                for obs, position in zip(observations, logits.argmax(-1).cpu().tolist())]

    def choose_action(self, observation):
        return self.choose_actions([observation])[0]


def make_agent(checkpoint: str, device="cpu", seed=0):
    return CombatAgent(checkpoint, device, seed)
