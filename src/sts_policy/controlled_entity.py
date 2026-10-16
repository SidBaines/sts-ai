"""Isolated feature/capacity controls; historical entity checkpoints stay intact.

Actor arithmetic is the original entity actor with one extra action-embedding
summand. ON/OFF both own a zero, bias-free 128->width projection; OFF feeds zero
and freezes that dormant projection. Critic parameters never read it. Source
and configuration identities are mandatory, with no historical exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from sts_policy import controlled_encoding, entity_encoding, entity_model
from sts_policy.entity_model import EntityActorCritic, EntityModelConfig

MODEL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ENTITY_MODEL_SHA256 = hashlib.sha256(Path(entity_model.__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ControlledEntityModelConfig(EntityModelConfig):
    legacy_action_features: bool = False


class _ControlledActor(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        # Reuse the just-initialized parent actor modules without reinitializing
        # them or changing shared state_dict keys. No hooks or monkeypatches.
        self.tower = original.tower
        self.action_numeric = original.action_numeric
        self.text = original.text
        self.score = original.score
        self.legacy_action_features = config.legacy_action_features
        # Linear's default initializer must not move the base model RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.legacy_action_projection = nn.Linear(128, config.width, bias=False)
        nn.init.zeros_(self.legacy_action_projection.weight)
        self.legacy_action_projection.requires_grad_(config.legacy_action_features)

    def legacy_embedding(self, features):
        if not self.legacy_action_features:
            features = torch.zeros_like(features)
        return self.legacy_action_projection(features)

    def forward(self, row):
        entities, state = self.tower.encode(row)
        action = (self.tower.identity(row['action_ids']) + self.tower.action_type(row['action_types'])
                  + self.tower.identity(row['action_aux']).sum(-2)
                  + self.action_numeric(row['action_numeric']) + self.text(row['action_text']))
        action = action + self.legacy_embedding(row['legacy_action_features'])
        members = row['action_source_members']
        mask = row['action_source_mask'].unsqueeze(-1)
        source = (entities[members.clamp_min(0)] * mask).sum(1) / mask.sum(1).clamp_min(1)
        targets = row['action_target']
        target = entities[targets.clamp_min(0)] * (targets >= 0).unsqueeze(-1)
        features = torch.cat((action, source, target, state.expand_as(action)), -1)
        return self.score(features).squeeze(-1)


class ControlledEntityActorCritic(EntityActorCritic):
    def __init__(self, width=128, layers=2, heads=4, dropout=0.0, phase='combat',
                 legacy_action_features=False):
        if type(legacy_action_features) is not bool:
            raise ValueError('legacy_action_features must be boolean')
        super().__init__(width=width, layers=layers, heads=heads, dropout=dropout, phase=phase)
        self.config = ControlledEntityModelConfig(width, layers, heads, dropout, phase, legacy_action_features)
        self.actor = _ControlledActor(self.actor, self.config)

    @staticmethod
    def _rows(batch):
        if batch['legacy_action_features'].shape != (*batch['action_mask'].shape, 128):
            raise ValueError('Expected legacy_action_features [B,A,128]')
        for i, row in enumerate(EntityActorCritic._rows(batch)):
            row['legacy_action_features'] = batch['legacy_action_features'][i, :len(row['action_ids'])]
            yield row

    def training_logits(self, batch):
        """Original vectorized arithmetic plus the controlled action summand."""
        b, tower = batch, self.actor.tower
        entities = (tower.identity(b['entity_ids']) + tower.entity_type(b['entity_types'])
                    + tower.identity(b['entity_aux']).sum(-2) + tower.numeric(b['entity_numeric']))
        size, count, width = entities.shape
        powers = tower.power_value(torch.cat((tower.power(b['power_ids']),
            b['power_amounts'].unsqueeze(-1), b['power_amount_available'].unsqueeze(-1).to(entities.dtype)), -1))
        owners = b['power_owner'].clamp_min(0) + torch.arange(size, device=entities.device)[:, None] * count
        additions = torch.zeros_like(entities).reshape(-1, width).index_add(
            0, owners.reshape(-1), (powers * b['power_mask'].unsqueeze(-1)).reshape(-1, width))
        entities = tower.attention(entities + additions.reshape_as(entities),
                                   src_key_padding_mask=~b['entity_mask'])
        mask = b['entity_mask'].unsqueeze(-1)
        pooled = (entities * mask).sum(1) / mask.sum(1).clamp_min(1)
        recent = (tower.identity(b['recent_ids']) + tower.action_type(b['recent_types'])
                  + tower.numeric(b['recent_numeric']))
        history, _ = tower.recent(recent)
        lengths = b['recent_mask'].sum(1)
        context = history[torch.arange(size, device=entities.device), (lengths - 1).clamp_min(0)]
        context = context * (lengths > 0).unsqueeze(-1)
        state = tower.pool(torch.cat((pooled, context), -1))
        action = (tower.identity(b['action_ids']) + tower.action_type(b['action_types'])
                  + tower.identity(b['action_aux']).sum(-2) + self.actor.action_numeric(b['action_numeric'])
                  + self.actor.text(b['action_text']))
        action = action + self.actor.legacy_embedding(b['legacy_action_features'])
        rows = torch.arange(size, device=entities.device)[:, None, None]
        members = entities[rows, b['action_source_members'].clamp_min(0)]
        member_mask = b['action_source_mask'].unsqueeze(-1)
        source = (members * member_mask).sum(2) / member_mask.sum(2).clamp_min(1)
        target = entities[rows.squeeze(-1), b['action_target'].clamp_min(0)]
        target = target * (b['action_target'] >= 0).unsqueeze(-1)
        features = torch.cat((action, source, target, state[:, None].expand_as(action)), -1)
        return self.actor.score(features).squeeze(-1).masked_fill(~b['action_mask'], -torch.inf)

    def parameter_counts(self):
        policy = sum(p.numel() for p in self.actor_parameters())
        dormant = 0 if self.config.legacy_action_features else self.actor.legacy_action_projection.weight.numel()
        return {'parameter_count': sum(p.numel() for p in self.parameters()),
                'trainable_parameter_count': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'policy_parameter_count': policy, 'policy_active_parameter_count': policy - dormant,
                'policy_dormant_parameter_count': dormant,
                'policy_trainable_parameter_count': sum(p.numel() for p in self.actor_parameters() if p.requires_grad)}

    def metadata(self):
        meta = super().metadata()
        meta.update(_source_identity())
        meta.update(self.parameter_counts())
        meta['model_config_sha256'] = _config_identity(meta['model_config'])
        return meta


def _source_identity():
    return {'kind': 'entity_controlled_v1',
            'encoder_version': controlled_encoding.ENCODER_VERSION,
            'encoder_sha256': controlled_encoding.ENCODER_SHA256,
            'controlled_model_sha256': MODEL_SHA256,
            'entity_encoder_version': entity_encoding.ENCODER_VERSION,
            'entity_encoder_sha256': entity_encoding.ENCODER_SHA256,
            'entity_model_sha256': ENTITY_MODEL_SHA256,
            'legacy_encoder_sha256': controlled_encoding.LEGACY_ENCODER_SHA256,
            'vocabulary_sha256': entity_encoding.VOCABULARY_SHA256,
            'vocabulary_size': entity_encoding.VOCAB_SIZE}


def _config_identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def policy_from_payload(payload: dict) -> ControlledEntityActorCritic:
    """Native/trainer-free strict inference loader; never converts old kinds."""
    meta = payload['metadata']
    for key, value in _source_identity().items():
        if meta.get(key) != value:
            raise ValueError(f'incompatible controlled checkpoint {key}: {meta.get(key)!r}')
    config = meta['model_config']
    if set(config) != set(ControlledEntityModelConfig.__dataclass_fields__):
        raise ValueError('incompatible controlled checkpoint model_config fields')
    if meta.get('model_config_sha256') != _config_identity(config):
        raise ValueError('incompatible controlled checkpoint model_config_sha256')
    # save_checkpoint duplicates model metadata at the top level. Reject split
    # identities rather than silently trusting one copy or a row-side condition.
    if 'model_config' in payload and payload['model_config'].get('model_config') != config:
        raise ValueError('incompatible controlled checkpoint model_config copies')
    training = meta.get('config')
    if training is not None:
        if training.get('architecture') != 'controlled':
            raise ValueError('incompatible controlled checkpoint training architecture')
        for key in ('phase', 'width', 'layers', 'heads', 'legacy_action_features'):
            if key not in training or type(training[key]) is not type(config[key]) or training[key] != config[key]:
                raise ValueError(f'incompatible controlled checkpoint training config {key}')
    model = ControlledEntityActorCritic(**config)
    model.load_state_dict(payload['model_state'], strict=True)
    return model


def describe_capacity(width=128, *, legacy_action_features=False, layers=2, heads=4):
    """Native/data-free counts without allocating weights or consuming RNG."""
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        model = ControlledEntityActorCritic(width=width, layers=layers, heads=heads,
                                           legacy_action_features=legacy_action_features)
    for p in model.critic_parameters():
        p.requires_grad_(False)
    return {'width': width, 'layers': layers, 'heads': heads,
            'legacy_action_features': legacy_action_features, **model.parameter_counts()}


def nearest_policy_width(target=2_035_713, *, legacy_action_features=False,
                         layers=2, heads=4, maximum_width=1024):
    """Count-only nearest active policy capacity on multiples of 8; no selection.

    Returns the nearest width (ties choose smaller) and adjacent counts. Root
    must fix the scientific width before fitting; this helper reads no data.
    """
    if type(target) is not int or target < 1 or maximum_width < 8:
        raise ValueError('positive target and maximum_width >= 8 required')
    counts = [describe_capacity(w, legacy_action_features=legacy_action_features, layers=layers, heads=heads)
              for w in range(8, maximum_width + 1, 8) if w % heads == 0]
    if not counts:
        raise ValueError('no widths divisible by heads in the requested range')
    best = min(range(len(counts)), key=lambda i: (abs(counts[i]['policy_active_parameter_count'] - target), counts[i]['width']))
    return {'target_policy_parameters': target, 'metric': 'policy_active_parameter_count',
            'nearest': counts[best], 'neighbors': counts[max(0, best-1):best+2]}
